"""
RunPod Serverless handler for PuLID-Flux identity-preserving image generation.

Generates a scene image where the person from a reference face photo
is placed into a new scene described by the prompt, preserving their identity.

Uses: PuLID v0.9.1 + Flux.1-schnell (Apache-2.0, commercial-safe)
      InsightFace antelopev2 for face embedding (requires commercial license for
      high-volume commercial use — see https://www.insightface.ai/services/models-commercial-licensing)

Accepts:
  {
    "face_image_url":   "https://...",      # reference portrait (JPG/PNG)
    "prompt":           "...",              # scene description
    "negative_prompt":  "...",              # optional
    "width":            1024,               # default 1024
    "height":           1024,               # default 1024
    "seed":             42,                 # optional, -1 = random
    "num_steps":        4,                  # schnell default 4, dev 20-25
    "guidance_scale":   1.0,               # schnell uses 1.0, dev 3.5-4.5
    "id_weight":        0.8                 # PuLID identity weight 0-1, default 0.8
  }

Returns:
  {"image_url": "https://...", "seed": 42}

ENV:
  HF_TOKEN             — HuggingFace token (required for model download)
  MODELS_DIR           — weights directory (default /workspace/models)
  R2_ENDPOINT_URL      — Cloudflare R2 endpoint
  R2_ACCESS_KEY_ID     — R2 access key
  R2_SECRET_ACCESS_KEY — R2 secret key
  R2_BUCKET            — R2 bucket name (default: vidgen-media)
  R2_CDN_BASE          — CDN base URL (e.g. https://cdn.vidgen-ai.com)
"""

import os
import sys
import time
import uuid
import json
import logging
import urllib.request
import urllib.error
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
)
logger = logging.getLogger("pulid_handler")

import runpod  # noqa: E402

# ── R2 upload ─────────────────────────────────────────────────────────────────

_R2_ENDPOINT = os.environ.get("R2_ENDPOINT_URL", "")
_R2_BUCKET   = os.environ.get("R2_BUCKET", "vidgen-media")
_R2_KEY_ID   = os.environ.get("R2_ACCESS_KEY_ID", "")
_R2_KEY_SEC  = os.environ.get("R2_SECRET_ACCESS_KEY", "")
_CDN_BASE    = os.environ.get("R2_CDN_BASE", "")

_r2 = None
if all([_R2_ENDPOINT, _R2_KEY_ID, _R2_KEY_SEC]):
    try:
        import boto3
        _r2 = boto3.client(
            "s3",
            endpoint_url=_R2_ENDPOINT,
            aws_access_key_id=_R2_KEY_ID,
            aws_secret_access_key=_R2_KEY_SEC,
            region_name="auto",
        )
        logger.info("R2 client initialized — bucket=%s", _R2_BUCKET)
    except Exception as e:
        logger.warning("R2 init failed: %s", e)


def _upload_r2(local_path: str, key: str, content_type: str = "image/jpeg") -> str:
    if _r2 is None:
        raise RuntimeError("R2 not configured")
    _r2.upload_file(
        local_path, _R2_BUCKET, key,
        ExtraArgs={"ContentType": content_type},
    )
    if _CDN_BASE:
        return f"{_CDN_BASE.rstrip('/')}/{key}"
    return _r2.generate_presigned_url(
        "get_object",
        Params={"Bucket": _R2_BUCKET, "Key": key},
        ExpiresIn=604800,
    )


# ── Model loading ─────────────────────────────────────────────────────────────

_MODELS_DIR = os.environ.get("MODELS_DIR", "/workspace/models")
_HF_TOKEN   = os.environ.get("HF_TOKEN", "")

_pipeline = None
_face_helper = None


def _ensure_models():
    """Download models on cold start if not present in network volume."""
    os.makedirs(_MODELS_DIR, exist_ok=True)

    # 1. PuLID weights
    pulid_path = os.path.join(_MODELS_DIR, "pulid", "pulid_flux_v0.9.1.safetensors")
    if not os.path.exists(pulid_path):
        logger.info("Downloading PuLID weights…")
        from huggingface_hub import hf_hub_download
        kwargs = dict(repo_id="guozinan/PuLID", filename="pulid_flux_v0.9.1.safetensors",
                      local_dir=os.path.join(_MODELS_DIR, "pulid"))
        if _HF_TOKEN:
            kwargs["token"] = _HF_TOKEN
        hf_hub_download(**kwargs)
        logger.info("PuLID weights downloaded")

    # 2. Flux.1-schnell (Apache-2.0, commercial-safe)
    schnell_dir = os.path.join(_MODELS_DIR, "flux1-schnell")
    if not os.path.isdir(schnell_dir) or not os.listdir(schnell_dir):
        logger.info("Downloading Flux.1-schnell (~24GB)…")
        from huggingface_hub import snapshot_download
        kwargs = dict(repo_id="black-forest-labs/FLUX.1-schnell",
                      local_dir=schnell_dir,
                      ignore_patterns=["*.md", "*.txt", "*.gitattributes"])
        if _HF_TOKEN:
            kwargs["token"] = _HF_TOKEN
        snapshot_download(**kwargs)
        logger.info("Flux.1-schnell downloaded")

    # 3. InsightFace antelopev2 (required by PuLID for face embedding)
    antelope_dir = os.path.join(_MODELS_DIR, "antelopev2")
    if not os.path.isdir(antelope_dir) or not os.listdir(antelope_dir):
        logger.info("Downloading InsightFace antelopev2…")
        from huggingface_hub import snapshot_download
        kwargs = dict(repo_id="DIAMONIK7777/antelopev2",
                      local_dir=antelope_dir)
        snapshot_download(**kwargs)
        logger.info("antelopev2 downloaded")

    logger.info("All models ready in %s", _MODELS_DIR)


def _load_pipeline():
    """Load PuLID + Flux.1-schnell pipeline into GPU (once per worker)."""
    global _pipeline, _face_helper

    if _pipeline is not None:
        return _pipeline, _face_helper

    _ensure_models()

    import torch
    from diffusers import FluxPipeline
    from pulid import attention_processor as attention
    from pulid.pipeline_flux import PuLIDPipeline

    logger.info("Loading Flux.1-schnell pipeline…")
    t0 = time.time()

    schnell_dir = os.path.join(_MODELS_DIR, "flux1-schnell")
    dtype = torch.bfloat16

    pipe = FluxPipeline.from_pretrained(
        schnell_dir,
        torch_dtype=dtype,
    ).to("cuda")

    # Load PuLID adapter
    pulid_ckpt = os.path.join(_MODELS_DIR, "pulid", "pulid_flux_v0.9.1.safetensors")
    pulid_pipe = PuLIDPipeline(pipe)
    pulid_pipe.load_pretrain(pulid_ckpt)

    # InsightFace face encoder
    import insightface
    face_analysis = insightface.app.FaceAnalysis(
        name="antelopev2",
        root=os.path.dirname(_MODELS_DIR),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    _pipeline = pulid_pipe
    _face_helper = face_analysis
    logger.info("PuLID-Flux loaded in %.1fs", time.time() - t0)
    return _pipeline, _face_helper


# ── Inference ─────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(dest, "wb") as f:
            f.write(resp.read())
    return dest


def _run_inference(
    face_image_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    num_steps: int,
    guidance_scale: float,
    id_weight: float,
    out_path: str,
) -> str:
    """Run PuLID inference. Returns output image path."""
    import torch
    import numpy as np
    from PIL import Image

    pipe, face_app = _load_pipeline()

    # Load + detect face
    face_img = np.array(Image.open(face_image_path).convert("RGB"))
    faces = face_app.get(face_img)
    if not faces:
        raise RuntimeError("No face detected in the reference image. "
                           "Please provide a clear portrait photo.")

    # Extract face embeddings from largest face by area
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                   reverse=True)
    face = faces[0]

    generator = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None

    logger.info("Running PuLID inference: %dx%d steps=%d guidance=%.1f id_weight=%.2f",
                width, height, num_steps, guidance_scale, id_weight)
    t0 = time.time()

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        id_image=face_img,
        id_weight=id_weight,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    elapsed = time.time() - t0
    logger.info("Inference done in %.1fs", elapsed)

    image = result.images[0]
    image.save(out_path, quality=95)
    return out_path


# ── RunPod handler ─────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    RunPod Serverless entrypoint.

    Input schema:
      face_image_url   str   — reference portrait URL (required)
      prompt           str   — scene description (required)
      negative_prompt  str   — optional
      width            int   — default 1024
      height           int   — default 1024
      seed             int   — optional, -1 = random
      num_steps        int   — default 4 (Flux.1-schnell)
      guidance_scale   float — default 1.0 (schnell)
      id_weight        float — PuLID identity weight, default 0.8

    Output:
      {"image_url": str, "seed": int}
    """
    job_input = job.get("input", {})

    face_image_url  = job_input.get("face_image_url", "")
    prompt          = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    width           = int(job_input.get("width", 1024))
    height          = int(job_input.get("height", 1024))
    seed            = int(job_input.get("seed", -1))
    num_steps       = int(job_input.get("num_steps", 4))
    guidance_scale  = float(job_input.get("guidance_scale", 1.0))
    id_weight       = float(job_input.get("id_weight", 0.8))

    if seed < 0:
        seed = int(uuid.uuid4().int % 2**31)

    if not face_image_url or not face_image_url.startswith("http"):
        return {"error": "face_image_url is required and must be a http(s) URL"}
    if not prompt or not prompt.strip():
        return {"error": "prompt is required"}

    # Clamp dimensions to multiples of 16
    width  = max(512, min(2048, (width  // 16) * 16))
    height = max(512, min(2048, (height // 16) * 16))

    job_id = job.get("id", uuid.uuid4().hex[:8])
    logger.info("job=%s face=%s... prompt=%s...", job_id, face_image_url[:60], prompt[:80])

    # Ensure pipeline loaded
    try:
        _load_pipeline()
    except Exception as e:
        logger.exception("pipeline load failed")
        return {"error": f"model load failed: {e}"}

    with tempfile.TemporaryDirectory() as tmp:
        face_path = os.path.join(tmp, "face.jpg")
        out_path  = os.path.join(tmp, "output.jpg")

        try:
            _download_file(face_image_url, face_path)
            logger.info("face downloaded: %d bytes", os.path.getsize(face_path))
        except Exception as e:
            return {"error": f"face_image download failed: {e}"}

        try:
            _run_inference(
                face_path, prompt, negative_prompt,
                width, height, seed, num_steps, guidance_scale, id_weight,
                out_path,
            )
        except Exception as e:
            logger.exception("inference error")
            return {"error": f"inference failed: {e}"}

        # Upload to R2
        r2_key = f"pulid/{job_id}/{uuid.uuid4().hex[:8]}.jpg"
        try:
            image_url = _upload_r2(out_path, r2_key, "image/jpeg")
            logger.info("uploaded to R2: %s", r2_key)
        except Exception as e:
            logger.error("R2 upload failed: %s — R2 not configured properly", e)
            return {"error": "R2 upload failed — check R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"}

    return {
        "image_url": image_url,
        "seed": seed,
        "width": width,
        "height": height,
    }


if __name__ == "__main__":
    logger.info("PuLID-Flux handler starting…")
    runpod.serverless.start({"handler": handler})
