# PuLID-Flux RunPod Serverless image
# Base: PyTorch 2.4 + CUDA 12.4 (matches our Wan I2V pattern)
#
# MODEL LICENSING NOTES:
#   - PuLID code/weights: Apache-2.0 ✓ (commercial OK)
#   - Flux.1-schnell: Apache-2.0 ✓ (commercial OK) — we use schnell, NOT dev
#   - InsightFace antelopev2 weights: non-commercial by default
#     → Contact recognition-oss-pack@insightface.ai for commercial license
#     → OR replace with mediapipe face detection (Apache-2.0)
#
# Build (requires DOCKER_HF_TOKEN secret for weight baking):
#   docker build \
#     --build-arg DOCKER_HF_TOKEN=hf_xxx \
#     -t ghcr.io/horrorme/vidgen-web/pulid-flux:latest \
#     .
#
# Run locally:
#   docker run --gpus all \
#     -e R2_ENDPOINT_URL=... -e R2_ACCESS_KEY_ID=... -e R2_SECRET_ACCESS_KEY=... \
#     -e R2_BUCKET=vidgen-media -e R2_CDN_BASE=https://cdn.vidgen-ai.com \
#     -p 8000:8000 \
#     ghcr.io/horrorme/vidgen-web/pulid-flux:latest

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

ARG DOCKER_HF_TOKEN=""
ENV HF_TOKEN=${DOCKER_HF_TOKEN}
ENV MODELS_DIR=/workspace/models
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone PuLID repo (Apache-2.0)
# We do NOT use "pip install -e /app/pulid" — modern pip refuses flat-layout repos
# with multiple top-level packages (flux, pulid, models, eva_clip, example_inputs).
# Instead we add /app/pulid to PYTHONPATH so imports work without editable install.
RUN git clone --depth 1 https://github.com/ToTheBeginning/PuLID /app/pulid
ENV PYTHONPATH=/app/pulid:${PYTHONPATH}

# Copy handler
COPY handler.py /app/handler.py

# Weights are NOT baked into the image — they exceed GHA runner disk (~14GB free).
# Flux.1-schnell (~24GB) + PuLID (~1GB) + antelopev2 (~1GB) are downloaded at
# cold start into /workspace/models, which is a RunPod network volume mount.
RUN mkdir -p /workspace/models

CMD ["python3", "-u", "/app/handler.py"]
