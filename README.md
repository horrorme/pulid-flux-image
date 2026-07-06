# PuLID Flux Image Handler

Containerized RunPod serverless handler for PuLID-Flux image generation. The repository is intentionally small: it packages the handler, dependencies and GitHub Actions workflow needed to build and publish the container image.

## Contents

- `handler.py` - RunPod request handler.
- `Dockerfile` - runtime image definition.
- `requirements.txt` - Python dependencies.
- `.github/workflows/build.yml` - GHCR build and publish workflow.

## Local Build

```bash
docker build -t pulid-flux-image .
```

## Notes

Runtime credentials and model storage configuration should be provided by the deployment platform. Do not commit provider tokens, bucket credentials or private model URLs to this repository.
