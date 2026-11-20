# Hugging Face Space (Docker SDK). scripts/hf_space.py deploy copies this file to the Space root together with
# api/, src/, configs/, prompts/ and inference.py; nothing else from the repository is uploaded.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 && \
    pip install --user -r requirements.txt

COPY --chown=user . .

# Fine-tuned weights (separate HF model repo, named in api/deploy.json) and both base models are downloaded while
# the image is built, so a restart does not download them again.
RUN python -m api.weights

EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
