# EcoConsensus API image. Build context = the app folder on the server (scripts/deploy_backend.sh copies only
# api/ src/ configs/ prompts/ inference.py .dockerignore there); .dockerignore repeats that allow-list.
#   docker build -f api/Dockerfile -t ecoconsensus-api .
# Weights and base models are NOT in the image: they are downloaded at startup into volumes (see api/server.sh).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

# uid 1000 = the default "ubuntu" user on the server, so the mounted volumes and .env are readable
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user api/requirements.txt api/requirements.txt
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 && \
    pip install --user -r api/requirements.txt

COPY --chown=user . .
# The classic Docker builder (docker.io on Ubuntu) creates WORKDIR as root, so the user could not add folders here
USER root
RUN mkdir -p models/finetuned outputs /home/user/.cache/huggingface && chown -R user:user /home/user
USER user

EXPOSE 7860
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
