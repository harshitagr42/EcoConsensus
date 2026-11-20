# EcoConsensus API

Backend of the EcoConsensus website (`web/`). Given the claim *"Climate change and global warming is a serious
concern."* and Hindi/Bengali comments, it predicts favour / neutral / against with the stacked ensemble: MuRIL and
IndicBERT v2 (LoRA) + two LLMs -> logistic-regression meta-model.

Runs in Docker on an EC2 server behind a Cloudflare Tunnel (`https://api.harshitagr42.com`). Setup and operation:
`docs/DEPLOYMENT.md`.

| File | Role |
|---|---|
| `app.py` | FastAPI routes: `/health`, `/limits`, `/inspect`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/results`, `/jobs/{id}/files/{name}` |
| `jobs.py` | uploads, rate limits, queue, the job runner (reuses `src/pipeline.py` and `inference.py`) |
| `settings.py` | limits (overridable in the server's `.env`) |
| `weights.py` | downloads the weights repo (`WEIGHTS_REPO`) and warms the base models at startup |
| `Dockerfile`, `requirements.txt` | the image (CPU torch); `../.dockerignore` is the allow-list |
| `server.sh` | on the server: `setup`, `tunnel`, `start`, `stop`, `logs`, `status` |

Local run: `uvicorn api.app:app --port 7860`.
