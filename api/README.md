---
title: EcoConsensus API
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Stance detection API for Hindi and Bengali comments
---

# EcoConsensus API

Backend of the EcoConsensus stance-detection website. Given the claim *"Climate change and global warming is a serious
concern."* and Hindi/Bengali comments, it predicts favour / neutral / against with a stacked ensemble: MuRIL and
IndicBERT v2 (LoRA fine-tuned) + two LLMs → logistic-regression meta-model.

This Space only holds the code. The fine-tuned weights are in a separate model repo (named in `api/deploy.json`); API
keys are Space secrets. Deployed with `scripts/hf_space.py deploy` from the project repository.

Endpoints: `GET /health`, `GET /limits`, `POST /inspect`, `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/results`,
`GET /jobs/{id}/files/{name}`.
