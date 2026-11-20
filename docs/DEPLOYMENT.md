# Deployment: website + API

```
Browser ─► EcoConsensus.harshitagr42.com     Cloudflare Pages, serves web/ only (always online)
              │ fetch (CORS: only this origin)
              ▼
           <you>-ecoconsensus-api.hf.space    Hugging Face Space (Docker, free CPU): api/ + src/ + configs/ + prompts/
              │ while the image is built      LLM keys = Space secrets
              ├─► <you>/ecoconsensus-weights  public HF model repo: LoRA adapters, meta-models, baselines (28 MB)
              └─► google/muril-base-cased, ai4bharat/IndicBERTv2-MLM-only
```

| Folder | Goes to | How it is selected |
|---|---|---|
| `web/` | Cloudflare Pages | Pages "build output directory" = `web` |
| `api/` + `src/` + `configs/` + `prompts/` + `inference.py` | HF Space | explicit list `BUNDLE` in `scripts/hf_space.py` |
| `models/finetuned/` | HF model repo | `scripts/hf_space.py weights` (git-ignored; never on GitHub) |

## 1. Hugging Face (once)

1. Create an account at huggingface.co. Settings → Access Tokens → **New token**, type **Write**.
2. Log in locally (the token stays on your Mac):
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt          # adds fastapi, uvicorn, huggingface_hub
   hf auth login                            # paste the token
   ```
3. Put both keys of each LLM in `.env` (comma-separated, no spaces needed):
   ```
   LLM_A_API_KEYS=gemini_key_account1,gemini_key_account2
   LLM_B_API_KEYS=groq_key_account1,groq_key_account2
   ```
   `LLM_A_API_KEY` (one key) still works when `LLM_A_API_KEYS` is not set. `LLM_<S>_DAILY_LIMIT` applies per key.
4. Upload the weights (public model repo `<you>/ecoconsensus-weights`):
   ```bash
   python scripts/hf_space.py weights
   ```
5. Create and deploy the Space (`<you>/ecoconsensus-api`, Docker, public). Check the file list first:
   ```bash
   python scripts/hf_space.py deploy --dry-run
   python scripts/hf_space.py deploy
   ```
6. Copy the LLM settings and keys to the Space (variables: provider, model, URL, limits; **secrets**: keys):
   ```bash
   python scripts/hf_space.py secrets
   ```
   Same thing by hand: Space → Settings → *Variables and secrets*.
7. Wait for the build (Space page → *Logs*, 5–10 min). Then:
   ```bash
   python scripts/hf_space.py status        # prints the API URL
   curl https://<you>-ecoconsensus-api.hf.space/health
   ```
   `"models": "ready"` and both LLMs `"configured": true` means it works.

## 2. Website

1. Put the two URLs printed by `status` into `web/config.js` (`API_BASE_URL`, `SPACE_PAGE_URL`) and commit.
2. Cloudflare Pages → connect the GitHub repo: framework *None*, build command *empty*, build output directory
   **`web`**. Build watch paths (Settings → Builds): include `web/*` so other commits do not redeploy.
3. Custom domain: `ecoconsensus.harshitagr42.com`. DNS is case-insensitive; `EcoConsensus.harshitagr42.com` works too.
4. If you use another domain or the `*.pages.dev` address, add it to the API:
   `python scripts/hf_space.py secrets --origin https://other.domain` (comma-separate several).

Local test: `uvicorn api.app:app --port 7860` and `cd web && python -m http.server 8080`, then open
`http://localhost:8080/?api=http://localhost:7860`.

## 3. Running it for the professor

| When | Command |
|---|---|
| Professor asks for the link | `python scripts/hf_space.py restart` (≈1 min), then check `/health` |
| After code changes | `python scripts/hf_space.py deploy` (rebuilds) |
| Professor is done | `python scripts/hf_space.py pause` (free, keys unused; the site shows "offline") |

A free Space also **sleeps after ~48 h without visits**; the website then shows "offline" with a link to the Space page,
which wakes it (about a minute). Paused Spaces only start with `restart`.

Before sending the link: the professor's ~1,000 rows need about `1000 / LLM_BATCH_SIZE` requests per LLM (plus
retries); check that this fits the free daily quota of one key (`python -m src.training.run_llm_batch --estimate`).
Check also that the providers' terms allow using two free accounts.

## 4. Limits (enforced by the API, shown on the website)

Defaults in `api/settings.py`; override any of them as a Space variable (e.g. `JOBS_PER_DAY=30`).

| Setting | Default |
|---|---|
| `MAX_ROWS` / `MAX_FILE_MB` | 2000 rows / 5 MB, CSV only |
| `JOBS_PER_IP_PER_HOUR` | 5 runs per visitor (IP) per hour |
| `JOBS_PER_DAY` | 20 runs per day, all visitors together (protects the LLM quota) |
| `UPLOADS_PER_IP_PER_HOUR` | 30 file previews (no model or LLM used) |
| `QUEUE_MAX` | 5 waiting runs; one runs at a time |
| `RESULT_TTL_HOURS` | 6 h, then uploads and results are deleted |

Counters live in memory: a restart resets them. Someone switching IPs gets around the per-IP limit, not the daily one.
A run that fails on the server's side is not counted.

## 5. Mapping on the website

After the upload the site shows two tables, always:

- **Columns**: comment (required), claim (default: the fixed claim; a detected claim/topic column is only used when
  picked), true stance (optional; `predict` / `prediction` / `target` are guessed when they hold few values), row ID.
- **Stance values**: every distinct value of the stance column and what it means (favour / neutral / against /
  ignore), with our id (0 favour, 1 neutral, 2 against). Text values (`FAVOR`, `none`, …) are pre-filled; **numbers are
  never guessed**, so a file where 0 means neutral is scored correctly. *Run* stays disabled until every value is
  mapped; with no stance column the site asks for confirmation and runs predictions only.

Predictions can be written in the file's own values (the mapping reversed), as Favour / None / Against, or as our ids.
