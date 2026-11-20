# Deployment: website + API

```
Browser ─► EcoConsensus.harshitagr42.com   Cloudflare Pages: web/ (always online)
             │ fetch (CORS: only this origin)
             ▼
          api.harshitagr42.com              Cloudflare Tunnel (HTTPS; no open ports on the server)
             ▼
          AWS EC2 t3.medium, Docker         api/ + src/ + configs/ + prompts/ + inference.py, .env mounted read-only
             └─ at first start ─► <hf-user>/ecoconsensus-weights (HF model repo) + both base models (kept on disk)
```

| What | Goes to | Selected by |
|---|---|---|
| `web/` | Cloudflare Pages | Pages output directory `web` |
| `api/ src/ configs/ prompts/ inference.py` | EC2 server | allow-list in `scripts/deploy_backend.sh` (and `.dockerignore`) |
| `models/finetuned/` | HF model repo | `scripts/hf_weights.py` (never GitHub, never the server's code folder) |
| `.env` (keys) | EC2 `~/ecoconsensus/.env` | `scripts/deploy_backend.sh … --env` (mode 600, outside the image) |

Cost: t3.medium ≈ USD 1/day on, 30 GB disk ≈ USD 2.5/month, public IPv4 ≈ USD 3.6/month. Three weeks ≈ USD 26.
A stopped instance costs only the disk.

## 0. Already done

Weights uploaded: `python scripts/hf_weights.py` (`--name` prints the repo id). Add to the local `.env`:
```
WEIGHTS_REPO=<hf-user>/ecoconsensus-weights
LLM_A_API_KEY_1=…   LLM_A_API_KEY_2=…   LLM_B_API_KEY_1=…   LLM_B_API_KEY_2=…
```
Check: `python -c "from src.llm_api import api_keys; print(len(api_keys('a')), len(api_keys('b')))"` → `2 2`.

## 1. EC2 server (AWS console)

1. **Billing → Budgets → Create budget**: monthly cost budget, e.g. USD 40, e-mail alert at 80 %.
2. **EC2 → Launch instance** (region close to you, e.g. ap-south-1 Mumbai):
   - Name `ecoconsensus`; AMI **Ubuntu Server 24.04 LTS (x86_64)**; type **t3.medium**.
   - Key pair: create `ecoconsensus` (RSA, .pem). Save it as `~/.ssh/ecoconsensus.pem`, then `chmod 400 ~/.ssh/ecoconsensus.pem`.
   - Network: auto-assign public IP **on**; security group: **SSH (22) from My IP** only. No HTTP/HTTPS rules
     (the tunnel connects outward).
   - Storage: **30 GB gp3**.
3. Note the instance's **public IPv4** (it changes after every stop/start; only SSH uses it).

## 2. Server setup (once)

From the project root on the Mac (`IP` = the public IPv4):
```bash
scripts/deploy_backend.sh ubuntu@IP -i ~/.ssh/ecoconsensus.pem --env --no-start   # code + .env
ssh -i ~/.ssh/ecoconsensus.pem ubuntu@IP
```
On the server:
```bash
bash ~/ecoconsensus/app/api/server.sh setup                        # swap, Docker, cloudflared (~3 min)
bash ~/ecoconsensus/app/api/server.sh tunnel api.harshitagr42.com  # prints a URL: open it on the Mac, pick harshitagr42.com
bash ~/ecoconsensus/app/api/server.sh start                        # build (~5 min) and start
bash ~/ecoconsensus/app/api/server.sh status                       # wait for "models":"ready" (first start ~2-3 min)
```
`tunnel` creates the DNS record `api.harshitagr42.com` in Cloudflare and installs the tunnel as a service.
Check from the Mac: `curl https://api.harshitagr42.com/health` → `"models":"ready"`, both LLMs `"keys":2`.

## 3. Website (Cloudflare Pages)

`web/config.js` already points to `https://api.harshitagr42.com`. Pages → connect the GitHub repo: framework *None*,
build command *empty*, output directory **`web`**, build watch paths include `web/*`. Custom domain
`ecoconsensus.harshitagr42.com` (DNS ignores case). Another origin (e.g. `*.pages.dev`) must be added to the server's
`.env`: `ALLOWED_ORIGINS=https://ecoconsensus.harshitagr42.com,https://<x>.pages.dev`, then `server.sh start`.

Local test: `uvicorn api.app:app --port 7860` and `cd web && python -m http.server 8080`, then
`http://localhost:8080/?api=http://localhost:7860`.

## 4. Running it

| When | What |
|---|---|
| Professor asks for the link | EC2 console → instance → **Start**. API and tunnel start by themselves (~2 min); check `/health` |
| Professor is done | EC2 console → **Stop** (the site then says the server is switched off). **Terminate** at the end of the project |
| Code changed | `scripts/deploy_backend.sh ubuntu@IP -i ~/.ssh/ecoconsensus.pem` (rebuild + restart) |
| `.env` changed (keys, limits) | same with `--env` |
| Logs | on the server: `bash ~/ecoconsensus/app/api/server.sh logs` |

Weights, base models and the LLM cache/daily counters live in `~/ecoconsensus/data/` and survive restarts and
rebuilds. After terminating the instance, also delete the tunnel (Cloudflare → Zero Trust → Networks → Tunnels) and
the `api` DNS record.

Before sending the link: 1,000 rows need about `1000 / LLM_BATCH_SIZE` requests per LLM (plus retries); check the free
daily quota of one key (`python -m src.training.run_llm_batch --estimate`), and that the providers' terms allow two
free accounts.

## 5. Limits (enforced by the API, shown on the website)

Defaults in `api/settings.py`; override in the server's `.env` (e.g. `JOBS_PER_DAY=30`), then `server.sh start`.

| Setting | Default |
|---|---|
| `MAX_ROWS` / `MAX_FILE_MB` | 2000 rows / 5 MB, CSV only |
| `JOBS_PER_IP_PER_HOUR` | 5 runs per visitor (IP, from Cloudflare's `CF-Connecting-IP`) per hour |
| `JOBS_PER_DAY` | 20 runs per day, all visitors together (protects the LLM quota) |
| `UPLOADS_PER_IP_PER_HOUR` | 30 file previews (no model or LLM used) |
| `QUEUE_MAX` | 5 waiting runs; one runs at a time |
| `RESULT_TTL_HOURS` | 6 h, then uploads and results are deleted |

Counters live in memory: a restart resets them. Switching IPs gets around the per-IP limit, not the daily one. A run
that fails on the server's side is not counted. Measured: a run peaks at ~1.5 GB RAM; 30 rows take ~25 s.

## 6. Mapping on the website

After the upload the site shows two tables, always:

- **Columns**: comment (required), claim (default: the fixed claim; a detected claim/topic column is only used when
  picked), true stance (optional; `predict` / `prediction` / `target` are guessed when they hold few values), row ID.
- **Stance values**: every distinct value of the stance column and what it means (favour / neutral / against /
  ignore), with our id (0 favour, 1 neutral, 2 against). Text values (`FAVOR`, `none`, …) are pre-filled; **numbers are
  never guessed**, so a file where 0 means neutral is scored correctly. *Run* stays disabled until every value is
  mapped; with no stance column the site asks for confirmation and runs predictions only.

Predictions can be written in the file's own values (the mapping reversed), as Favour / None / Against, or as our ids.
