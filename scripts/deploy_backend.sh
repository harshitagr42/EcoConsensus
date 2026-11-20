#!/usr/bin/env bash
# Runs on the Mac: copy the backend to the EC2 server and restart it there.
#
#   scripts/deploy_backend.sh ubuntu@<server-ip> -i ~/.ssh/ecoconsensus.pem          # code, then rebuild + restart
#   scripts/deploy_backend.sh ubuntu@<server-ip> -i ~/.ssh/ecoconsensus.pem --env    # also copy .env (keys)
#   scripts/deploy_backend.sh ubuntu@<server-ip> -i ~/.ssh/ecoconsensus.pem --no-start   # copy only (before setup)
#
# Only the allow-list below is copied (to ~/ecoconsensus/app); with --env the local .env goes to ~/ecoconsensus/.env
# (mode 600, outside the code folder and never inside the image). Weights are not copied: the server downloads them
# from the HF model repo named by WEIGHTS_REPO in that .env.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${1:?usage: scripts/deploy_backend.sh ubuntu@<server-ip> [-i key.pem] [--env] [--no-start]}"
shift
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
WITH_ENV=0
START=1
while [ $# -gt 0 ]; do
  case "$1" in
    -i) SSH_OPTS+=(-i "$2"); shift 2 ;;
    --env) WITH_ENV=1; shift ;;
    --no-start) START=0; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
ALLOW=(api src configs prompts inference.py .dockerignore)

ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p ~/ecoconsensus/app ~/ecoconsensus/data"
rsync -az --delete --relative -e "ssh ${SSH_OPTS[*]}" \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' --exclude 'src/training/' \
  "${ALLOW[@]}" "$HOST:ecoconsensus/app/"
echo "code copied: ${ALLOW[*]}"

if [ "$WITH_ENV" = 1 ]; then
  grep -q '^WEIGHTS_REPO=' .env || { echo "add WEIGHTS_REPO=<hf-user>/ecoconsensus-weights to .env first"; exit 1; }
  rsync -a --chmod=F600 -e "ssh ${SSH_OPTS[*]}" .env "$HOST:ecoconsensus/.env"
  echo ".env copied (mode 600)"
fi

if [ "$START" = 1 ]; then
  ssh "${SSH_OPTS[@]}" "$HOST" "bash ~/ecoconsensus/app/api/server.sh start"
fi
