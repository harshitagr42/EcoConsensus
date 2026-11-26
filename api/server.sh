#!/usr/bin/env bash
# Runs ON the EC2 server (Ubuntu). The Mac copies the code with scripts/deploy_backend.sh, which then calls `start`.
#
#   bash ~/ecoconsensus/app/api/server.sh setup                       # once: swap, Docker, cloudflared
#   bash ~/ecoconsensus/app/api/server.sh tunnel api.harshitagr42.com # once: Cloudflare Tunnel -> this server
#   bash ~/ecoconsensus/app/api/server.sh start                       # build the image and (re)start the API
#   bash ~/ecoconsensus/app/api/server.sh stop | logs | status
#
# Layout: ~/ecoconsensus/app (code), ~/ecoconsensus/.env (keys, WEIGHTS_REPO), ~/ecoconsensus/data/{models,hf-cache,
# outputs} (weights, base models, LLM cache and daily counters; kept across restarts and rebuilds).
# The API listens on 127.0.0.1:7860 only; the tunnel is the one way in. Container and tunnel start on boot.
set -euo pipefail

BASE="$HOME/ecoconsensus"
APP="$BASE/app"
DATA="$BASE/data"
NAME="ecoconsensus-api"
TUNNEL="ecoconsensus"
DOCKER="sudo docker"

cmd_setup() {
  if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
  fi
  # non-interactive: no "which services should be restarted?" dialog on Ubuntu
  sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y docker.io curl
  sudo systemctl enable --now docker
  if ! command -v cloudflared >/dev/null; then
    curl -fsSL -o /tmp/cloudflared.deb \
      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$(dpkg --print-architecture).deb"
    sudo dpkg -i /tmp/cloudflared.deb
  fi
  mkdir -p "$DATA/models" "$DATA/hf-cache" "$DATA/outputs"
  echo "setup done: docker $($DOCKER --version | cut -d' ' -f3) cloudflared $(cloudflared --version | cut -d' ' -f3)"
}

cmd_tunnel() {
  local host="${1:?usage: server.sh tunnel api.harshitagr42.com}"
  [ -f "$HOME/.cloudflared/cert.pem" ] || cloudflared tunnel login     # prints a URL: open it, pick the domain
  cloudflared tunnel list | awk '{print $2}' | grep -qx "$TUNNEL" || cloudflared tunnel create "$TUNNEL"
  local id
  id=$(cloudflared tunnel list | awk -v n="$TUNNEL" '$2==n {print $1}')
  cloudflared tunnel route dns --overwrite-dns "$TUNNEL" "$host"
  sudo mkdir -p /etc/cloudflared
  sudo cp "$HOME/.cloudflared/$id.json" /etc/cloudflared/
  sudo tee /etc/cloudflared/config.yml >/dev/null <<EOF
tunnel: $id
credentials-file: /etc/cloudflared/$id.json
ingress:
  - hostname: $host
    service: http://localhost:7860
  - service: http_status:404
EOF
  systemctl list-unit-files | grep -q '^cloudflared.service' || sudo cloudflared service install
  sudo systemctl enable cloudflared && sudo systemctl restart cloudflared
  echo "tunnel $TUNNEL ($id): https://$host -> http://localhost:7860"
}

cmd_start() {
  [ -f "$BASE/.env" ] || { echo "missing $BASE/.env (scripts/deploy_backend.sh HOST --env copies it)"; exit 1; }
  grep -q '^WEIGHTS_REPO=' "$BASE/.env" || { echo "add WEIGHTS_REPO=<hf-user>/ecoconsensus-weights to $BASE/.env"; exit 1; }
  mkdir -p "$DATA/models" "$DATA/hf-cache" "$DATA/outputs"
  $DOCKER build -f "$APP/api/Dockerfile" -t "$NAME" "$APP"
  $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
  $DOCKER run -d --name "$NAME" --restart unless-stopped -p 127.0.0.1:7860:7860 \
    -v "$BASE/.env:/home/user/app/.env:ro" \
    -v "$DATA/models:/home/user/app/models/finetuned" \
    -v "$DATA/hf-cache:/home/user/.cache/huggingface" \
    -v "$DATA/outputs:/home/user/app/outputs" \
    "$NAME" >/dev/null
  $DOCKER image prune -f >/dev/null
  echo "started; models load in ~1-3 min (first start downloads ~2 GB). Check: bash $APP/api/server.sh status"
}

cmd_stop()   { $DOCKER rm -f "$NAME" >/dev/null 2>&1 && echo "stopped" || echo "not running"; }
cmd_logs()   { $DOCKER logs -f --tail 100 "$NAME"; }
cmd_status() {
  echo "container: $($DOCKER inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)"
  echo "tunnel   : $(systemctl is-active cloudflared 2>/dev/null || echo missing)"
  echo "health   : $(curl -s --max-time 5 http://127.0.0.1:7860/health || echo 'no answer')"
}

case "${1:-}" in
  setup) cmd_setup ;;
  tunnel) shift; cmd_tunnel "$@" ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  logs) cmd_logs ;;
  status) cmd_status ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
