#!/usr/bin/env bash
set -euo pipefail

# Deploy sports-betting-bot to Hostinger VPS via SSH
# Usage: ./deploy.sh [user@host]

REMOTE="${1:?Usage: deploy.sh user@host}"
REMOTE_DIR="/opt/sports-betting-bot"

echo "==> Syncing files to ${REMOTE}:${REMOTE_DIR}"
rsync -avz --exclude 'venv/' --exclude '__pycache__/' --exclude '.env' \
    --exclude 'data/raw/' --exclude 'logs/' --exclude '.git/' \
    ../ "${REMOTE}:${REMOTE_DIR}"

echo "==> Building and restarting containers"
ssh "${REMOTE}" "cd ${REMOTE_DIR}/docker && docker compose pull && docker compose up -d --build"

echo "==> Deploy complete"
