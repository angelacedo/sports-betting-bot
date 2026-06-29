#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================"
echo "  Sports Betting Bot - VPS Deployment"
echo "============================================"
echo ""

echo "[1/4] Pulling latest changes..."
git pull origin main
echo ""

echo "[2/4] Building Docker image (no cache)..."
docker compose build --no-cache
echo ""

echo "[3/4] Starting services..."
docker compose up -d --force-recreate --remove-orphans
echo ""

echo "[4/4] Cleaning up..."
docker image prune -f
docker volume prune -f
echo ""

echo "============================================"
echo "  Deployment complete!"
echo "============================================"
echo ""
echo "Services status:"
docker compose ps
echo ""
echo "View logs: docker compose logs -f bot"
echo "Stop: docker compose down"
