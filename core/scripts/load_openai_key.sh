#!/usr/bin/env bash
# 从 MINER-My/.env.openai 加载 OPENAI_API_KEY（若存在）
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env.openai"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "[OK] OPENAI_API_KEY loaded from .env.openai (${#OPENAI_API_KEY} chars)"
    return 0 2>/dev/null || exit 0
  fi
fi
echo "[WARN] No .env.openai or empty OPENAI_API_KEY — use stub/structured or create .env.openai"
return 1 2>/dev/null || exit 1
