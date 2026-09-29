#!/bin/zsh
# Täglicher Transfermarkt-Snapshot (von launchd aufgerufen). Log: data/logs/snapshot.log
set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  /opt/homebrew/bin/uv run fussball snapshot transfermarkt
  echo "exit=$?"
} >> data/logs/snapshot.log 2>&1
