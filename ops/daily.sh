#!/bin/zsh
# Täglicher Job (von launchd aufgerufen): Transfermarkt-Snapshot, dann Vorhersage des
# nächsten Spieltags. Log: data/logs/daily.log
set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs
UV=/opt/homebrew/bin/uv
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  $UV run fussball snapshot transfermarkt; echo "snapshot exit=$?"
  $UV run fussball predict;                echo "predict exit=$?"
} >> data/logs/daily.log 2>&1
