#!/bin/zsh
# Geplanter Job (von launchd gestartet, beendet sich danach): Transfermarkt-Snapshot, Nachtraining,
# Vorhersage des nächsten Spieltags, Dashboard. Log: data/logs/daily.log
set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs
UV=/opt/homebrew/bin/uv
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  $UV run fussball snapshot transfermarkt; echo "snapshot exit=$?"
  $UV run fussball backtest >/dev/null;   echo "backtest exit=$?"   # Modelle mit neuesten Spielen nachtrainieren
  $UV run fussball predict;                echo "predict exit=$?"
  $UV run fussball dashboard;              echo "dashboard exit=$?"
} >> data/logs/daily.log 2>&1
