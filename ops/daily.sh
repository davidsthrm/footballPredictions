#!/bin/zsh
# Geplanter Job (von launchd gestartet, beendet sich danach): Transfermarkt-Snapshot, Nachtraining,
# Vorhersage des nächsten Spieltags, Dashboard. Log: data/logs/daily.log
set -u
cd "$(dirname "$0")/.."
mkdir -p data/logs
UV=/opt/homebrew/bin/uv
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  # Nach dem Aufwachen ist das Netz oft noch nicht da: bis zu 10 Minuten warten
  for i in {1..20}; do
    curl -s -o /dev/null --max-time 10 https://www.football-data.co.uk && break
    echo "warte auf Netzwerk ($i)"; sleep 30
  done
  $UV run fussball snapshot transfermarkt; echo "snapshot exit=$?"
  $UV run fussball backtest >/dev/null;   echo "backtest exit=$?"   # Modelle mit neuesten Spielen nachtrainieren
  $UV run fussball predict;                echo "predict exit=$?"
  $UV run fussball dashboard;              echo "dashboard exit=$?"
} >> data/logs/daily.log 2>&1
