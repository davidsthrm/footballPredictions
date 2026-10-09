#!/bin/zsh
# Geplanter Job (von launchd gestartet, beendet sich danach): Transfermarkt-Snapshot, Nachtraining,
# Vorhersage des nächsten Spieltags, Dashboard. Log: data/logs/daily.log
# Manuell: Knopf „Jetzt aktualisieren“ im Dashboard (ops/install_refresh.sh) oder
#   launchctl kickstart gui/$(id -u)/de.fussball.daily
set -u
# Solange der Lauf dauert, nicht in den Ruhezustand gehen (wirkt nicht bei zugeklapptem Laptop im Akkubetrieb)
[[ "${1:-}" == "--awake" ]] || exec /usr/bin/caffeinate -i "$0" --awake
cd "$(dirname "$0")/.."
mkdir -p data/logs
UV=/opt/homebrew/bin/uv
step() { echo "$(date '+%H:%M:%S') $1 exit=$2"; }
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  # Nach dem Aufwachen ist das Netz oft noch nicht da: bis zu 10 Minuten warten
  for i in {1..20}; do
    curl -s -o /dev/null --max-time 10 https://www.football-data.co.uk && break
    echo "warte auf Netzwerk ($i)"; sleep 30
  done
  $UV run fussball snapshot transfermarkt; step snapshot $?
  $UV run fussball backtest >/dev/null;   step backtest $?   # Modelle mit neuesten Spielen nachtrainieren
  $UV run fussball predict;                ok=$?; step predict $ok
  $UV run fussball dashboard;              step dashboard $?
  if grep -q '^CLOUDFLARE_API_TOKEN=.' .env 2>/dev/null; then  # Web-App aktualisieren (nur wenn eingerichtet)
    $UV run fussball webapp --deploy;      step webapp $?
  fi
} >> data/logs/daily.log 2>&1
msg=$([[ $ok == 0 ]] && echo "Neue Tipps sind da – Dashboard neu laden." || echo "Vorhersage fehlgeschlagen, siehe data/logs/daily.log")
/usr/bin/osascript -e "display notification \"$msg\" with title \"Tippspiel-Assistent\"" 2>/dev/null
exit 0
