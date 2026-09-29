#!/bin/zsh
# Plant den Job ops/daily.sh über launchd ein. launchd startet das Skript nur zu den
# angegebenen Zeitpunkten; es läuft ~10 Minuten und beendet sich. War der Mac zu dem
# Zeitpunkt aus oder im Ruhezustand, wird der Lauf beim Aufwachen nachgeholt.
#
#   ops/install_launchd.sh              täglich 08:00
#   ops/install_launchd.sh 2 5          nur Di und Fr (0=So, 1=Mo, ... 6=Sa)
#   HOUR=7 ops/install_launchd.sh 2 5   Di und Fr um 07:00
#
# Entfernen: launchctl bootout gui/$(id -u)/de.fussball.daily && rm ~/Library/LaunchAgents/de.fussball.daily.plist
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=de.fussball.daily
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
HOUR=${HOUR:-8}
mkdir -p "$REPO/data/logs" "$HOME/Library/LaunchAgents"

if (( $# == 0 )); then
  INTERVAL="<dict><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>0</integer></dict>"
else
  INTERVAL="<array>"
  for d in "$@"; do
    INTERVAL+="<dict><key>Weekday</key><integer>$d</integer><key>Hour</key><integer>$HOUR</integer><key>Minute</key><integer>0</integer></dict>"
  done
  INTERVAL+="</array>"
fi

cat > "$TARGET" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$REPO/ops/daily.sh</string></array>
  <key>StartCalendarInterval</key>$INTERVAL
  <key>StandardErrorPath</key><string>$REPO/data/logs/launchd.err</string>
</dict></plist>
PLIST
plutil -lint "$TARGET" >/dev/null
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
echo "Eingeplant: $TARGET"
echo "Sofort testen: launchctl kickstart gui/$(id -u)/$LABEL   ·  Log: $REPO/data/logs/daily.log"
