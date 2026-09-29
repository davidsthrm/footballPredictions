#!/bin/zsh
# Installiert den täglichen Job (Snapshot + Vorhersage) als launchd-Job für den aktuellen Benutzer.
# Deinstallieren: launchctl bootout gui/$(id -u)/de.fussball.daily && rm ~/Library/LaunchAgents/de.fussball.daily.plist
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$HOME/Library/LaunchAgents/de.fussball.daily.plist"
mkdir -p "$REPO/data/logs" "$HOME/Library/LaunchAgents"
sed "s|__REPO__|$REPO|g" "$REPO/ops/de.fussball.daily.plist.template" > "$TARGET"
launchctl bootout "gui/$(id -u)/de.fussball.daily" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$TARGET"
echo "Installiert: $TARGET (täglich 08:00). Test sofort: launchctl kickstart gui/$(id -u)/de.fussball.daily"
