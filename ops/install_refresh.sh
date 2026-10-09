#!/bin/zsh
# Installiert ~/Applications/Tippspiel Refresh.app: öffnet sich bei Links tippspiel://refresh
# (Knopf „Jetzt aktualisieren“ im Dashboard, nur am Mac sichtbar) und startet den launchd-Job
# sofort – derselbe Lauf wie um 08:00, kein Dauerprozess. Läuft der Job schon, passiert nichts.
#
# Entfernen: rm -rf ~/Applications/"Tippspiel Refresh.app"
set -eu
LABEL=de.fussball.daily
APP="$HOME/Applications/Tippspiel Refresh.app"
PLIST="$APP/Contents/Info.plist"
SRC=$(mktemp -t tippspiel).applescript
mkdir -p "$HOME/Applications"
cat > "$SRC" <<APPLESCRIPT
on refreshNow()
	do shell script "launchctl kickstart gui/\$(id -u)/$LABEL"
	display notification "Läuft, dauert etwa 10 Minuten." with title "Tippspiel-Assistent"
end refreshNow

on open location theURL
	refreshNow()
end open location

on run
	refreshNow()
end run
APPLESCRIPT
rm -rf "$APP"
osacompile -o "$APP" "$SRC"
rm -f "$SRC"
/usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string de.fussball.refresh" "$PLIST" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier de.fussball.refresh" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleURLTypes array" \
  -c "Add :CFBundleURLTypes:0 dict" \
  -c "Add :CFBundleURLTypes:0:CFBundleURLName string Tippspiel Refresh" \
  -c "Add :CFBundleURLTypes:0:CFBundleURLSchemes array" \
  -c "Add :CFBundleURLTypes:0:CFBundleURLSchemes:0 string tippspiel" "$PLIST"
codesign --force --sign - "$APP" >/dev/null 2>&1      # Info.plist geändert -> neu (ad hoc) signieren
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP"
echo "Installiert: $APP"
echo "Testen: open tippspiel://refresh   ·  Log: data/logs/daily.log"
