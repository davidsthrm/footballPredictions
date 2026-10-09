# Fußball-Ergebnisvorhersage

Vorhersage des exakten Endergebnisses für den nächsten Spieltag der Top-5-Ligen.
Baseline: Dixon-Coles mit Time-Decay, bewertet per Walk-forward-Backtest.
Regeln und Konventionen stehen in [CLAUDE.md](CLAUDE.md).

## Setup
```bash
brew install uv libomp      # libomp: OpenMP-Laufzeit für LightGBM
uv sync
```

## Daten laden (idempotent, alles wird unter `data/raw/` gecacht)
```bash
uv run fussball ingest football-data
uv run fussball ingest understat
uv run fussball ingest clubelo --suggest   # Vorschläge in data_seed/team_name_map.csv prüfen
uv run fussball ingest europe --suggest    # Europapokal-Termine, Vereins-IDs prüfen
uv run fussball check
```

## Nächster Spieltag
```bash
uv run fussball predict        # frischt die laufende Saison auf, tippt den nächsten Spieltag
uv run fussball scorecard      # echte Trefferbilanz der gespeicherten Vorhersagen
```
Der Spielplan kommt aus Understat (kein API-Key nötig). Jede Vorhersage wird mit
Cutoff-Datum in `predictions` gespeichert und nie überschrieben; die Scorecard wertet
je Spiel die letzte Vorhersage vor dem Spieltag aus. Export fürs Dashboard:
`data/reports/next_matchday.json`.

## Dashboard
```bash
uv run fussball dashboard      # -> data/dashboard/index.html (eine Datei, ohne JavaScript)
```
Mobil-first; im Zentrum der nächste Spieltag je Liga (Tipp, 1X2, erwartete Tore,
Ergebnismatrix, Ausfälle), darunter Live-Bilanz und Backtest-Einordnung. Fürs Handy:
in `.env` `FUSSBALL_DASHBOARD_COPY` auf eine Datei in iCloud Drive setzen, dann liegt
das Dashboard nach jedem Lauf auch dort (Dateien-App auf dem iPhone).

## Auf dem iPhone wie eine App
Das Dashboard liegt nach jedem Lauf in iCloud Drive (`FUSSBALL_DASHBOARD_COPY`), daneben
`Tippspiel-Symbol.png`. Einmalig in der Kurzbefehle-App:
1. Neuer Kurzbefehl „Tippspiel“: Aktion **Datei abrufen** → iCloud Drive › Fussball › `index.html`
   (Option „Ordnerauswahl anzeigen“ aus), danach Aktion **Übersicht**.
2. `Tippspiel-Symbol.png` in der Dateien-App öffnen → Teilen → „Bild sichern“ (landet in Fotos).
3. Kurzbefehl → Teilen → **Zum Home-Bildschirm** → Symbol antippen → Foto wählen → „Hinzufügen“.
Der Kurzbefehl holt bei jedem Öffnen die aktuelle Datei aus iCloud (die Dateien-App zeigt dagegen
manchmal noch eine zwischengespeicherte ältere Kopie).

## Geplanter Lauf (Snapshot + Vorhersage + Dashboard)
Ablauf: Snapshot → Nachtraining (`backtest`, ~30 s) → Vorhersage → Dashboard.
Nichts läuft dauerhaft: launchd startet `ops/daily.sh` zu festen Zeitpunkten, das Skript
arbeitet ~10 Minuten und beendet sich. Verpasste Termine (Mac aus) werden beim Aufwachen
nachgeholt.
```bash
ops/install_launchd.sh          # täglich 08:00
ops/install_launchd.sh 2 5      # nur Di + Fr
ops/install_refresh.sh          # Knopf „Jetzt aktualisieren“ im Dashboard (nur am Mac)
```
Ein Lauf braucht rund 3 Minuten Rechenzeit (Nachtraining ~1,5 min auf bis zu 5 Kernen, ~1,2 GB RAM)
plus einige Minuten höfliches Warten beim Transfermarkt-Snapshot; am Ende kommt eine Mitteilung.
Log: `data/logs/daily.log`. Transfermarkt: ca. 100 Requests pro Lauf; bei Blockade (403)
bricht der Scraper mit Exit-Code 2 ab. Verfügbarkeiten lassen sich nicht nachträglich
rekonstruieren – je seltener der Snapshot, desto gröber die Ausfall-Historie.
Neue Vereine (Aufsteiger) meldet `fussball check` als ungemappt → in
`data_seed/team_name_map.csv` ergänzen, dann `uv run fussball teams`.

## Europapokal (CL, EL, Conference League)
```bash
uv run fussball ingest other-leagues   # weitere Ligen für die Stärke der Gegner
uv run fussball ingest europe          # Spielpläne + Ergebnisse (Transfermarkt)
uv run fussball backtest-europe --tune # Parameter auf 2016–2018, Test ab 2019/20
```
Danach tippt `fussball predict` auch die nächste Europapokal-Runde (Reiter „Europa“ im Dashboard).

## Backtest
```bash
uv run fussball backtest --tune   # ξ/reg auf 2016–2018 tunen, Report ab 2019/20
uv run fussball backtest          # mit gespeicherten Parametern
```
Ergebnis: `data/reports/backtest_report.md`, `backtest_summary.csv`,
`backtest_predictions.csv`, `score_distribution.png`.
