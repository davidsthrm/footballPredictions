# Fußball-Ergebnisvorhersage

Vorhersage des exakten Endergebnisses für den nächsten Spieltag der Top-5-Ligen.
Baseline: Dixon-Coles mit Time-Decay, bewertet per Walk-forward-Backtest.
Regeln und Konventionen stehen in [CLAUDE.md](CLAUDE.md).

## Setup
```bash
brew install uv
uv sync
```

## Daten laden (idempotent, alles wird unter `data/raw/` gecacht)
```bash
uv run fussball ingest football-data
uv run fussball ingest understat
uv run fussball ingest clubelo --suggest   # Vorschläge in data_seed/team_name_map.csv prüfen
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

## Täglicher Job (Snapshot + Vorhersage)
Verfügbarkeiten lassen sich nicht nachträglich rekonstruieren, deshalb läuft der
Transfermarkt-Snapshot täglich (ca. 100 Requests, ~9 Minuten), danach `predict`:
```bash
uv run fussball snapshot transfermarkt     # manuell
ops/install_launchd.sh                     # als launchd-Job, täglich 08:00
```
Log: `data/logs/daily.log`. Bei Blockade (403) bricht der Scraper mit Exit-Code 2 ab.
Neue Vereine (Aufsteiger) meldet `fussball check` als ungemappt → in
`data_seed/team_name_map.csv` ergänzen, dann `uv run fussball teams`.

## Backtest
```bash
uv run fussball backtest --tune   # ξ/reg auf 2016–2018 tunen, Report ab 2019/20
uv run fussball backtest          # mit gespeicherten Parametern
```
Ergebnis: `data/reports/backtest_report.md`, `backtest_summary.csv`,
`backtest_predictions.csv`, `score_distribution.png`.
