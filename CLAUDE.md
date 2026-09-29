# Fußball-Ergebnisvorhersage (Top 5 Ligen)

Privates ML-Projekt: exaktes Endergebnis jedes Spiels des **nächsten Spieltags** in
Premier League (E0), La Liga (SP1), Serie A (I1), Ligue 1 (F1) und Bundesliga (D1).
Vorhersage am Vortag, ohne bekannte Aufstellungen. Erfolgsmetrik: exakte Treffer;
Modellauswahl über Log-Loss auf der Ergebnismatrix (Trefferquote ist zu verrauscht).

## Harte Regeln
- **Keine Wettquoten als Features.** Quoten liegen nur in `odds_benchmark` und werden
  ausschließlich in `src/fussball/backtest/benchmark.py` gelesen (RPS-Benchmark).
  Nie in Feature- oder Fit-Daten joinen. Abgesichert durch `tests/test_no_odds_leakage.py`
  – neue Module mit Quoten-Zugriff dort NICHT einfach auf die Whitelist setzen.
- **Point-in-time-Korrektheit.** Features/Fits für ein Spiel am Tag D nutzen nur Daten
  mit Zeitstempel < D (Cutoff = Vortag nach Spielschluss, `lag_days=1`). ClubElo-Ratings
  mit `valid_from = D` sind erst nach den Spielen von D-1 bekannt. Kein zufälliger
  Train/Test-Split, nur Walk-forward.
- **Nur kostenlose Datenquellen.**
- **Stack bewusst simpel:** Python/uv, pandas, SQLite, requests + BeautifulSoup, scipy,
  scikit-learn, LightGBM, matplotlib, pytest. Kein Polars/Spark/DuckDB/Cloud.
  Zusätzliche Dependencies nur mit Begründung.
- **Höflich scrapen:** `fussball.http.Fetcher` benutzen (Rate-Limit, eigener User-Agent,
  Cache unter `data/raw/`, nie doppelt abrufen). Bei 403/Bot-Schutz abbrechen, nichts
  umgehen (keine Proxies, keine gefälschten Browser-Header).
- Secrets (`FOOTBALL_DATA_ORG_API_KEY`) nur in `.env`, nie committen.

## Befehle
```bash
uv run pytest                                  # Tests (ohne Netzwerk)
uv run fussball ingest football-data           # Ergebnisse + Statistik, 1. und 2. Ligen
uv run fussball ingest understat [--suggest]   # xG; --suggest schlägt Teamnamen vor
uv run fussball ingest clubelo [--suggest]     # Elo-Historien der gemappten Vereine
uv run fussball snapshot transfermarkt         # täglicher Snapshot Verletzungen/Sperren
uv run fussball teams                          # data_seed/team_name_map.csv -> DB
uv run fussball check [-v]                     # Datenqualität, Exit 1 bei ERROR
uv run fussball backtest [--tune]              # Walk-forward + Report in data/reports/
```

## Aufbau
- `src/fussball/config.py` Ligen, Saisons (Startjahr: 2014 = 2014/15), Pfade
- `db.py` Schema + `upsert` (idempotent); `teams.py` Namens-Mapping
- `ingest/` je Quelle ein Modul: `fetch`/`parse`/`store`/`ingest`
- `models/dixon_coles.py` Baseline; `backtest/` Walk-forward, Metriken, Report, Benchmark
- `data/` ist gitignored (Cache, `fussball.sqlite`, Reports); `data_seed/` ist kuratiert
  und committet.

## Konventionen
- Kanonische `team_id` = Slug des football-data.co.uk-Namens. Namen anderer Quellen
  nur über `data_seed/team_name_map.csv` (Vorschläge prüfen, dann committen).
- 2. Ligen (E1, SP2, I2, F2, D2) sind reine Fit-Daten (Aufsteiger-Stärke), keine Ziele.
- Zeitliche Aufteilung im Backtest: 2014–15 Burn-in, 2016–2018 Validierung (Tuning),
  ab 2019/20 Test. Nie auf Testsaisons tunen.
- Code und Kommentare auf Deutsch, Bezeichner auf Englisch.
