# Fußball-Ergebnisvorhersage (Top 5 Ligen)

Privates ML-Projekt: exaktes Endergebnis jedes Spiels des **nächsten Spieltags** in
Premier League (E0), La Liga (SP1), Serie A (I1), Ligue 1 (F1) und Bundesliga (D1).
Vorhersage am Vortag, ohne bekannte Aufstellungen. Erfolgsmetrik: exakte Treffer
bei realistischer Tipp-Verteilung (Tendenz, Tippspiel-Punkte 4/3/2 im Blick);
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
uv run fussball ingest europe [--suggest]      # Europapokal-Termine (Transfermarkt)
uv run fussball ingest squad-values [--suggest] # Kaderwerte abgeschlossener Saisons (1. + 2. Ligen)
uv run fussball snapshot transfermarkt         # täglicher Snapshot Verletzungen/Sperren
uv run fussball teams                          # data_seed/team_name_map.csv -> DB
uv run fussball check [-v]                     # Datenqualität, Exit 1 bei ERROR
uv run fussball backtest [--tune]              # Walk-forward + Report in data/reports/
uv run fussball predict [--no-refresh]         # nächster Spieltag -> predictions + JSON
uv run fussball scorecard                      # echte Trefferbilanz
uv run fussball dashboard                      # data/dashboard/index.html (+ FUSSBALL_DASHBOARD_COPY)
```
Geplanter Lauf: `ops/daily.sh` via launchd (`ops/install_launchd.sh`), kein Dauerprozess.

## Aufbau
- `src/fussball/config.py` Ligen, Saisons (Startjahr: 2014 = 2014/15), Pfade
- `db.py` Schema + `upsert` (idempotent); `teams.py` Namens-Mapping
- `ingest/` je Quelle ein Modul: `fetch`/`parse`/`store`/`ingest`
- `models/dixon_coles.py` Dixon-Coles; `models/blend.py` aktuelles Modell: Dixon-Coles auf
  Toren + Dixon-Coles auf xG (eigener, stärkerer Time-Decay), log-λ gewichtet gemischt
- `features/form.py` point-in-time Team-Features (Ruhetage, xG-/Tor-Form, Chancenverwertung,
  Heim/Auswärts-Form, Tabelle, Saisonfortschritt); `models/lgbm.py` LightGBM-Korrekturstufe
  (Poisson, DC-log-λ als Offset), je Saison nur auf Vorsaisons trainiert
- `ingest/europe.py` Europapokal-Termine (CL/EL/ECL, Transfermarkt-Gesamtspielplan, ~32 Seiten);
  Vereine über TM-ID gemappt (Quelle `transfermarkt_id` in der Seed-Map). Belastungs-Features
  (`form.EU_FEATURES`) sind berechnet, aber NICHT im Modell: im Backtest kein Gewinn.
  Im Dashboard nur als Hinweis („Europapokal vor/in n Tagen“).
- `features/availability.py` Ausfall-Gewicht je Team: Anteil fehlender Minuten / xG+xA /
  Defensivminuten (Understat-Spielerdaten, Tabelle player_season_stats) aus dem letzten
  Snapshot vor dem Spieltag. Anpassung der Raten ist eine NICHT validierte Schätzung
  (keine Ausfall-Historie vor 2026-09-29): `predict` speichert jede Vorhersage doppelt
  (Modell mit Suffix ` +ausfaelle` und ohne) → Live-A/B im Dashboard. Sobald genug
  Snapshots vorliegen, Ausfälle datenbasiert ins LightGBM übernehmen.
- Ensemble (`EnsembleDixonColes`): Mittel der log-λ der 3 besten Tore+xG-Parametersätze
  der Validierung (model_params.json → "ensemble"); Test: ΔLog-Loss −0,0005 (signifikant).
  LightGBM-Runden per Early Stopping gemittelt über alle Validierungssaisons (nicht eine).
  Regel: Varianten nie nachträglich nach Testergebnis auswählen.
- `backtest/` Walk-forward, Metriken, Report, Benchmark; Parameter in
  data/reports/model_params.json (per `backtest --tune` nur auf Validierung gewählt),
  Live-LightGBM in data/reports/lgbm_model.txt (bei jedem `backtest` neu trainiert).
  Fehlt die Datei, sagt `predict` ohne LightGBM vorher.
- macOS: LightGBM braucht `brew install libomp`
- `data/` ist gitignored (Cache, `fussball.sqlite`, Reports); `data_seed/` ist kuratiert
  und committet.

## Konventionen
- Kanonische `team_id` = Slug des football-data.co.uk-Namens. Namen anderer Quellen
  nur über `data_seed/team_name_map.csv` (Vorschläge prüfen, dann committen).
- `predictions` ist append-only (Schlüssel model + Spiel + cutoff_date); nie nachträglich
  ändern, sonst ist die Live-Bilanz wertlos.
- Spielplan und frische Ergebnisse kommen aus Understat (UTC-Zeiten); football-data.co.uk
  hinkt einige Tage hinterher und ist die Quelle für Historie und 2. Ligen.
- 2. Ligen (E1, SP2, I2, F2, D2) sind reine Fit-Daten (Aufsteiger-Stärke), keine Ziele.
- Zeitliche Aufteilung im Backtest: 2014–15 Burn-in, 2016–2018 Validierung (Tuning),
  ab 2019/20 Test. Nie auf Testsaisons tunen.
- Dashboard ist eine statische HTML-Datei ohne JavaScript (muss in der iOS-Dateien-
  Vorschau funktionieren); Auf-/Zuklappen per Checkbox+Label (kein <details>), standard-
  mäßig eingeklappt. Zahlen im deutschen Format, Zeiten in Europe/Berlin.
- Anzeigenamen deutsch über `data_seed/team_display_names.csv` (Aufsteiger ergänzen);
  Liga-Farben in `dashboard.THEMES`; Wappen/Logos von Transfermarkt, einmalig gecacht
  unter data/raw/transfermarkt/{crests,logos}/ und base64 eingebettet.
- Haupttipp = realistische Strategie (`models/tipping.py`): erst Ausgang über p·w, dann
  wahrscheinlichstes Ergebnis darin; Gewichte w so, dass die Tipps sich wie echte Ergebnisse
  verteilen (Backtest: je Saison aus 3 Vorsaisons; live: data/reports/tip_weights.json).
  Test ab 2019: Tendenz 50,8 % statt 40,1 %, exakt 12,1 % statt 12,9 %, Punkte 1,36 statt 1,10.
  Das wahrscheinlichste Einzelergebnis (argmax, meist 1:1) bleibt als pred_h/pred_a gespeichert.
- Getestet und verworfen (kein Gewinn im Backtest, Details im Git-Log):
  Zell-Kalibrierung der Matrix; Europapokal-Belastung; direkte Duelle (letzte 6 H2H,
  Tordifferenz/Torsumme); „Tor-Umfeld“ (Torsumme der letzten Spiele beider Teams).
  Tor-/Gegentor-Form ist bereits über form_gf/ga und form_xgf/xga im LightGBM.
  Außerdem (2026-09-30): npxG statt xG (schlechter: Elfmeter sind teils Können),
  teamspezifischer Heimvorteil mit L2 (schlechter, auch stark regularisiert),
  realistische Tordifferenz-Verteilung der Tipps (−0,8 Pp. exakt, −2,4 Pp. Tordifferenz).
  Weitere LightGBM-Kandidaten ohne robusten Gewinn (Validierung schlechter, Test n.s.):
  Aufsteiger-Flag, Englische Woche, Schiedsrichter (nur E0), Tage seit Trainerwechsel,
  Kaderwert der Vorsaison (`ingest/squad_values.py`, Tabelle squad_values_history; hilft im
  Test am Saisonbeginn, auf der Validierung nicht → in einem Jahr mit mehr Saisons erneut prüfen).
  Kaderwerte vergangener Saisons auf Transfermarkt sind Werte NACH dem Saisonverlauf →
  immer nur die Vorsaison verwenden.
- Code und Kommentare auf Deutsch, Bezeichner auf Englisch.
