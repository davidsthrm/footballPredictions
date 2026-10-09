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
uv run fussball ingest other-leagues           # 16 weitere Ligen (NED, POR, TUR, AUT, …) für das Europa-Modell
uv run fussball ingest understat-matches       # Spielberichte (Tore, Spielerwerte) der laufenden Saison
uv run fussball snapshot transfermarkt         # täglicher Snapshot Verletzungen/Sperren
uv run fussball teams                          # data_seed/team_name_map.csv -> DB
uv run fussball check [-v]                     # Datenqualität, Exit 1 bei ERROR
uv run fussball backtest [--tune]              # Walk-forward + Report in data/reports/
uv run fussball backtest-europe [--tune]       # Walk-forward CL/EL/ECL -> europe_params.json, tip_weights_eu.json
uv run fussball predict [--no-refresh]         # nächster Spieltag -> predictions + JSON
uv run fussball scorecard                      # echte Trefferbilanz
uv run fussball dashboard                      # data/dashboard/index.html (+ FUSSBALL_DASHBOARD_COPY)
```
Geplanter Lauf: `ops/daily.sh` via launchd (`ops/install_launchd.sh`), kein Dauerprozess.
Sofort-Lauf: Knopf „Jetzt aktualisieren“ im Dashboard (nur Mac) öffnet `tippspiel://refresh` →
`~/Applications/Tippspiel Refresh.app` (`ops/install_refresh.sh`) → `launchctl kickstart` desselben Jobs.
`predict` tippt angepfiffene Spiele nicht neu (sonst überschriebe ein später Lauf den Tipp vor Anpfiff).

## Aufbau
- `src/fussball/config.py` Ligen, Saisons (Startjahr: 2014 = 2014/15), Pfade
- `db.py` Schema + `upsert` (idempotent); `teams.py` Namens-Mapping
- `ingest/` je Quelle ein Modul: `fetch`/`parse`/`store`/`ingest`
- `models/dixon_coles.py` Dixon-Coles; `models/blend.py` aktuelles Modell: Dixon-Coles auf
  Toren + Dixon-Coles auf xG (eigener, stärkerer Time-Decay), log-λ gewichtet gemischt
- `features/form.py` point-in-time Team-Features (Ruhetage, xG-/Tor-Form, Chancenverwertung,
  Heim/Auswärts-Form, Tabelle, Saisonfortschritt, Form der Deep Completions und PPDA aus Understat –
  seit 2026-10-08, Validierung ΔLog-Loss −0,0002, Test −0,0006 bei z −3,8); `models/lgbm.py` LightGBM-Korrekturstufe
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
- Europapokal (CL, EL, Conference League): `ingest/europe.py` liest aus den Transfermarkt-
  Gesamtspielplänen auch Ergebnisse und Namen (keine Anstoßzeiten, Verlängerung nicht markiert);
  `ingest/other_leagues.py` holt 16 weitere Ligen (nur Datum/Teams/Tore, Tabelle other_matches);
  `models/europe.py` ein gemeinsamer Dixon-Coles über alle Ligen + Europapokal (Heimvorteil je
  Wettbewerb, Europapokal-Spiele verbinden die Ligen; Vereine ohne Ligadaten als `tm-<id>`).
  Vereine über TM-ID gemappt (geprüfte Vorschläge in der Seed-Map, Zypern & Co. bewusst nicht).
  Backtest (`backtest/europe.py`, Tuning 2016–2018): Test ab 2019/20, 2.895 Spiele, Tendenz 52,2 %,
  exakt 10,5 %, 1,35 Punkte; nur Heimvorteil 39,2 % / 1,05; weitere Ligen ΔLog-Loss −0,021 (z −3,4).
  Live: `predict_europe.py` (Tabelle europe_predictions, append-only; angepfiffene Spiele nicht neu
  tippen – Anstoßzeit aus den Kopfzeilen der Transfermarkt-Seite, sobald angesetzt),
  Dashboard-Reiter „Europa“ (`dashboard_europe.py`, Farben je Wettbewerb in THEMES).
  `ingest/europe.py` speichert je Spiel Runde (Ligaphase/K.-o.-Runde, Hin-/Rückspiel), Spieltag
  (aus Terminblöcken) und Anstoß; Spiele, die nicht mehr auf der Seite stehen (verlegt), werden gelöscht.
  `tournament.py`: Turnierprognose (10.000 Simulationen der Ligaphase + K.-o.-Baum nach UEFA-Setzliste,
  Format ab 2024/25), im Dashboard als Baum + Titelchancen, nur solange die Ligaphase läuft.
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
  Gestaltung: Titel „Dein Tippspiel-Assistent“, Schrift Plus Jakarta Sans (SIL OFL, variable woff2
  unter data_seed/fonts/, base64 eingebettet). Erklärtexte nur, wo nötig. Reiter Spiele/Ergebnisse/
  Tabelle/Fakten mit Liga-Sprungknöpfen; Spiele und Ergebnisse je Liga und Tag gruppiert, eine
  Zeile pro Spiel, Details (1X2, Vergleichstabelle Heim/Auswärts, Hinweise, Matrix) erst nach
  Antippen. Emojis sparsam als Icons (Fakten, Detail-Zeilen, Punkte), keine Schatten/Verläufe,
  Texte sachlich. Spielinfo je Spiel (Spieltag aus Zahl der Ligaspiele, Derbys in `dashboard.DERBIES`,
  „Banker“ 🔥 ab 70 % Tendenz). 1X2-Leiste hebt den wahrscheinlichsten Ausgang hervor, der Tipp ist
  markiert (weicht er ab, kurzer Hinweis 🎲 zur Tipp-Strategie). Animationen und Easter Eggs nur per CSS
  in `dashboard_fx.py` (Titel 3× antippen, 🏟️ Flutlicht-Modus in der Fußzeile, Pokal im Baum; aus bei
  „Bewegung reduzieren“). Ergebnisse aufklappbar: Tore/Vorlagen und „Top-Spieler laut Daten“ aus den
  Understat-Spielberichten (`ingest/understat_matches.py`, Tabellen match_goals/match_players, kein
  Modell-Input; offizielle Spieler des Spiels gibt es nicht frei), Schüsse/Ecken/Karten von football-data.
  Modell-Bilanz (gesamt, je Liga, Saison, Spieltag, Modellvergleich, Europapokal) im Reiter Ergebnisse.
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
  Außerdem (2026-10-08, kein Gewinn auf der Validierung): eigenes Elo aus Ergebnissen (K 15/20/25,
  mit Tordifferenz, 1.+2. Liga als ein Pool, auch mit Momentum), xG-Elo, Pi-Ratings, Form der
  Schüsse aufs Tor, xG-Trend (kurze gegen lange Form). Dixon-Coles misst die Stärke schon.
  ClubElo-API (api.clubelo.com) lieferte am 2026-10-08 durchgehend 502; die neue Website zeigt
  Verläufe erst ab 2022 → für Validierung 2016–2018 unbrauchbar.
- Code und Kommentare auf Deutsch, Bezeichner auf Englisch.
