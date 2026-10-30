# Dein Tippspiel-Assistent

Ein privater Tipp-Helfer für Fußball: Er sagt das **exakte Endergebnis** jedes Spiels am
nächsten Spieltag von Premier League, La Liga, Serie A, Ligue 1 und Bundesliga voraus – und
dazu die nächste Runde von Champions League, Europa League und Conference League.

Gedacht fürs Tippspiel unter Freunden (4/3/2 Punkte für exakt / Tordifferenz / Tendenz):
Die Tipps sollen nicht nur oft stimmen, sondern auch so aussehen wie echte Ergebnisse –
also nicht jedes Mal 1:1.

## Was man bekommt

Jeden Morgen um 8 Uhr ein frisches Dashboard, auf dem iPhone als Web-App wie eine
native App, hinter einem Login.

- **Spiele** – der nächste Spieltag je Liga mit Tipp und Tendenz-Wahrscheinlichkeit. Nach
  Antippen: 1X2, erwartete Tore, Form, Ausfälle, Hinweise (Derby, „Banker“,
  Europapokal unter der Woche) und die komplette Ergebnismatrix.
- **Europa** – die nächste Runde der drei Europapokale und, solange die Ligaphase läuft,
  eine Turnierprognose mit K.-o.-Baum und Titelchancen.
- **Ergebnisse** – alle Spiele der Saison mit Tipp und Punkten, Torschützen und
  „Top-Spieler laut Daten“, dazu die Bilanz des Modells nach Liga, Saison und Spieltag.
- **Tabellen** – aktuelle Tabellen mit Prognose der Endplatzierung.
- **Fakten** – auffällige Zahlen der Woche.

## Wie gut es tippt

Bewertet wird ehrlich: Jede Vorhersage nutzt nur Daten bis zum Vortag, getestet wird
auf Saisons, die beim Entwickeln nie angefasst wurden (ab 2019/20, über 12.000 Spiele).

|                    | Tendenz richtig | exakt  | Punkte pro Spiel |
|--------------------|-----------------|--------|------------------|
| Top-5-Ligen        | 50,8 %          | 12,1 % | 1,36             |
| Europapokal        | 52,2 %          | 10,5 % | 1,35             |

Live-Tipps werden gespeichert und nie nachträglich geändert, die echte Bilanz steht im
Dashboard.

## Wie es funktioniert

- **Teamstärke** aus Toren und Expected Goals (xG), jüngere Spiele zählen mehr
  (Dixon-Coles, mehrere Varianten gemittelt).
- **Korrektur** durch LightGBM mit Form, Ruhetagen, Heim-/Auswärtsform, Tabellenlage u. a.
- **Ausfälle** (Verletzungen, Sperren) aus einem täglichen Snapshot – noch als Live-A/B-Test,
  weil es dafür keine Historie gibt.
- **Europapokal**: ein gemeinsames Modell über 26 Ligen, verbunden über die
  Europapokal-Spiele.
- **Tipp-Strategie**: erst den Ausgang wählen, dann das wahrscheinlichste Ergebnis darin –
  so verteilen sich die Tipps wie echte Ergebnisse.

Bewusst **ohne Wettquoten** und nur mit **kostenlosen Daten** (football-data.co.uk,
Understat, Transfermarkt). Was getestet und verworfen wurde, steht in [CLAUDE.md](CLAUDE.md).

## Technik

Python (uv), pandas, SQLite, scipy, scikit-learn, LightGBM. Läuft auf dem Mac per launchd
einmal am Tag, kein Server; das Dashboard ist eine einzige HTML-Datei ohne JavaScript,
gehostet auf Cloudflare Pages.

```bash
uv sync && uv run pytest        # einrichten und testen (LightGBM braucht brew install libomp)
uv run fussball predict         # nächsten Spieltag tippen
uv run fussball dashboard       # Dashboard bauen
```

Alle Befehle und Regeln des Projekts: [CLAUDE.md](CLAUDE.md).
