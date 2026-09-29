"""Datenqualitäts-Checks mit klarer Ausgabe. Exit-Code 1, wenn ein ERROR auftritt.

Geprüft: Duplikate, Spiele pro Liga/Saison gegen Sollwerte, symmetrischer Spielplan,
xG-/Elo-/Quoten-Abdeckung, ungemappte Teamnamen, Stand der Transfermarkt-Snapshots.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from fussball import config, db

# Abweichungen vom Standard (18 Teams -> 306, 20 Teams -> 380 Spiele)
EXPECTED_SPECIAL = {
    ("F1", 2019): 279,  # Ligue 1 2019/20 wegen COVID abgebrochen
    ("F2", 2019): 280,
    ("I2", 2018): 342,  # Serie B 2018/19 mit 19 Teams
}


@dataclass
class Result:
    level: str  # OK | INFO | WARN | ERROR
    check: str
    message: str


def expected_matches(league: str, season: int, n_teams: int) -> int:
    return EXPECTED_SPECIAL.get((league, season), n_teams * (n_teams - 1))


def check_duplicates(con) -> list[Result]:
    dup = db.read(con, """SELECT league, season, home_id, away_id, COUNT(*) n FROM matches
                          GROUP BY 1,2,3,4 HAVING n > 1""")
    if len(dup):
        return [Result("ERROR", "duplikate", f"{len(dup)} Paarungen mehrfach pro Saison:\n{dup.head(20).to_string(index=False)}")]
    xdup = db.read(con, "SELECT match_id, COUNT(*) n FROM team_xg WHERE match_id IS NOT NULL GROUP BY 1 HAVING n <> 2")
    if len(xdup):
        return [Result("ERROR", "duplikate", f"{len(xdup)} Spiele mit ≠ 2 xG-Zeilen")]
    return [Result("OK", "duplikate", "keine doppelten Paarungen / xG-Zeilen")]


def check_match_counts(con) -> list[Result]:
    df = db.read(con, """SELECT league, season, COUNT(*) n,
                         COUNT(DISTINCT home_id) teams FROM matches GROUP BY 1, 2""")
    current = config.current_season()
    out = []
    for r in df.itertuples():
        lg = config.LEAGUES[r.league]
        exp = expected_matches(r.league, r.season, r.teams)
        label = f"{r.league} {config.season_label(r.season)}"
        if r.season == current:
            out.append(Result("INFO", "spielanzahl", f"{label}: laufende Saison, {r.n}/{exp} Spiele"))
        elif r.n != exp:
            level = "ERROR" if lg.tier == 1 else "WARN"
            out.append(Result(level, "spielanzahl", f"{label}: {r.n} Spiele, erwartet {exp} ({r.teams} Teams)"))
    missing = [(lg, s) for lg in config.LEAGUES for s in config.seasons()
               if not ((df.league == lg) & (df.season == s)).any()]
    for lg, s in missing:
        level = "ERROR" if config.LEAGUES[lg].tier == 1 else "WARN"
        out.append(Result(level, "spielanzahl", f"{lg} {config.season_label(s)}: keine Spiele geladen"))
    if not any(r.level in ("ERROR", "WARN") for r in out):
        out.append(Result("OK", "spielanzahl", f"{len(df)} Liga-Saisons mit Sollwert (abgeschlossene Saisons)"))
    return out


def check_schedule_symmetry(con) -> list[Result]:
    """Abgeschlossene Saisons: jedes Team gleich viele Heim- und Auswärtsspiele."""
    df = db.read(con, """SELECT league, season, team, SUM(h) home, SUM(a) away FROM (
                           SELECT league, season, home_id team, 1 h, 0 a FROM matches
                           UNION ALL SELECT league, season, away_id, 0, 1 FROM matches)
                         GROUP BY 1,2,3""")
    df = df[df.season < config.current_season()]
    df = df[~df.set_index(["league", "season"]).index.isin(list(EXPECTED_SPECIAL))]
    bad = df[df.home != df.away]
    if len(bad):
        return [Result("WARN", "spielplan", f"{len(bad)} Team-Saisons mit ungleich vielen Heim-/Auswärtsspielen:\n"
                                            f"{bad.head(10).to_string(index=False)}")]
    return [Result("OK", "spielplan", "Heim-/Auswärtsspiele pro Team ausgeglichen")]


def check_xg_coverage(con) -> list[Result]:
    df = db.read(con, f"""SELECT m.league, m.season, COUNT(*) n, SUM(x.cnt IS NULL) missing FROM matches m
                         LEFT JOIN (SELECT match_id, COUNT(*) cnt FROM team_xg GROUP BY 1) x USING (match_id)
                         WHERE m.league IN ({",".join("?" * len(config.TOP_LEAGUES))})
                         GROUP BY 1, 2""", config.TOP_LEAGUES)
    bad = df[df.missing > 0]
    out = [Result("WARN", "xg", f"{r.league} {config.season_label(r.season)}: {r.missing}/{r.n} Spiele ohne xG")
           for r in bad.itertuples()]
    # jüngere Spiele fehlen bei football-data.co.uk oft noch (Update-Verzug) -> erst nach 14 Tagen melden
    unlinked = con.execute("SELECT COUNT(*) FROM team_xg WHERE match_id IS NULL "
                           "AND date < date('now', '-14 days')").fetchone()[0]
    if unlinked:
        out.append(Result("WARN", "xg", f"{unlinked // 2} Understat-Spiele ohne Gegenstück in matches"))
    if not len(df):
        out.append(Result("ERROR", "xg", "keine xG-Daten geladen"))
    elif not out:
        out.append(Result("OK", "xg", f"xG für alle {int(df.n.sum())} Top-5-Spiele vorhanden"))
    return out


def check_unmapped(con) -> list[Result]:
    out = []
    tm = db.read(con, """SELECT DISTINCT league, tm_club_name FROM squad_value_snapshots
                         WHERE team_id IS NULL""")
    if len(tm):
        out.append(Result("ERROR", "teamnamen", "Transfermarkt-Vereine ohne Mapping: "
                          + ", ".join(f"{r.tm_club_name} ({r.league})" for r in tm.itertuples())))
    cur = config.current_season()
    elo_missing = db.read(con, f"""SELECT DISTINCT m.league, t.name FROM matches m
                                  JOIN teams t ON t.team_id IN (m.home_id, m.away_id)
                                  WHERE m.season >= ? AND m.league IN ({",".join("?" * len(config.TOP_LEAGUES))})
                                  AND t.team_id NOT IN (SELECT team_id FROM team_name_map WHERE source = 'clubelo')""",
                          [cur - 1] + config.TOP_LEAGUES)
    if len(elo_missing):
        out.append(Result("WARN", "teamnamen", f"{len(elo_missing)} Top-5-Teams (seit {cur - 1}) ohne ClubElo-Mapping, "
                          "z.B. " + ", ".join(elo_missing.name.head(8))))
    if not out:
        out.append(Result("OK", "teamnamen", "alle Namen gemappt"))
    return out


def check_elo(con) -> list[Result]:
    df = db.read(con, f"""SELECT m.league, COUNT(*) n, SUM(e.team_id IS NULL) missing FROM matches m
                         LEFT JOIN elo e ON e.team_id = m.home_id AND date(m.date, '-1 day')
                              BETWEEN e.valid_from AND e.valid_to
                         WHERE m.league IN ({",".join("?" * len(config.TOP_LEAGUES))}) GROUP BY 1""",
                  config.TOP_LEAGUES)
    if not con.execute("SELECT COUNT(*) FROM elo").fetchone()[0]:
        return [Result("WARN", "elo", "keine Elo-Daten geladen (fussball ingest clubelo --suggest)")]
    bad = df[df.missing > 0]
    if len(bad):
        return [Result("WARN", "elo", f"{r.league}: {r.missing}/{r.n} Heimteams ohne Elo am Vortag")
                for r in bad.itertuples()]
    return [Result("OK", "elo", "Elo am Vortag für alle Top-5-Heimteams vorhanden")]


def check_odds(con) -> list[Result]:
    df = db.read(con, f"""SELECT m.league, COUNT(*) n, SUM(o.match_id IS NULL) missing FROM matches m
                         LEFT JOIN (SELECT DISTINCT match_id FROM odds_benchmark WHERE kind = 'pre') o USING (match_id)
                         WHERE m.league IN ({",".join("?" * len(config.TOP_LEAGUES))}) GROUP BY 1""",
                  config.TOP_LEAGUES)
    return [Result("INFO" if r.missing else "OK", "quoten",
                   f"{r.league}: Benchmark-Quoten für {r.n - r.missing}/{r.n} Spiele") for r in df.itertuples()]


def check_snapshots(con) -> list[Result]:
    df = db.read(con, """SELECT s.league, s.snapshot_date, COUNT(*) clubs,
                                (SELECT COUNT(*) FROM availability_snapshots a
                                 WHERE a.snapshot_date = s.snapshot_date AND a.league = s.league) absences
                         FROM squad_value_snapshots s
                         WHERE s.snapshot_date = (SELECT MAX(snapshot_date) FROM squad_value_snapshots)
                         GROUP BY 1, 2""")
    if df.empty:
        return [Result("WARN", "snapshots", "noch kein Transfermarkt-Snapshot")]
    n_days = con.execute("SELECT COUNT(DISTINCT snapshot_date) FROM squad_value_snapshots").fetchone()[0]
    last = df.snapshot_date.iloc[0]
    age = (pd.Timestamp.today().normalize() - pd.Timestamp(last)).days
    out = [Result("WARN" if age > 1 else "OK", "snapshots",
                  f"letzter Snapshot {last} ({age} Tage alt), {n_days} Snapshot-Tage insgesamt")]
    for r in df.itertuples():
        exp = 18 if r.league in ("D1", "F1") else 20
        level = "OK" if r.clubs == exp else "WARN"
        out.append(Result(level, "snapshots", f"{r.league}: {r.clubs}/{exp} Vereine, {r.absences} Ausfälle"))
    return out


CHECKS = [check_duplicates, check_match_counts, check_schedule_symmetry, check_xg_coverage,
          check_unmapped, check_elo, check_odds, check_snapshots]


def run(con: sqlite3.Connection, verbose: bool = False) -> int:
    results = [r for c in CHECKS for r in c(con)]
    icons = {"OK": "✓", "INFO": "·", "WARN": "!", "ERROR": "✗"}
    for r in results:
        if r.level == "INFO" and not verbose:
            continue
        print(f"{icons[r.level]} {r.level:<5} [{r.check}] {r.message}")
    counts = pd.Series([r.level for r in results]).value_counts().to_dict()
    print(f"\n{counts.get('ERROR', 0)} Fehler, {counts.get('WARN', 0)} Warnungen, {counts.get('OK', 0)} OK")
    return 1 if counts.get("ERROR") else 0
