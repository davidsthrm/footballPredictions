"""SQLite-Schema und Hilfsfunktionen.

Regel: `odds_benchmark` hat keine Beziehung zu Feature-Tabellen und wird nur in
`fussball.backtest.benchmark` gelesen (abgesichert durch tests/test_no_odds_leakage.py).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from fussball import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id   TEXT PRIMARY KEY,          -- kanonischer Slug, abgeleitet vom football-data-Namen
    name      TEXT NOT NULL,
    country   TEXT NOT NULL
);

-- Vereinheitlicht Teamnamen über alle Quellen (football-data, understat, clubelo, transfermarkt)
CREATE TABLE IF NOT EXISTS team_name_map (
    source      TEXT NOT NULL,
    source_name TEXT NOT NULL,
    team_id     TEXT NOT NULL REFERENCES teams(team_id),
    PRIMARY KEY (source, source_name)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id  INTEGER PRIMARY KEY,
    league    TEXT NOT NULL,
    season    INTEGER NOT NULL,           -- Startjahr, 2014 = 2014/15
    date      TEXT NOT NULL,              -- ISO-Datum (Ortszeit UK laut Quelle)
    kickoff   TEXT,                       -- HH:MM, ab 2019/20
    home_id   TEXT NOT NULL REFERENCES teams(team_id),
    away_id   TEXT NOT NULL REFERENCES teams(team_id),
    fthg INTEGER NOT NULL, ftag INTEGER NOT NULL,
    hthg INTEGER, htag INTEGER,
    hs INTEGER, "as" INTEGER, hst INTEGER, ast INTEGER,
    hf INTEGER, af INTEGER, hc INTEGER, ac INTEGER,
    hy INTEGER, ay INTEGER, hr INTEGER, ar INTEGER,
    referee TEXT,
    UNIQUE (league, date, home_id, away_id)
);
CREATE INDEX IF NOT EXISTS ix_matches_league_date ON matches(league, date);

-- xG pro Team und Spiel (Understat). match_id verlinkt auf matches, NULL = nicht zuordenbar
CREATE TABLE IF NOT EXISTS team_xg (
    understat_match_id INTEGER NOT NULL,
    team_id    TEXT NOT NULL REFERENCES teams(team_id),
    match_id   INTEGER REFERENCES matches(match_id),
    league     TEXT NOT NULL,
    season     INTEGER NOT NULL,
    date       TEXT NOT NULL,
    is_home    INTEGER NOT NULL,
    opponent_id TEXT NOT NULL,
    goals INTEGER, goals_against INTEGER,
    xg REAL, xga REAL, npxg REAL, npxga REAL,
    ppda_att INTEGER, ppda_def INTEGER, ppda_allowed_att INTEGER, ppda_allowed_def INTEGER,
    deep INTEGER, deep_allowed INTEGER,
    PRIMARY KEY (understat_match_id, team_id)
);
CREATE INDEX IF NOT EXISTS ix_team_xg_team_date ON team_xg(team_id, date);

-- ClubElo: Rating gilt im Intervall [valid_from, valid_to]
CREATE TABLE IF NOT EXISTS elo (
    team_id    TEXT NOT NULL REFERENCES teams(team_id),
    valid_from TEXT NOT NULL,
    valid_to   TEXT NOT NULL,
    elo        REAL NOT NULL,
    level      INTEGER,
    PRIMARY KEY (team_id, valid_from)
);

-- NUR Evaluations-Benchmark. Niemals in Feature-/Fit-Daten joinen.
CREATE TABLE IF NOT EXISTS odds_benchmark (
    match_id  INTEGER NOT NULL,
    bookmaker TEXT NOT NULL,              -- z.B. PS (Pinnacle), Avg, B365
    kind      TEXT NOT NULL CHECK (kind IN ('pre', 'closing')),
    odds_h REAL, odds_d REAL, odds_a REAL,
    PRIMARY KEY (match_id, bookmaker, kind)
);

-- Tägliche Transfermarkt-Snapshots (historisch nicht rekonstruierbar!)
CREATE TABLE IF NOT EXISTS availability_snapshots (
    snapshot_date TEXT NOT NULL,          -- Kalendertag des Snapshots (Europe/Berlin)
    snapshot_ts   TEXT NOT NULL,          -- exakter Abrufzeitpunkt UTC
    league        TEXT NOT NULL,
    tm_club_id    INTEGER NOT NULL,
    team_id       TEXT REFERENCES teams(team_id),
    tm_player_id  INTEGER NOT NULL,
    player        TEXT NOT NULL,
    position      TEXT,
    age           INTEGER,
    category      TEXT NOT NULL,          -- injury | suspension | other | suspension_threat
    reason        TEXT,
    since         TEXT,
    until         TEXT,
    missed_games  INTEGER,
    PRIMARY KEY (snapshot_date, tm_club_id, tm_player_id, category)
);

-- Aktueller Spielplan (noch nicht gespielte Spiele, Quelle Understat). Wird pro
-- Liga/Saison bei jedem Ingest komplett ersetzt, Verlegungen werden so übernommen.
CREATE TABLE IF NOT EXISTS fixtures (
    understat_match_id INTEGER PRIMARY KEY,
    league      TEXT NOT NULL,
    season      INTEGER NOT NULL,
    kickoff_utc TEXT NOT NULL,
    date        TEXT NOT NULL,
    home_id     TEXT NOT NULL REFERENCES teams(team_id),
    away_id     TEXT NOT NULL REFERENCES teams(team_id),
    fetched_at  TEXT
);

-- Gespeicherte Live-Vorhersagen. Nie überschreiben: jede Vorhersage ist mit Cutoff
-- versioniert, damit die echte Trefferbilanz ehrlich messbar bleibt.
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,            -- UTC
    model       TEXT NOT NULL,            -- z.B. 'dixon-coles xi=0.002 reg=1.0'
    cutoff_date TEXT NOT NULL,            -- Fit nutzte nur Ergebnisse mit date < cutoff_date
    understat_match_id INTEGER NOT NULL,
    league      TEXT NOT NULL,
    season      INTEGER NOT NULL,
    kickoff_utc TEXT NOT NULL,
    date        TEXT NOT NULL,
    home_id     TEXT NOT NULL,
    away_id     TEXT NOT NULL,
    lam_h REAL, lam_a REAL, rho REAL,
    p_h REAL, p_d REAL, p_a REAL,
    pred_h INTEGER NOT NULL, pred_a INTEGER NOT NULL, p_pred REAL,
    matrix      TEXT NOT NULL,            -- JSON, 10x10 Wahrscheinlichkeiten (Heim-Zeilen)
    UNIQUE (model, understat_match_id, cutoff_date)
);

CREATE TABLE IF NOT EXISTS squad_value_snapshots (
    snapshot_date TEXT NOT NULL,
    snapshot_ts   TEXT NOT NULL,
    league        TEXT NOT NULL,
    tm_club_id    INTEGER NOT NULL,
    tm_club_name  TEXT NOT NULL,
    team_id       TEXT REFERENCES teams(team_id),
    squad_size    INTEGER,
    avg_age       REAL,
    total_value_eur REAL,
    PRIMARY KEY (snapshot_date, tm_club_id)
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, table: str, rows: Iterable[dict] | pd.DataFrame,
           key: Sequence[str]) -> int:
    """INSERT … ON CONFLICT(key) DO UPDATE. Idempotent; bestehende Primärschlüssel bleiben stabil."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.astype(object).where(rows.notna(), None).to_dict("records")
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0])
    quoted = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c not in key)
    conflict = "DO UPDATE SET " + updates if updates else "DO NOTHING"
    sql = (f'INSERT INTO {table} ({quoted}) VALUES ({placeholders}) '
           f'ON CONFLICT ({", ".join(key)}) {conflict}')
    with con:
        con.executemany(sql, rows)
    return len(rows)


def read(con: sqlite3.Connection, sql: str, params: Sequence | dict = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)
