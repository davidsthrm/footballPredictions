"""Weitere europäische Ligen von football-data.co.uk -> other_matches (nur Ergebnisse).

Zweck: Stärke der Europapokal-Gegner außerhalb der Top-5-Ligen (Benfica, PSV, Celtic, …)
schätzen. Gelesen werden ausschließlich Datum, Teams und Tore – keine weiteren Spalten.

- Hauptligen im gewohnten Format, eine Datei je Saison: mmz4281/{saison}/{code}.csv
- "Extra"-Ligen, eine Datei mit allen Saisons: new/{code}.csv (teils Kalenderjahr-Saisons)

Teams bekommen wie gewohnt team_id = Slug des football-data-Namens (Land = Code unten).
"""

from __future__ import annotations

import io
import logging
import sqlite3
import time

import pandas as pd

from fussball import config, db, teams
from fussball.http import Fetcher
from fussball.ingest.football_data import _decode, _parse_date, season_code

log = logging.getLogger(__name__)

MAIN = {"N1": "NED", "B1": "BEL", "P1": "POR", "SC0": "SCO", "T1": "TUR", "G1": "GRE"}
EXTRA = {"AUT": "AUT", "SWZ": "SUI", "DNK": "DEN", "NOR": "NOR", "SWE": "SWE", "POL": "POL",
         "ROU": "ROU", "RUS": "RUS", "FIN": "FIN", "IRL": "IRL"}
MAIN_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
EXTRA_URL = "https://www.football-data.co.uk/new/{league}.csv"
MAX_AGE_H = 24
MIN_DATE = f"{config.FIRST_SEASON}-07-01"

SCHEMA = """
CREATE TABLE IF NOT EXISTS other_matches (
    league  TEXT NOT NULL,
    date    TEXT NOT NULL,
    home_id TEXT NOT NULL REFERENCES teams(team_id),
    away_id TEXT NOT NULL REFERENCES teams(team_id),
    hg      INTEGER NOT NULL,
    ag      INTEGER NOT NULL,
    PRIMARY KEY (league, date, home_id, away_id)
);
"""
# Spaltennamen der beiden Formate -> einheitlich
COLUMNS = {"Date": "date", "HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag",
           "Home": "home", "Away": "away", "HG": "hg", "AG": "ag"}


def parse(raw: bytes, league: str) -> pd.DataFrame:
    """Gespielte Spiele (date, home, away, hg, ag) ab der ersten Projektsaison."""
    head = pd.read_csv(io.StringIO(_decode(raw)), nrows=0).columns
    use = [c for c in head if c.strip() in COLUMNS]
    df = pd.read_csv(io.StringIO(_decode(raw)), usecols=use, dtype=str)
    df = df.rename(columns=lambda c: COLUMNS[c.strip()]).dropna(subset=["home", "away", "hg", "ag"])
    df["date"] = df.date.map(_parse_date)
    df["hg"] = pd.to_numeric(df.hg, errors="coerce")
    df["ag"] = pd.to_numeric(df.ag, errors="coerce")
    df = df.dropna(subset=["hg", "ag"])
    df = df[df.date >= MIN_DATE].assign(home=df.home.str.strip(), away=df.away.str.strip(), league=league)
    return df.astype({"hg": int, "ag": int}).reset_index(drop=True)


def store(con: sqlite3.Connection, df: pd.DataFrame, country: str) -> int:
    if df.empty:
        return 0
    con.executescript(SCHEMA)
    teams.register_canonical(con, {n: country for n in pd.unique(pd.concat([df.home, df.away]))})
    rows = df.assign(home_id=df.home.map(teams.slug), away_id=df.away.map(teams.slug))
    db.upsert(con, "other_matches", rows[["league", "date", "home_id", "away_id", "hg", "ag"]],
              key=["league", "date", "home_id", "away_id"])
    return len(rows)


def _get(fetcher: Fetcher, url: str, name: str, current: bool) -> bytes | None:
    path = fetcher.path(name)
    refresh = current and path.exists() and time.time() - path.stat().st_mtime > MAX_AGE_H * 3600
    try:
        return fetcher.get(url, name, refresh=refresh)
    except FileNotFoundError:
        log.warning("Keine Datei: %s", url)
        return None


def ingest(con: sqlite3.Connection, seasons: list[int] | None = None,
           fetcher: Fetcher | None = None) -> pd.DataFrame:
    fetcher = fetcher or Fetcher("football-data", min_interval=2.0)
    current = config.current_season()
    summary = []
    for season in seasons or config.seasons():
        for league, country in MAIN.items():
            code = season_code(season)
            raw = _get(fetcher, MAIN_URL.format(code=code, league=league), f"{code}/{league}.csv", season == current)
            if raw:
                summary.append({"league": league, "season": season, "matches": store(con, parse(raw, league), country)})
    for league, country in EXTRA.items():
        raw = _get(fetcher, EXTRA_URL.format(league=league), f"new/{league}.csv", True)
        if raw:
            summary.append({"league": league, "season": "alle", "matches": store(con, parse(raw, league), country)})
    return pd.DataFrame(summary)
