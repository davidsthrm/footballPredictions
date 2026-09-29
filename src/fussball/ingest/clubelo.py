"""ClubElo: Elo-Historie pro Verein -> elo.

API (CSV, Spalten Rank,Club,Country,Level,Elo,From,To):
- api.clubelo.com/YYYY-MM-DD  -> alle Vereine an einem Tag (für die Namensfindung)
- api.clubelo.com/<Club>       -> komplette Historie eines Vereins (Name ohne Leerzeichen)

Point-in-time-Hinweis für Features: Ein Rating mit valid_from = D ist erst nach den
Spielen von D-1 bekannt; für ein Spiel am Tag D nur Ratings mit valid_from <= D-1 nutzen.
"""

from __future__ import annotations

import difflib
import io
import logging
import sqlite3
from datetime import date

import pandas as pd

from fussball import config, db, teams
from fussball.http import Fetcher

log = logging.getLogger(__name__)

URL = "http://api.clubelo.com/{what}"
SOURCE = "clubelo"
MIN_DATE = f"{config.FIRST_SEASON - 1}-07-01"  # ein Jahr Vorlauf vor der ersten Saison


def parse(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={"Club": "club", "Country": "country", "Level": "level",
                              "Elo": "elo", "From": "valid_from", "To": "valid_to"})


def snapshot_dates(seasons: list[int]) -> list[str]:
    today = date.today().isoformat()
    return [d for s in seasons for d in (f"{s}-09-01", f"{s + 1}-03-01") if d <= today]


def discover_clubs(fetcher: Fetcher, seasons: list[int]) -> pd.DataFrame:
    """Alle Vereine der relevanten Länder in Liga 1/2 an Stichtagen je Saison."""
    countries = {lg.country for lg in config.LEAGUES.values()}
    frames = []
    for d in snapshot_dates(seasons):
        df = parse(fetcher.get(URL.format(what=d), f"day/{d}.csv"))
        frames.append(df[df.country.isin(countries) & df.level.isin([1, 2])][["club", "country"]])
    return pd.concat(frames).drop_duplicates().reset_index(drop=True)


def suggest_names(con: sqlite3.Connection, clubs: pd.DataFrame) -> list[dict]:
    """Exakte Treffer (nach slug) + beste difflib-Ähnlichkeit ≥ 0.75 innerhalb desselben Landes."""
    t = db.read(con, "SELECT team_id, name, country FROM teams")
    known = teams.name_map(con, SOURCE)
    out = []
    for c in clubs.itertuples():
        if c.club in known:
            continue
        cand = t[t.country == c.country]
        exact = cand[cand.team_id == teams.slug(c.club)]
        if len(exact):
            out.append({"source": SOURCE, "source_name": c.club, "team_id": exact.team_id.iloc[0]})
            continue
        best = difflib.get_close_matches(c.club, cand.name.tolist(), n=1, cutoff=0.75)
        if best:
            tid = cand[cand.name == best[0]].team_id.iloc[0]
            out.append({"source": SOURCE, "source_name": c.club, "team_id": tid, "fuzzy": True})
    return out


def store(con: sqlite3.Connection, hist: pd.DataFrame, team_id: str) -> int:
    h = hist[hist.valid_to >= MIN_DATE].assign(team_id=team_id)
    return db.upsert(con, "elo", h[["team_id", "valid_from", "valid_to", "elo", "level"]],
                     key=["team_id", "valid_from"])


def ingest(con: sqlite3.Connection, fetcher: Fetcher | None = None, refresh: bool = False) -> pd.DataFrame:
    """Lädt die Historie aller gemappten Vereine. `refresh` holt die Historien neu (z.B. wöchentlich)."""
    fetcher = fetcher or Fetcher("clubelo", min_interval=2.0)
    summary = []
    for club, team_id in sorted(teams.name_map(con, SOURCE).items()):
        key = club.replace(" ", "")
        raw = fetcher.get(URL.format(what=key), f"club/{key}.csv", refresh=refresh)
        n = store(con, parse(raw), team_id)
        summary.append({"club": club, "team_id": team_id, "rows": n})
    return pd.DataFrame(summary)
