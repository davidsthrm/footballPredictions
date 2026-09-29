"""football-data.co.uk: Ergebnisse + Spielstatistiken -> matches, 1X2-Quoten -> odds_benchmark.

Quoten-Spalten werden direkt beim Parsen abgetrennt; der Match-DataFrame enthält
ausschließlich die Whitelist-Spalten aus STAT_COLUMNS.
"""

from __future__ import annotations

import csv
import io
import logging
import sqlite3
import time
from datetime import datetime

import pandas as pd

from fussball import config, db, teams
from fussball.http import Fetcher

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
CURRENT_SEASON_MAX_AGE_H = 12

# Quellspalte -> DB-Spalte. Alles andere gilt als Quote bzw. wird verworfen.
STAT_COLUMNS = {
    "Div": "div", "Date": "date", "Time": "kickoff", "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "fthg", "FTAG": "ftag", "HTHG": "hthg", "HTAG": "htag", "Referee": "referee",
    "HS": "hs", "AS": "as", "HST": "hst", "AST": "ast", "HF": "hf", "AF": "af",
    "HC": "hc", "AC": "ac", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
}
INT_COLUMNS = ["fthg", "ftag", "hthg", "htag", "hs", "as", "hst", "ast", "hf", "af",
               "hc", "ac", "hy", "ay", "hr", "ar"]
BOOKMAKER_ALIASES = {"BbAv": "Avg", "BbMx": "Max"}


def season_code(season: int) -> str:
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_rows(raw: bytes) -> pd.DataFrame:
    """CSV robust einlesen: manche Dateien haben Zeilen mit mehr/weniger Feldern als der Header."""
    rows = list(csv.reader(io.StringIO(_decode(raw))))
    header = [h.strip() for h in rows[0]]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in rows[1:] if any(c.strip() for c in r)]
    df = pd.DataFrame(body, columns=header)
    df = df.loc[:, [c for c in df.columns if c]]
    return df[df["HomeTeam"].str.strip() != ""].reset_index(drop=True)


def _parse_date(s: str) -> str:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unbekanntes Datumsformat: {s!r}")


def odds_triplets(columns: list[str]) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Findet 1X2-Quoten-Tripel: {(bookmaker, kind): (col_h, col_d, col_a)}."""
    cols = set(columns) - set(STAT_COLUMNS)
    prefixes = {c[:-1] for c in cols if c.endswith("H") and c[:-1] + "D" in cols and c[:-1] + "A" in cols}
    out = {}
    for p in prefixes:
        if p.endswith("C") and p[:-1] in prefixes:
            book, kind = p[:-1], "closing"
        else:
            book, kind = p, "pre"
        out[(BOOKMAKER_ALIASES.get(book, book), kind)] = (p + "H", p + "D", p + "A")
    return out


def parse(raw: bytes, league: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Liefert (matches, odds). odds ist über die Zeilenposition `row` mit matches verknüpft."""
    df = _read_rows(raw)
    m = df[[c for c in STAT_COLUMNS if c in df.columns]].rename(columns=STAT_COLUMNS)
    for c in STAT_COLUMNS.values():
        if c not in m.columns:
            m[c] = None
    m = m.replace({"": None})
    m["date"] = m["date"].map(_parse_date)
    m["home"] = m["home"].str.strip()
    m["away"] = m["away"].str.strip()
    for c in INT_COLUMNS:
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("Int64")
    m = m[m.fthg.notna() & m.ftag.notna()].copy()  # nur gespielte Spiele
    m["league"] = league
    m["season"] = season
    m["row"] = m.index

    odds = []
    for (book, kind), (ch, cd, ca) in odds_triplets(list(df.columns)).items():
        o = df.loc[m.index, [ch, cd, ca]].apply(pd.to_numeric, errors="coerce")
        o.columns = ["odds_h", "odds_d", "odds_a"]
        o = o.dropna(how="all")
        o["bookmaker"], o["kind"], o["row"] = book, kind, o.index
        odds.append(o)
    odds_df = pd.concat(odds, ignore_index=True) if odds else pd.DataFrame(
        columns=["odds_h", "odds_d", "odds_a", "bookmaker", "kind", "row"])
    return m.drop(columns=["div"]).reset_index(drop=True), odds_df


def store(con: sqlite3.Connection, matches: pd.DataFrame, odds: pd.DataFrame) -> int:
    country = config.LEAGUES[matches.league.iloc[0]].country
    names = pd.unique(pd.concat([matches.home, matches.away]))
    teams.register_canonical(con, {n: country for n in names})
    rows = matches.assign(home_id=matches.home.map(teams.slug), away_id=matches.away.map(teams.slug))
    cols = ["league", "season", "date", "kickoff", "home_id", "away_id", "referee"] + INT_COLUMNS
    db.upsert(con, "matches", rows[cols], key=["league", "date", "home_id", "away_id"])

    ids = db.read(con, "SELECT match_id, league, date, home_id, away_id FROM matches "
                       "WHERE league = ? AND season = ?", (rows.league.iloc[0], int(rows.season.iloc[0])))
    row_to_id = rows.merge(ids, on=["league", "date", "home_id", "away_id"]).set_index("row").match_id
    if len(odds):
        o = odds.assign(match_id=odds.row.map(row_to_id)).dropna(subset=["match_id"])
        db.upsert(con, "odds_benchmark",
                  o[["match_id", "bookmaker", "kind", "odds_h", "odds_d", "odds_a"]].astype({"match_id": int}),
                  key=["match_id", "bookmaker", "kind"])
    return len(rows)


def ingest(con: sqlite3.Connection, leagues: list[str] | None = None,
           seasons: list[int] | None = None, fetcher: Fetcher | None = None,
           max_age_h: float = CURRENT_SEASON_MAX_AGE_H) -> pd.DataFrame:
    fetcher = fetcher or Fetcher("football-data", min_interval=2.0)
    leagues = leagues or list(config.LEAGUES)
    seasons = seasons or config.seasons()
    current = config.current_season()
    summary = []
    for season in seasons:
        for league in leagues:
            code = season_code(season)
            name = f"{code}/{league}.csv"
            path = fetcher.path(name)
            refresh = (season == current and path.exists()
                       and time.time() - path.stat().st_mtime > max_age_h * 3600)
            try:
                raw = fetcher.get(BASE_URL.format(code=code, league=league), name, refresh=refresh)
            except FileNotFoundError:
                log.warning("Keine Datei für %s %s", league, config.season_label(season))
                continue
            matches, odds = parse(raw, league, season)
            n = store(con, matches, odds) if len(matches) else 0
            summary.append({"league": league, "season": season, "matches": n, "odds_rows": len(odds)})
    return pd.DataFrame(summary)
