"""Understat: xG pro Team und Spiel -> team_xg.

Die Seite lädt ihre Daten per AJAX von getLeagueData/<liga>/<saison> (JSON mit
`dates` = Spiele, `teams` = Team-Historie inkl. npxG/PPDA/deep, `players` =
Saison-Aggregate). Das komplette JSON wird gecacht; Spielerebene folgt in Phase 2.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time

import pandas as pd

from fussball import config, db, teams
from fussball.http import Fetcher

log = logging.getLogger(__name__)

URL = "https://understat.com/getLeagueData/{league}/{season}"
SOURCE = "understat"
CURRENT_SEASON_MAX_AGE_H = 12


def fetch(fetcher: Fetcher, league: str, season: int,
          max_age_h: float = CURRENT_SEASON_MAX_AGE_H) -> dict:
    ul = config.LEAGUES[league].understat
    name = f"league/{ul}_{season}.json"
    path = fetcher.path(name)
    refresh = (season == config.current_season() and path.exists()
               and time.time() - path.stat().st_mtime > max_age_h * 3600)
    raw = fetcher.get(URL.format(league=ul, season=season), name, refresh=refresh, headers={
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{ul}/{season}",
    })
    return json.loads(raw)


def parse(data: dict, league: str, season: int) -> pd.DataFrame:
    """Eine Zeile pro (Spiel, Team) mit Understat-Namen in `team`/`opponent`."""
    hist = {}
    for t in data["teams"].values():
        for h in t["history"]:
            hist[(t["title"], h["date"])] = h

    rows = []
    for m in data["dates"]:
        if not m.get("isResult"):
            continue
        for side, other in (("h", "a"), ("a", "h")):
            team = m[side]["title"]
            h = hist.get((team, m["datetime"]), {})
            ppda, ppda_a = h.get("ppda") or {}, h.get("ppda_allowed") or {}
            rows.append({
                "understat_match_id": int(m["id"]),
                "league": league, "season": season,
                "datetime": m["datetime"],
                "date": m["datetime"][:10],
                "is_home": int(side == "h"),
                "team": team, "opponent": m[other]["title"],
                "goals": int(m["goals"][side]), "goals_against": int(m["goals"][other]),
                "xg": float(m["xG"][side]), "xga": float(m["xG"][other]),
                "npxg": h.get("npxG"), "npxga": h.get("npxGA"),
                "ppda_att": ppda.get("att"), "ppda_def": ppda.get("def"),
                "ppda_allowed_att": ppda_a.get("att"), "ppda_allowed_def": ppda_a.get("def"),
                "deep": h.get("deep"), "deep_allowed": h.get("deep_allowed"),
            })
    return pd.DataFrame(rows)


def link_matches(con: sqlite3.Connection, xg: pd.DataFrame) -> pd.Series:
    """match_id per (league, home_id, away_id, Datum ±1 Tag). Understat-Zeiten sind UTC."""
    league, season = xg.league.iloc[0], int(xg.season.iloc[0])
    m = db.read(con, "SELECT match_id, date, home_id, away_id FROM matches WHERE league = ? "
                     "AND season BETWEEN ? AND ?", (league, season - 1, season + 1))
    m["d"] = pd.to_datetime(m.date)
    home = xg.assign(h=xg.team_id.where(xg.is_home == 1, xg.opponent_id),
                     a=xg.opponent_id.where(xg.is_home == 1, xg.team_id),
                     d=pd.to_datetime(xg.date))
    j = home.reset_index().merge(m, left_on=["h", "a"], right_on=["home_id", "away_id"], how="left")
    j = j[(j.d_x - j.d_y).abs() <= pd.Timedelta(days=1)].drop_duplicates("index")
    return j.set_index("index").match_id.reindex(xg.index)


def store(con: sqlite3.Connection, xg: pd.DataFrame) -> tuple[int, set[str]]:
    mp = teams.name_map(con, SOURCE)
    unmapped = set(xg.team) - set(mp)
    xg = xg.assign(team_id=xg.team.map(mp), opponent_id=xg.opponent.map(mp))
    xg = xg.dropna(subset=["team_id", "opponent_id"])
    if xg.empty:
        return 0, unmapped
    xg["match_id"] = link_matches(con, xg).astype("Int64")
    cols = ["understat_match_id", "team_id", "match_id", "league", "season", "date", "is_home",
            "opponent_id", "goals", "goals_against", "xg", "xga", "npxg", "npxga", "ppda_att",
            "ppda_def", "ppda_allowed_att", "ppda_allowed_def", "deep", "deep_allowed"]
    db.upsert(con, "team_xg", xg[cols], key=["understat_match_id", "team_id"])
    return len(xg), unmapped


def suggest_names(con: sqlite3.Connection, xg: pd.DataFrame) -> list[dict]:
    """Leitet Namens-Mappings über identische Spiele ab (gleicher Tag, gleiches Ergebnis).

    Nur Spiele mit genau einem Kandidaten zählen (sonst ist die Zuordnung mehrdeutig).
    Ein Understat-Name wird vorgeschlagen, wenn ≥3 Stimmen und ≥80 % davon auf dasselbe
    football-data-Team zeigen.
    """
    league = xg.league.iloc[0]
    m = db.read(con, "SELECT date, home_id, away_id, fthg, ftag FROM matches WHERE league = ?", (league,))
    home = xg[xg.is_home == 1]
    j = home.merge(m, left_on=["date", "goals", "goals_against"], right_on=["date", "fthg", "ftag"])
    j = j[j.groupby("understat_match_id").understat_match_id.transform("size") == 1]
    votes = list(zip(j.team, j.home_id)) + list(zip(j.opponent, j.away_id))
    v = pd.DataFrame(votes, columns=["source_name", "team_id"]).value_counts().reset_index(name="n")
    v["share"] = v.n / v.groupby("source_name").n.transform("sum")
    best = v.sort_values("n", ascending=False).drop_duplicates("source_name")
    best = best[(best.n >= 3) & (best.share >= 0.8)]
    return [{"source": SOURCE, "source_name": r.source_name, "team_id": r.team_id} for r in best.itertuples()]


def parse_fixtures(data: dict, league: str, season: int) -> pd.DataFrame:
    """Noch nicht gespielte Spiele (isResult = false). Understat-Zeiten sind UTC."""
    rows = [{"understat_match_id": int(m["id"]), "league": league, "season": season,
             "kickoff_utc": m["datetime"], "date": m["datetime"][:10],
             "home": m["h"]["title"], "away": m["a"]["title"]}
            for m in data["dates"] if not m.get("isResult")]
    return pd.DataFrame(rows, columns=["understat_match_id", "league", "season", "kickoff_utc",
                                       "date", "home", "away"])


def store_fixtures(con: sqlite3.Connection, fx: pd.DataFrame, league: str, season: int,
                   fetched_at: str | None = None) -> set[str]:
    """Ersetzt den Spielplan einer Liga/Saison. Liefert ungemappte Namen."""
    mp = teams.name_map(con, SOURCE)
    unmapped = (set(fx.home) | set(fx.away)) - set(mp)
    fx = fx.assign(home_id=fx.home.map(mp), away_id=fx.away.map(mp), fetched_at=fetched_at)
    fx = fx.dropna(subset=["home_id", "away_id"])
    with con:
        con.execute("DELETE FROM fixtures WHERE league = ? AND season = ?", (league, season))
    db.upsert(con, "fixtures", fx[["understat_match_id", "league", "season", "kickoff_utc", "date",
                                   "home_id", "away_id", "fetched_at"]], key=["understat_match_id"])
    return unmapped


def ingest(con: sqlite3.Connection, leagues: list[str] | None = None,
           seasons: list[int] | None = None, fetcher: Fetcher | None = None,
           suggest: bool = False, max_age_h: float = CURRENT_SEASON_MAX_AGE_H,
           ) -> tuple[pd.DataFrame, list[dict]]:
    fetcher = fetcher or Fetcher("understat", min_interval=3.0)
    leagues = leagues or config.TOP_LEAGUES
    summary, suggestions = [], []
    for season in seasons or config.seasons():
        for league in leagues:
            data = fetch(fetcher, league, season, max_age_h)
            unmapped = set()
            if season == config.current_season():
                name = f"league/{config.LEAGUES[league].understat}_{season}.json"
                unmapped |= store_fixtures(con, parse_fixtures(data, league, season), league, season,
                                           fetcher.fetched_at(name))
            xg = parse(data, league, season)
            if xg.empty:
                continue
            if suggest:
                suggestions += suggest_names(con, xg)
            n, unmapped_xg = store(con, xg)
            unmapped |= unmapped_xg
            summary.append({"league": league, "season": season, "rows": n,
                            "unmapped": ", ".join(sorted(unmapped))})
    return pd.DataFrame(summary), suggestions
