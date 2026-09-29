"""Historische Kaderwerte je Liga und Saison (Transfermarkt-Ligaübersicht).

Achtung Point-in-time: Die Übersicht einer vergangenen Saison zeigt Marktwerte nach dem
Saisonverlauf (z.B. Leverkusen 2023/24: 658 Mio., zu Saisonbeginn deutlich weniger).
Für Saison s wird deshalb ausschließlich der Wert der Vorsaison s-1 verwendet
(`team_values`), der vor Saisonbeginn feststeht. Sommertransfers fehlen dadurch.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from fussball import config, db, teams
from fussball.http import Fetcher
from fussball.ingest import europe
from fussball.ingest.transfermarkt import BASE, parse_league

# football-data-Liga -> (Transfermarkt-Slug, Transfermarkt-Code)
TM_LEAGUES = {
    "E0": ("premier-league", "GB1"), "SP1": ("laliga", "ES1"), "I1": ("serie-a", "IT1"),
    "F1": ("ligue-1", "FR1"), "D1": ("bundesliga", "L1"),
    "E1": ("championship", "GB2"), "SP2": ("laliga2", "ES2"), "I2": ("serie-b", "IT2"),
    "F2": ("ligue-2", "FR2"), "D2": ("2-bundesliga", "L2"),
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS squad_values_history (
    season INTEGER NOT NULL, league TEXT NOT NULL, tm_club_id INTEGER NOT NULL,
    slug TEXT, name TEXT, squad_size INTEGER, avg_age REAL, total_value_eur REAL,
    PRIMARY KEY (season, league, tm_club_id)
);
"""


def ingest(con: sqlite3.Connection, seasons: list[int] | None = None,
           fetcher: Fetcher | None = None) -> pd.DataFrame:
    """Lädt abgeschlossene Saisons (nie die laufende – deren Werte ändern sich noch)."""
    con.executescript(SCHEMA)
    fetcher = fetcher or Fetcher("transfermarkt", min_interval=5.0)
    done = [s for s in (seasons or config.seasons()) if s < config.current_season()]
    summary = []
    for season in done:
        for league, (slug, code) in TM_LEAGUES.items():
            try:
                html = fetcher.get(f"{BASE}/{slug}/startseite/wettbewerb/{code}/saison_id/{season}",
                                   f"history/{code}_{season}.html").decode()
            except FileNotFoundError:
                continue
            clubs = parse_league(html)
            db.upsert(con, "squad_values_history", [
                {"season": season, "league": league, "tm_club_id": c.tm_club_id, "slug": c.slug,
                 "name": c.name, "squad_size": c.squad_size, "avg_age": c.avg_age,
                 "total_value_eur": c.total_value_eur} for c in clubs],
                key=["season", "league", "tm_club_id"])
            summary.append({"season": season, "league": league, "clubs": len(clubs)})
    return pd.DataFrame(summary)


def suggest_ids(con: sqlite3.Connection) -> list[dict]:
    """TM-ID -> team_id über Namensähnlichkeit mit den Teams derselben Liga und Saison."""
    con.executescript(SCHEMA)
    known = teams.name_map(con, europe.SOURCE)
    names = teams.display_names(con)
    played = db.read(con, "SELECT DISTINCT league, season, home_id AS team_id FROM matches")
    h = db.read(con, "SELECT season, league, tm_club_id, slug, name FROM squad_values_history")
    out = []
    for cid, g in h.groupby("tm_club_id"):
        if str(cid) in known:
            continue
        cand = played.merge(g[["league", "season"]], on=["league", "season"]).team_id.unique()
        tok = europe._tokens(g.slug.iloc[0]) | europe._tokens(g.name.iloc[0])
        best, score = None, 0.0
        for t in cand:
            ct = europe._tokens(names.get(t, t)) | europe._tokens(t)
            j = len(tok & ct) / max(len(tok | ct), 1)
            if j > score:
                best, score = t, j
        if best and score >= 0.34:
            out.append({"source": europe.SOURCE, "source_name": str(cid), "team_id": best,
                        "slug": g.slug.iloc[0], "score": round(score, 2)})
    return out


def team_values(con: sqlite3.Connection) -> pd.DataFrame:
    """Je (team_id, season): Kaderwert der VORSAISON, log-relativ zum Mittel der Liga der Vorsaison.

    Spalten: team_id, season, mv_log (log Mio. €), mv_rel (log-Abstand zum Ligamittel).
    """
    con.executescript(SCHEMA)
    mp = {int(k): v for k, v in teams.name_map(con, europe.SOURCE).items()}
    h = db.read(con, "SELECT season, league, tm_club_id, total_value_eur FROM squad_values_history "
                     "WHERE total_value_eur > 0")
    h["team_id"] = h.tm_club_id.map(mp)
    h = h.dropna(subset=["team_id"])
    h["mv_log"] = np.log(h.total_value_eur / 1e6)
    top = h[h.league.isin(config.TOP_LEAGUES)]
    league_mean = top.groupby(["season", "league"]).mv_log.mean().rename("lg_mean")
    h = h.join(league_mean, on=["season", "league"])
    # Zweitligisten: relativ zum Mittel der 1. Liga desselben Landes
    country_top = {config.LEAGUES[c].country: c for c in config.TOP_LEAGUES}
    lg1 = h.league.map(lambda c: country_top[config.LEAGUES[c].country])
    h["lg_mean"] = [league_mean.get((s, l), np.nan) for s, l in zip(h.season, lg1)]
    h["mv_rel"] = h.mv_log - h.lg_mean
    h = h.sort_values("league").drop_duplicates(["team_id", "season"])  # 1. Liga vor 2. Liga
    return h.assign(season=h.season + 1)[["team_id", "season", "mv_log", "mv_rel"]]
