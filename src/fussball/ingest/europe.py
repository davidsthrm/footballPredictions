"""Europapokal-Spielpläne (Champions League, Europa League, Conference League) von Transfermarkt.

Eine Seite pro Wettbewerb und Saison ("Gesamtspielplan", ~32 Seiten für 2014–heute), inkl.
bereits angesetzter, noch nicht gespielter Spiele. Nur die Termine werden gebraucht
(Belastung/Rotation), nicht die Ergebnisse. Qualifikationsrunden sind nicht enthalten.

Vereine werden über die Transfermarkt-ID auf team_id gemappt: bekannte IDs aus den
Snapshots, übrige über Namens-Ähnlichkeit gegen die Teams der Top-5-Ligen derselben
Saison (Vorschläge landen zur Prüfung in data_seed/team_name_map.csv, Quelle
'transfermarkt_id').
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from fussball import config, db, teams
from fussball.http import Fetcher

log = logging.getLogger(__name__)

BASE = "https://www.transfermarkt.de"
COMPETITIONS = {  # Code -> (URL-Slug, erste Saison)
    "CL": ("uefa-champions-league", config.FIRST_SEASON),
    "EL": ("europa-league", config.FIRST_SEASON),
    "UCOL": ("uefa-conference-league", 2021),
}
SOURCE = "transfermarkt_id"
DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
CLUB_RE = re.compile(r"/([^/]+)/(?:startseite|spielplan)/verein/(\d+)")
CURRENT_MAX_AGE_H = 24

SCHEMA = """
CREATE TABLE IF NOT EXISTS europe_matches (
    competition TEXT NOT NULL,
    season      INTEGER NOT NULL,
    date        TEXT NOT NULL,
    tm_home_id  INTEGER NOT NULL,
    tm_away_id  INTEGER NOT NULL,
    home_slug   TEXT, away_slug TEXT,
    PRIMARY KEY (competition, season, date, tm_home_id, tm_away_id)
);
"""


def parse(html: str) -> pd.DataFrame:
    """Alle Spiele (Datum, Heim-ID, Auswärts-ID) einer Gesamtspielplan-Seite."""
    soup = BeautifulSoup(html, "html.parser")
    rows, current = [], None
    for tr in soup.select("div.box table tbody tr"):
        text = tr.get_text(" ", strip=True)
        m = DATE_RE.search(text)
        if m:
            current = datetime.strptime(m.group(1), "%d.%m.%Y").date().isoformat()
        clubs = []
        for a in tr.select("a[href*='/verein/']"):
            cm = CLUB_RE.search(a["href"])
            if cm and (not clubs or clubs[-1][1] != int(cm.group(2))):
                clubs.append((cm.group(1), int(cm.group(2))))
        # Spielzeile: zwei verschiedene Vereine und ein Ergebnis/Uhrzeit-Feld; Tabellenzeilen haben nur einen
        if len(clubs) == 2 and current and re.search(r"\d+:\d+|-:-", text):
            rows.append({"date": current, "home_slug": clubs[0][0], "tm_home_id": clubs[0][1],
                         "away_slug": clubs[1][0], "tm_away_id": clubs[1][1]})
    return pd.DataFrame(rows, columns=["date", "home_slug", "tm_home_id", "away_slug", "tm_away_id"]).drop_duplicates()


def ingest(con: sqlite3.Connection, seasons: list[int] | None = None,
           fetcher: Fetcher | None = None) -> pd.DataFrame:
    con.executescript(SCHEMA)
    fetcher = fetcher or Fetcher("transfermarkt", min_interval=5.0)
    current = config.current_season()
    summary = []
    for code, (slug, first) in COMPETITIONS.items():
        for season in seasons or config.seasons():
            if season < first:
                continue
            name = f"europe/{code}_{season}.html"
            path = fetcher.path(name)
            refresh = (season == current and path.exists()
                       and time.time() - path.stat().st_mtime > CURRENT_MAX_AGE_H * 3600)
            try:
                html = fetcher.get(f"{BASE}/{slug}/gesamtspielplan/pokalwettbewerb/{code}/saison_id/{season}",
                                   name, refresh=refresh).decode()
            except FileNotFoundError:
                continue
            df = parse(html).assign(competition=code, season=season)
            db.upsert(con, "europe_matches", df, key=["competition", "season", "date", "tm_home_id", "tm_away_id"])
            summary.append({"competition": code, "season": season, "matches": len(df)})
    return pd.DataFrame(summary)


def _tokens(s: str) -> set[str]:
    stop = {"fc", "ac", "as", "sc", "sv", "vfb", "vfl", "cf", "cd", "ud", "rc", "ss", "ssc", "us", "afc",
            "de", "la", "real", "club", "calcio", "1", "04", "05", "07", "1899", "1907", "1913"}
    return {t for t in re.split(r"[^a-z0-9]+", teams.slug(s).replace("-", " ")) if t and t not in stop}


def suggest_ids(con: sqlite3.Connection) -> list[dict]:
    """Mapping-Vorschläge für TM-Vereins-IDs aus Europapokal-Spielen."""
    con.executescript(SCHEMA)
    known = teams.name_map(con, SOURCE)
    snap = dict(con.execute("SELECT tm_club_id, team_id FROM squad_value_snapshots "
                            "WHERE team_id IS NOT NULL GROUP BY tm_club_id").fetchall())
    names = teams.display_names(con)
    tops = db.read(con, f"""SELECT DISTINCT season, home_id AS team_id FROM matches
                           WHERE league IN ({",".join("?" * len(config.TOP_LEAGUES))})""", config.TOP_LEAGUES)
    clubs = db.read(con, """SELECT season, tm_home_id AS id, home_slug AS slug FROM europe_matches
                            UNION SELECT season, tm_away_id, away_slug FROM europe_matches""")
    out = []
    for cid, g in clubs.groupby("id"):
        if str(cid) in known:
            continue
        if cid in snap:
            out.append({"source": SOURCE, "source_name": str(cid), "team_id": snap[cid]})
            continue
        cand = tops[tops.season.isin(g.season)].team_id.unique()
        tok = _tokens(g.slug.iloc[0])
        best, score = None, 0.0
        for t in cand:
            ct = _tokens(names.get(t, t)) | _tokens(t)
            j = len(tok & ct) / max(len(tok | ct), 1)
            if j > score:
                best, score = t, j
        if best and score >= 0.5:
            out.append({"source": SOURCE, "source_name": str(cid), "team_id": best,
                        "slug": g.slug.iloc[0], "score": round(score, 2)})
    return out


def team_dates(con: sqlite3.Connection) -> pd.DataFrame:
    """Europapokal-Termine je gemapptem Team (Spalten team, date) für die Features."""
    con.executescript(SCHEMA)
    mp = {int(k): v for k, v in teams.name_map(con, SOURCE).items()}
    e = db.read(con, "SELECT date, tm_home_id, tm_away_id FROM europe_matches")
    long = pd.concat([e.assign(team=e.tm_home_id.map(mp)), e.assign(team=e.tm_away_id.map(mp))])
    long = long.dropna(subset=["team"])[["team", "date"]].drop_duplicates()
    return long.assign(date=pd.to_datetime(long.date)).reset_index(drop=True)
