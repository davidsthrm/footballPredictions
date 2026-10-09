"""Europapokal-Spielpläne (Champions League, Europa League, Conference League) von Transfermarkt.

Eine Seite pro Wettbewerb und Saison ("Gesamtspielplan", ~32 Seiten für 2014–heute), inkl.
bereits angesetzter, noch nicht gespielter Spiele. Gespeichert werden Termine (Belastungs-
Features), Ergebnisse und Vereinsnamen (Europa-Modell). Qualifikationsrunden sind nicht enthalten,
Anstoßzeiten stehen nicht auf der Seite. Ergebnisse nach Verlängerung sind nicht markiert und
zählen mit dem Endstand (betrifft ~1 % der Spiele, nur K.-o.-Runde).

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
# Später ergänzte Spalten (bestehende Datenbanken werden per ALTER TABLE nachgerüstet)
EXTRA_COLUMNS = {"tm_match_id": "INTEGER", "home_name": "TEXT", "away_name": "TEXT",
                 "hg": "INTEGER", "ag": "INTEGER", "kickoff": "TEXT", "round": "TEXT", "leg": "TEXT",
                 "matchday": "INTEGER"}
SCORE_RE = re.compile(r"^(\d+):(\d+)$")


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(europe_matches)")}
    for col, typ in EXTRA_COLUMNS.items():
        if col not in have:
            con.execute(f"ALTER TABLE europe_matches ADD COLUMN {col} {typ}")


DATETIME_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})(?:\s+(\d{1,2}:\d{2}))?")
KO_RE = re.compile(r"(Zwischenrunde|Sechzehntelfinale|Achtelfinale|Viertelfinale|Halbfinale|Finale|Play-?offs?)"
                   r"(?:\s*-\s*(Hin|Rück)spiele)?", re.I)
MATCHDAY_GAP_DAYS = 4   # Spiele einer Runde liegen höchstens so weit auseinander (Di–Do)


def _phase(headline: str) -> str | None:
    if headline == "Gruppe GP":
        return "Ligaphase"
    if headline.startswith("Gruppe "):
        return "Gruppenphase"
    return None


def parse(html: str) -> pd.DataFrame:
    """Alle Spiele einer Gesamtspielplan-Seite: Datum, Anstoß (Ortszeit, falls angesetzt), Runde,
    Vereine (ID, Slug, Name), Ergebnis. Runde = "Ligaphase"/"Gruppenphase" bzw. K.-o.-Runde, leg =
    Hin-/Rückspiel; der Spieltag der Liga-/Gruppenphase wird in `add_matchdays` ergänzt."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for box in soup.select("div.box"):
        hl = box.select_one("h2")
        phase = _phase(hl.get_text(" ", strip=True) if hl else "")
        current, kickoff, rnd, leg = None, None, phase, None
        for tr in box.select("table tbody tr"):
            text = tr.get_text(" ", strip=True)
            clubs = []
            for a in tr.select("a[href*='/verein/']"):
                cm = CLUB_RE.search(a["href"])
                if cm and (not clubs or clubs[-1][1] != int(cm.group(2))):
                    clubs.append((cm.group(1), int(cm.group(2)), a.get("title") or a.get_text(strip=True)))
            m = DATETIME_RE.search(text)
            if len(clubs) < 2:      # Zwischenzeile: Runde und/oder Termin
                ko = KO_RE.search(text) if not phase else None
                if ko and not m:
                    rnd, leg = ko.group(1), (f"{ko.group(2)}spiel" if ko.group(2) else None)
                if m:
                    current = datetime.strptime(m.group(1), "%d.%m.%Y").date().isoformat()
                    kickoff = m.group(2)
                continue
            if m:                   # ältere Seiten: Datum in der Spielzeile, ohne Uhrzeit
                d = datetime.strptime(m.group(1), "%d.%m.%Y").date().isoformat()
                if d != current:
                    current, kickoff = d, m.group(2)
            # Spielzeile: zwei verschiedene Vereine und ein Ergebnis/Uhrzeit-Feld
            if len(clubs) == 2 and current and re.search(r"\d+:\d+|-:-", text):
                link = tr.select_one("a[href*='/spielbericht/']")
                mid = re.search(r"/spielbericht/(\d+)", link["href"]) if link else None
                score = SCORE_RE.match(link.get_text(strip=True)) if link else None
                rows.append({"date": current, "kickoff": kickoff, "round": rnd, "leg": leg,
                             "home_slug": clubs[0][0], "tm_home_id": clubs[0][1], "home_name": clubs[0][2],
                             "away_slug": clubs[1][0], "tm_away_id": clubs[1][1], "away_name": clubs[1][2],
                             "tm_match_id": int(mid.group(1)) if mid else None,
                             "hg": int(score.group(1)) if score else None, "ag": int(score.group(2)) if score else None})
    cols = ["date", "kickoff", "round", "leg", "home_slug", "tm_home_id", "home_name", "away_slug", "tm_away_id",
            "away_name", "tm_match_id", "hg", "ag"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates(["date", "tm_home_id", "tm_away_id"])
    return add_matchdays(df)


def add_matchdays(df: pd.DataFrame) -> pd.DataFrame:
    """Spieltag der Liga-/Gruppenphase aus Terminblöcken (neuer Block nach > MATCHDAY_GAP_DAYS Tagen)."""
    df = df.assign(matchday=pd.NA)
    grp = df["round"].isin(["Ligaphase", "Gruppenphase"])
    if grp.any():
        days = pd.Series(sorted(pd.to_datetime(df.loc[grp, "date"]).unique()))
        md = (days.diff().dt.days.fillna(0) > MATCHDAY_GAP_DAYS).cumsum() + 1
        lookup = dict(zip(days.dt.strftime("%Y-%m-%d"), md))
        df.loc[grp, "matchday"] = df.loc[grp, "date"].map(lookup)
    return df


def ingest(con: sqlite3.Connection, seasons: list[int] | None = None,
           fetcher: Fetcher | None = None) -> pd.DataFrame:
    ensure_schema(con)
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
            if df.empty:     # Layout geändert o.ä.: nichts löschen, Bestand behalten
                log.warning("Keine Spiele erkannt: %s %s", code, season)
                continue
            # Seite = vollständiger Stand der Saison: verlegte oder entfallene Spiele nicht stehen lassen
            keep = {(d, h, a) for d, h, a in zip(df.date, df.tm_home_id, df.tm_away_id)}
            old = con.execute("SELECT date, tm_home_id, tm_away_id FROM europe_matches "
                              "WHERE competition = ? AND season = ?", (code, season)).fetchall()
            con.executemany("DELETE FROM europe_matches WHERE competition = ? AND season = ? AND date = ? "
                            "AND tm_home_id = ? AND tm_away_id = ?",
                            [(code, season, *k) for k in old if tuple(k) not in keep])
            db.upsert(con, "europe_matches", df, key=["competition", "season", "date", "tm_home_id", "tm_away_id"])
            summary.append({"competition": code, "season": season, "matches": len(df)})
    return pd.DataFrame(summary)


def fetch_images(con: sqlite3.Connection, season: int, fetcher: Fetcher | None = None) -> int:
    """Wappen der Vereine einer Europapokal-Saison und Wettbewerbs-Logos, einmalig gecacht."""
    from fussball.ingest.transfermarkt import IMG_BASE, crest_name, logo_name
    fetcher = fetcher or Fetcher("transfermarkt", min_interval=2.0)
    ids = [r[0] for r in con.execute("""SELECT tm_home_id FROM europe_matches WHERE season = ?
                                        UNION SELECT tm_away_id FROM europe_matches WHERE season = ?""",
                                     (season, season))]
    todo = [(f"{IMG_BASE}/logo/homepageWappen150x150/{code.lower()}.png", logo_name(code)) for code in COMPETITIONS]
    todo += [(f"{IMG_BASE}/wappen/medium/{i}.png", crest_name(i)) for i in ids]
    new = [(u, n) for u, n in todo if not fetcher.cached(n)]
    for url, name in new:
        try:
            fetcher.get(url, name)
        except FileNotFoundError:
            log.warning("Bild fehlt: %s", url)
    return len(new)


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
