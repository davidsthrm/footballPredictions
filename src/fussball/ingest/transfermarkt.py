"""Transfermarkt: täglicher Snapshot von Verletzungen/Sperren + Kader-Marktwerten.

Pro Liga: Startseite (Vereine, Kadergröße, Gesamtmarktwert), pro Verein die Seite
"Sperren und Verletzungen" (Verletzungen, Sperren, drohende Sperren).
~100 Requests/Tag bei 5 s Abstand. Roh-HTML liegt unter data/raw/transfermarkt/<datum>/,
ein zweiter Lauf am selben Tag nutzt nur den Cache. Bei 403/Bot-Schutz: Abbruch,
es wird nichts umgangen.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup, Tag

from fussball import config, db, teams
from fussball.http import Fetcher

log = logging.getLogger(__name__)

BASE = "https://www.transfermarkt.de"
IMG_BASE = "https://img.a.transfermarkt.technology"
SOURCE = "transfermarkt"
LEAGUE_SLUGS = {"GB1": "premier-league", "ES1": "laliga", "IT1": "serie-a", "FR1": "ligue-1", "L1": "bundesliga"}
MIN_INTERVAL = 5.0

HEADER_FIELDS = {
    "spieler": "player", "alter": "age", "grund": "reason", "seit": "since",
    "bis voraussichtlich": "until", "bis ca.": "until", "verpasste spiele": "missed_games",
}


@dataclass
class Club:
    tm_club_id: int
    name: str
    slug: str
    squad_size: int | None
    avg_age: float | None
    total_value_eur: float | None


def parse_value(s: str) -> float | None:
    """'1,04 Mrd. €' -> 1.04e9, '45,11 Mio. €' -> 4.511e7, '500 Tsd. €' -> 5e5."""
    m = re.search(r"([\d.,]+)\s*(Mrd|Mio|Tsd)?", s or "")
    if not m or not s.strip() or s.strip() == "-":
        return None
    num = float(m.group(1).replace(".", "").replace(",", "."))
    return num * {"Mrd": 1e9, "Mio": 1e6, "Tsd": 1e3, None: 1}[m.group(2)]


def _num(s: str, typ=int):
    s = (s or "").strip().replace(",", ".")
    try:
        return typ(s)
    except ValueError:
        return None


def _date(s: str) -> str | None:
    try:
        return datetime.strptime(s.strip(), "%d.%m.%Y").date().isoformat()
    except ValueError:
        return None


def parse_league(html: str) -> list[Club]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.items")
    if table is None:
        raise ValueError("Ligaseite: Tabelle 'items' nicht gefunden (Layout geändert?)")
    clubs = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        a = tr.select_one("td.hauptlink a[href*='/verein/']")
        if a is None:
            continue
        m = re.match(r"/([^/]+)/startseite/verein/(\d+)", a["href"])
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        clubs.append(Club(int(m.group(2)), a.get("title") or a.get_text(strip=True), m.group(1),
                          _num(tds[2]), _num(tds[3], float), parse_value(tds[-1])))
    return clubs


def _section_heading(table: Tag) -> str:
    h = table.find_previous("h2")
    return h.get_text(" ", strip=True) if h else ""


def _category(section: str, heading: str, reason: str = "") -> str:
    if "spielberechtigung" in reason.lower():
        # z.B. nicht für den Champions-League-Kader gemeldet – für die Liga ohne Bedeutung
        return "registration"
    if "droht" in heading.lower():
        return "suspension_threat"
    s = section.lower()
    if "verletz" in s:
        return "injury"
    if "sperr" in s:
        return "suspension"
    return "other"


def parse_club(html: str) -> list[dict]:
    """Alle Ausfall-Einträge einer Vereinsseite 'Sperren und Verletzungen'."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for table in soup.select("table.items"):
        heading = _section_heading(table)
        headers = [th.get_text(" ", strip=True).lower() for th in table.select("thead th")]
        fields = [HEADER_FIELDS.get(h, h) for h in headers]
        section = ""
        tbody = table.find("tbody")
        if tbody is None:
            continue
        for tr in tbody.find_all("tr", recursive=False):
            tds = tr.find_all("td", recursive=False)
            link = tr.select_one("td.hauptlink a[href*='/spieler/']")
            if link is None:  # Abschnittszeile, z.B. "Verletzungen" / "Sperren"
                section = tr.get_text(" ", strip=True)
                continue
            inline = tds[0].select("table.inline-table tr")
            position = inline[1].get_text(" ", strip=True) if len(inline) > 1 else None
            rec = {"tm_player_id": int(re.search(r"/spieler/(\d+)", link["href"]).group(1)),
                   "player": link.get("title") or link.get_text(strip=True),
                   "position": position, "category": _category(section, heading)}
            extra = []
            for f, td in zip(fields[1:], tds[1:]):
                text = td.get_text(" ", strip=True)
                if f in ("age", "missed_games"):
                    rec[f] = _num(text)
                elif f in ("since", "until"):
                    rec[f] = _date(text)
                elif f == "reason":
                    rec[f] = text or None
                    rec["category"] = _category(section, heading, text)
                elif text:
                    extra.append(f"{f}: {text}")
            if extra and not rec.get("reason"):
                rec["reason"] = "; ".join(extra)
            out.append(rec)
    return out


def crest_name(tm_club_id: int) -> str:
    return f"crests/{tm_club_id}.png"


def logo_name(league: str) -> str:
    return f"logos/{league}.png"


def fetch_images(clubs: list[Club], league: str, fetcher: Fetcher | None = None) -> int:
    """Wappen (58x76) und Liga-Logo einmalig laden; Vorhandenes wird nie neu abgerufen."""
    fetcher = fetcher or Fetcher(SOURCE, min_interval=2.0)
    code = config.LEAGUES[league].transfermarkt
    todo = [(f"{IMG_BASE}/logo/homepageWappen150x150/{code.lower()}.png", logo_name(league))]
    todo += [(f"{IMG_BASE}/wappen/medium/{c.tm_club_id}.png", crest_name(c.tm_club_id)) for c in clubs]
    new = [(u, n) for u, n in todo if not fetcher.cached(n)]
    for url, name in new:
        fetcher.get(url, name)
    return len(new)


def snapshot(con: sqlite3.Connection, leagues: list[str] | None = None,
             fetcher: Fetcher | None = None, now: datetime | None = None) -> pd.DataFrame:
    fetcher = fetcher or Fetcher(SOURCE, min_interval=MIN_INTERVAL)
    now = now or datetime.now(timezone.utc)
    snap_date = now.astimezone(ZoneInfo("Europe/Berlin")).date().isoformat()
    tm_map = teams.name_map(con, SOURCE)
    summary = []
    for league in leagues or config.TOP_LEAGUES:
        code = config.LEAGUES[league].transfermarkt
        name = f"{snap_date}/{code}_startseite.html"
        html = fetcher.get(f"{BASE}/{LEAGUE_SLUGS[code]}/startseite/wettbewerb/{code}", name).decode()
        clubs = parse_league(html)
        fetch_images(clubs, league)
        base = {"snapshot_date": snap_date, "snapshot_ts": fetcher.fetched_at(name), "league": league}
        db.upsert(con, "squad_value_snapshots", [
            {**base, "tm_club_id": c.tm_club_id, "tm_club_name": c.name, "team_id": tm_map.get(c.name),
             "squad_size": c.squad_size, "avg_age": c.avg_age, "total_value_eur": c.total_value_eur}
            for c in clubs], key=["snapshot_date", "tm_club_id"])
        n_abs = 0
        for c in clubs:
            name = f"{snap_date}/club_{c.tm_club_id}.html"
            html = fetcher.get(f"{BASE}/{c.slug}/sperrenundverletzungen/verein/{c.tm_club_id}", name).decode()
            recs = [{**base, "snapshot_ts": fetcher.fetched_at(name), "tm_club_id": c.tm_club_id, "team_id": tm_map.get(c.name), **r}
                    for r in parse_club(html)]
            cols = ["snapshot_date", "snapshot_ts", "league", "tm_club_id", "team_id", "tm_player_id",
                    "player", "position", "age", "category", "reason", "since", "until", "missed_games"]
            recs = [{k: r.get(k) for k in cols} for r in recs]
            db.upsert(con, "availability_snapshots", recs,
                      key=["snapshot_date", "tm_club_id", "tm_player_id", "category"])
            n_abs += len(recs)
        summary.append({"league": league, "clubs": len(clubs), "absences": n_abs,
                        "unmapped_clubs": sum(c.name not in tm_map for c in clubs)})
    return pd.DataFrame(summary)
