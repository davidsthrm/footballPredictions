"""Understat-Spielberichte (getMatchData/<id>): Tore und Spielerwerte je Spiel der Top-5-Ligen.

Für die Spieldetails im Dashboard (Torschützen mit Minute/Vorlage, Elfmeter/Eigentore,
Spielerwerte für den „Top-Spieler laut Daten“). Nur gespielte Spiele der laufenden Saison,
jedes Spiel genau einmal abgerufen (Bericht ändert sich danach nicht mehr). Kein Modell-Input.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date

import pandas as pd

from fussball import config, db
from fussball.http import Fetcher

log = logging.getLogger(__name__)

URL = "https://understat.com/getMatchData/{id}"
MAX_PER_RUN = 150   # höflich bleiben: Rest beim nächsten Lauf

SCHEMA = """
CREATE TABLE IF NOT EXISTS match_goals (
    understat_match_id INTEGER NOT NULL,
    shot_id     INTEGER NOT NULL,
    minute      INTEGER, h_a TEXT, player TEXT, assist TEXT, situation TEXT,
    own_goal    INTEGER, xg REAL,
    PRIMARY KEY (understat_match_id, shot_id)
);
CREATE TABLE IF NOT EXISTS match_players (
    understat_match_id INTEGER NOT NULL,
    player_id   INTEGER NOT NULL,
    h_a TEXT, player TEXT, position TEXT, minutes INTEGER, goals INTEGER, own_goals INTEGER,
    assists INTEGER, shots INTEGER, key_passes INTEGER, xg REAL, xa REAL, yellow INTEGER, red INTEGER,
    PRIMARY KEY (understat_match_id, player_id)
);
"""


def parse(data: dict, match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Tore, Spieler). Eigentore stehen bei Understat auf der Seite des begünstigten Teams."""
    goals = [{"understat_match_id": match_id, "shot_id": int(s["id"]), "minute": int(s["minute"]),
              "h_a": side, "player": s["player"], "assist": s.get("player_assisted") or None,
              "situation": s["situation"], "own_goal": int(s["result"] == "OwnGoal"), "xg": float(s["xG"])}
             for side in ("h", "a") for s in data["shots"][side] if s["result"] in ("Goal", "OwnGoal")]
    players = [{"understat_match_id": match_id, "player_id": int(p["player_id"]), "h_a": side,
                "player": p["player"], "position": p["position"], "minutes": int(p["time"]),
                "goals": int(p["goals"]), "own_goals": int(p["own_goals"]), "assists": int(p["assists"]),
                "shots": int(p["shots"]), "key_passes": int(p["key_passes"]), "xg": float(p["xG"]),
                "xa": float(p["xA"]), "yellow": int(p["yellow_card"]), "red": int(p["red_card"])}
               for side in ("h", "a") for p in data["rosters"][side].values()]
    return pd.DataFrame(goals), pd.DataFrame(players)


def ingest(con: sqlite3.Connection, season: int | None = None, fetcher: Fetcher | None = None,
           limit: int = MAX_PER_RUN) -> int:
    """Holt fehlende Spielberichte der Saison (nur Spiele vor heute). Liefert die Anzahl neuer Berichte."""
    con.executescript(SCHEMA)
    fetcher = fetcher or Fetcher("understat", min_interval=2.0)
    season = season or config.current_season()
    todo = [r[0] for r in con.execute(
        f"""SELECT DISTINCT understat_match_id FROM team_xg
            WHERE season = ? AND date < ? AND league IN ({",".join("?" * len(config.TOP_LEAGUES))})
              AND understat_match_id NOT IN (SELECT understat_match_id FROM match_players)
            ORDER BY date DESC""", (season, date.today().isoformat(), *config.TOP_LEAGUES))][:limit]
    n = 0
    for mid in todo:
        name = f"match/{mid}.json"
        try:
            raw = fetcher.get(URL.format(id=mid), name, headers={
                "X-Requested-With": "XMLHttpRequest", "Referer": f"https://understat.com/match/{mid}"})
            goals, players = parse(json.loads(raw), mid)
        except Exception as e:  # noqa: BLE001 – einzelnes Spiel überspringen, nächster Lauf versucht es erneut
            log.warning("Spielbericht %s: %s", mid, e)
            continue
        if len(goals):
            db.upsert(con, "match_goals", goals, key=["understat_match_id", "shot_id"])
        if len(players):
            db.upsert(con, "match_players", players, key=["understat_match_id", "player_id"])
            n += 1
    return n
