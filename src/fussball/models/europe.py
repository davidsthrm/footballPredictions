"""Europa-Modell: ein gemeinsamer Dixon-Coles über alle Ligen plus Europapokal-Spiele.

Getrennte Länder-Modelle (wie für die Ligen) kennen kein gemeinsames Niveau. Hier stehen alle
Ligaspiele (Top 5, 2. Ligen, weitere Ligen aus other_matches) und die Europapokal-Spiele in
einem Fit; die Europapokal-Spiele verbinden die Ligen. Heimvorteil je Wettbewerb (tier =
Liga-Code bzw. "EU"). Europapokal-Spiele können stärker gewichtet werden (w_eu).

Vereine ohne Ligadaten (z.B. aus Zypern, Serbien, Kroatien) laufen als "tm-<Transfermarkt-ID>"
und sind nur aus ihren Europapokal-Spielen bekannt; der L2-Term zieht sie zum Mittel.
Nur Tore, kein xG (gibt es für Europapokal und die weiteren Ligen nicht).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fussball import db, teams
from fussball.models.dixon_coles import DixonColes, score_matrix

EU_TIER = "EU"
COMPETITIONS = {"CL": "Champions League", "EL": "Europa League", "UCOL": "Conference League"}
DEFAULT_PARAMS = {"xi": 0.002, "reg": 1.0, "w_eu": 2.0}


def club_ids(con: sqlite3.Connection) -> dict[int, str]:
    """Transfermarkt-ID -> team_id (über die Seed-Map, Quelle 'transfermarkt_id')."""
    return {int(k): v for k, v in teams.name_map(con, "transfermarkt_id").items()}


def europe_frame(con: sqlite3.Connection) -> pd.DataFrame:
    """Alle Europapokal-Spiele (auch künftige) mit team_id bzw. 'tm-<id>' und Transfermarkt-Namen."""
    e = db.read(con, """SELECT competition, season, date, kickoff, round, leg, matchday, tm_match_id,
                               tm_home_id, tm_away_id, home_name, away_name, hg, ag FROM europe_matches""")
    ids = club_ids(con)
    e["home"] = [ids.get(i, f"tm-{i}") for i in e.tm_home_id]
    e["away"] = [ids.get(i, f"tm-{i}") for i in e.tm_away_id]
    e["date"] = pd.to_datetime(e.date)
    return e


def fit_data(con: sqlite3.Connection, include_other: bool = True) -> pd.DataFrame:
    """Gespielte Spiele aller Wettbewerbe: date, home, away, hg, ag, tier, eu (bool)."""
    dom = db.read(con, """SELECT date, league AS tier, home_id AS home, away_id AS away,
                                 fthg AS hg, ftag AS ag FROM matches""")
    parts = [dom]
    if include_other:
        parts.append(db.read(con, """SELECT date, league AS tier, home_id AS home, away_id AS away, hg, ag
                                     FROM other_matches"""))
    e = europe_frame(con).dropna(subset=["hg", "ag"])
    parts.append(e[["date", "home", "away", "hg", "ag"]].assign(tier=EU_TIER))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df.date)
    df["eu"] = df.tier == EU_TIER
    return df.astype({"hg": float, "ag": float}).sort_values("date").reset_index(drop=True)


@dataclass
class EuropeModel:
    xi: float = DEFAULT_PARAMS["xi"]
    reg: float = DEFAULT_PARAMS["reg"]
    w_eu: float = DEFAULT_PARAMS["w_eu"]
    dc: DixonColes | None = None

    @classmethod
    def from_params(cls, p: dict) -> EuropeModel:
        return cls(**{k: p[k] for k in ("xi", "reg", "w_eu") if k in p})

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp, warm_start: EuropeModel | None = None) -> EuropeModel:
        d = df.assign(weight=np.where(df.eu, self.w_eu, 1.0))
        self.dc = DixonColes(self.xi, self.reg).fit(d, cutoff, warm_start.dc if warm_start else None)
        return self

    @property
    def rho(self) -> float:
        return self.dc.params_["rho"]

    def rates(self, home: str, away: str) -> tuple[float, float]:
        return self.dc.rates(home, away, tier=EU_TIER)

    def matrix(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.rates(home, away)
        return score_matrix(lam, mu, self.rho)

    def known(self, team: str) -> bool:
        return team in self.dc.params_["attack"]
