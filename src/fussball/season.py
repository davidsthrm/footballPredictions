"""Tabellen und Saisonprognose (Monte-Carlo-Simulation der Restsaison).

Aktuelle Tabelle aus allen gespielten Ligaspielen der laufenden Saison. Für alle noch
offenen Spiele (Spielplan aus Understat) liefert das Modell eine Ergebnismatrix; daraus
werden N komplette Restsaisons gezogen. Ergebnis je Team: erwartete Punkte und Platz,
Wahrscheinlichkeiten für Meisterschaft, Champions-League-Plätze und Abstieg.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from fussball import config, db
from fussball.models.blend import EnsembleDixonColes
from fussball.models.dixon_coles import score_matrix


@dataclass(frozen=True)
class Rules:
    cl: int          # Champions-League-Plätze (ohne Zusatzplätze)
    releg: int       # direkte Absteiger
    playoff: int     # Relegationsplatz (0 = keiner)


RULES = {"E0": Rules(4, 3, 0), "SP1": Rules(4, 3, 0), "I1": Rules(4, 3, 0),
         "D1": Rules(4, 2, 1), "F1": Rules(3, 2, 1)}
N_SIMS = 10_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS season_projection (
    created_at TEXT NOT NULL, league TEXT NOT NULL, season INTEGER NOT NULL, team_id TEXT NOT NULL,
    pos INTEGER, played INTEGER, won INTEGER, drawn INTEGER, lost INTEGER, gf INTEGER, ga INTEGER, pts INTEGER,
    exp_pts REAL, exp_pos REAL, p_title REAL, p_cl REAL, p_playoff REAL, p_releg REAL,
    pos_dist TEXT,
    PRIMARY KEY (created_at, league, team_id)
);
"""


def standings(results: pd.DataFrame) -> pd.DataFrame:
    """Tabelle aus Ergebnissen (Spalten home, away, hg, ag). Sortierung: Punkte, Tordifferenz, Tore."""
    rows = []
    for side, gf, ga in (("home", "hg", "ag"), ("away", "ag", "hg")):
        r = results[[side, gf, ga]].rename(columns={side: "team_id", gf: "gf", ga: "ga"})
        rows.append(r)
    t = pd.concat(rows)
    t = t.assign(won=(t.gf > t.ga).astype(int), drawn=(t.gf == t.ga).astype(int), lost=(t.gf < t.ga).astype(int))
    t = t.groupby("team_id").agg(played=("gf", "size"), won=("won", "sum"), drawn=("drawn", "sum"),
                                 lost=("lost", "sum"), gf=("gf", "sum"), ga=("ga", "sum")).reset_index()
    t["pts"] = 3 * t.won + t.drawn
    t = t.assign(gd=t.gf - t.ga).sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
    t["pos"] = np.arange(1, len(t) + 1)
    return t


def simulate(table: pd.DataFrame, fixtures: pd.DataFrame, rules: Rules, n: int = N_SIMS,
             seed: int = 0) -> pd.DataFrame:
    """fixtures: home_id, away_id, lam_h, lam_a, rho. Liefert Wahrscheinlichkeiten je Team."""
    rng = np.random.default_rng(seed)
    teams = list(table.team_id)
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    pts = np.tile(table.pts.to_numpy(float), (n, 1))
    gd = np.tile(table.gd.to_numpy(float), (n, 1))
    gf = np.tile(table.gf.to_numpy(float), (n, 1))
    for f in fixtures.itertuples():
        if f.home_id not in idx or f.away_id not in idx:
            continue
        cum = np.cumsum(score_matrix(f.lam_h, f.lam_a, f.rho).ravel())
        cell = np.minimum(np.searchsorted(cum, rng.random(n) * cum[-1]), 99)
        hg, ag = cell // 10, cell % 10
        h, a = idx[f.home_id], idx[f.away_id]
        pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gd[:, h] += hg - ag
        gd[:, a] += ag - hg
        gf[:, h] += hg
        gf[:, a] += ag
    # Platz je Simulation: Punkte, dann Tordifferenz, dann Tore (+ Zufall gegen Gleichstand)
    key = pts * 1e6 + (gd + 500) * 1e3 + gf + rng.random((n, k)) * 1e-3
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n)[:, None], order] = np.arange(1, k + 1)
    dist = np.stack([(pos == p).mean(0) for p in range(1, k + 1)], axis=1)   # (k, k)
    out = pd.DataFrame({
        "team_id": teams, "exp_pts": pts.mean(0), "exp_pos": pos.mean(0),
        "p_title": (pos == 1).mean(0), "p_cl": (pos <= rules.cl).mean(0),
        "p_releg": (pos > k - rules.releg).mean(0),
        "p_playoff": ((pos > k - rules.releg - rules.playoff) & (pos <= k - rules.releg)).mean(0),
    })
    out["pos_dist"] = [json.dumps(np.round(d, 4).tolist()) for d in dist]
    return out


def project(con: sqlite3.Connection, results: pd.DataFrame, data: pd.DataFrame, params: dict,
            today: date | None = None, n: int = N_SIMS) -> pd.DataFrame:
    """Prognose für alle Top-Ligen; speichert in season_projection und gibt sie zurück.

    results: gespielte Spiele (league, season, home, away, hg, ag), data: Fit-Daten (wie predict).
    """
    con.executescript(SCHEMA)
    today = today or date.today()
    season = config.current_season(today)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fixtures = db.read(con, "SELECT * FROM fixtures WHERE season = ?", (season,))
    out = []
    for league in config.TOP_LEAGUES:
        res = results[(results.league == league) & (results.season == season)]
        fx = fixtures[fixtures.league == league]
        if res.empty and fx.empty:
            continue
        table = standings(res)
        missing = sorted((set(fx.home_id) | set(fx.away_id)) - set(table.team_id))
        if missing:  # Teams ohne gespieltes Spiel
            table = pd.concat([table, pd.DataFrame({"team_id": missing})]).fillna(0)
            for c in ("played", "won", "drawn", "lost", "gf", "ga", "pts", "gd"):
                table[c] = table[c].astype(int)
            table["pos"] = np.arange(1, len(table) + 1)
        country = config.LEAGUES[league].country
        model = EnsembleDixonColes.from_params(params).fit(data[data.country == country], pd.Timestamp(today))
        rates = [model.rates(h, a) for h, a in zip(fx.home_id, fx.away_id)]
        fx = fx.assign(lam_h=[r[0] for r in rates], lam_a=[r[1] for r in rates], rho=model.rho)
        sim = simulate(table, fx, RULES[league], n)
        t = table.merge(sim, on="team_id").assign(created_at=created, league=league, season=season)
        out.append(t)
    proj = pd.concat(out, ignore_index=True)
    cols = ["created_at", "league", "season", "team_id", "pos", "played", "won", "drawn", "lost", "gf", "ga",
            "pts", "exp_pts", "exp_pos", "p_title", "p_cl", "p_playoff", "p_releg", "pos_dist"]
    db.upsert(con, "season_projection", proj[cols], key=["created_at", "league", "team_id"])
    return proj


def latest(con: sqlite3.Connection) -> pd.DataFrame:
    con.executescript(SCHEMA)
    return db.read(con, """SELECT * FROM season_projection
                           WHERE created_at = (SELECT MAX(created_at) FROM season_projection)""")
