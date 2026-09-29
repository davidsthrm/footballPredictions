"""Walk-forward-Backtest: für jeden Spieltag (Anstoßdatum) D wird nur mit Spielen
mit Datum <= D - lag_days gefittet und dann alle Spiele an D vorhergesagt.

lag_days=1 entspricht einer Vorhersage am Vortag nach Spielschluss. Kein
zufälliger Split, keine Informationen aus der Zukunft.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fussball import config, db
from fussball.backtest import metrics
from fussball.models.dixon_coles import MAX_GOALS, DixonColes

BURN_IN_SEASONS = 2


def load_fit_data(con: sqlite3.Connection) -> pd.DataFrame:
    """Nur Ergebnisse – bewusst keine anderen Tabellen (insb. keine Quoten)."""
    df = db.read(con, "SELECT match_id, league, season, date, home_id AS home, away_id AS away, "
                      "fthg AS hg, ftag AS ag FROM matches")
    df["date"] = pd.to_datetime(df.date)
    df["tier"] = df.league.map(lambda c: config.LEAGUES[c].tier)
    df["country"] = df.league.map(lambda c: config.LEAGUES[c].country)
    return df


def predict_league(data: pd.DataFrame, league: str, seasons: list[int], xi: float,
                   reg: float = 1.0, lag_days: int = 1) -> pd.DataFrame:
    """Walk-forward-Vorhersagen für alle Spiele von `league` in `seasons`."""
    country = config.LEAGUES[league].country
    train = data[data.country == country]
    target = data[(data.league == league) & data.season.isin(seasons)].sort_values("date")
    prev = None
    out = []
    for day, games in target.groupby("date", sort=True):
        cutoff = day - pd.Timedelta(days=lag_days - 1)  # fit nutzt date < cutoff
        model = DixonColes(xi=xi, reg=reg).fit(train, cutoff, warm_start=prev)
        prev = model
        for g in games.itertuples():
            lam, mu = model.rates(g.home, g.away, tier=1)
            m = model.predict_matrix(g.home, g.away, tier=1)
            ph, pd_, pa = metrics.outcome_probs(m)
            i, j = np.unravel_index(np.argmax(m), m.shape)
            out.append({
                "match_id": g.match_id, "league": league, "season": g.season, "date": day.date(),
                "home": g.home, "away": g.away, "hg": g.hg, "ag": g.ag,
                "lam_h": lam, "lam_a": mu, "rho": model.params_["rho"],
                "p_actual": m[min(g.hg, MAX_GOALS), min(g.ag, MAX_GOALS)],
                "p_h": ph, "p_d": pd_, "p_a": pa, "pred_h": int(i), "pred_a": int(j),
                "p_pred": m[i, j], "home_known": g.home in model.params_["attack"],
                "away_known": g.away in model.params_["attack"],
            })
    return pd.DataFrame(out)


def run(data: pd.DataFrame, leagues: list[str], seasons: list[int], xi: float,
        reg: float = 1.0, lag_days: int = 1, workers: int | None = None) -> pd.DataFrame:
    """Alle Ligen parallel (je Liga ein Prozess)."""
    with ProcessPoolExecutor(max_workers=workers or len(leagues)) as ex:
        futs = [ex.submit(predict_league, data, lg, seasons, xi, reg, lag_days) for lg in leagues]
        return pd.concat([f.result() for f in futs], ignore_index=True)
