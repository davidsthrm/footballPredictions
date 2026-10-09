"""Live-Vorhersage der nächsten Europapokal-Runde (Champions League, Europa League, Conference League).

Je Wettbewerb die Spiele ab dem nächsten Spieltermin (bis zu 2 Tage später, z.B. CL Di + Mi),
vorhergesagt mit dem gemeinsamen Europa-Modell (fussball.models.europe) aus allen Spielen vor
heute. Gespeichert in `europe_predictions` (append-only wie `predictions`: Schlüssel Modell +
Spiel + Cutoff). Angepfiffene Spiele werden nicht neu getippt: bekannte Anstoßzeit (Ortszeit
Berlin) <= jetzt; ohne Anstoßzeit gilt ein Spiel von heute als begonnen, sobald es heute schon
einen Tipp gibt – ein Lauf am Abend überschreibt also nichts nach dem Anpfiff.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fussball import db
from fussball.backtest import metrics
from fussball.backtest.europe import TIP_WEIGHTS_FILE, load_params
from fussball.ingest import europe, other_leagues
from fussball.models import tipping
from fussball.models.europe import EuropeModel, europe_frame, fit_data

ROUND_SPAN_DAYS = 2
TZ = ZoneInfo("Europe/Berlin")   # Transfermarkt.de zeigt Anstoßzeiten in deutscher Ortszeit

SCHEMA = """
CREATE TABLE IF NOT EXISTS europe_predictions (
    model        TEXT NOT NULL,
    tm_match_id  INTEGER NOT NULL,
    cutoff_date  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    competition  TEXT NOT NULL,
    season       INTEGER NOT NULL,
    date         TEXT NOT NULL,
    home_id      TEXT NOT NULL, away_id TEXT NOT NULL,
    tm_home_id   INTEGER NOT NULL, tm_away_id INTEGER NOT NULL,
    home_name    TEXT, away_name TEXT,
    lam_h REAL, lam_a REAL, rho REAL, p_h REAL, p_d REAL, p_a REAL,
    pred_h INTEGER, pred_a INTEGER, tip_h INTEGER, tip_a INTEGER, p_tip REAL,
    home_known INTEGER, away_known INTEGER,
    matrix       TEXT NOT NULL,
    PRIMARY KEY (model, tm_match_id, cutoff_date)
);
"""


def ensure_schema(con: sqlite3.Connection) -> None:
    """Tabellen anlegen, die sonst erst die Ingest-Schritte erzeugen (frische DB, Tests)."""
    europe.ensure_schema(con)
    con.executescript(other_leagues.SCHEMA + SCHEMA)


def model_name(p: dict) -> str:
    return f"europa xi={p['xi']} reg={p['reg']} w_eu={p['w_eu']}"


def next_round(fixtures: pd.DataFrame, today: date) -> pd.DataFrame:
    """Je Wettbewerb die offenen Spiele vom nächsten Termin bis ROUND_SPAN_DAYS Tage danach."""
    fx = fixtures[fixtures.hg.isna() & (fixtures.date >= pd.Timestamp(today))]
    parts = []
    for _, g in fx.groupby("competition"):
        first = g.date.min()
        parts.append(g[g.date <= first + pd.Timedelta(days=ROUND_SPAN_DAYS)])
    return pd.concat(parts) if parts else fx


def predict(con: sqlite3.Connection, today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    ensure_schema(con)
    params = load_params()
    name = model_name(params)
    fx = next_round(europe_frame(con), today)
    if fx.empty:
        return pd.DataFrame()
    cutoff = today.isoformat()
    done_today = {r[0] for r in con.execute(
        "SELECT tm_match_id FROM europe_predictions WHERE model = ? AND cutoff_date = ? AND date = ?",
        (name, cutoff, cutoff))}
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    ko = pd.to_datetime(fx.date.dt.strftime("%Y-%m-%d") + " " + fx.kickoff.fillna("23:59"))
    started = (ko <= now) if today == date.today() else pd.Series(False, index=fx.index)
    started |= fx.kickoff.isna() & (fx.date == pd.Timestamp(today)) & fx.tm_match_id.isin(done_today)
    fx = fx[~started]
    model = EuropeModel.from_params(params).fit(fit_data(con), pd.Timestamp(today))
    weights = tipping.load_weights(TIP_WEIGHTS_FILE)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for f in fx.itertuples():
        lam, mu = model.rates(f.home, f.away)
        m = model.matrix(f.home, f.away)
        ph, pd_, pa = metrics.outcome_probs(m)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        ti, tj = tipping.tip(m, weights)
        rows.append({"model": name, "tm_match_id": int(f.tm_match_id), "cutoff_date": cutoff, "created_at": created,
                     "competition": f.competition, "season": int(f.season), "date": f.date.date().isoformat(),
                     "home_id": f.home, "away_id": f.away, "tm_home_id": int(f.tm_home_id),
                     "tm_away_id": int(f.tm_away_id), "home_name": f.home_name, "away_name": f.away_name,
                     "lam_h": lam, "lam_a": mu, "rho": model.rho, "p_h": ph, "p_d": pd_, "p_a": pa,
                     "pred_h": int(i), "pred_a": int(j), "tip_h": ti, "tip_a": tj, "p_tip": float(m[ti, tj]),
                     "home_known": int(model.known(f.home)), "away_known": int(model.known(f.away)),
                     "matrix": json.dumps(np.round(m, 6).tolist())})
    out = pd.DataFrame(rows)
    if len(out):
        db.upsert(con, "europe_predictions", out, key=["model", "tm_match_id", "cutoff_date"])
    return out


def upcoming(con: sqlite3.Connection, today: date) -> pd.DataFrame:
    """Jüngster Tipp je noch offenem Europapokal-Spiel (für das Dashboard)."""
    ensure_schema(con)
    p = db.read(con, """SELECT p.*, e.kickoff, e.round, e.leg, e.matchday FROM europe_predictions p
        JOIN europe_matches e ON e.tm_match_id = p.tm_match_id
        WHERE e.hg IS NULL AND p.date >= ?""", (today.isoformat(),))
    return p.sort_values("cutoff_date").drop_duplicates("tm_match_id", keep="last")


def scorecard(con: sqlite3.Connection) -> pd.DataFrame:
    """Echte Bilanz: je Spiel der letzte Tipp mit Cutoff am oder vor dem Spieltag."""
    ensure_schema(con)
    p = db.read(con, """SELECT p.*, e.hg, e.ag FROM europe_predictions p
        JOIN europe_matches e ON e.tm_match_id = p.tm_match_id
        WHERE e.hg IS NOT NULL AND p.cutoff_date <= p.date""")
    if p.empty:
        return p
    p = p.sort_values("cutoff_date").drop_duplicates(["model", "tm_match_id"], keep="last")
    return p.assign(points=metrics.tippspiel_points(p.tip_h, p.tip_a, p.hg, p.ag))
