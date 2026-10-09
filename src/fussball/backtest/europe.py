"""`fussball backtest-europe`: Walk-forward für Champions League, Europa League, Conference League.

Gleiche Regeln wie für die Ligen: für jeden Europapokal-Spieltag D nur Daten mit Datum < D;
Parameter (ξ, L2, Gewicht der Europapokal-Spiele) nur auf den Validierungssaisons 2016–2018
gewählt, berichtet ab 2019/20. Vergleich mit dem Modell ohne weitere Ligen und einer
Baseline, die nur den Heimvorteil kennt.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fussball import config
from fussball.backtest import metrics, walkforward
from fussball.models import tipping
from fussball.models.dixon_coles import MAX_GOALS, score_matrix
from fussball.models.europe import EuropeModel, europe_frame, fit_data

GRID = {"xi": [0.0015, 0.002, 0.003], "reg": [0.5, 1.0, 2.0], "w_eu": [1.0, 2.0, 4.0]}
FIRST_EVAL = config.FIRST_SEASON + walkforward.BURN_IN_SEASONS
TEST_FROM = 2019
PARAMS_FILE = config.REPORT_DIR / "europe_params.json"
TIP_WEIGHTS_FILE = config.REPORT_DIR / "tip_weights_eu.json"
PRED_FILE = config.REPORT_DIR / "europe_backtest.csv"
COMPARISON_FILE = config.REPORT_DIR / "europe_comparison.csv"


def load_params() -> dict:
    from fussball.models.europe import DEFAULT_PARAMS
    return json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else dict(DEFAULT_PARAMS)


def predict_season(data: pd.DataFrame, targets: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Walk-forward über die Spieltage einer Saison (Warmstart von Spieltag zu Spieltag)."""
    prev, out = None, []
    for day, games in targets.groupby("date", sort=True):
        model = EuropeModel.from_params(params).fit(data, day, warm_start=prev)
        prev = model
        for g in games.itertuples():
            lam, mu = model.rates(g.home, g.away)
            out.append({"tm_match_id": g.tm_match_id, "competition": g.competition, "season": g.season,
                        "date": day.date(), "home": g.home, "away": g.away, "hg": int(g.hg), "ag": int(g.ag),
                        "lam_h": lam, "lam_a": mu, "rho": model.rho,
                        "home_known": model.known(g.home), "away_known": model.known(g.away)})
    return pd.DataFrame(out)


def home_baseline(data: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Kennt nur den Heimvorteil: mittlere Heim-/Auswärtstore aller früheren Europapokal-Spiele."""
    eu = data[data.eu].sort_values("date")
    rows = []
    for day, games in targets.groupby("date"):
        past = eu[eu.date < day]
        rows.append(games.assign(lam_h=past.hg.mean(), lam_a=past.ag.mean(), rho=0.0))
    return pd.concat(rows)


def run(data: pd.DataFrame, targets: pd.DataFrame, params_list: list[dict]) -> list[pd.DataFrame]:
    """Je Parametersatz alle Saisons von `targets`, parallel über (Parametersatz, Saison)."""
    seasons = sorted(targets.season.unique())
    with ProcessPoolExecutor() as ex:
        futs = {(i, s): ex.submit(predict_season, data, targets[targets.season == s], p)
                for i, p in enumerate(params_list) for s in seasons}
        res = {k: f.result() for k, f in futs.items()}
    return [pd.concat([res[(i, s)] for s in seasons], ignore_index=True) for i in range(len(params_list))]


def scored(pred: pd.DataFrame) -> pd.DataFrame:
    """Matrix-Kennzahlen, realistische Tipps (Gewichte aus Vorsaisons) und Punkte."""
    rows = []
    for r in pred.itertuples():
        m = score_matrix(r.lam_h, r.lam_a, r.rho)
        rows.append((m[min(r.hg, MAX_GOALS), min(r.ag, MAX_GOALS)], *metrics.outcome_probs(m)))
    p = pred.assign(**dict(zip(["p_actual", "p_h", "p_d", "p_a"], map(list, zip(*rows)))))
    p, _ = walkforward.add_tips(p)
    h = metrics.hits(p.tip_h, p.tip_a, p.hg, p.ag)
    return p.assign(logloss=-np.log(p.p_actual),
                    rps=metrics.rps(p[["p_h", "p_d", "p_a"]].to_numpy(), metrics.outcome_index(p.hg, p.ag)),
                    exact=h["exact"], tendency=h["tendency"],
                    points=metrics.tippspiel_points(p.tip_h, p.tip_a, p.hg, p.ag))


def summarize(p: pd.DataFrame) -> dict:
    return {"n": len(p), "logloss": p.logloss.mean(), "rps": p.rps.mean(), "tend_%": 100 * p.tendency.mean(),
            "exact_%": 100 * p.exact.mean(), "pts": p.points.mean()}


def main(con: sqlite3.Connection, tune_params: bool = False) -> int:
    data = fit_data(con)
    e = europe_frame(con).dropna(subset=["hg", "ag"])
    targets = e[e.season >= FIRST_EVAL].astype({"hg": int, "ag": int})
    valid = targets[targets.season < TEST_FROM]
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if tune_params:
        grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
        res = run(data, valid, grid)
        tab = pd.DataFrame([{**g, **summarize(scored(r))} for g, r in zip(grid, res)]).sort_values("logloss")
        print(f"Tuning auf Validierung {FIRST_EVAL}–{TEST_FROM - 1} ({len(valid)} Spiele):")
        print(tab.head(8).to_string(index=False))
        params = {k: float(tab.iloc[0][k]) for k in GRID} | {"tuned_on": list(range(FIRST_EVAL, TEST_FROM))}
        PARAMS_FILE.write_text(json.dumps(params, indent=1))
    else:
        params = load_params()
    print(f"Parameter: {params}")

    model_p = {k: params[k] for k in GRID}
    full = scored(run(data, targets, [model_p])[0])
    no_other = scored(run(fit_data(con, include_other=False), targets, [model_p])[0])
    base = scored(home_baseline(data, targets))
    rows = []
    for name, p in (("Europa-Modell", full), ("ohne weitere Ligen", no_other), ("nur Heimvorteil", base)):
        for part, mask in (("VALID", p.season < TEST_FROM), ("TEST", p.season >= TEST_FROM)):
            rows.append({"modell": name, "teil": part, **summarize(p[mask])})
    comp = pd.DataFrame(rows)
    d = full.set_index("tm_match_id").logloss - no_other.set_index("tm_match_id").logloss
    test_ids = full[full.season >= TEST_FROM].tm_match_id
    dt = d.loc[test_ids]
    print(comp.to_string(index=False, float_format="{:.3f}".format))
    print(f"Weitere Ligen, Test ΔLog-Loss {dt.mean():+.4f} (z {dt.mean() / (dt.std() / np.sqrt(len(dt))):+.1f})")
    print(full[full.season >= TEST_FROM].groupby("competition").apply(
        lambda g: pd.Series(summarize(g)), include_groups=False).to_string(float_format="{:.3f}".format))

    done = sorted(s for s in full.season.unique() if s < config.current_season())[-3:]
    hist = full[full.season.isin(done)]
    M = np.stack([score_matrix(r.lam_h, r.lam_a, r.rho) for r in hist.itertuples()])
    w = tipping.fit_weights(tipping.outcome_matrix_probs(M), metrics.outcome_index(hist.hg, hist.ag))
    tipping.save_weights(w, TIP_WEIGHTS_FILE, [int(s) for s in done])
    full.to_csv(PRED_FILE, index=False)
    comp.to_csv(COMPARISON_FILE, index=False)
    print(f"Tipp-Gewichte Europa (aus {done[0]}–{done[-1]}): {np.round(w, 2).tolist()}")
    return 0
