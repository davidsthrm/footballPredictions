"""`fussball backtest`: ξ/Regularisierung tunen (nur Validierungssaisons) und Report erzeugen.

Zeitliche Aufteilung (fest, um Leakage durch Tuning zu vermeiden):
  2014/15–2015/16  Burn-in (nur Trainingsdaten)
  2016/17–test_from-1  Validierung (Tuning von ξ und reg)
  ab test_from      Test (berichtete Zahlen)
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fussball import config
from fussball.backtest import benchmark, report, walkforward

XI_GRID = [0.0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006]
REG_GRID = [0.3, 1.0, 3.0]
DEFAULT_PARAMS = {"xi": 0.002, "reg": 1.0}
PARAMS_FILE = config.REPORT_DIR / "dixon_coles_params.json"


def tune(data: pd.DataFrame, leagues: list[str], seasons: list[int]) -> pd.DataFrame:
    grid = list(itertools.product(XI_GRID, REG_GRID))
    with ProcessPoolExecutor() as ex:
        futs = {(xi, reg, lg): ex.submit(walkforward.predict_league, data, lg, seasons, xi, reg)
                for xi, reg in grid for lg in leagues}
        res = {k: f.result() for k, f in futs.items()}
    rows = []
    for xi, reg in grid:
        p = pd.concat([res[(xi, reg, lg)] for lg in leagues])
        rows.append({"xi": xi, "reg": reg, "n": len(p), "logloss": float(-np.log(p.p_actual).mean())})
    return pd.DataFrame(rows).sort_values("logloss")


def main(con: sqlite3.Connection, leagues: list[str] | None = None, xi: float | None = None,
         tune_params: bool = False, test_from: int = 2019) -> int:
    leagues = leagues or config.TOP_LEAGUES
    data = walkforward.load_fit_data(con)
    first_eval = config.FIRST_SEASON + walkforward.BURN_IN_SEASONS
    valid = list(range(first_eval, test_from))
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if tune_params:
        grid = tune(data, leagues, valid)
        grid.to_csv(config.REPORT_DIR / "tuning_grid.csv", index=False)
        print(f"Tuning auf Validierung {valid[0]}–{valid[-1]} (Log-Loss, kleiner = besser):")
        print(grid.head(8).to_string(index=False))
        best = grid.iloc[0]
        params = {"xi": float(best.xi), "reg": float(best.reg), "tuned_on": valid}
        PARAMS_FILE.write_text(json.dumps(params, indent=1))
    else:
        params = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else dict(DEFAULT_PARAMS)
    if xi is not None:
        params["xi"] = xi
    print(f"\nParameter: ξ = {params['xi']}/Tag (Halbwertszeit "
          f"{np.log(2) / params['xi'] / 365:.1f} J.), reg = {params['reg']}" if params["xi"] else
          f"\nParameter: ξ = 0 (kein Decay), reg = {params['reg']}")

    seasons = list(range(first_eval, config.current_season() + 1))
    pred = walkforward.run(data, leagues, seasons, params["xi"], params["reg"])
    p = report.enrich(pred, benchmark.implied_probs(con))
    summ = report.summary(p, test_from)
    test = p[p.season >= test_from]
    dist = report.score_distribution(test)

    out = config.REPORT_DIR
    p.to_csv(out / "backtest_predictions.csv", index=False)
    summ.to_csv(out / "backtest_summary.csv", index=False)
    dist.to_csv(out / "score_distribution.csv", index=False)
    report.plot_distribution(dist, out / "score_distribution.png",
                             f"Ergebnisverteilung Testzeitraum ab {config.season_label(test_from)} "
                             f"({len(test):,} Spiele)")
    headline = summ[summ.season.isin(["TEST", "VALID"])].sort_values(["season", "league"])
    md = [f"# Backtest Dixon-Coles-Baseline\n",
          f"ξ = {params['xi']}, reg = {params['reg']}, Validierung {valid[0]}–{valid[-1]}, "
          f"Test ab {test_from}. Log-Loss auf 10×10-Ergebnismatrix, RPS auf 1X2, "
          f"Treffer des argmax-Tipps. `rps_odds` = Pinnacle/Markt vor Schluss, "
          f"`rps_model_same` = Modell-RPS auf denselben Spielen.\n",
          "## Validierung / Test\n", report.to_markdown(headline),
          "\n## Pro Liga und Saison\n", report.to_markdown(summ[~summ.season.isin(["TEST", "VALID"])]),
          "\n## Ergebnisverteilung (Test)\n", report.to_markdown(dist),
          "\n![Verteilung](score_distribution.png)\n"]
    (out / "backtest_report.md").write_text("\n".join(md))

    cols = ["league", "season", "n", "logloss", "rps", "rps_model_same", "rps_odds", "exact_%",
            "diff_%", "tend_%", "exact_1:1_%", "exact_2:1_%"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(headline[[c for c in cols if c in headline]].to_string(index=False))
        print("\nErgebnisverteilung Test (%):")
        print(dist.to_string(index=False))
    print(f"\nReport: {out / 'backtest_report.md'}")
    return 0
