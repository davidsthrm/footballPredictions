"""`fussball backtest`: Modellparameter tunen (nur Validierungssaisons) und Report erzeugen.

Zeitliche Aufteilung (fest, um Leakage durch Tuning zu vermeiden):
  2014/15–2015/16  Burn-in (nur Trainingsdaten)
  2016/17–test_from-1  Validierung (Tuning)
  ab test_from      Test (berichtete Zahlen)

Modell: BlendedDixonColes (Tore + xG), darauf eine LightGBM-Korrekturstufe (Form,
Tabelle, Ruhetage). LightGBM wird je Saison nur mit den Vorsaisons trainiert; die Anzahl
Boosting-Runden per Early Stopping auf der letzten Validierungssaison. Zum Vergleich
laufen die reine Tor-Baseline und DC+xG ohne LightGBM mit (model_comparison.csv).
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fussball import config
from fussball.backtest import benchmark, metrics, report, walkforward
from fussball.features import form
from fussball.models import lgbm, tipping
from fussball.models.dixon_coles import score_matrix

GRID = {
    "xi": [0.0015, 0.002, 0.003],
    "reg": [0.5, 1.0, 2.0],
    "w_xg": [0.4, 0.5, 0.6, 0.7],
    "xi_xg": [0.004, 0.005, 0.007],
}
BASELINE_PARAMS = {"xi": 0.002, "reg": 1.0, "w_xg": 0.0}
DEFAULT_PARAMS = {"xi": 0.002, "reg": 1.0, "w_xg": 0.5, "xi_xg": 0.005}
PARAMS_FILE = config.REPORT_DIR / "model_params.json"
COMPARISON_FILE = config.REPORT_DIR / "model_comparison.csv"
LGBM_FILE = config.REPORT_DIR / "lgbm_model.txt"
TIP_WEIGHTS_FILE = config.REPORT_DIR / "tip_weights.json"


def load_params() -> dict:
    """Getunte Parameter aus data/reports/, sonst Defaults."""
    return json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else dict(DEFAULT_PARAMS)


def model_name(p: dict, with_lgbm: bool = False) -> str:
    if not p.get("w_xg"):
        return f"dixon-coles xi={p['xi']} reg={p['reg']}"
    name = f"dc+xg xi={p['xi']} reg={p['reg']} w_xg={p['w_xg']} xi_xg={p['xi_xg']}"
    return name + " +lgbm" if with_lgbm else name


def _half_life(xi: float) -> str:
    return f"{np.log(2) / xi / 30.4:.0f} Monate" if xi else "∞"


def tune(data: pd.DataFrame, leagues: list[str], seasons: list[int]) -> pd.DataFrame:
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())] + [BASELINE_PARAMS]
    with ProcessPoolExecutor() as ex:
        futs = {(i, lg): ex.submit(walkforward.predict_league, data, lg, seasons, p)
                for i, p in enumerate(grid) for lg in leagues}
        res = {k: f.result() for k, f in futs.items()}
    rows = []
    for i, p in enumerate(grid):
        d = pd.concat([res[(i, lg)] for lg in leagues])
        rows.append({**p, "n": len(d), "logloss": float(-np.log(d.p_actual).mean()),
                     "exact_%": 100 * float(((d.pred_h == d.hg) & (d.pred_a == d.ag)).mean())})
    return pd.DataFrame(rows).sort_values("logloss")


def evaluate(con: sqlite3.Connection, pred: pd.DataFrame, test_from: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred, _ = walkforward.add_tips(pred)
    p = report.enrich(pred, benchmark.implied_probs(con))
    return p, report.summary(p, test_from)


def lgbm_stage(data: pd.DataFrame, pred: pd.DataFrame, first_eval: int, test_from: int,
               save_to=None) -> tuple[pd.DataFrame, int]:
    """Korrigiert die DC-Vorhersagen je Saison mit einem nur auf Vorsaisons trainierten LightGBM."""
    rows = lgbm.to_team_rows(form.build(data), pred)
    rounds = lgbm.LGBMCorrector.tune_rounds(rows[rows.season.between(first_eval, test_from - 2)],
                                            rows[rows.season == test_from - 1])
    corrected = []
    for s in sorted(pred.season.unique()):
        if s <= first_eval:
            continue  # keine Vorsaison mit Vorhersagen zum Trainieren
        m = lgbm.LGBMCorrector(rounds).fit(rows[rows.season < s])
        te = rows[rows.season == s]
        corrected.append(lgbm.match_rates(te, m.rates(te)))
    new = pd.concat(corrected).set_index("match_id")
    out = pred.set_index("match_id")
    out.update(new)
    if save_to is not None:
        lgbm.LGBMCorrector(rounds).fit(rows).save(save_to)
    return walkforward.apply_rates(out.reset_index()), rounds


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
        params = {k: float(best[k]) for k in GRID if pd.notna(best.get(k))}
        params["tuned_on"] = valid
        PARAMS_FILE.write_text(json.dumps(params, indent=1))
    else:
        params = load_params()
    if xi is not None:
        params["xi"] = xi
    print(f"\nModell: {model_name(params)}  (Halbwertszeit Tore {_half_life(params['xi'])}"
          + (f", xG {_half_life(params['xi_xg'])})" if params.get("w_xg") else ")"))

    seasons = list(range(first_eval, config.current_season() + 1))
    dc_pred = walkforward.run(data, leagues, seasons, params)
    stacked, rounds = lgbm_stage(data, dc_pred, first_eval, test_from, save_to=LGBM_FILE)
    print(f"LightGBM-Korrektur: {rounds} Runden (Early Stopping auf {test_from - 1})")
    p, summ = evaluate(con, stacked, test_from)
    _, dc = evaluate(con, dc_pred, test_from)
    _, base = evaluate(con, walkforward.run(data, leagues, seasons, BASELINE_PARAMS), test_from)
    cols = ["n", "logloss", "rps", "rps_odds", "exact_%", "diff_%", "tend_%", "pts_argmax",
            "tip_exact_%", "tip_diff_%", "tip_tend_%", "pts_tip", "tip_draw_%", "exact_1:1_%"]
    test_row = lambda s: s.set_index(["league", "season"]).loc[("ALL", "TEST"), cols].to_dict()
    comparison = pd.DataFrame([
        {"model": model_name(BASELINE_PARAMS), "rolle": "Baseline (nur Tore)", **test_row(base)},
        {"model": model_name(params), "rolle": "Tore + xG", **test_row(dc)},
        {"model": model_name(params, with_lgbm=True), "rolle": "aktuell", **test_row(summ)},
    ])

    # Gewichte für Live-Tipps: letzte 3 abgeschlossene Saisons des aktuellen Modells
    done = sorted(s for s in p.season.unique() if s < config.current_season())[-3:]
    hist = p[p.season.isin(done)]
    M = np.stack([score_matrix(r.lam_h, r.lam_a, r.rho) for r in hist.itertuples()])
    w = tipping.fit_weights(tipping.outcome_matrix_probs(M), metrics.outcome_index(hist.hg, hist.ag))
    tipping.save_weights(w, TIP_WEIGHTS_FILE, [int(s) for s in done])
    print(f"Tipp-Gewichte Heim/Remis/Auswärts (aus {done[0]}–{done[-1]}): {np.round(w, 2).tolist()}")
    test = p[p.season >= test_from]
    dist = report.score_distribution(test)
    out = config.REPORT_DIR
    p.to_csv(out / "backtest_predictions.csv", index=False)
    summ.to_csv(out / "backtest_summary.csv", index=False)
    comparison.to_csv(COMPARISON_FILE, index=False)
    dist.to_csv(out / "score_distribution.csv", index=False)
    report.plot_distribution(dist, out / "score_distribution.png",
                             f"Ergebnisverteilung Testzeitraum ab {config.season_label(test_from)} "
                             f"({len(test):,} Spiele)")
    headline = summ[summ.season.isin(["TEST", "VALID"])].sort_values(["season", "league"])
    md = [f"# Backtest {model_name(params, with_lgbm=True)}\n",
          f"Validierung {valid[0]}–{valid[-1]}, Test ab {test_from}. Log-Loss auf 10×10-Ergebnismatrix, "
          f"RPS auf 1X2, Treffer des argmax-Tipps. `rps_odds` = Pinnacle/Markt vor Schluss, "
          f"`rps_model_same` = Modell-RPS auf denselben Spielen.\n",
          "## Vergleich mit der Baseline (Test)\n", report.to_markdown(comparison),
          "\n## Validierung / Test\n", report.to_markdown(headline),
          "\n## Pro Liga und Saison\n", report.to_markdown(summ[~summ.season.isin(["TEST", "VALID"])]),
          "\n## Ergebnisverteilung (Test)\n", report.to_markdown(dist),
          "\n![Verteilung](score_distribution.png)\n"]
    (out / "backtest_report.md").write_text("\n".join(md))

    show = ["league", "season", "n", "logloss", "rps", "rps_odds", "exact_%", "tend_%", "pts_argmax",
            "tip_exact_%", "tip_tend_%", "pts_tip", "tip_draw_%"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\nVergleich Test (alle Ligen):")
        print(comparison.drop(columns=["model"]).to_string(index=False))
        print()
        print(headline[[c for c in show if c in headline]].to_string(index=False))
    print(f"\nReport: {out / 'backtest_report.md'}")
    return 0
