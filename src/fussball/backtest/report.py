"""Report-Tabellen und Plot für den Walk-forward-Backtest."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fussball.backtest import metrics
from fussball.models.dixon_coles import score_matrix

BASELINE_TIPS = {"1:1": (1, 1), "2:1": (2, 1)}


def enrich(pred: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    """Hängt Metriken je Spiel an (Modell, Baselines, Quoten-RPS)."""
    p = pred.copy()
    outcome = metrics.outcome_index(p.hg, p.ag)
    p["logloss"] = -np.log(p.p_actual)
    p["rps"] = metrics.rps(p[["p_h", "p_d", "p_a"]].to_numpy(), outcome)
    for k, v in metrics.hits(p.pred_h, p.pred_a, p.hg, p.ag).items():
        p[k] = v
    for name, (h, a) in BASELINE_TIPS.items():
        for k, v in metrics.hits(h, a, p.hg, p.ag).items():
            p[f"{k}_{name}"] = v
    if odds is not None and len(odds):
        p = p.merge(odds[["match_id", "p_h", "p_d", "p_a"]].rename(
            columns={"p_h": "o_h", "p_d": "o_d", "p_a": "o_a"}), on="match_id", how="left")
        has = p.o_h.notna().to_numpy()
        p["rps_odds"] = np.nan
        p.loc[has, "rps_odds"] = metrics.rps(p.loc[has, ["o_h", "o_d", "o_a"]].to_numpy(), outcome[has])
        p["rps_model_w_odds"] = p.rps.where(has)
    return p


def _agg(g: pd.DataFrame) -> pd.Series:
    s = {
        "n": len(g),
        "logloss": g.logloss.mean(),
        "rps": g.rps.mean(),
        "exact_%": 100 * g.exact.mean(),
        "diff_%": 100 * g["diff"].mean(),
        "tend_%": 100 * g.tendency.mean(),
    }
    for name in BASELINE_TIPS:
        s[f"exact_{name}_%"] = 100 * g[f"exact_{name}"].mean()
        s[f"tend_{name}_%"] = 100 * g[f"tendency_{name}"].mean()
    if "rps_odds" in g:
        s["rps_odds"] = g.rps_odds.mean()
        s["rps_model_same"] = g.rps_model_w_odds.mean()
        s["odds_cov_%"] = 100 * g.rps_odds.notna().mean()
    return pd.Series(s)


def summary(p: pd.DataFrame, test_from: int) -> pd.DataFrame:
    """Pro Liga und Saison, plus ALL je Saison und Validierung/Test-Blöcke."""
    p = p.assign(split=np.where(p.season >= test_from, "TEST", "VALID"))
    parts = [
        p.groupby(["league", "season"]).apply(_agg, include_groups=False).reset_index(),
        p.groupby("season").apply(_agg, include_groups=False).reset_index().assign(league="ALL"),
        p.groupby(["league", "split"]).apply(_agg, include_groups=False).reset_index()
         .rename(columns={"split": "season"}),
        p.groupby("split").apply(_agg, include_groups=False).reset_index()
         .rename(columns={"split": "season"}).assign(league="ALL"),
    ]
    out = pd.concat(parts, ignore_index=True)
    out["season"] = out.season.astype(str)
    out["n"] = out.n.astype(int)
    return out[["league", "season"] + [c for c in out.columns if c not in ("league", "season")]]


def score_distribution(p: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Anteil je Ergebnis: tatsächlich, als Tipp (argmax) und mittlere Modellwahrscheinlichkeit."""
    n = len(p)
    actual = p.groupby([p.hg.clip(upper=9), p.ag.clip(upper=9)]).size() / n
    tipped = p.groupby(["pred_h", "pred_a"]).size() / n
    mean_m = sum(score_matrix(r.lam_h, r.lam_a, r.rho) for r in p.itertuples()) / n
    idx = pd.MultiIndex.from_product([range(10), range(10)], names=["h", "a"])
    df = pd.DataFrame({
        "actual_%": 100 * actual.reindex(idx, fill_value=0).to_numpy(),
        "tipped_%": 100 * tipped.reindex(idx, fill_value=0).to_numpy(),
        "model_prob_%": 100 * mean_m.ravel(),
    }, index=idx).reset_index()
    df.insert(0, "score", df.h.astype(str) + ":" + df.a.astype(str))
    return df.sort_values("actual_%", ascending=False).head(top).drop(columns=["h", "a"])


def plot_distribution(dist: pd.DataFrame, path: Path, title: str) -> None:
    colors = {"actual_%": "#2a78d6", "tipped_%": "#eb6834", "model_prob_%": "#1baf7a"}
    labels = {"actual_%": "Tatsächlich", "tipped_%": "Getippt (argmax)",
              "model_prob_%": "Ø Modell-Wahrscheinlichkeit"}
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = np.arange(len(dist))
    w = 0.27
    for k, (col, c) in enumerate(colors.items()):
        ax.bar(x + (k - 1) * w, dist[col], width=w * 0.92, color=c, label=labels[col], zorder=2)
    ax.set_xticks(x, dist.score)
    ax.set_ylabel("Anteil der Spiele (%)", color="#52514e")
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=12)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimaler Markdown-Export (ohne tabulate-Dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}")
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
