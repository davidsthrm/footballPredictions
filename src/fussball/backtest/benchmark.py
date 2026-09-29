"""Quoten-Benchmark. EINZIGES Modul, das `odds_benchmark` lesen darf.

Die Quoten dienen ausschließlich als Vergleichsmaßstab (RPS auf 1X2) und fließen
nie in Features oder Modell-Fits ein (siehe tests/test_no_odds_leakage.py).

Vorrang: Pinnacle vor Schluss (bei football-data.co.uk Fr./Di. nachmittags erfasst,
passt zum Vortags-Cutoff) -> Marktdurchschnitt -> Bet365.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from fussball import db

PREFERENCE = [("PS", "pre"), ("Avg", "pre"), ("B365", "pre")]


def implied_probs(con: sqlite3.Connection, kind: str = "pre") -> pd.DataFrame:
    """match_id, p_h, p_d, p_a, bookmaker – Marge proportional herausgerechnet."""
    pref = PREFERENCE if kind == "pre" else [("PS", "closing"), ("Avg", "closing")]
    o = db.read(con, "SELECT match_id, bookmaker, kind, odds_h, odds_d, odds_a FROM odds_benchmark "
                     "WHERE odds_h > 1 AND odds_d > 1 AND odds_a > 1")
    rank = {bk: i for i, bk in enumerate(pref)}
    o["rank"] = [rank.get((b, k)) for b, k in zip(o.bookmaker, o.kind)]
    o = o.dropna(subset=["rank"]).sort_values("rank").drop_duplicates("match_id")
    inv = 1 / o[["odds_h", "odds_d", "odds_a"]].to_numpy()
    inv = inv / inv.sum(axis=1, keepdims=True)
    return pd.DataFrame({"match_id": o.match_id.to_numpy(), "p_h": inv[:, 0], "p_d": inv[:, 1],
                         "p_a": inv[:, 2], "bookmaker": o.bookmaker.to_numpy()})
