"""Bewertungsmetriken für Ergebnismatrizen und 1X2-Wahrscheinlichkeiten."""

from __future__ import annotations

import numpy as np


def outcome_probs(m: np.ndarray) -> np.ndarray:
    """Ergebnismatrix -> [P(Heimsieg), P(Remis), P(Auswärtssieg)]."""
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def outcome_index(hg, ag) -> np.ndarray:
    """0 = Heimsieg, 1 = Remis, 2 = Auswärtssieg."""
    return np.select([np.asarray(hg) > np.asarray(ag), np.asarray(hg) == np.asarray(ag)], [0, 1], 2)


def rps(p: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Ranked Probability Score je Spiel für geordnete 1X2-Ausgänge (kleiner = besser)."""
    p = np.atleast_2d(p)
    o = np.eye(3)[outcome]
    return ((np.cumsum(p, axis=1) - np.cumsum(o, axis=1))[:, :2] ** 2).sum(axis=1) / 2


def hits(pred_h, pred_a, hg, ag) -> dict[str, np.ndarray]:
    """Trefferindikatoren eines Tipps: exaktes Ergebnis, Tordifferenz, Tendenz."""
    pred_h, pred_a, hg, ag = map(np.asarray, (pred_h, pred_a, hg, ag))
    return {
        "exact": (pred_h == hg) & (pred_a == ag),
        "diff": (pred_h - pred_a) == (hg - ag),
        "tendency": np.sign(pred_h - pred_a) == np.sign(hg - ag),
    }


def tippspiel_points(pred_h, pred_a, hg, ag) -> np.ndarray:
    """Übliches Tippspiel-Schema: 4 exakt, 3 richtige Tordifferenz (Sieg), 2 richtige Tendenz."""
    h = hits(pred_h, pred_a, hg, ag)
    winner = np.asarray(pred_h) != np.asarray(pred_a)
    return np.where(h["exact"], 4, np.where(h["diff"] & winner, 3, np.where(h["tendency"], 2, 0)))
