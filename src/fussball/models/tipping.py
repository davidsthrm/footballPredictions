"""Tipp-Strategien: aus einer Ergebnismatrix einen konkreten Tipp machen.

- argmax:      wahrscheinlichstes Einzelergebnis. Maximiert exakte Treffer, tippt aber bei
               ausgeglichenen Spielen fast immer 1:1 (unrealistisch, schwache Tendenz).
- realistisch: erst den Ausgang (Heim/Remis/Auswärts) wählen, dann das wahrscheinlichste
               Ergebnis innerhalb dieses Ausgangs. Der Ausgang maximiert p_k · w_k; die
               Gewichte w werden so bestimmt, dass sich die Tipps über frühere Saisons so
               verteilen wie die echten Ergebnisse (sonst würde nie Remis getippt).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GOALS = np.arange(10)
_H, _A = np.meshgrid(GOALS, GOALS, indexing="ij")
OUTCOME = np.sign(_H - _A)        # 1 Heimsieg, 0 Remis, -1 Auswärtssieg je Matrixzelle
ORDER = (1, 0, -1)                 # Index 0/1/2 = Heim/Remis/Auswärts
DEFAULT_WEIGHTS = np.array([1.0, 1.5, 1.25])


def outcome_matrix_probs(M: np.ndarray) -> np.ndarray:
    """(n, 10, 10) -> (n, 3) Wahrscheinlichkeiten Heim/Remis/Auswärts."""
    return np.stack([M[:, OUTCOME == k].sum(1) for k in ORDER], axis=1)


def fit_weights(P: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Gewichte (Heim = 1), mit denen die Tipp-Verteilung der echten Verteilung am nächsten kommt.

    P: (n, 3) Modellwahrscheinlichkeiten, actual: (n,) Ausgang 0/1/2.
    """
    target = np.bincount(actual, minlength=3) / len(actual)
    best_err, best_w = np.inf, DEFAULT_WEIGHTS
    for wd in np.arange(0.8, 2.2, 0.01):
        for wa in np.arange(0.8, 1.8, 0.01):
            w = np.array([1.0, wd, wa])
            share = np.bincount(np.argmax(P * w, 1), minlength=3) / len(P)
            err = np.abs(share - target).sum()
            if err < best_err - 1e-12:
                best_err, best_w = err, w
    return best_w


def tip(m: np.ndarray, weights: np.ndarray | None = None) -> tuple[int, int]:
    """Realistischer Tipp für eine Matrix (argmax, falls keine Gewichte)."""
    if weights is None:
        i, j = np.unravel_index(np.argmax(m), m.shape)
        return int(i), int(j)
    p = np.array([m[OUTCOME == k].sum() for k in ORDER])
    k = ORDER[int(np.argmax(p * weights))]
    i, j = np.unravel_index(np.argmax(np.where(OUTCOME == k, m, 0)), m.shape)
    return int(i), int(j)


def save_weights(w: np.ndarray, path: Path, fitted_on: list[int]) -> None:
    path.write_text(json.dumps({"weights": [round(float(x), 3) for x in w], "fitted_on": fitted_on}))


def load_weights(path: Path) -> np.ndarray | None:
    return np.array(json.loads(path.read_text())["weights"]) if path.exists() else None
