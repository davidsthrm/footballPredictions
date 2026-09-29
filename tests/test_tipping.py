import numpy as np
import pytest

from fussball.backtest import metrics
from fussball.models import tipping
from fussball.models.dixon_coles import score_matrix


def test_argmax_fallback_and_realistic_tip():
    m = score_matrix(1.45, 1.05, -0.05)  # ausgeglichen-leicht Heim: argmax ist 1:1
    assert tipping.tip(m, None) == (1, 1)
    h, a = tipping.tip(m, np.array([1.0, 1.0, 1.0]))  # Heimsieg ist wahrscheinlichster Ausgang
    assert h > a and (h, a) in {(1, 0), (2, 1)}


def test_draw_weight_produces_draw_tips():
    m = score_matrix(1.2, 1.15, -0.08)
    assert tipping.tip(m, np.array([1.0, 3.0, 1.0])) == (1, 1)


def test_fit_weights_matches_target_distribution():
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.8, 2.2, (600, 2))
    M = np.stack([score_matrix(a, b, -0.05) for a, b in lam])
    P = tipping.outcome_matrix_probs(M)
    actual = np.array([0] * 270 + [1] * 150 + [2] * 180)
    w = tipping.fit_weights(P, actual)
    share = np.bincount(np.argmax(P * w, 1), minlength=3) / len(P)
    assert share == pytest.approx([0.45, 0.25, 0.30], abs=0.05)


def test_save_load_weights(tmp_path):
    f = tmp_path / "w.json"
    assert tipping.load_weights(f) is None
    tipping.save_weights(np.array([1.0, 1.8, 1.3]), f, [2023, 2024, 2025])
    assert tipping.load_weights(f).tolist() == [1.0, 1.8, 1.3]


def test_tippspiel_points():
    pts = metrics.tippspiel_points([2, 2, 1, 1, 0], [1, 0, 1, 1, 2], [2, 3, 2, 1, 1], [1, 1, 2, 0, 0])
    assert pts.tolist() == [4, 3, 2, 0, 0]
