import numpy as np
import pytest

from fussball import checks
from fussball.backtest import metrics
from fussball.models.dixon_coles import score_matrix


def test_rps_known_values():
    # perfekte Vorhersage -> 0, maximal falsch (Heimsieg sicher, Auswärtssieg tritt ein) -> 1
    assert metrics.rps(np.array([[1, 0, 0]]), np.array([0]))[0] == 0
    assert metrics.rps(np.array([[1, 0, 0]]), np.array([2]))[0] == pytest.approx(1.0)
    assert metrics.rps(np.array([[1 / 3] * 3]), np.array([1]))[0] == pytest.approx((1 / 9 + 1 / 9) / 2)


def test_outcome_probs_sum_to_one_and_order():
    p = metrics.outcome_probs(score_matrix(2.0, 0.8, -0.05))
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[2]


def test_hits():
    # Tipp 2:1 / 1:1 / 0:2 gegen Ergebnis 3:2 / 0:0 / 0:1
    h = metrics.hits([2, 1, 0], [1, 1, 2], [3, 0, 0], [2, 0, 1])
    assert h["exact"].tolist() == [False, False, False]
    assert h["diff"].tolist() == [True, True, False]
    assert h["tendency"].tolist() == [True, True, True]


@pytest.mark.parametrize("league,season,teams,exp", [
    ("E0", 2020, 20, 380), ("D1", 2020, 18, 306), ("F1", 2023, 18, 306), ("F1", 2019, 20, 279)])
def test_expected_matches(league, season, teams, exp):
    assert checks.expected_matches(league, season, teams) == exp
