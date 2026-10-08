import numpy as np
import pandas as pd
import pytest

from fussball.features import form
from fussball.models.lgbm import LGBMCorrector, match_rates, to_team_rows


def league_data(n_teams=6, rounds=4, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows, day, mid = [], pd.Timestamp("2021-08-01"), 0
    for r in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                xh, xa = rng.gamma(2, 0.7), rng.gamma(2, 0.55)
                rows.append({"match_id": mid, "league": "D1", "season": 2021 + r // 2, "date": day,
                             "home": h, "away": a, "hg": rng.poisson(xh), "ag": rng.poisson(xa),
                             "xg_h": xh, "xg_a": xa, "tier": 1, "country": "GER",
                             "deep_h": rng.poisson(6), "deep_a": rng.poisson(4),
                             "ppda_h": rng.gamma(4, 3), "ppda_a": rng.gamma(4, 3)})
                mid += 1
                day += pd.Timedelta(days=2)
    return pd.DataFrame(rows)


def test_features_use_only_earlier_matches():
    data = league_data()
    k = len(data) // 2
    target = data.match_id.iloc[k]
    before = form.build(data)
    poisoned = data.copy()
    poisoned.loc[poisoned.index >= k, ["hg", "ag", "xg_h", "xg_a", "deep_h", "ppda_a"]] = [9, 9, 9.0, 9.0, 99, 99.0]
    after = form.build(poisoned)
    cols = form.FEATURES
    a = before[before.match_id == target].sort_values("team")[cols].reset_index(drop=True)
    b = after[after.match_id == target].sort_values("team")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # ... aber spätere Spiele sehen die Änderung
    later = data.match_id.iloc[k + 20]
    assert not before[before.match_id == later][cols].equals(after[after.match_id == later][cols])


def test_table_features_before_matchday():
    data = league_data(n_teams=4, rounds=1)
    f = form.build(data)
    first = f[f.match_id == data.match_id.iloc[0]]
    assert first.ppg.isna().all() and (first.progress == 0).all()
    assert set(f.columns) >= set(form.FEATURES) | {"goals"}


def test_lgbm_corrector_offset_roundtrip(tmp_path):
    data = league_data(rounds=6)
    feats = form.build(data)
    base = data[["match_id"]].assign(lam_h=1.5, lam_a=1.1)
    rows = to_team_rows(feats, base)
    empty = LGBMCorrector(rounds=0)
    assert np.allclose(empty.rates(rows), np.where(rows.is_home == 1, 1.5, 1.1))

    model = LGBMCorrector(rounds=20).fit(rows)
    lam = model.rates(rows)
    assert np.isfinite(lam).all() and (lam > 0).all()
    path = tmp_path / "m.txt"
    model.save(path)
    loaded = LGBMCorrector.load(path)
    assert loaded.rounds == 20 and np.allclose(loaded.rates(rows), lam)
    per_match = match_rates(rows, lam)
    assert len(per_match) == len(data) and per_match[["lam_h", "lam_a"]].notna().all().all()
    assert LGBMCorrector.load(tmp_path / "fehlt.txt") is None


def test_lgbm_correct_for_pending_fixtures():
    from fussball import predict
    data = league_data(rounds=6)
    feats = form.build(data)
    model = LGBMCorrector(rounds=10).fit(to_team_rows(feats, data[["match_id"]].assign(lam_h=1.4, lam_a=1.0)))
    base = pd.DataFrame({"understat_match_id": [1, 2], "league": "D1", "season": 2023,
                         "date": ["2030-01-05", "2030-01-05"], "home_id": ["T0", "T2"],
                         "away_id": ["T1", "T3"], "lam_h": [1.4, 1.4], "lam_a": [1.0, 1.0], "rho": -0.05})
    out = predict.lgbm_correct(data, base, model)
    assert out[["lam_h", "lam_a"]].notna().all().all()
    assert out.lam_h.to_numpy() == pytest.approx([1.4, 1.4], rel=0.3)
