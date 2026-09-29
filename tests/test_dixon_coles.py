import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from fussball.models.dixon_coles import DixonColes, score_matrix


def synthetic_league(n_teams=12, seasons=3, seed=0, rho=-0.08, home=0.3):
    rng = np.random.default_rng(seed)
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.2, n_teams)
    att -= att.mean()
    rows, day = [], pd.Timestamp("2020-08-01")
    for _ in range(seasons):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lam = np.exp(0.1 + home + att[h] + dfn[a])
                mu = np.exp(0.1 + att[a] + dfn[h])
                m = score_matrix(lam, mu, rho, max_goals=12)
                k = rng.choice(m.size, p=m.ravel())
                rows.append({"date": day, "home": f"T{h}", "away": f"T{a}",
                             "hg": k // 13, "ag": k % 13, "tier": 1})
                day += pd.Timedelta(hours=8)
    return pd.DataFrame(rows), att, dfn


def test_score_matrix_normalised_and_rho_effect():
    m0 = score_matrix(1.4, 1.1, 0.0)
    m1 = score_matrix(1.4, 1.1, -0.1)
    assert m0.shape == (10, 10)
    assert m0.sum() == pytest.approx(1.0) and m1.sum() == pytest.approx(1.0)
    # negatives ρ erhöht 0:0 und 1:1, senkt 1:0 und 0:1
    assert m1[0, 0] > m0[0, 0] and m1[1, 1] > m0[1, 1]
    assert m1[1, 0] < m0[1, 0] and m1[0, 1] < m0[0, 1]


def test_recovers_parameters_on_synthetic_data():
    df, att, dfn = synthetic_league()
    cutoff = df.date.max() + pd.Timedelta(days=1)
    model = DixonColes(xi=0.0, reg=0.01, window_days=10_000).fit(df, cutoff)
    p = model.params_
    est_att = np.array([p["attack"][f"T{i}"] for i in range(len(att))])
    assert np.corrcoef(est_att, att)[0, 1] > 0.8
    assert p["home"][1] == pytest.approx(0.3, abs=0.12)
    assert -0.3 <= p["rho"] < 0.05
    assert p["converged"]


def test_gradient_matches_finite_differences(monkeypatch):
    df, _, _ = synthetic_league(n_teams=6, seasons=1, seed=1)
    df["tier"] = np.where(df.index % 3 == 0, 2, 1)
    captured = {}
    import fussball.models.dixon_coles as dc

    real_minimize = dc.minimize

    def spy(fun, x0, **kw):
        captured["fun"], captured["x0"] = fun, x0
        return real_minimize(fun, x0, **kw)

    monkeypatch.setattr(dc, "minimize", spy)
    DixonColes(xi=0.003, reg=0.5).fit(df, df.date.max() + pd.Timedelta(days=1))
    fun = captured["fun"]
    rng = np.random.default_rng(3)
    theta = captured["x0"] + rng.normal(0, 0.1, captured["x0"].size)
    theta[-1] = -0.07
    err = check_grad(lambda t: fun(t)[0], lambda t: fun(t)[1], theta)
    assert err < 1e-3 * np.linalg.norm(fun(theta)[1]) + 1e-4


def test_fit_ignores_matches_on_or_after_cutoff():
    df, _, _ = synthetic_league(n_teams=6, seasons=1)
    cutoff = df.date.iloc[len(df) // 2]
    a = DixonColes(xi=0.002).fit(df, cutoff).params_
    poisoned = df.copy()
    poisoned.loc[poisoned.date >= cutoff, ["hg", "ag"]] = [9, 0]
    b = DixonColes(xi=0.002).fit(poisoned, cutoff).params_
    assert a["attack"] == pytest.approx(b["attack"])
    assert a["n_matches"] == (df.date < cutoff).sum()


def test_blend_without_xg_equals_goals_model_and_uses_xg():
    from fussball.models.blend import BlendedDixonColes
    df, _, _ = synthetic_league(n_teams=6, seasons=2, seed=4)
    cutoff = df.date.max() + pd.Timedelta(days=1)
    df["xg_h"], df["xg_a"] = np.nan, np.nan
    plain = DixonColes(xi=0.002, reg=1.0).fit(df, cutoff)
    blend0 = BlendedDixonColes(xi=0.002, reg=1.0, w_xg=0.0).fit(df, cutoff)
    assert blend0.rates("T0", "T1") == pytest.approx(plain.rates("T0", "T1"))

    # xG sagt: T0 ist viel stärker als die Tore vermuten lassen -> höheres λ für T0
    df.loc[df.home == "T0", "xg_h"] = 3.0
    df.loc[df.away == "T0", "xg_a"] = 3.0
    blend = BlendedDixonColes(xi=0.002, reg=1.0, w_xg=0.5, xi_xg=0.005).fit(df, cutoff)
    assert blend.rates("T0", "T1")[0] > plain.rates("T0", "T1")[0]
    assert blend.rho == pytest.approx(plain.params_["rho"])
    assert blend.xg.params_["rho"] == 0.0
    m = blend.predict_matrix("T0", "T1")
    assert m.sum() == pytest.approx(1.0)


def test_ensemble_mean_of_members():
    from fussball.models.blend import BlendedDixonColes, EnsembleDixonColes
    df, _, _ = synthetic_league(n_teams=6, seasons=2, seed=5)
    df["xg_h"], df["xg_a"] = df.hg * 0.9 + 0.2, df.ag * 0.9 + 0.2
    cutoff = df.date.max() + pd.Timedelta(days=1)
    p1 = {"xi": 0.002, "reg": 0.5, "w_xg": 0.6, "xi_xg": 0.005}
    p2 = {"xi": 0.0015, "reg": 0.5, "w_xg": 0.5, "xi_xg": 0.007}
    assert isinstance(EnsembleDixonColes.from_params(p1), BlendedDixonColes)   # ohne Ensemble: Einzelmodell
    ens = EnsembleDixonColes.from_params({**p1, "ensemble": [p1, p2]}).fit(df, cutoff)
    a = BlendedDixonColes.from_params(p1).fit(df, cutoff).rates("T0", "T1")
    b = BlendedDixonColes.from_params(p2).fit(df, cutoff).rates("T0", "T1")
    assert ens.rates("T0", "T1") == pytest.approx(tuple(np.sqrt(np.multiply(a, b))))
    assert ens.predict_matrix("T0", "T1").sum() == pytest.approx(1.0)
