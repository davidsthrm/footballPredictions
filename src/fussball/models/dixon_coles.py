"""Dixon-Coles (1997) mit exponentiellem Time-Decay.

    log λ_heim     = μ + h_tier + α_heim + β_ausw
    log λ_auswärts = μ         + α_ausw + β_heim

α = Angriffsstärke, β = Abwehrschwäche. Eine Liga wird gemeinsam mit ihrer 2. Liga
gefittet (Aufsteiger haben so eine Historie); Heimvorteil je Spielklasse. Ein
L2-Term auf α/β macht das Modell identifizierbar und zieht Teams mit wenig Daten
zum Mittel. ρ korrigiert die Abhängigkeit bei 0:0, 1:0, 0:1, 1:1.

Gewicht eines Spiels: exp(-ξ · Tage vor dem Cutoff).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 9
RHO_BOUNDS = (-0.3, 0.3)


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(Heim = i, Auswärts = j) für i, j in 0..max_goals, renormiert auf Summe 1."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


@dataclass
class DixonColes:
    xi: float = 0.002          # Time-Decay pro Tag
    reg: float = 1.0           # L2-Stärke auf α/β
    window_days: int = 3 * 365
    params_: dict = field(default_factory=dict, repr=False)

    # --- Fit -------------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp, warm_start: DixonColes | None = None) -> DixonColes:
        """df: Spalten date (Timestamp), home, away, hg, ag, tier. Nutzt nur date < cutoff."""
        d = df[(df.date < cutoff) & (df.date >= cutoff - pd.Timedelta(days=self.window_days))]
        if d.empty:
            raise ValueError("Keine Trainingsdaten vor dem Cutoff")
        team_list = pd.unique(pd.concat([d.home, d.away]))
        teams = {t: i for i, t in enumerate(team_list)}
        tiers = sorted(d.tier.unique())
        tier_idx = {t: i for i, t in enumerate(tiers)}
        n, nt = len(teams), len(tiers)

        hi = d.home.map(teams).to_numpy()
        ai = d.away.map(teams).to_numpy()
        ti = d.tier.map(tier_idx).to_numpy()
        x = d.hg.to_numpy(dtype=float)
        y = d.ag.to_numpy(dtype=float)
        age = (cutoff - d.date).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * age)

        # Parametervektor: [μ, h_tier..., α_1..n, β_1..n, ρ]
        i_h, i_a, i_b, i_r = 1, 1 + nt, 1 + nt + n, 1 + nt + 2 * n
        x0 = np.zeros(i_r + 1)
        x0[0], x0[i_h:i_a] = 0.1, 0.25
        if warm_start is not None and warm_start.params_:
            p = warm_start.params_
            x0[0] = p["mu"]
            for t, k in tier_idx.items():
                x0[i_h + k] = p["home"].get(t, 0.25)
            for t, k in teams.items():
                x0[i_a + k] = p["attack"].get(t, 0.0)
                x0[i_b + k] = p["defence"].get(t, 0.0)
            x0[i_r] = p["rho"]

        is00, is01, is10, is11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)
        reg = self.reg

        def obj(theta):
            mu0, home = theta[0], theta[i_h:i_a]
            att, dfn, rho = theta[i_a:i_b], theta[i_b:i_r], theta[i_r]
            ll_ = mu0 + home[ti] + att[hi] + dfn[ai]
            la_ = mu0 + att[ai] + dfn[hi]
            lam, mu = np.exp(ll_), np.exp(la_)

            t = np.ones_like(lam)
            t[is00] = 1 - lam[is00] * mu[is00] * rho
            t[is01] = 1 + lam[is01] * rho
            t[is10] = 1 + mu[is10] * rho
            t[is11] = 1 - rho
            t = np.maximum(t, 1e-10)

            ll = np.log(t) + x * ll_ - lam + y * la_ - mu
            f = -(w * ll).sum() + reg * (att @ att + dfn @ dfn)

            # ∂log τ / ∂λ, ∂μ, ∂ρ
            dl = np.zeros_like(lam); dm = np.zeros_like(lam); dr = np.zeros_like(lam)
            dl[is00] = -mu[is00] * rho / t[is00]; dm[is00] = -lam[is00] * rho / t[is00]
            dr[is00] = -lam[is00] * mu[is00] / t[is00]
            dl[is01] = rho / t[is01]; dr[is01] = lam[is01] / t[is01]
            dm[is10] = rho / t[is10]; dr[is10] = mu[is10] / t[is10]
            dr[is11] = -1 / t[is11]

            g_lh = w * (lam * dl + x - lam)   # ∂ll/∂log λ_heim
            g_la = w * (mu * dm + y - mu)     # ∂ll/∂log λ_ausw
            grad = np.empty_like(theta)
            grad[0] = -(g_lh.sum() + g_la.sum())
            grad[i_h:i_a] = -np.bincount(ti, g_lh, nt)
            grad[i_a:i_b] = -(np.bincount(hi, g_lh, n) + np.bincount(ai, g_la, n)) + 2 * reg * att
            grad[i_b:i_r] = -(np.bincount(ai, g_lh, n) + np.bincount(hi, g_la, n)) + 2 * reg * dfn
            grad[i_r] = -(w * dr).sum()
            return f, grad

        bounds = [(None, None)] * i_r + [RHO_BOUNDS]
        res = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "gtol": 1e-6})
        th = res.x
        self.params_ = {
            "mu": th[0],
            "home": {t: th[i_h + k] for t, k in tier_idx.items()},
            "attack": {t: th[i_a + k] for t, k in teams.items()},
            "defence": {t: th[i_b + k] for t, k in teams.items()},
            "rho": th[i_r],
            "converged": bool(res.success),
            "n_matches": len(d),
        }
        return self

    # --- Vorhersage ------------------------------------------------------------------------
    def rates(self, home: str, away: str, tier: int = 1) -> tuple[float, float]:
        p = self.params_
        a, b = p["attack"], p["defence"]
        lam = np.exp(p["mu"] + p["home"].get(tier, 0.0) + a.get(home, 0.0) + b.get(away, 0.0))
        mu = np.exp(p["mu"] + a.get(away, 0.0) + b.get(home, 0.0))
        return float(lam), float(mu)

    def predict_matrix(self, home: str, away: str, tier: int = 1) -> np.ndarray:
        lam, mu = self.rates(home, away, tier)
        return score_matrix(lam, mu, self.params_["rho"])
