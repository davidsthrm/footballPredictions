"""Dixon-Coles auf Toren, kombiniert mit einem Dixon-Coles auf xG.

    log λ = w · log λ_xG + (1 - w) · log λ_Tore

xG misst die Qualität der Chancen und ist weniger zufallsabhängig als Tore; das
xG-Modell darf daher einen eigenen (typischerweise stärkeren) Time-Decay haben.
ρ (Korrektur für 0:0/1:0/0:1/1:1) stammt aus dem Tor-Modell, das xG-Modell ist reines
Poisson. Spiele ohne xG (2. Ligen) gehen mit ihren Toren ins xG-Modell ein.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fussball.models.dixon_coles import DixonColes, score_matrix


@dataclass
class BlendedDixonColes:
    xi: float = 0.002
    reg: float = 1.0
    w_xg: float = 0.0
    xi_xg: float | None = None
    window_days: int = 3 * 365
    goals: DixonColes | None = None
    xg: DixonColes | None = None

    @classmethod
    def from_params(cls, p: dict) -> BlendedDixonColes:
        return cls(**{k: p[k] for k in ("xi", "reg", "w_xg", "xi_xg", "window_days") if k in p})

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp,
            warm_start: BlendedDixonColes | None = None) -> BlendedDixonColes:
        ws = warm_start
        self.goals = DixonColes(self.xi, self.reg, self.window_days).fit(
            df, cutoff, ws.goals if ws else None)
        if self.w_xg > 0:
            xdf = df.assign(hg=df.xg_h.fillna(df.hg), ag=df.xg_a.fillna(df.ag))
            self.xg = DixonColes(self.xi_xg if self.xi_xg is not None else self.xi, self.reg,
                                 self.window_days, fit_rho=False).fit(xdf, cutoff, ws.xg if ws else None)
        return self

    @property
    def params_(self) -> dict:
        return self.goals.params_

    @property
    def rho(self) -> float:
        return self.goals.params_["rho"]

    def rates(self, home: str, away: str, tier: int = 1) -> tuple[float, float]:
        lg, mg = self.goals.rates(home, away, tier)
        if self.w_xg <= 0 or self.xg is None:
            return lg, mg
        lx, mx = self.xg.rates(home, away, tier)
        w = self.w_xg
        return float(np.exp(w * np.log(lx) + (1 - w) * np.log(lg))), float(np.exp(w * np.log(mx) + (1 - w) * np.log(mg)))

    def predict_matrix(self, home: str, away: str, tier: int = 1) -> np.ndarray:
        lam, mu = self.rates(home, away, tier)
        return score_matrix(lam, mu, self.rho)


@dataclass
class EnsembleDixonColes:
    """Mittelt mehrere BlendedDixonColes (unterschiedliche Decay-/Gewichts-Parameter).

    log λ = Mittel der log λ der Mitglieder, ρ = Mittel. Verringert die Varianz der
    Parameterwahl, ohne neue freie Parameter einzuführen.
    """
    members: list[BlendedDixonColes]

    @classmethod
    def from_params(cls, p: dict) -> BlendedDixonColes | EnsembleDixonColes:
        if not p.get("ensemble"):
            return BlendedDixonColes.from_params(p)
        return cls([BlendedDixonColes.from_params(m) for m in p["ensemble"]])

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp,
            warm_start: EnsembleDixonColes | None = None) -> EnsembleDixonColes:
        prev = warm_start.members if warm_start else [None] * len(self.members)
        self.members = [m.fit(df, cutoff, w) for m, w in zip(self.members, prev)]
        return self

    @property
    def params_(self) -> dict:
        return self.members[0].params_

    @property
    def rho(self) -> float:
        return float(np.mean([m.rho for m in self.members]))

    def rates(self, home: str, away: str, tier: int = 1) -> tuple[float, float]:
        r = np.log([m.rates(home, away, tier) for m in self.members])
        return tuple(float(x) for x in np.exp(r.mean(axis=0)))

    def predict_matrix(self, home: str, away: str, tier: int = 1) -> np.ndarray:
        lam, mu = self.rates(home, away, tier)
        return score_matrix(lam, mu, self.rho)
