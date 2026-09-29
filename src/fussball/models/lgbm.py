"""LightGBM-Korrekturstufe auf dem Dixon-Coles-Modell (Tore + xG).

Pro Team und Spiel eine Zeile; Ziel = eigene Tore, Poisson-Objective. Das log-λ des
Dixon-Coles-Modells geht als Offset (init_score) ein, LightGBM lernt also nur
Abweichungen davon – aus Features wie Saisonfortschritt, xG-Form, Tabellensituation und
Ruhetagen (siehe fussball.features.form). Stark regularisiert und früh gestoppt, weil die
Korrekturen klein sind und die Daten begrenzt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fussball.features.form import FEATURES

PARAMS = dict(objective="poisson", learning_rate=0.02, num_leaves=8, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, seed=1, num_threads=4)
MAX_ROUNDS = 2000


@dataclass
class LGBMCorrector:
    rounds: int = 45
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    booster: lgb.Booster | None = None

    @staticmethod
    def offset(rows: pd.DataFrame) -> np.ndarray:
        return np.log(np.where(rows.is_home == 1, rows.lam_h, rows.lam_a))

    def fit(self, rows: pd.DataFrame) -> LGBMCorrector:
        ds = lgb.Dataset(rows[self.features], rows.goals, init_score=self.offset(rows))
        self.booster = lgb.train(PARAMS, ds, num_boost_round=self.rounds)
        return self

    @classmethod
    def tune_rounds(cls, train: pd.DataFrame, valid: pd.DataFrame) -> int:
        """Anzahl Boosting-Runden per Early Stopping auf einer Validierungssaison."""
        feats = list(FEATURES)
        dtr = lgb.Dataset(train[feats], train.goals, init_score=cls.offset(train))
        dva = lgb.Dataset(valid[feats], valid.goals, init_score=cls.offset(valid), reference=dtr)
        b = lgb.train(PARAMS, dtr, MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return max(int(b.best_iteration), 1)

    def rates(self, rows: pd.DataFrame) -> np.ndarray:
        """Korrigiertes λ je Team-Zeile."""
        raw = self.booster.predict(rows[self.features], raw_score=True) if self.booster else 0.0
        return np.exp(self.offset(rows) + raw)

    def save(self, path: Path) -> None:
        self.booster.save_model(str(path))
        path.with_suffix(".json").write_text(json.dumps({"rounds": self.rounds, "features": self.features}))

    @classmethod
    def load(cls, path: Path) -> LGBMCorrector | None:
        if not path.exists():
            return None
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(rounds=meta["rounds"], features=meta["features"],
                   booster=lgb.Booster(model_file=str(path)))


def to_team_rows(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Features (Team-Zeilen) + λ des Basismodells je Spiel (match_id, lam_h, lam_a)."""
    return features.merge(matches[["match_id", "lam_h", "lam_a"]], on="match_id")


def match_rates(rows: pd.DataFrame, lam: np.ndarray) -> pd.DataFrame:
    """Team-Zeilen mit korrigiertem λ -> je Spiel (match_id, lam_h, lam_a)."""
    r = rows[["match_id", "is_home"]].assign(lam=lam)
    h = r[r.is_home == 1].set_index("match_id").lam.rename("lam_h")
    a = r[r.is_home == 0].set_index("match_id").lam.rename("lam_a")
    return pd.concat([h, a], axis=1).dropna().reset_index()
