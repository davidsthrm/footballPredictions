"""Point-in-time-Features pro Team und Spiel (Team-Perspektive, zwei Zeilen pro Spiel).

Jedes Feature eines Spiels am Tag D nutzt ausschließlich Spiele des Teams mit Datum < D
(Rolling-Werte werden vor dem Mitteln um ein Spiel verschoben). Keine Quoten.

- rest:         Tage seit dem letzten Ligaspiel (nur Ligaspiele bekannt, max. 30)
- form_xgf/xga: exponentiell gewichtete xG-Form für/gegen (Halbwertszeit FORM_HL Spiele)
- form_gf/ga:   dasselbe für Tore
- form_deep/deep_allowed: Form der Deep Completions für/gegen (Understat)
- form_ppda/ppda_allowed: Form der PPDA eigen/Gegner (Pressing-Intensität)
- finishing:    Tore minus xG (Chancenverwertung; neigt zur Rückkehr zum Mittel)
- venue_xgf/xga: xG-Form nur aus Heim- bzw. Auswärtsspielen
- ppg, rank_frac: Tabellensituation der laufenden Saison vor dem Spieltag
- n_hist:       Anzahl bekannter Spiele (wenig Historie = unsichere Stärke)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORM_HL = 5        # Halbwertszeit in Spielen
FINISH_HL = 10
VENUE_HL = 4
MAX_REST = 30

TEAM_FEATURES = ["rest", "form_xgf", "form_xga", "form_gf", "form_ga", "finishing",
                 "venue_xgf", "venue_xga", "ppg", "rank_frac", "n_hist",
                 "form_deep", "form_deep_allowed", "form_ppda", "form_ppda_allowed"]
# Spielstil-Spalten je Spiel (Heim/Auswärts); fehlen sie in den Daten, bleiben die Features NaN.
# Deep/PPDA im Backtest (2026-10-08): Validierung ΔLog-Loss −0,0002, Test −0,0006 (z −3,8);
# eigenes Elo, xG-Elo, Pi-Ratings, Schüsse aufs Tor und xG-Trend ohne Gewinn auf der Validierung.
STYLE = {"deep": ("deep_h", "deep_a"), "deep_allowed": ("deep_a", "deep_h"),
         "ppda": ("ppda_h", "ppda_a"), "ppda_allowed": ("ppda_a", "ppda_h")}
FEATURES = ["is_home", "progress"] + TEAM_FEATURES + [f"opp_{c}" for c in TEAM_FEATURES]


def team_long(data: pd.DataFrame) -> pd.DataFrame:
    """Spiele -> Team-Zeilen (team, opp, is_home, gf, ga, xgf, xga, Spielstil)."""
    base = ["match_id", "league", "season", "date", "tier"]
    col = lambda c: data[c] if c in data else np.nan
    h = data[base].assign(team=data.home, opp=data.away, is_home=1, gf=data.hg, ga=data.ag,
                          xgf=data.xg_h, xga=data.xg_a, **{k: col(hc) for k, (hc, _) in STYLE.items()})
    a = data[base].assign(team=data.away, opp=data.home, is_home=0, gf=data.ag, ga=data.hg,
                          xgf=data.xg_a, xga=data.xg_h, **{k: col(ac) for k, (_, ac) in STYLE.items()})
    long = pd.concat([h, a], ignore_index=True)
    return long.sort_values(["team", "date", "is_home"]).reset_index(drop=True)


def _ewm_prev(s: pd.Series, hl: float) -> pd.Series:
    return s.shift(1).ewm(halflife=hl, ignore_na=True).mean()


def rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team", sort=False)
    out = long.copy()
    out["rest"] = (long.date - g.date.shift(1)).dt.days.clip(upper=MAX_REST).fillna(MAX_REST)
    out["n_hist"] = g.cumcount().clip(upper=100)
    for col in ("xgf", "xga", "gf", "ga", *STYLE):
        out[f"form_{col}"] = g[col].transform(lambda s: _ewm_prev(s, FORM_HL))
    out["finishing"] = (long.gf - long.xgf).groupby(long.team).transform(lambda s: _ewm_prev(s, FINISH_HL))
    gv = long.groupby(["team", "is_home"], sort=False)
    out["venue_xgf"] = gv.xgf.transform(lambda s: _ewm_prev(s, VENUE_HL))
    out["venue_xga"] = gv.xga.transform(lambda s: _ewm_prev(s, VENUE_HL))
    return out


def table_features(long: pd.DataFrame) -> pd.DataFrame:
    """Punkte pro Spiel, relativer Tabellenplatz und Saisonfortschritt vor dem Spieltag."""
    long = long.assign(pts=np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0))
    parts = []
    for _, g in long.groupby(["league", "season"], sort=False):
        teams = sorted(g.team.unique())
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        dates = np.sort(g.date.unique())
        di = np.searchsorted(dates, g.date.to_numpy())
        ti = g.team.map(idx).to_numpy()
        # Zeile k = Stand vor dem k-ten Spieltermin der Saison
        P = np.zeros((len(dates) + 1, n))
        G = np.zeros((len(dates) + 1, n))
        np.add.at(P, (di + 1, ti), g.pts.to_numpy())
        np.add.at(G, (di + 1, ti), 1)
        P, G = P.cumsum(0), G.cumsum(0)
        p_before, g_before = P[di, ti], G[di, ti]
        rank = (P[di] > p_before[:, None]).sum(1)       # 0 = Tabellenführer
        parts.append(pd.DataFrame({
            "match_id": g.match_id.to_numpy(), "team": g.team.to_numpy(),
            "ppg": np.where(g_before > 0, p_before / np.maximum(g_before, 1), np.nan),
            "rank_frac": np.where(g_before > 0, rank / max(n - 1, 1), np.nan),
            "progress": g_before / (2 * (n - 1)),
        }))
    return pd.concat(parts, ignore_index=True)


def build(data: pd.DataFrame) -> pd.DataFrame:
    """Team-Zeilen mit eigenen und gegnerischen Features; Ziel `goals` = eigene Tore (NaN = offen)."""
    long = rolling_form(team_long(data))
    long = long.merge(table_features(long), on=["match_id", "team"], how="left")
    opp = long[["match_id", "team"] + TEAM_FEATURES].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in TEAM_FEATURES}})
    cols = ["match_id", "league", "season", "date", "tier", "team", "opp", "is_home", "progress"] + TEAM_FEATURES
    return long[cols].assign(goals=long.gf).merge(opp, on=["match_id", "opp"], how="left")
