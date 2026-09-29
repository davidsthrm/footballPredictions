"""Point-in-time-Features pro Team und Spiel (Team-Perspektive, zwei Zeilen pro Spiel).

Jedes Feature eines Spiels am Tag D nutzt ausschließlich Spiele des Teams mit Datum < D
(Rolling-Werte werden vor dem Mitteln um ein Spiel verschoben). Keine Quoten.

- rest:         Tage seit dem letzten Ligaspiel (nur Ligaspiele bekannt, max. 30)
- form_xgf/xga: exponentiell gewichtete xG-Form für/gegen (Halbwertszeit FORM_HL Spiele)
- form_gf/ga:   dasselbe für Tore
- finishing:    Tore minus xG (Chancenverwertung; neigt zur Rückkehr zum Mittel)
- venue_xgf/xga: xG-Form nur aus Heim- bzw. Auswärtsspielen
- ppg, rank_frac: Tabellensituation der laufenden Saison vor dem Spieltag
- n_hist:       Anzahl bekannter Spiele (wenig Historie = unsichere Stärke)
- eu_since/eu_until: Tage seit dem letzten / bis zum nächsten Europapokal-Spiel (max. 30);
                Termine stehen vorab im Spielplan fest
- eu_14d:       Europapokal-Spiele in den 14 Tagen vor dem Spieltag
- rest_all:     Ruhetage über Liga und Europapokal
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORM_HL = 5        # Halbwertszeit in Spielen
FINISH_HL = 10
VENUE_HL = 4
MAX_REST = 30

TEAM_FEATURES = ["rest", "form_xgf", "form_xga", "form_gf", "form_ga", "finishing",
                 "venue_xgf", "venue_xga", "ppg", "rank_frac", "n_hist"]
# Europapokal-Belastung: berechnet, aber nicht im Modell – im Backtest (2026-09-29) kein Gewinn,
# auch nicht bei Spielen direkt vor/nach Europapokal-Einsätzen (Log-Loss 2.9234 vs. 2.9235).
EU_FEATURES = ["eu_since", "eu_until", "eu_14d", "rest_all"]
FEATURES = ["is_home", "progress"] + TEAM_FEATURES + [f"opp_{c}" for c in TEAM_FEATURES]


def team_long(data: pd.DataFrame) -> pd.DataFrame:
    """Spiele -> Team-Zeilen (team, opp, is_home, gf, ga, xgf, xga)."""
    base = ["match_id", "league", "season", "date", "tier"]
    h = data[base].assign(team=data.home, opp=data.away, is_home=1, gf=data.hg, ga=data.ag,
                          xgf=data.xg_h, xga=data.xg_a)
    a = data[base].assign(team=data.away, opp=data.home, is_home=0, gf=data.ag, ga=data.hg,
                          xgf=data.xg_a, xga=data.xg_h)
    long = pd.concat([h, a], ignore_index=True)
    return long.sort_values(["team", "date", "is_home"]).reset_index(drop=True)


def _ewm_prev(s: pd.Series, hl: float) -> pd.Series:
    return s.shift(1).ewm(halflife=hl, ignore_na=True).mean()


def rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team", sort=False)
    out = long.copy()
    out["rest"] = (long.date - g.date.shift(1)).dt.days.clip(upper=MAX_REST).fillna(MAX_REST)
    out["n_hist"] = g.cumcount().clip(upper=100)
    for col in ("xgf", "xga", "gf", "ga"):
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


def europe_features(long: pd.DataFrame, europe: pd.DataFrame | None) -> pd.DataFrame:
    """Belastung durch Europapokal-Spiele (europe: Spalten team, date)."""
    out = long.assign(eu_since=float(MAX_REST), eu_until=float(MAX_REST), eu_14d=0.0)
    if europe is not None and len(europe):
        eu = {t: np.sort(g.date.to_numpy()) for t, g in europe.groupby("team")}
        for team, idx in out.groupby("team").groups.items():
            dates = eu.get(team)
            if dates is None:
                continue
            d = out.loc[idx, "date"].to_numpy()
            before = np.searchsorted(dates, d, side="left")      # Europa-Spiele strikt vor d
            after = np.searchsorted(dates, d, side="right")      # Europa-Spiele strikt nach d
            prev = np.where(before > 0, (d - dates[np.maximum(before - 1, 0)]) / np.timedelta64(1, "D"), np.inf)
            nxt = np.where(after < len(dates), (dates[np.minimum(after, len(dates) - 1)] - d) / np.timedelta64(1, "D"), np.inf)
            out.loc[idx, "eu_since"] = np.minimum(prev, MAX_REST)
            out.loc[idx, "eu_until"] = np.minimum(nxt, MAX_REST)
            out.loc[idx, "eu_14d"] = before - np.searchsorted(dates, d - np.timedelta64(14, "D"), side="left")
    out["rest_all"] = np.minimum(out.rest, out.eu_since)
    return out


def build(data: pd.DataFrame, europe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Team-Zeilen mit eigenen und gegnerischen Features; Ziel `goals` = eigene Tore (NaN = offen).

    europe: optionale Europapokal-Termine (Spalten team, date) für die Belastungs-Features.
    """
    long = rolling_form(team_long(data))
    long = europe_features(long, europe)
    long = long.merge(table_features(long), on=["match_id", "team"], how="left")
    opp = long[["match_id", "team"] + TEAM_FEATURES].rename(
        columns={"team": "opp", **{c: f"opp_{c}" for c in TEAM_FEATURES}})
    cols = ["match_id", "league", "season", "date", "tier", "team", "opp", "is_home", "progress"] + TEAM_FEATURES + EU_FEATURES
    return long[cols].assign(goals=long.gf).merge(opp, on=["match_id", "opp"], how="left")
