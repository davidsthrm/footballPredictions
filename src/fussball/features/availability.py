"""Gewicht aktueller Ausfälle je Team (aus dem letzten Transfermarkt-Snapshot vor dem Spieltag).

Die Bedeutung eines Spielers kommt aus Understat (laufende + vorige Saison beim selben
Verein): Anteil an den Einsatzminuten, an xG+xA (Offensive) und – für Verteidiger und
Torhüter – an den Minuten der Defensive. Transfermarkt- und Understat-Namen werden
innerhalb des Teams abgeglichen (Akzente/Schreibweisen); nicht gefundene Spieler
(meist Reservisten) zählen mit 0.

Historische Ausfälle sind nicht rekonstruierbar (Snapshots erst ab 2026-09-29). Die
Anpassung in `adjust_rates` ist daher eine vorsichtige, nicht per Backtest validierte
Schätzung und wird live gegen das Modell ohne Anpassung verglichen (A/B).
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

import numpy as np
import pandas as pd

from fussball import db

ATTACK_FACTOR = 0.5    # Ersatzspieler fängt etwa die Hälfte des fehlenden xG+xA auf
DEFENCE_FACTOR = 0.3   # fehlende Defensive erhöht die Gegentore moderat
MAX_ATTACK_CUT = 0.25
MAX_DEFENCE_RISE = 0.20
IGNORE_REASONS = ("keine Spielberechtigung",)  # meist nur Europapokal-Kaderliste


TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "Ae", "å": "a", "ß": "ss", "đ": "d",
                          "Đ": "D", "ł": "l", "Ł": "L", "ı": "i", "þ": "th", "ð": "d"})


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.translate(TRANSLIT)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]+", " ", s).strip()


def match_player(tm_name: str, candidates: pd.Series) -> int | None:
    """Index des passenden Understat-Spielers (Kandidaten: normalisierte Namen) oder None."""
    n = _norm(tm_name)
    hit = candidates[candidates == n]
    if len(hit) == 1:
        return hit.index[0]
    tok = set(n.split())
    subset = candidates[candidates.map(lambda c: bool(tok) and (tok <= set(c.split()) or set(c.split()) <= tok))]
    if len(subset) == 1:
        return subset.index[0]
    last, first = n.split()[-1] if n else "", n[:1]
    same_last = candidates[candidates.map(lambda c: c.split()[-1:] == [last] and c[:1] == first)]
    return same_last.index[0] if len(same_last) == 1 else None


def player_importance(con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Je Spieler und Team: Minuten-, Offensiv- und Defensivanteil (laufende + vorige Saison)."""
    p = db.read(con, """SELECT team_id, player, position, SUM(minutes) AS minutes, SUM(xg + xa) AS xgxa
                        FROM player_season_stats WHERE season IN (?, ?) AND team_id IS NOT NULL
                        GROUP BY team_id, understat_player_id""", (season, season - 1))
    p["is_def"] = p.position.fillna("").str.contains(r"\bD\b|GK")
    tot = p.groupby("team_id").agg(team_min=("minutes", "sum"), team_xgxa=("xgxa", "sum"))
    def_min = p[p.is_def].groupby("team_id").minutes.sum().rename("def_min")
    p = p.join(tot, on="team_id").join(def_min, on="team_id")
    p["lineup_share"] = 11 * p.minutes / p.team_min          # ~1 = Stammspieler
    p["att_share"] = p.xgxa / p.team_xgxa
    p["def_share"] = np.where(p.is_def, p.minutes / p.def_min, 0.0)
    p["norm"] = p.player.map(_norm)
    return p


def team_absences(con: sqlite3.Connection, match_date: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pro Team: fehlende Anteile, pro Spieler: Zuordnung) aus dem letzten Snapshot vor match_date."""
    snap = con.execute("SELECT MAX(snapshot_date) FROM availability_snapshots WHERE snapshot_date < ?",
                       (match_date,)).fetchone()[0]
    empty = pd.DataFrame(columns=["team_id", "n_absent", "missing_lineup", "missing_att", "missing_def"])
    if snap is None:
        return empty, pd.DataFrame()
    a = db.read(con, """SELECT team_id, tm_player_id, player, reason, until FROM availability_snapshots
                        WHERE snapshot_date = ? AND category IN ('injury', 'suspension')
                          AND team_id IS NOT NULL""", (snap,))
    a = a[~a.reason.fillna("").isin(IGNORE_REASONS)]
    a = a[a.until.isna() | (a.until >= match_date)].drop_duplicates(["team_id", "tm_player_id"])
    imp = player_importance(con, season)
    rows = []
    for r in a.itertuples():
        cand = imp[imp.team_id == r.team_id]
        idx = match_player(r.player, cand.norm)
        hit = cand.loc[idx] if idx is not None else None
        rows.append({"team_id": r.team_id, "player": r.player, "matched": hit is not None,
                     "lineup_share": float(hit.lineup_share) if hit is not None else 0.0,
                     "att_share": float(hit.att_share) if hit is not None else 0.0,
                     "def_share": float(hit.def_share) if hit is not None else 0.0})
    players = pd.DataFrame(rows)
    if players.empty:
        return empty, players
    per_team = players.groupby("team_id").agg(
        n_absent=("player", "size"), missing_lineup=("lineup_share", lambda s: s.sum() / 11),
        missing_att=("att_share", "sum"), missing_def=("def_share", "sum")).reset_index()
    return per_team, players


def adjust_rates(lam_h: float, lam_a: float, home: dict, away: dict) -> tuple[float, float]:
    """Erwartete Tore an Ausfälle anpassen (home/away: missing_att, missing_def)."""
    def cut(t):
        return 1 - min(ATTACK_FACTOR * t.get("missing_att", 0.0), MAX_ATTACK_CUT)

    def rise(t):
        return 1 + min(DEFENCE_FACTOR * t.get("missing_def", 0.0), MAX_DEFENCE_RISE)

    return lam_h * cut(home) * rise(away), lam_a * cut(away) * rise(home)
