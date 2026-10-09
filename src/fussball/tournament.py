"""Turnierprognose für Champions League, Europa League und Conference League (Format ab 2024/25).

Ligaphase: Für alle offenen Spiele liefert das Europa-Modell eine Ergebnismatrix; daraus werden
N_SIMS komplette Ligaphasen gezogen (Tabelle nach Punkten, Tordifferenz, Toren, danach Zufall –
die weiteren UEFA-Kriterien sind vereinfacht). Plätze 1–8: Achtelfinale, 9–24: Play-offs, ab 25: raus.

K.-o.-Baum nach der festen UEFA-Setzliste (ausgelost wird nur innerhalb der Paare):
  Play-offs    9/10 – 23/24, 11/12 – 21/22, 13/14 – 19/20, 15/16 – 17/18
  Achtelfinale 1/2 – Sieger 15/16–17/18, 3/4 – 13/14–19/20, 5/6 – 11/12–21/22, 7/8 – 9/10–23/24
  Die Teams eines Paares (1/2, 3/4, …) landen in verschiedenen Hälften des Baums.
Hin- und Rückspiel, Rückspiel beim Besserplatzierten; bei Gleichstand Verlängerung (λ/3) und
Elfmeterschießen (50:50). Finale auf neutralem Platz. Tore als Poisson ohne ρ-Korrektur.

Gilt nur, solange die Ligaphase läuft; danach (echte Paarungen) zeigt das Dashboard keinen Baum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fussball.models.europe import EuropeModel

N_SIMS = 10_000
ROUNDS = ["top8", "playoff", "r16", "qf", "sf", "final", "win"]
PO_SETS = [((15, 16), (17, 18)), ((13, 14), (19, 20)), ((11, 12), (21, 22)), ((9, 10), (23, 24))]
SEED_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8)]       # gehört jeweils zum PO-Set mit gleichem Index
HALF_ORDER = [0, 3, 2, 1]                            # Reihenfolge im Baum: 1/2, 7/8, 5/6, 3/4


@dataclass
class Forecast:
    table: pd.DataFrame      # je Team: aktuelle Tabelle + Wahrscheinlichkeiten je Runde
    bracket: list[list[dict]]  # wahrscheinlichster Baum: Runden -> Paarungen (a, b, p_a)


def current_table(played: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    rows = {t: {"team": t, "played": 0, "pts": 0, "gf": 0, "ga": 0} for t in teams}
    for r in played.itertuples():
        for t, f, a in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            x = rows[t]
            x["played"] += 1
            x["gf"] += f
            x["ga"] += a
            x["pts"] += 3 if f > a else (1 if f == a else 0)
    t = pd.DataFrame(rows.values())
    t = t.assign(gd=t.gf - t.ga).sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
    return t.assign(pos=np.arange(1, len(t) + 1))


def _rates(model: EuropeModel, teams: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """L[i, j] = erwartete Tore von i zu Hause gegen j; N[i, j] = dasselbe auf neutralem Platz."""
    p = model.dc.params_
    att = np.array([p["attack"].get(t, 0.0) for t in teams])
    dfn = np.array([p["defence"].get(t, 0.0) for t in teams])
    base = p["mu"] + att[:, None] + dfn[None, :]
    return np.exp(base + p["home"].get("EU", 0.0)), np.exp(base)


def _tie(rng, L, N, a, b, two_legs: bool = True) -> np.ndarray:
    """Sieger je Simulation. a, b: Teamindizes (Arrays); b hat das Rückspiel zu Hause.

    Gastgeber trifft mit L (inkl. Heimvorteil), Gast mit N (ohne) – wie im Dixon-Coles-Modell.
    """
    if two_legs:
        ga = rng.poisson(L[a, b]) + rng.poisson(N[a, b])      # Hinspiel bei a, Rückspiel bei b
        gb = rng.poisson(N[b, a]) + rng.poisson(L[b, a])
        et_a, et_b = N[a, b] / 3, L[b, a] / 3                 # Verlängerung im Rückspiel
    else:
        ga, gb = rng.poisson(N[a, b]), rng.poisson(N[b, a])
        et_a, et_b = N[a, b] / 3, N[b, a] / 3
    level = ga == gb
    ga = ga + np.where(level, rng.poisson(et_a), 0)
    gb = gb + np.where(level, rng.poisson(et_b), 0)
    pens = rng.random(len(a)) < 0.5
    return np.where(ga > gb, a, np.where(gb > ga, b, np.where(pens, a, b)))


def simulate(model: EuropeModel, league: pd.DataFrame, n: int = N_SIMS, seed: int = 1) -> Forecast:
    """league: alle Ligaphasen-Spiele eines Wettbewerbs (home, away, hg, ag; offene mit NaN)."""
    rng = np.random.default_rng(seed)
    teams = sorted(set(league.home) | set(league.away))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    played = league.dropna(subset=["hg", "ag"])
    table = current_table(played, teams)
    pts = np.zeros((n, k))
    gd = np.zeros((n, k))
    gf = np.zeros((n, k))
    for r in table.itertuples():
        i = idx[r.team]
        pts[:, i], gd[:, i], gf[:, i] = r.pts, r.gd, r.gf
    for f in league[league.hg.isna()].itertuples():          # offene Spiele aus der Ergebnismatrix ziehen
        m = model.matrix(f.home, f.away)
        cell = rng.choice(m.size, size=n, p=m.ravel() / m.sum())
        h, a = np.divmod(cell, m.shape[1])
        i, j = idx[f.home], idx[f.away]
        pts[:, i] += np.where(h > a, 3, np.where(h == a, 1, 0))
        pts[:, j] += np.where(a > h, 3, np.where(h == a, 1, 0))
        gd[:, i] += h - a
        gd[:, j] += a - h
        gf[:, i] += h
        gf[:, j] += a
    key = pts * 1e6 + gd * 1e3 + gf + rng.random((n, k))
    order = np.argsort(-key, axis=1)                          # order[s, p] = Team auf Platz p+1
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(1, k + 1)[None, :].repeat(n, 0), axis=1)

    L, N = _rates(model, teams)
    reach = {r: np.zeros(k) for r in ROUNDS}
    at = lambda p: order[:, p - 1]                            # Team auf Platz p (je Simulation)
    rows = np.arange(n)
    halves = {0: [], 1: []}
    for g, ((h1, h2), (l1, l2)) in enumerate(PO_SETS):
        cross = rng.random(n) < 0.5                           # Auslosung innerhalb der Paare
        lo_a, lo_b = np.where(cross, at(l1), at(l2)), np.where(cross, at(l2), at(l1))
        w1, w2 = _tie(rng, L, N, lo_a, at(h1)), _tie(rng, L, N, lo_b, at(h2))
        s1, s2 = SEED_PAIRS[g]
        flip = rng.random(n) < 0.5
        seed_top, seed_bot = np.where(flip, at(s1), at(s2)), np.where(flip, at(s2), at(s1))
        opp_top = np.where(rng.random(n) < 0.5, w1, w2)
        opp_bot = np.where(opp_top == w1, w2, w1)
        halves[0].append(_tie(rng, L, N, opp_top, seed_top))
        halves[1].append(_tie(rng, L, N, opp_bot, seed_bot))
        for w in (w1, w2):
            np.add.at(reach["r16"], w, 1)
    for p in range(1, 9):
        np.add.at(reach["top8"], at(p), 1)
        np.add.at(reach["r16"], at(p), 1)
    for p in range(9, 25):
        np.add.at(reach["playoff"], at(p), 1)

    def better_home(a, b):        # Rückspiel beim Besserplatzierten der Ligaphase
        swap = pos[rows, a] < pos[rows, b]
        return np.where(swap, b, a), np.where(swap, a, b)

    sf_teams = []
    for half in (0, 1):
        r16 = [halves[half][g] for g in HALF_ORDER]
        for w in r16:
            np.add.at(reach["qf"], w, 1)
        qf = [_tie(rng, L, N, *better_home(r16[0], r16[1])), _tie(rng, L, N, *better_home(r16[2], r16[3]))]
        for w in qf:
            np.add.at(reach["sf"], w, 1)
        f = _tie(rng, L, N, *better_home(qf[0], qf[1]))
        np.add.at(reach["final"], f, 1)
        sf_teams.append(f)
    champ = _tie(rng, L, N, sf_teams[0], sf_teams[1], two_legs=False)
    np.add.at(reach["win"], champ, 1)

    out = table.copy()
    ti = out.team.map(idx).to_numpy()
    out["exp_pos"] = pos.mean(0)[ti]
    out["exp_pts"] = pts.mean(0)[ti]
    for r in ROUNDS:
        out[f"p_{r}"] = reach[r][ti] / n
    out["p_out"] = 1 - out.p_top8 - out.p_playoff
    return Forecast(out, likely_bracket(out, L, N, teams, rng))


def _p_tie(rng, L, N, a: int, b: int, two_legs: bool = True, n: int = 4000) -> float:
    """P(a gewinnt die Paarung gegen b), b mit Rückspiel zu Hause."""
    w = _tie(rng, L, N, np.full(n, a), np.full(n, b), two_legs)
    return float((w == a).mean())


def likely_bracket(table: pd.DataFrame, L, N, teams: list[str], rng) -> list[list[dict]]:
    """Baum aus der erwarteten Abschlusstabelle; in jeder Paarung kommt der Favorit weiter."""
    idx = {t: i for i, t in enumerate(teams)}
    seed = table.sort_values("exp_pos").team.tolist()       # seed[p-1] = erwarteter Platz p
    at = lambda p: seed[p - 1]

    def tie(a: str, b: str, two_legs: bool = True) -> dict:
        p = _p_tie(rng, L, N, idx[a], idx[b], two_legs)
        return {"a": a, "b": b, "p_a": p, "winner": a if p >= 0.5 else b}

    po, r16 = {0: [], 1: []}, {0: [], 1: []}
    for g, ((h1, h2), (l1, l2)) in enumerate(PO_SETS):
        t1, t2 = tie(at(l2), at(h1)), tie(at(l1), at(h2))     # 15 – 18, 16 – 17 usw.
        po[0].append(t1)
        po[1].append(t2)
        s1, s2 = SEED_PAIRS[g]
        r16[0].append((t1["winner"], at(s1)))
        r16[1].append((t2["winner"], at(s2)))
    # Play-off-Paarung i speist Achtelfinale i (gleiche Reihenfolge wie im Baum)
    rounds = [[po[h][g] for h in (0, 1) for g in HALF_ORDER], [], [], [], []]
    for half in (0, 1):
        ties = [tie(*r16[half][g]) for g in HALF_ORDER]
        rounds[1] += ties
        qf = [tie(ties[0]["winner"], ties[1]["winner"]), tie(ties[2]["winner"], ties[3]["winner"])]
        rounds[2] += qf
        rounds[3].append(tie(qf[0]["winner"], qf[1]["winner"]))
    rounds[4].append(tie(rounds[3][0]["winner"], rounds[3][1]["winner"], two_legs=False))
    return rounds
