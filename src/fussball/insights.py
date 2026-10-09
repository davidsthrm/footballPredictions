"""Statistische Einordnungen fürs Dashboard: Form, direkte Duelle, Spielfakten, Wochen-Fakten.

Alles wird aus den vorhandenen Ergebnissen, xG- und Spielerdaten berechnet und ändert sich
automatisch mit jedem neuen Spieltag. Reine Darstellung – nichts davon fließt ins Modell.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from fussball import config, db


@dataclass
class Fact:
    icon: str
    title: str
    text: str
    ranking: list[tuple[str, str]]   # (team_id oder Spielername, Wert als Text)
    kind: str = "team"               # team | player | match


TOP_N = 5   # Plätze je Fakten-Rangliste


def _fmt(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def results_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Gespielte Spiele (alle Ligen) mit xG, aufsteigend nach Datum."""
    r = data.dropna(subset=["hg", "ag"]).sort_values("date")
    return r[["league", "season", "date", "home", "away", "hg", "ag", "xg_h", "xg_a", "tier"]].reset_index(drop=True)


def team_games(res: pd.DataFrame, team: str, before, league_only: str | None = None) -> pd.DataFrame:
    """Spiele eines Teams vor einem Datum aus Team-Sicht (gf, ga, xgf, xga, result S/U/N)."""
    r = res[((res.home == team) | (res.away == team)) & (res.date < before)]
    if league_only:
        r = r[r.league == league_only]
    home = r.home == team
    g = pd.DataFrame({
        "date": r.date, "league": r.league, "season": r.season, "is_home": home,
        "opp": np.where(home, r.away, r.home),
        "gf": np.where(home, r.hg, r.ag), "ga": np.where(home, r.ag, r.hg),
        "xgf": np.where(home, r.xg_h, r.xg_a), "xga": np.where(home, r.xg_a, r.xg_h)})
    g["res"] = np.select([g.gf > g.ga, g.gf == g.ga], ["S", "U"], "N")
    return g


def form(res: pd.DataFrame, team: str, before, n: int = 5) -> list[str]:
    """Letzte n Ligaspiele, ältestes zuerst: 'S' Sieg, 'U' Unentschieden, 'N' Niederlage."""
    return team_games(res, team, before).tail(n).res.tolist()


def streak(games: pd.DataFrame) -> tuple[str, int]:
    """Aktuelle Serie: ('ungeschlagen'|'sieglos'|'siegreich', Länge)."""
    seq = games.res.tolist()[::-1]
    def run(pred):
        k = 0
        for x in seq:
            if not pred(x):
                break
            k += 1
        return k
    wins, unbeaten, winless = run(lambda x: x == "S"), run(lambda x: x != "N"), run(lambda x: x != "S")
    if wins >= 3:
        return "siegreich", wins
    if unbeaten >= 4:
        return "ungeschlagen", unbeaten
    if winless >= 4:
        return "sieglos", winless
    return "", 0


def match_facts(res: pd.DataFrame, players: pd.DataFrame, home: str, away: str, match_date,
                season: int, names: dict, lam_h: float, lam_a: float, context: dict) -> list[str]:
    """Bis zu 5 kurze Fakten zu einem Spiel (Sätze, bereits mit deutschen Teamnamen)."""
    d = pd.Timestamp(match_date)
    nh, na = names.get(home, home), names.get(away, away)
    facts = []
    # Direkte Duelle
    h2h = res[(((res.home == home) & (res.away == away)) | ((res.home == away) & (res.away == home)))
              & (res.date < d)].tail(5)
    if len(h2h):
        w_h = int((((h2h.home == home) & (h2h.hg > h2h.ag)) | ((h2h.away == home) & (h2h.ag > h2h.hg))).sum())
        w_a = int((((h2h.home == away) & (h2h.hg > h2h.ag)) | ((h2h.away == away) & (h2h.ag > h2h.hg))).sum())
        dr = len(h2h) - w_h - w_a
        last = h2h.iloc[-1]
        last_txt = (f"{names.get(last.home, last.home)} – {names.get(last.away, last.away)} "
                    f"{int(last.hg)}:{int(last.ag)} ({last.date:%d.%m.%Y})")
        facts.append(f"⚔️ Letzte {len(h2h)} Duelle: {w_h}× {nh}, {dr}× Remis, {w_a}× {na}. Zuletzt: {last_txt}.")
    else:
        facts.append("⚔️ Kein Ligaduell der beiden seit 2014/15.")
    for team, name in ((home, nh), (away, na)):
        g = team_games(res, team, d)
        kind, k = streak(g)
        if kind:
            facts.append(f"{'🔥' if kind != 'sieglos' else '🥶'} {name} ist seit {k} Ligaspielen {kind}.")
        cur = g[g.season == season]
        if len(cur) >= 3 and cur.xgf.notna().all():
            luck = cur.gf.sum() - cur.xgf.sum()
            if luck >= 2.5:
                facts.append(f"🍀 {name} hat diese Saison {_fmt(luck)} Tore mehr erzielt als die xG.")
            elif luck <= -2.5:
                facts.append(f"😬 {name} liegt diese Saison {_fmt(abs(luck))} Tore unter den eigenen xG.")
        top = players[(players.team_id == team) & (players.goals > 0)].sort_values(["goals", "xg"], ascending=False)
        if len(top) and top.goals.iloc[0] >= 3:
            p = top.iloc[0]
            facts.append(f"👟 Bester Torschütze bei {name}: {p.player} mit {int(p.goals)} Toren ({_fmt(p.xg)} xG).")
    total = lam_h + lam_a
    if context.get("max_total") and abs(total - context["max_total"]) < 1e-9:
        facts.append(f"🥅 Torreichstes Spiel des Spieltags laut Modell ({_fmt(total)} Tore erwartet).")
    elif context.get("min_total") and abs(total - context["min_total"]) < 1e-9:
        facts.append(f"🧱 Torärmstes Spiel des Spieltags laut Modell ({_fmt(total)} Tore erwartet).")
    return facts[:6]


def weekly_facts(res: pd.DataFrame, players: pd.DataFrame, backtest: pd.DataFrame | None,
                 names: dict, season: int, today: date) -> list[Fact]:
    """Statistische Highlights der laufenden Saison über alle Top-5-Ligen."""
    cur = res[(res.season == season) & res.league.isin(config.TOP_LEAGUES)]
    facts: list[Fact] = []
    if cur.empty:
        return facts
    rows = []
    for side, o in (("home", "away"), ("away", "home")):
        h = side == "home"
        rows.append(pd.DataFrame({"team": cur[side], "league": cur.league,
                                  "gf": cur.hg if h else cur.ag, "ga": cur.ag if h else cur.hg,
                                  "xgf": cur.xg_h if h else cur.xg_a, "xga": cur.xg_a if h else cur.xg_h}))
    t = pd.concat(rows).groupby(["team", "league"]).agg(n=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"),
                                                          xgf=("xgf", "sum"), xga=("xga", "sum")).reset_index()
    t = t[t.n >= 3]
    nm = lambda x: names.get(x, x)
    top = lambda df, col, asc, fmt: [(r.team, fmt(r)) for r in df.sort_values(col, ascending=asc).head(TOP_N).itertuples()]
    t["gpg"], t["gapg"], t["luck"], t["dluck"] = t.gf / t.n, t.ga / t.n, t.gf - t.xgf, t.xga - t.ga
    if len(t):
        b = t.sort_values("gpg", ascending=False).iloc[0]
        facts.append(Fact("🥅", "Meiste Tore", f"{nm(b.team)}: {_fmt(b.gpg)} Tore pro Spiel.",
                          top(t, "gpg", False, lambda r: f"{_fmt(r.gpg)} Tore/Spiel")))
        b = t.sort_values("gapg").iloc[0]
        facts.append(Fact("🧱", "Wenigste Gegentore", f"{nm(b.team)}: {_fmt(b.gapg)} Gegentore pro Spiel.",
                          top(t, "gapg", True, lambda r: f"{_fmt(r.gapg)} Gegentore/Spiel")))
        b = t.sort_values("luck", ascending=False).iloc[0]
        facts.append(Fact("🍀", "Mehr Tore als xG", f"{nm(b.team)} liegt {_fmt(b.luck)} Tore über den eigenen xG.", top(t, "luck", False, lambda r: f"+{_fmt(r.luck)} Tore vs. xG")))
        b = t.sort_values("luck").iloc[0]
        facts.append(Fact("😬", "Weniger Tore als xG", f"{nm(b.team)} liegt {_fmt(abs(b.luck))} Tore unter den eigenen xG.", top(t, "luck", True, lambda r: f"{_fmt(r.luck)} Tore vs. xG")))
        b = t.sort_values("dluck", ascending=False).iloc[0]
        facts.append(Fact("🧤", "Weniger Gegentore als xG", f"{nm(b.team)} hat {_fmt(b.dluck)} Gegentore weniger "
                          f"kassiert, als die Chancen der Gegner erwarten ließen.", top(t, "dluck", False, lambda r: f"{_fmt(r.dluck)} verhinderte Tore")))
    # Serien
    serien = []
    for team in pd.unique(pd.concat([cur.home, cur.away])):
        g = team_games(res[res.league.isin(config.TOP_LEAGUES)], team, pd.Timestamp(today))
        kind, k = streak(g)
        if kind in ("ungeschlagen", "siegreich"):
            k_unb = streak_len(g, lambda x: x != "N")
            serien.append((team, k_unb))
    if serien:
        serien.sort(key=lambda x: -x[1])
        facts.append(Fact("📈", "Ungeschlagen", f"{nm(serien[0][0])}: seit {serien[0][1]} Ligaspielen ohne Niederlage.",
                          [(tm, f"{k} Spiele") for tm, k in serien[:TOP_N]]))
    # Spieler
    pl = players[players.goals > 0]
    if len(pl):
        b = pl.sort_values(["goals", "xg"], ascending=False).iloc[0]
        facts.append(Fact("👟", "Torjäger", f"{b.player} führt mit {int(b.goals)} Toren.",
                          [(r.player, f"{int(r.goals)} Tore · {_fmt(r.xg)} xG") for r in
                           pl.sort_values(["goals", "xg"], ascending=False).head(TOP_N).itertuples()], "player"))
        pl = pl.assign(over=pl.goals - pl.xg)
        b = pl.sort_values("over", ascending=False).iloc[0]
        facts.append(Fact("🎯", "Spieler über xG", f"{b.player}: {_fmt(b.over)} Tore mehr als seine xG.",
                          [(r.player, f"{int(r.goals)} Tore aus {_fmt(r.xg)} xG") for r in
                           pl.sort_values("over", ascending=False).head(TOP_N).itertuples()], "player"))
    # Überraschung und Berechenbarkeit (Walk-forward-Vorhersagen)
    if backtest is not None and len(backtest):
        bt = backtest[backtest.season == season].copy()
        if len(bt):
            bt["p_out"] = np.select([bt.hg > bt.ag, bt.hg == bt.ag], [bt.p_h, bt.p_d], bt.p_a)
            last_day = bt.date.max()
            recent = bt[pd.to_datetime(bt.date) >= pd.to_datetime(last_day) - pd.Timedelta(days=4)]
            s = recent.sort_values("p_out").iloc[0]
            outcome = ("Sieg " + nm(s.home)) if s.hg > s.ag else ("Sieg " + nm(s.away)) if s.hg < s.ag else "ein Remis"
            facts.append(Fact("🤯", "Überraschung",
                              f"{nm(s.home)} – {nm(s.away)} {int(s.hg)}:{int(s.ag)}: Das Modell gab {outcome} "
                              f"nur {s.p_out * 100:.0f} %.",
                              [(f"{nm(r.home)} – {nm(r.away)}", f"{int(r.hg)}:{int(r.ag)} · {r.p_out * 100:.0f} %")
                               for r in recent.sort_values("p_out").head(TOP_N).itertuples()], "match"))
            bt["tend"] = np.sign(bt.tip_h - bt.tip_a) == np.sign(bt.hg - bt.ag)
            lg = bt.groupby("league").tend.mean().sort_values(ascending=False)
            facts.append(Fact("🔮", "Tendenz je Liga",
                              f"Am besten lag das Modell in der {config.LEAGUES[lg.index[0]].name}: "
                              f"{lg.iloc[0] * 100:.0f} % richtige Tendenzen.",
                              [(config.LEAGUES[k].name, f"{v * 100:.0f} % Tendenz") for k, v in lg.items()], "league"))
    home_rate = cur.groupby("league").apply(lambda g: (g.hg > g.ag).mean(), include_groups=False).sort_values(ascending=False)
    facts.append(Fact("🏟️", "Heimsiege", f"Die meisten Heimsiege gibt es in der "
                      f"{config.LEAGUES[home_rate.index[0]].name}: {home_rate.iloc[0] * 100:.0f} %.",
                      [(config.LEAGUES[k].name, f"{v * 100:.0f} % Heimsiege") for k, v in home_rate.items()], "league"))
    return facts


def streak_len(games: pd.DataFrame, pred) -> int:
    k = 0
    for x in games.res.tolist()[::-1]:
        if not pred(x):
            break
        k += 1
    return k


def load_players(con: sqlite3.Connection, season: int) -> pd.DataFrame:
    return db.read(con, "SELECT player, team_id, league, goals, xg, assists, xa, minutes FROM player_season_stats "
                        "WHERE season = ? AND team_id IS NOT NULL", (season,))
