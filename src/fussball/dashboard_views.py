"""Weitere Dashboard-Ansichten: Ergebnisse, Tabelle mit Saisonprognose, Fakten.

Rendering-Helfer (Wappen, Zahlenformat, Liga-Farben) kommen aus fussball.dashboard.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from fussball import config
from fussball.insights import Fact

RECENT_DAYS = 14   # Ergebnisse der letzten zwei Wochen offen, ältere eingeklappt


def _d():
    from fussball import dashboard as d  # späte Bindung, vermeidet Zirkelimport
    return d


def form_letters(seq: list[str]) -> str:
    """Letzte Ligaspiele als Buchstaben S/U/N, ältestes zuerst."""
    if not seq:
        return ""
    return (f"<span class='frm' aria-label='Form, ältestes zuerst: {' '.join(seq)}'>"
            + "".join(f"<span class='f{x}'>{x}</span>" for x in seq) + "</span>")


def heat(p: float, color: str) -> str:
    """Prozentzelle, Hintergrund umso kräftiger, je wahrscheinlicher."""
    if p < 0.005:
        return "<td class='pc'>–</td>"
    txt = ">99" if p > 0.995 else f"{p * 100:.0f}"
    return (f"<td class='pc' style='background:color-mix(in srgb,var({color}) {8 + 52 * p:.0f}%,transparent)'>"
            f"{txt}</td>")


# --- Ergebnisse ---------------------------------------------------------------------------

def _points(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["exact"] = (p.tip_h == p.hg) & (p.tip_a == p.ag)
    p["tend"] = np.sign(p.tip_h - p.tip_a) == np.sign(p.hg - p.ag)
    p["pts"] = np.where(p.exact, 4, np.where(((p.tip_h - p.tip_a) == (p.hg - p.ag)) & (p.tip_h != p.tip_a), 3,
                                             np.where(p.tend, 2, 0)))
    return p


PT_ICON = {4: "🎯", 3: "✅", 2: "✅", 0: "–"}


def match_stats(con, played: pd.DataFrame) -> dict[tuple, str]:
    """Details je gespieltem Spiel (Schlüssel: Datum, Heim-ID, Gast-ID): Tore, xG, Statistik, Top-Spieler.

    Tore und Spielerwerte aus den Understat-Spielberichten, Schüsse/Ecken/Karten von football-data.
    """
    from fussball import db
    from fussball.ingest.understat_matches import SCHEMA
    d = _d()
    if played.empty:
        return {}
    con.executescript(SCHEMA)
    ids = db.read(con, """SELECT date, team_id AS home_id, opponent_id AS away_id, understat_match_id AS mid,
                                 xg, xga FROM team_xg WHERE is_home = 1 AND season = ?""", (int(played.season.max()),))
    fd = db.read(con, """SELECT date, home_id, away_id, hs, "as", hst, ast, hc, ac, hy, ay, hr, ar FROM matches
                         WHERE season = ?""", (int(played.season.max()),)).set_index(["date", "home_id", "away_id"])
    goals = db.read(con, "SELECT * FROM match_goals").groupby("understat_match_id")
    players = db.read(con, "SELECT * FROM match_players").groupby("understat_match_id")
    gids, pids = set(goals.groups), set(players.groups)
    out = {}
    for r in ids.itertuples():
        key = (r.date, r.home_id, r.away_id)
        rows = [f"<tr><th>📊 xG</th><td>{d._num(r.xg, 2)}</td><td>{d._num(r.xga, 2)}</td></tr>"]
        if r.mid in gids:
            g = goals.get_group(r.mid).sort_values("minute")
            side = lambda h: "".join(
                f"<li>{x.minute}' {escape(x.player)}"
                f"{' <small>(Eigentor)</small>' if x.own_goal else (' <small>(Elfmeter)</small>' if x.situation == 'Penalty' else '')}"
                f"{f'<small>Vorlage {escape(x.assist)}</small>' if isinstance(x.assist, str) and not x.own_goal else ''}</li>"
                for x in g[g.h_a == h].itertuples()) or "<span class='muted'>–</span>"
            rows.insert(0, f"<tr><th>⚽ Tore</th><td><ul class='abl'>{side('h')}</ul></td><td><ul class='abl'>{side('a')}</ul></td></tr>")
        if key in fd.index:
            f = fd.loc[key]
            if pd.notna(f.hs):
                rows += [f"<tr><th>🥅 Schüsse <small>aufs Tor</small></th><td>{int(f.hs)} ({int(f.hst)})</td><td>{int(f['as'])} ({int(f.ast)})</td></tr>",
                         f"<tr><th>🚩 Ecken</th><td>{int(f.hc)}</td><td>{int(f.ac)}</td></tr>",
                         f"<tr><th>🟨 Karten</th><td>{int(f.hy)}{f' 🟥{int(f.hr)}' if f.hr else ''}</td>"
                         f"<td>{int(f.ay)}{f' 🟥{int(f.ar)}' if f.ar else ''}</td></tr>"]
        star = ""
        if r.mid in pids:
            pl = players.get_group(r.mid)
            pl = pl.assign(score=3 * pl.goals + 2 * pl.assists + pl.xg + pl.xa + 0.2 * pl.key_passes - 2 * pl.own_goals)
            b = pl.sort_values("score", ascending=False).iloc[0]
            parts = [f"{b.goals} Tor{'e' if b.goals != 1 else ''}" if b.goals else "",
                     f"{b.assists} Vorlage{'n' if b.assists != 1 else ''}" if b.assists else "",
                     f"{d._num(b.xg + b.xa, 2)} xG+xA"]
            star = f"<p class='star'>⭐ Top-Spieler laut Daten: <b>{escape(b.player)}</b> · {' · '.join(x for x in parts if x)}</p>"
        out[key] = f"<table class='dt'><tbody>{''.join(rows)}</tbody></table>{star}"
    return out


def _result_line(r, ctx, stats: dict | None = None) -> str:
    d = _d()
    home, away = ctx.names.get(r.home_id, r.home_id), ctx.names.get(r.away_id, r.away_id)
    wh, wa = (" win", "") if r.hg > r.ag else (("", " win") if r.hg < r.ag else ("", ""))
    live = "<span class='live' title='live getippt'>●</span>" if r.source == "live" else ""
    line = (f"<span class='tm h{wh}'><span class='nm'>{escape(home)}</span>"
            f"{d.crest(r.home_id, ctx.crests, home)}</span><span class='sc'>{int(r.hg)}:{int(r.ag)}</span>"
            f"<span class='tm a{wa}'>{d.crest(r.away_id, ctx.crests, away)}<span class='nm'>{escape(away)}"
            f"</span></span><span class='tp'>{live}{int(r.tip_h)}:{int(r.tip_a)}</span>"
            f"<span class='pt p{r.pts}' title='{r.pts} Punkte'>{PT_ICON[r.pts]}<small>{r.pts}</small></span>")
    det = (stats or {}).get((str(r.date)[:10], r.home_id, r.away_id))
    if not det:
        return f"<li class='res'>{line}</li>"
    uid = f"rs-{str(r.date)[:10]}-{r.home_id}"
    return (f"<li class='resw'><input type='checkbox' id='{uid}' class='tg'><label for='{uid}' class='res'>{line}</label>"
            f"<div class='det rdet'>{det}</div></li>")


def results_view(played: pd.DataFrame, ctx, live_html: str = "", record_html: str = "",
                 stats: dict | None = None) -> str:
    """played: league, date, home_id, away_id, hg, ag, tip_h, tip_a, source ('live'|'backtest').

    Je Liga die Spieltage der letzten RECENT_DAYS Tage offen, ältere pro Liga eingeklappt.
    """
    d = _d()
    if played.empty:
        return f"{live_html}<p class='muted'>Noch keine Ergebnisse in dieser Saison.</p>"
    p = _points(played)
    p["day"] = pd.to_datetime(p.date).dt.date
    newest = p.day.max()
    summary = lambda g: (f"Tendenz {d._pct(g.tend.mean())} · exakt {d._pct(g.exact.mean(), 1)} · "
                         f"{d._num(g.pts.mean(), 2)} Punkte pro Spiel")
    leagues, sections = [], []
    for lg in config.TOP_LEAGUES:
        lp = p[p.league == lg]
        if lp.empty:
            continue
        leagues.append(lg)
        recent, older = [], []
        for day, g in sorted(lp.groupby("day"), key=lambda x: x[0], reverse=True):
            block = (f"<h3 class='day'>{d._day_title(day)} <span class='muted'>· {int(g.pts.sum())} Punkte</span></h3>"
                     f"<ul class='reslist'>{''.join(_result_line(r, ctx, stats) for r in g.sort_values('home_id').itertuples())}</ul>")
            (recent if (newest - day).days < RECENT_DAYS else older).append(block)
        more = (f"<input type='checkbox' id='old-{lg}' class='tg'><label for='old-{lg}' class='more'>Ältere Spieltage</label>"
                f"<div class='old'>{''.join(older)}</div>") if older else ""
        sections.append(f"<section id='erg-{lg}' class='league' style='--lg:{d.THEMES[lg].primary}'>"
                        f"<h2><span class='logo lg-{lg}'></span>{escape(config.LEAGUES[lg].name)}</h2>"
                        f"<p class='muted small'>{summary(lp)}</p>{''.join(recent)}{more}</section>")
    note = "<p class='muted small'>● = live getippt</p>" if (p.source == "live").any() else ""
    return (f"{d.chips([(f'erg-{lg}', lg) for lg in leagues])}{record_html}<section class='plain'>{live_html}{note}</section>"
            + "".join(sections))


# --- Modell-Bilanz ------------------------------------------------------------------------

BT_COLS = ["league", "season", "date", "home", "away", "hg", "ag", "tip_h", "tip_a"]   # bewusst ohne Quoten-Spalten
TEST_FROM = 2019


def _kpis(g: pd.DataFrame) -> dict:
    return {"n": len(g), "tend": g.tend.mean(), "exact": g.exact.mean(), "pts": g.pts.mean()}


def _kpi_row(label: str, g: pd.DataFrame, note: str = "") -> str:
    d, k = _d(), _kpis(g)
    return (f"<tr><td>{label}{f' <small class=muted>{note}</small>' if note else ''}</td><td>{d._num(k['n'], 0)}</td>"
            f"<td>{d._pct(k['tend'])}</td><td>{d._pct(k['exact'], 1)}</td><td>{d._num(k['pts'], 2)}</td></tr>")


def _kpi_table(rows: str, first: str) -> str:
    return (f"<table class='stats'><thead><tr><th>{first}</th><th>Spiele</th><th>Tendenz</th><th>Exakt</th>"
            f"<th>Punkte/Spiel</th></tr></thead><tbody>{rows}</tbody></table>")


def matchdays(p: pd.DataFrame) -> pd.Series:
    """Spieltag je Spiel einer Liga-Saison: 1 + meiste vorher gespielte Spiele der beiden Teams."""
    seen, out = {}, {}
    for r in p.sort_values("date").itertuples():
        out[r.Index] = 1 + max(seen.get(r.home_id, 0), seen.get(r.away_id, 0))
        seen[r.home_id] = seen.get(r.home_id, 0) + 1
        seen[r.away_id] = seen.get(r.away_id, 0) + 1
    return pd.Series(out)


def model_record(played: pd.DataFrame, comparison: pd.DataFrame | None, season: int) -> str:
    """Trefferbilanz des Modells: gesamt, je Liga, je Saison, je Spieltag (laufende Saison), Modellvergleich."""
    from fussball.backtest import europe as bt_eu
    d = _d()
    f = config.REPORT_DIR / "backtest_predictions.csv"
    if not f.exists():
        return ""
    bt = pd.read_csv(f, usecols=BT_COLS).rename(columns={"home": "home_id", "away": "away_id"}).dropna(subset=["tip_h"])
    bt = _points(bt[bt.season < season].astype({"tip_h": int, "tip_a": int}))
    cur = _points(played) if len(played) else played
    test = bt[bt.season >= TEST_FROM]
    k_test, k_cur = _kpis(test), (_kpis(cur) if len(cur) else None)
    tiles = (f"<div class='kpis'><div><strong>{d._pct(k_test['tend'])}</strong><span>Tendenz richtig</span></div>"
             f"<div><strong>{d._pct(k_test['exact'], 1)}</strong><span>exakt</span></div>"
             f"<div><strong>{d._num(k_test['pts'], 2)}</strong><span>Punkte pro Spiel</span></div></div>"
             f"<p class='muted small'>Seit 2019/20, {d._num(k_test['n'], 0)} Spiele, jeweils nur mit Daten bis zum Vortag"
             + (f" · diese Saison: Tendenz {d._pct(k_cur['tend'])}, {d._num(k_cur['pts'], 2)} Punkte pro Spiel" if k_cur else "")
             + "</p>")
    by_league = "".join(_kpi_row(f"<span class='logo lg-{lg} mini'></span>{escape(config.LEAGUES[lg].name)}", g)
                        for lg, g in test.groupby("league"))
    seasons = pd.concat([bt, cur.assign(season=season)]) if len(cur) else bt
    by_season = "".join(_kpi_row(config.season_label(int(s)), g,
                                 "läuft" if s == season else ("Validierung" if s < TEST_FROM else ""))
                        for s, g in sorted(seasons.groupby("season"), key=lambda x: -x[0]))
    md_html = ""
    if len(cur):
        cur = cur.assign(md=pd.concat([matchdays(g) for _, g in cur.groupby("league")]))
        grid = cur.pivot_table(index="md", columns="league", values="pts", aggfunc="mean")
        tot = cur.groupby("md").pts.mean()
        lgs = [lg for lg in config.TOP_LEAGUES if lg in grid.columns]
        head = "".join(f"<th><span class='logo lg-{lg} mini'></span></th>" for lg in lgs)
        rows = "".join(f"<tr><td>{int(md)}.</td>" + "".join(
            f"<td class='pc' style='background:color-mix(in srgb,var(--good) {v / 4 * 70:.0f}%,transparent)'>{d._num(v, 1)}</td>"
            if pd.notna(v) else "<td class='pc'>–</td>" for v in (grid.loc[md, lg] for lg in lgs))
            + f"<td class='pc'><b>{d._num(tot.loc[md], 2)}</b></td></tr>" for md in grid.index)
        md_html = (f"<table class='ltable mdgrid'><thead><tr><th>Sp.</th>{head}<th>Alle</th></tr></thead><tbody>{rows}</tbody></table>"
                   "<p class='muted small'>Ø Punkte pro Spiel je Spieltag (4 exakt, 3 Tordifferenz, 2 Tendenz)</p>")
    cmp_html = ""
    if comparison is not None and "pts_tip" in comparison.columns:
        label = {"aktuell": "Tore + xG + LightGBM (aktuell)"}
        cmp_html = _kpi_table("".join(
            f"<tr><td>{escape(label.get(role, role))}</td><td>{d._num(r['n'], 0)}</td><td>{d._num(r['tip_tend_%'])} %</td>"
            f"<td>{d._num(r['tip_exact_%'])} %</td><td>{d._num(r['pts_tip'], 2)}</td></tr>" for role, r in comparison.iterrows()), "Modell")
    eu_html = ""
    if bt_eu.PRED_FILE.exists():
        e = pd.read_csv(bt_eu.PRED_FILE)
        e = e[e.season >= TEST_FROM].rename(columns={"tendency": "tend"}).assign(pts=lambda x: x.points)
        from fussball.models.europe import COMPETITIONS
        eu_html = _kpi_table("".join(_kpi_row(f"<span class='logo lg-{c} mini'></span>{escape(COMPETITIONS[c])}", g)
                                     for c, g in e.groupby("competition")) + _kpi_row("<b>Alle</b>", e), "Wettbewerb")

    def fold(i: str, title: str, body: str) -> str:
        return (f"<input type='checkbox' id='rec-{i}' class='tg'><label for='rec-{i}' class='more'>{title}</label>"
                f"<div class='old'>{body}</div>") if body else ""

    return (f"<section class='plain record'><h2>📊 Modell-Bilanz</h2>{tiles}"
            + fold("lg", "Nach Liga", _kpi_table(by_league, "Liga") + "<p class='muted small'>seit 2019/20</p>")
            + fold("season", "Nach Saison", _kpi_table(by_season, "Saison"))
            + fold("md", "Nach Spieltag (diese Saison)", md_html)
            + fold("cmp", "Modelle im Vergleich", cmp_html + "<p class='muted small'>seit 2019/20</p>" if cmp_html else "")
            + fold("eu", "Europapokal", eu_html + "<p class='muted small'>seit 2019/20</p>" if eu_html else "")
            + "</section>")


# --- Tabelle mit Prognose ---------------------------------------------------------------

def table_view(proj: pd.DataFrame, ctx) -> str:
    d = _d()
    if proj.empty:
        return "<p class='muted'>Noch keine Saisonprognose – erst <code>fussball predict</code> ausführen.</p>"
    from fussball.season import RULES
    leagues, out = [], []
    for lg in config.TOP_LEAGUES:
        t = proj[proj.league == lg].sort_values("pos")
        if t.empty:
            continue
        leagues.append(lg)
        rules, n = RULES[lg], len(t)
        rows = []
        for r in t.itertuples():
            zone = ("z-cl" if r.pos <= rules.cl else "z-rel" if r.pos > n - rules.releg else
                    "z-po" if r.pos > n - rules.releg - rules.playoff else "")
            team = ctx.names.get(r.team_id, r.team_id)
            rows.append(f"<tr class='{zone}'><td class='pos'>{r.pos}</td>"
                        f"<td class='tn'>{d.crest(r.team_id, ctx.crests, team)}<span>{escape(team)}</span></td>"
                        f"<td>{r.played}</td><td>{r.gf - r.ga:+d}</td><td class='pts'>{r.pts}</td>"
                        f"<td class='prog'>{r.exp_pts:.0f}</td>{heat(r.p_title, '--title')}{heat(r.p_cl, '--title')}"
                        f"{heat(r.p_releg + r.p_playoff, '--down')}</tr>")
        po = " oder Relegation" if rules.playoff else ""
        out.append(f"""<section id="tab-{lg}" class="league" style="--lg:{d.THEMES[lg].primary}">
<h2><span class="logo lg-{lg}"></span>{escape(config.LEAGUES[lg].name)}</h2>
<table class="ltable"><thead><tr><th class="pos">#</th><th class="tn"></th><th>Sp</th><th>TD</th>
<th>Pkt</th><th class="prog">Erw.</th><th>Meister</th><th>CL</th><th>Abstieg</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="muted small">Erw. = erwartete Punkte am Saisonende</p></section>""")
    return d.chips([(f"tab-{lg}", lg) for lg in leagues]) + "".join(out)


# --- Fakten -----------------------------------------------------------------------------

def facts_view(facts: list[Fact], ctx) -> str:
    d = _d()
    if not facts:
        return "<p class='muted'>Noch keine Fakten.</p>"
    items = []
    for f in facts:
        rank = []
        for i, (key, val) in enumerate(f.ranking, 1):
            if f.kind == "team":
                label = f"{d.crest(key, ctx.crests, ctx.names.get(key, key))}{escape(ctx.names.get(key, key))}"
            else:
                label = escape(key)
            rank.append(f"<li><span class='fr'>{i}.</span><span class='fl'>{label}</span>"
                        f"<span class='fv'>{escape(val)}</span></li>")
        items.append(f"<div class='fact'><span class='ficon'>{f.icon}</span><div><h3>{escape(f.title)}</h3>"
                     f"<ol>{''.join(rank)}</ol></div></div>")
    return (f"<section class='plain'><h2>💡 Saison bisher</h2><div>{''.join(items)}</div></section>")


CSS = """
/* Ergebnisse */
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0 6px}
.kpis div{border:1px solid var(--line);border-radius:12px;padding:10px 12px;background:var(--surface)}
.kpis strong{display:block;font-size:22px;font-weight:800;letter-spacing:-.02em}.kpis span{font-size:11px;color:var(--muted)}
.reslist{list-style:none;padding:0;margin:0}
.resw>.res{cursor:pointer}.resw>.tg:checked+.res+.rdet{display:block}
.rdet .dt{margin-top:0}.star{margin:10px 0 0}
.res{display:grid;grid-template-columns:1fr 40px 1fr 32px 34px;align-items:center;gap:6px;padding:7px 0;
border-bottom:1px solid var(--line);font-size:13px}
.res .tm{font-size:13px}
.sc{text-align:center;font-weight:700;font-variant-numeric:tabular-nums}
.tp{text-align:right;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.pt{text-align:right;font-size:12px;white-space:nowrap}.pt small{font-size:10px;color:var(--muted);margin-left:2px}
.pt.p4 small{color:var(--good);font-weight:700}.pt.p0{opacity:.55}
.live{color:var(--title);margin-right:3px;font-size:9px;vertical-align:1px}
.old{display:none}.tg:checked+.more+.old{display:block}
/* Tabelle */
.ltable{width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px;font-variant-numeric:tabular-nums;margin-top:6px}
.ltable th{font-size:10px;color:var(--muted);font-weight:500;padding:6px 2px;text-align:center}
.ltable td{padding:5px 2px;text-align:center;border-top:1px solid var(--line)}
.ltable .pos{width:24px;color:var(--muted)}
.ltable td.tn{text-align:left;display:flex;align-items:center;gap:6px;min-width:0}
.ltable td.tn span{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.ltable .tn .crest{width:16px;height:16px}
.ltable th:nth-child(3),.ltable th:nth-child(4){width:26px}.ltable th:nth-child(5){width:28px}
.ltable th.prog{width:30px}.ltable th:nth-child(n+7){width:44px}
.ltable td.pts{font-weight:700}.ltable td.prog{color:var(--muted)}
.ltable td.pc{font-size:11px}
.mdgrid{table-layout:fixed}.mdgrid th,.mdgrid td{text-align:center}.mdgrid th:first-child{width:36px}
.mdgrid td:first-child{color:var(--muted)}.mdgrid th:last-child{width:52px}
.record .more{display:block;margin-top:12px}
.ltable tr.z-cl td.pos{box-shadow:inset 2px 0 0 var(--title)}
.ltable tr.z-rel td.pos,.ltable tr.z-po td.pos{box-shadow:inset 2px 0 0 var(--down)}
/* Fakten */
.fact{display:flex;gap:12px;border-top:1px solid var(--line);padding:4px 0 10px}.fact>div{flex:1;min-width:0}
.ficon{font-size:26px;line-height:1;margin-top:10px}
.fact h3{margin:10px 0 2px}.fact p{margin:0 0 6px;font-size:13px}
.fact ol{list-style:none;margin:0;padding:0;font-size:12px;color:var(--muted)}
.fact li{display:flex;align-items:baseline;gap:6px;padding:2px 0}
.fr{width:14px;flex:none}.fl{flex:1;min-width:0;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fl .crest{width:14px;height:14px;vertical-align:-2px;margin-right:5px;display:inline-block}
.fv{white-space:nowrap;font-variant-numeric:tabular-nums}
"""
