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


def _result_line(r, ctx) -> str:
    d = _d()
    home, away = ctx.names.get(r.home_id, r.home_id), ctx.names.get(r.away_id, r.away_id)
    wh, wa = (" win", "") if r.hg > r.ag else (("", " win") if r.hg < r.ag else ("", ""))
    live = "<span class='live' title='live getippt'>●</span>" if r.source == "live" else ""
    return (f"<li class='res'><span class='tm h{wh}'><span class='nm'>{escape(home)}</span>"
            f"{d.crest(r.home_id, ctx.crests, home)}</span><span class='sc'>{int(r.hg)}:{int(r.ag)}</span>"
            f"<span class='tm a{wa}'>{d.crest(r.away_id, ctx.crests, away)}<span class='nm'>{escape(away)}"
            f"</span></span><span class='tp'>{live}{int(r.tip_h)}:{int(r.tip_a)}</span>"
            f"<span class='pt p{r.pts}' title='{r.pts} Punkte'>{PT_ICON[r.pts]}<small>{r.pts}</small></span></li>")


def results_view(played: pd.DataFrame, ctx, live_html: str = "") -> str:
    """played: league, date, home_id, away_id, hg, ag, tip_h, tip_a, source ('live'|'backtest').

    Je Liga die Spieltage der letzten RECENT_DAYS Tage offen, ältere pro Liga eingeklappt.
    """
    d = _d()
    if played.empty:
        return f"{live_html}<p class='muted'>Noch keine Ergebnisse in dieser Saison.</p>"
    p = _points(played)
    p["day"] = pd.to_datetime(p.date).dt.date
    newest = p.day.max()
    season = config.season_label(int(p.season.max()))
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
                     f"<ul class='reslist'>{''.join(_result_line(r, ctx) for r in g.sort_values('home_id').itertuples())}</ul>")
            (recent if (newest - day).days < RECENT_DAYS else older).append(block)
        more = (f"<input type='checkbox' id='old-{lg}' class='tg'><label for='old-{lg}' class='more'>Ältere Spieltage</label>"
                f"<div class='old'>{''.join(older)}</div>") if older else ""
        sections.append(f"<section id='erg-{lg}' class='league' style='--lg:{d.THEMES[lg].primary}'>"
                        f"<h2><span class='logo lg-{lg}'></span>{escape(config.LEAGUES[lg].name)}</h2>"
                        f"<p class='muted small'>{summary(lp)}</p>{''.join(recent)}{more}</section>")
    lead = f"<p class='lead'>📊 Saison {season}: {summary(p)}</p>"
    note = "<p class='muted small'>● = live getippt</p>" if (p.source == "live").any() else ""
    return (f"{d.chips([(f'erg-{lg}', lg) for lg in leagues])}<section class='plain'>{live_html}{lead}{note}</section>"
            + "".join(sections))


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
.reslist{list-style:none;padding:0;margin:0}
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
