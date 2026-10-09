"""Dashboard-Reiter „Europa“: nächste Runde von Champions League, Europa League und
Conference League, gruppiert nach Wettbewerb und Tag, im Farbschema des Wettbewerbs.

Wappen kommen direkt über die Transfermarkt-ID (auch Vereine außerhalb der Top-5-Ligen).
Form über alle Wettbewerbe (Liga, weitere Ligen, Europapokal); Ausfälle gibt es nur für
Top-5-Vereine.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from html import escape

import numpy as np
import pandas as pd

from fussball import config, db, predict_europe, tournament
from fussball.backtest import europe as bt
from fussball.ingest import transfermarkt
from fussball.models.europe import COMPETITIONS, EuropeModel, europe_frame, fit_data


@dataclass(frozen=True)
class CompTheme:
    c1: str       # Hauptfarbe (Band, Balken)
    c2: str       # Zweitfarbe (Band)
    accent: str   # Akzent für Text/Hervorhebung, in hell und dunkel lesbar


THEMES = {
    "CL": CompTheme("#1b2a6b", "#7b2fbe", "#7b4fd6"),
    "EL": CompTheme("#f37021", "#f37021", "#f37021"),
    "UCOL": CompTheme("#00b85c", "#00b85c", "#00a854"),
}


SHORT = {"CL": "CL", "EL": "EL", "UCOL": "ECL"}


def _d():
    from fussball import dashboard as d  # späte Bindung, vermeidet Zirkelimport
    return d


def crest(tm_id: int, name: str) -> str:
    if (_d().IMG_DIR / transfermarkt.crest_name(tm_id)).exists():
        return f'<span class="crest c{tm_id}" role="img" aria-label="Wappen {escape(name)}"></span>'
    return f'<span class="crest none">{escape(name[:1])}</span>'


TOP: set[str] = set()   # Vereine der Top-5-Ligen (kuratierte Namen, Ausfall-Daten); in europe_view gesetzt


def _name(ctx, team_id: str, tm_name: str) -> str:
    """Deutsche Anzeigenamen der Top-5-Vereine, sonst der (deutsche) Transfermarkt-Name."""
    return ctx.names.get(team_id, tm_name) if team_id in TOP else tm_name


KO_LABEL = {"Zwischenrunde": "Play-offs"}   # Format ab 2024/25 (vorher: Zwischenrunde/Sechzehntelfinale)


def round_label(rnd, leg, matchday, season: int) -> str:
    """'Ligaphase, 2. Spieltag' / 'Achtelfinale, Hinspiel' / 'Finale'."""
    if not isinstance(rnd, str):
        return ""
    if rnd in ("Ligaphase", "Gruppenphase"):
        return f"{rnd}, {int(matchday)}. Spieltag" if pd.notna(matchday) else rnd
    name = KO_LABEL.get(rnd, rnd) if season >= 2024 else rnd
    return f"{name}, {leg}" if isinstance(leg, str) else name


def form_all(games: pd.DataFrame, team: str, before: pd.Timestamp, n: int = 5) -> list[str]:
    """Letzte n Spiele über alle Wettbewerbe als S/U/N, ältestes zuerst."""
    g = games[((games.home == team) | (games.away == team)) & (games.date < before)].tail(n)
    gf = np.where(g.home == team, g.hg, g.ag)
    ga = np.where(g.home == team, g.ag, g.hg)
    return ["S" if f > a else ("U" if f == a else "N") for f, a in zip(gf, ga)]


def match_row(r, ctx, games: pd.DataFrame) -> str:
    d, views = _d(), __import__("fussball.dashboard_views", fromlist=["x"])
    home, away = _name(ctx, r.home_id, r.home_name), _name(ctx, r.away_id, r.away_name)
    m = np.array(json.loads(r.matrix))
    uid = f"eu{r.tm_match_id}"
    th, ta, p_tend, p_exact = d._tip(r)
    pick, text = d.verdict(th, ta, home, away)
    hcls, acls = {1: (" win", ""), 0: ("", ""), -1: ("", " win")}[pick]
    day = pd.Timestamp(r.date)
    fh, fa = (views.form_letters(form_all(games, t, day)) for t in (r.home_id, r.away_id))
    trs = [f"<tr><th>⚽ Erw. Tore</th><td>{d._num(r.lam_h)}</td><td>{d._num(r.lam_a)}</td></tr>",
           f"<tr><th>📈 Form <small>alle Wettbewerbe</small></th><td>{fh or '–'}</td><td>{fa or '–'}</td></tr>"]
    if {r.home_id, r.away_id} & TOP:
        cell = lambda t: d.absence_cell(ctx, t) if t in TOP else "<span class='muted'>keine Daten</span>"
        trs.append(f"<tr><th>🚑 Ausfälle</th><td>{cell(r.home_id)}</td><td>{cell(r.away_id)}</td></tr>")
    notes = [f"🎯 Tipp {th}:{ta} trifft exakt mit {d._pct(p_exact, 1)}", d.tip_reason(r, pick, home, away),
             d.blowout(m, home, away)]
    notes += [f"🔍 Wenig Daten zu {escape(n)} – Schätzung nahe am Durchschnitt"
              for n, k in ((home, r.home_known), (away, r.away_known)) if not k]
    det = (d.outcome_strip(r.p_h, r.p_d, r.p_a, pick)
           + f"<table class='dt'><thead><tr><th></th><th>{escape(home)}</th><th>{escape(away)}</th></tr></thead>"
             f"<tbody>{''.join(trs)}</tbody></table>"
           + "<ul class='notes'>" + "".join(f"<li>{n}</li>" for n in notes if n) + "</ul>"
           + f'<input type="checkbox" id="x{uid}" class="tg"><label for="x{uid}" class="more">📊 Ergebnismatrix</label>'
             f'<div class="mx">{d.matrix_table(m, home, away, (th, ta))}</div>')
    return f"""<div class="match"><input type="checkbox" id="m{uid}" class="tg">
<label for="m{uid}" class="row" aria-label="{escape(home)} gegen {escape(away)}, Tipp {th}:{ta}, {escape(text)}">
<span class="ko">{r.kickoff if isinstance(r.kickoff, str) else ""}</span>
<span class="tm h{hcls}"><span class="nm">{escape(home)}</span>{crest(r.tm_home_id, home)}</span>
<span class="tip{' zz' if th == ta == 0 else ''}">{th}:{ta}</span>
<span class="tm a{acls}">{crest(r.tm_away_id, away)}<span class="nm">{escape(away)}</span></span>
{d.prob_cell(p_tend)}{d.row_tags(r.home_id, r.away_id)}</label>
<div class="det">{det}</div></div>"""


def _result_line(r, ctx) -> str:
    from fussball.dashboard_views import PT_ICON
    home, away = _name(ctx, r.home, r.home_name), _name(ctx, r.away, r.away_name)
    wh, wa = (" win", "") if r.hg > r.ag else (("", " win") if r.hg < r.ag else ("", ""))
    live = "<span class='live' title='live getippt'>●</span>" if r.source == "live" else ""
    return (f"<li class='res'><span class='tm h{wh}'><span class='nm'>{escape(home)}</span>"
            f"{crest(r.tm_home_id, home)}</span><span class='sc'>{int(r.hg)}:{int(r.ag)}</span>"
            f"<span class='tm a{wa}'>{crest(r.tm_away_id, away)}<span class='nm'>{escape(away)}</span></span>"
            f"<span class='tp'>{live}<small class='prl'>Tipp </small>{int(r.tip_h)}:{int(r.tip_a)}</span>"
            f"<span class='pt p{r.points}' title='{r.points} Punkte'>{PT_ICON[r.points]}<small>{r.points}</small></span></li>")


def season_results(con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Gespielte Europapokal-Spiele der Saison mit Tipp: live, sonst Walk-forward aus dem Backtest."""
    cols = ["tm_match_id", "competition", "date", "home", "away", "hg", "ag", "tip_h", "tip_a", "points", "source"]
    parts = []
    if bt.PRED_FILE.exists():
        b = pd.read_csv(bt.PRED_FILE)
        parts.append(b[b.season == season].assign(source="backtest")[cols])
    sc = predict_europe.scorecard(con)
    if len(sc):
        parts.append(sc[sc.season == season].rename(columns={"home_id": "home", "away_id": "away"})
                     .assign(source="live")[cols])
    if not parts:
        return pd.DataFrame(columns=cols)
    p = pd.concat(parts).sort_values("source", ascending=False).drop_duplicates("tm_match_id")
    e = db.read(con, "SELECT tm_match_id, tm_home_id, tm_away_id, home_name, away_name FROM europe_matches")
    return p.merge(e, on="tm_match_id")


def europe_view(con: sqlite3.Connection, ctx, today: date) -> tuple[str, set[int]]:
    """HTML des Reiters und die Transfermarkt-IDs der gezeigten Wappen."""
    d = _d()
    TOP.clear()
    TOP.update(r[0] for r in con.execute(
        f"SELECT DISTINCT home_id FROM matches WHERE season >= ? AND league IN ({','.join('?' * len(config.TOP_LEAGUES))})",
        (config.current_season(today) - 1, *config.TOP_LEAGUES)))
    pred = predict_europe.upcoming(con, today)
    played = season_results(con, config.current_season(today))
    shown = set(pred.tm_home_id) | set(pred.tm_away_id) | set(played.tm_home_id) | set(played.tm_away_id)
    if pred.empty and played.empty:
        return "<p class='muted'>Keine Europapokal-Spiele angesetzt.</p>", shown
    games = fit_data(con)
    forecasts = tournament_forecasts(con, games, today)
    comps = [c for c in COMPETITIONS if (pred.competition == c).any() or (played.competition == c).any()]
    sections = []
    for c in comps:
        th = THEMES[c]
        g = pred[pred.competition == c].assign(day=lambda x: pd.to_datetime(x.date).dt.date).sort_values(["date", "home_name"])
        body = "".join(f"<h3 class='day'>{d._day_title(day)} · {round_label(dg['round'].iloc[0], dg.leg.iloc[0], dg.matchday.iloc[0], int(dg.season.iloc[0]))}</h3>"
                       + "<div class='group'>" + "".join(match_row(r, ctx, games) for r in dg.sort_values(["kickoff", "home_name"]).itertuples()) + "</div>"
                       for day, dg in g.groupby("day", sort=True))
        pc = played[played.competition == c]
        res = ""
        if len(pc):
            pc = pc.assign(day=pd.to_datetime(pc.date).dt.date)
            days = "".join(f"<h3 class='day'>{d._day_title(day)} <span class='muted'>· {int(dg.points.sum())} Punkte</span></h3>"
                           f"<ul class='reslist group'>{''.join(_result_line(r, ctx) for r in dg.sort_values('home_name').itertuples())}</ul>"
                           for day, dg in sorted(pc.groupby("day"), key=lambda x: x[0], reverse=True))
            res = (f"<input type='checkbox' id='eures-{c}' class='tg'><label for='eures-{c}' class='more'>"
                   f"✅ Ergebnisse dieser Saison ({len(pc)} Spiele, Ø {d._num(pc.points.mean(), 2)} Punkte)</label>"
                   f"<div class='old'>{days}</div>")
        sections.append(f"<section id='eu-{c}' class='league comp' style='--c1:{th.c1};--c2:{th.c2};--accent:{th.accent}'>"
                        f"<h2><span class='logo lg-{c}'></span>{escape(COMPETITIONS[c])}</h2>{d.column_head()}"
                        f"{body or '<p class=muted>Kein Spiel in den nächsten Tagen.</p>'}"
                        f"{forecast_html(c, forecasts.get(c), ctx)}{res}</section>")
    chips = "<nav class='chips'>" + "".join(
        f'<a href="#eu-{c}" title="{escape(COMPETITIONS[c])}"><span class="logo lg-{c}"></span>{SHORT[c]}</a>'
        for c in comps) + "</nav>"
    return chips + "".join(sections), shown


def tournament_forecasts(con: sqlite3.Connection, games: pd.DataFrame, today: date) -> dict:
    """Turnierprognose je Wettbewerb, solange dessen Ligaphase (Format ab 2024/25) läuft."""
    e = europe_frame(con)
    season = config.current_season(today)
    e = e[(e.season == season) & (e["round"] == "Ligaphase")]
    if e.empty or season < 2024:
        return {}
    model = EuropeModel.from_params(bt.load_params()).fit(games, pd.Timestamp(today))
    out = {}
    for c, lg in e.groupby("competition"):
        if lg.hg.isna().any():            # nach der Ligaphase gelten die echten Paarungen
            out[c] = (tournament.simulate(model, lg), lg)
    return out


ROUND_NAMES = ["Play-offs", "Achtelfinale", "Viertelfinale", "Halbfinale", "Finale"]


def forecast_html(c: str, fc, ctx) -> str:
    """Baum (wahrscheinlichster Verlauf), Titelchancen und Ligaphasen-Prognose, eingeklappt."""
    if fc is None:
        return ""
    d, views = _d(), __import__("fussball.dashboard_views", fromlist=["x"])
    f, lg = fc
    tm = dict(zip(lg.home, lg.tm_home_id)) | dict(zip(lg.away, lg.tm_away_id))
    nm = {t: _name(ctx, t, n) for t, n in (dict(zip(lg.home, lg.home_name)) | dict(zip(lg.away, lg.away_name))).items()}
    t = f.table.set_index("team")

    def slot(team: str, p: float, win: bool) -> str:
        return (f"<div class='bt{' w' if win else ''}'>{crest(tm[team], nm[team])}<span class='nm'>{escape(nm[team])}</span>"
                f"<b>{d._pct(p)}</b></div>")

    cols = []
    for i, ties in enumerate(f.bracket):
        cards = "".join(f"<div class='tie'>{slot(x['a'], x['p_a'], x['winner'] == x['a'])}"
                        f"{slot(x['b'], 1 - x['p_a'], x['winner'] == x['b'])}</div>" for x in ties)
        cols.append(f"<div class='bcol'><h4>{ROUND_NAMES[i]}</h4><div class='bties'>{cards}</div></div>")
    champ = f.bracket[-1][0]["winner"]
    cols.append(f"<div class='bcol champ'><h4>Sieger</h4><div class='bties'><label for='cup-{c}' class='cup'>"
                f"<span class='trophy'>🏆</span>{crest(tm[champ], nm[champ])}<b>{escape(nm[champ])}</b>"
                f"<small>{d._pct(t.loc[champ, 'p_win'])} Titelchance</small></label></div></div>")
    top = f.table.sort_values("p_win", ascending=False).head(10)
    chance_rows = "".join(
        f"<tr><td class='tn'>{crest(tm[r.team], nm[r.team])}<span>{escape(nm[r.team])}</span></td>"
        + "".join(views.heat(getattr(r, k), "--accent") for k in ("p_top8", "p_r16", "p_qf", "p_sf", "p_final", "p_win"))
        + "</tr>" for r in top.itertuples())
    phase_rows = "".join(
        f"<tr><td class='pos'>{r.pos}</td><td class='tn'>{crest(tm[r.team], nm[r.team])}<span>{escape(nm[r.team])}</span></td>"
        f"<td>{r.played}</td><td class='pts'>{r.pts}</td><td class='prog'>{r.exp_pos:.0f}</td>"
        + views.heat(r.p_top8, "--accent") + views.heat(r.p_playoff, "--accent") + views.heat(r.p_out, "--down")
        + "</tr>" for r in f.table.itertuples())
    return f"""<input type='checkbox' id='tree-{c}' class='tg'><label for='tree-{c}' class='more tree-btn'>🏆 Turnierbaum &amp; Titelchancen</label>
<div class='old tree'><input type='checkbox' id='cup-{c}' class='tg cupegg'>
<div class='bracket'>{''.join(cols)}</div>
<h3>Titelchancen</h3><table class='ltable chances'><thead><tr><th class='tn'></th><th>Top 8</th><th>AF</th><th>VF</th>
<th>HF</th><th>Finale</th><th>Sieg</th></tr></thead><tbody>{chance_rows}</tbody></table>
<input type='checkbox' id='lp-{c}' class='tg'><label for='lp-{c}' class='more'>📋 Ligaphase-Prognose (alle Teams)</label>
<div class='old'><table class='ltable phase'><thead><tr><th class='pos'>#</th><th class='tn'></th><th>Sp</th><th>Pkt</th>
<th class='prog'>Erw.</th><th>Top 8</th><th>Play-off</th><th>Aus</th></tr></thead><tbody>{phase_rows}</tbody></table>
<p class='muted small'>Erw. = erwarteter Platz am Ende der Ligaphase</p></div></div>"""


CSS = """
/* Turnierbaum */
.tree-btn{font-size:13px}
.bracket{display:flex;gap:12px;overflow-x:auto;padding:8px 2px 16px;scroll-snap-type:x proximity}
.bcol{flex:none;width:152px;display:flex;flex-direction:column;scroll-snap-align:start}
.bcol h4{margin:0 0 6px;font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.bties{flex:1;display:flex;flex-direction:column;justify-content:space-around;gap:6px}
.tie{position:relative;border-radius:10px;background:var(--surface);box-shadow:var(--shadow);border:1px solid var(--line)}
.tie::after{content:"";position:absolute;right:-13px;top:50%;width:12px;border-top:1.5px solid color-mix(in srgb,var(--accent) 45%,var(--line))}
.bt{display:flex;align-items:center;gap:5px;padding:5px 7px;font-size:11px;color:var(--muted);min-width:0}
.bt+.bt{border-top:1px solid var(--line)}
.bt .crest{width:14px;height:14px}.bt .nm{flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.bt b{font-weight:600;font-variant-numeric:tabular-nums}
.bt.w{color:var(--text);font-weight:600}.bt.w b{color:var(--accent)}
.cup{display:flex;flex-direction:column;align-items:center;gap:6px;text-align:center;padding:16px 10px;border-radius:var(--radius);
border:1.5px solid var(--accent);background:var(--surface);box-shadow:var(--shadow);cursor:pointer;position:relative;overflow:hidden}
.cup .crest{width:28px;height:28px}.cup small{color:var(--muted)}
.trophy{font-size:38px;display:inline-block;filter:drop-shadow(0 4px 10px color-mix(in srgb,var(--accent) 50%,transparent))}
.cup::before{content:"";position:absolute;inset:0;background:linear-gradient(110deg,transparent 35%,rgba(255,255,255,.35) 50%,transparent 65%);
background-size:250% 100%;animation:shimmer 3.5s linear infinite}
.tree .bcol{animation:drawin .5s both}.tree .bcol:nth-child(2){animation-delay:.12s}.tree .bcol:nth-child(3){animation-delay:.24s}
.tree .bcol:nth-child(4){animation-delay:.36s}.tree .bcol:nth-child(5){animation-delay:.48s}.tree .bcol:nth-child(6){animation-delay:.6s}
.cupegg:checked~.bracket .trophy{animation:lift 1.2s ease-in-out 2}
.chances td.tn,.phase td.tn{text-align:left;display:flex;align-items:center;gap:6px;min-width:0}
.chances td.tn span,.phase td.tn span{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.chances .crest,.phase .crest{width:16px;height:16px}
.chances th:first-child{width:auto}.chances th:nth-child(n+2){width:42px}
/* Europa: Farbschema je Wettbewerb */
.comp h2::after{width:56px;background:linear-gradient(90deg,var(--c1),var(--c2))}
.comp .tip{color:var(--accent)}



@media (prefers-color-scheme:dark){.comp h2::after{background:linear-gradient(90deg,color-mix(in srgb,var(--c1) 60%,#8aa0ff),var(--c2))}}
"""
