"""Statisches Dashboard (eine HTML-Datei, ohne JavaScript, mobil-first, bewusst schlicht).

Reiter Spiele / Ergebnisse / Tabelle / Fakten. Spiele: je Liga eine Zeile pro Spiel (Anstoß,
Teams, Tipp, Wahrscheinlichkeit der Tendenz); 1X2, erwartete Tore, Form, Ausfälle, Fakten und
Ergebnismatrix erst nach Antippen.

Aufklappen per CSS (Checkbox + Label) statt <details>: startet in jedem Viewer eingeklappt,
auch in der Vorschau der iOS-Dateien-App. Wappen und Logos werden einmal als CSS-Klassen
eingebettet (base64), die Datei lädt also nichts nach.
"""

from __future__ import annotations

import base64
import re
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fussball import config, db, insights, predict, season, teams
from fussball import dashboard_views as views
from fussball.features import availability
from fussball.ingest import europe, transfermarkt

OUT_FILE = config.DATA_DIR / "dashboard" / "index.html"
IMG_DIR = config.RAW_DIR / transfermarkt.SOURCE
TZ = ZoneInfo("Europe/Berlin")
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
MATRIX_SHOW = 6  # 0..5 Tore je Team


@dataclass(frozen=True)
class Theme:
    primary: str   # Hauptfarbe der Liga (Kopfband, Akzente)
    accent: str    # zweite Markenfarbe (Linie im Kopfband)
    ink: str = "#ffffff"  # Textfarbe auf primary


# An die offiziellen Logos angelehnte Markenfarben
THEMES = {
    "E0": Theme("#3D195B", "#00FF85"),
    "SP1": Theme("#FF4B44", "#1A1A1A"),
    "I1": Theme("#1B4FC4", "#2BC8F2"),
    "F1": Theme("#141414", "#FFFFFF"),
    "D1": Theme("#D20515", "#FFFFFF"),
}


@dataclass
class Ctx:
    names: dict
    crests: dict
    ab: pd.DataFrame                 # Ausfälle je Spieler (letzter Snapshot)
    impact: dict                     # (datum, team_id) -> fehlende Anteile
    key_players: set                 # (team_id, Spielername) mit Stammplatz (>= 50 % der Minuten)
    europe: dict                     # team_id -> sortierte Europapokal-Termine
    res: pd.DataFrame | None = None  # gespielte Spiele (Form, Duelle, Fakten)
    players: pd.DataFrame | None = None
    totals: dict | None = None       # min/max erwartete Torsumme des Spieltags


# --- Daten ------------------------------------------------------------------------------

def current_model(con: sqlite3.Connection) -> str | None:
    """Modell des jüngsten Predict-Laufs."""
    row = con.execute("""SELECT model FROM predictions
        WHERE created_at = (SELECT MAX(created_at) FROM predictions)
        ORDER BY model LIKE '%+ausfaelle' DESC LIMIT 1""").fetchone()
    return row[0] if row else None


def upcoming_predictions(con: sqlite3.Connection, today: date, model: str | None) -> pd.DataFrame:
    """Jüngste Vorhersage des aktuellen Modells je noch nicht gespieltem Spiel.

    lam_h0/lam_a0 = erwartete Tore desselben Laufs ohne Ausfall-Anpassung (falls vorhanden).
    """
    base = model.removesuffix(" +ausfaelle") if model else None
    p = db.read(con, """SELECT p.*, b.lam_h AS lam_h0, b.lam_a AS lam_a0 FROM predictions p
        LEFT JOIN predictions b ON b.model = :b AND b.understat_match_id = p.understat_match_id
             AND b.cutoff_date = p.cutoff_date AND :b != :m
        WHERE p.model = :m AND p.date >= :d
          AND p.cutoff_date = (SELECT MAX(cutoff_date) FROM predictions WHERE model = :m)""",
                {"m": model, "b": base, "d": today.isoformat()})
    return p.sort_values(["league", "kickoff_utc", "home_id"])


def absence_impact(con: sqlite3.Connection, pred: pd.DataFrame) -> tuple[dict, set]:
    impact, key = {}, set()
    for (d, season), _ in pred.groupby(["date", "season"]):
        per_team, players = availability.team_absences(con, d, int(season))
        for r in per_team.itertuples():
            impact[(d, r.team_id)] = {"lineup": r.missing_lineup, "att": r.missing_att}
        if len(players):
            key |= set(map(tuple, players[players.lineup_share >= 0.5][["team_id", "player"]].to_numpy()))
    return impact, key


def europe_note(ctx: Ctx, team_id: str, team: str, match_date: str) -> str:
    """'X spielte vor 3 Tagen im Europapokal' (Belastung wirkt nach dem Spiel; nur wenn nah dran)."""
    dates = ctx.europe.get(team_id)
    if dates is None:
        return ""
    d = np.datetime64(match_date)
    i = np.searchsorted(dates, d)
    if i > 0 and (d - dates[i - 1]) / np.timedelta64(1, "D") <= 4:
        n = int((d - dates[i - 1]) / np.timedelta64(1, "D"))
        return f"{escape(team)} spielte vor {n} Tag{'en' if n != 1 else ''} im Europapokal."
    return ""


def absences(con: sqlite3.Connection) -> pd.DataFrame:
    """Ausfälle aus dem letzten Snapshot, ein Eintrag pro Spieler (Gründe zusammengefasst)."""
    a = db.read(con, """
        SELECT team_id, tm_player_id, player, position, reason, until FROM availability_snapshots
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM availability_snapshots)
          AND category IN ('injury', 'suspension') AND team_id IS NOT NULL""")
    a["reason"] = a.reason.fillna("Ausfall")
    return (a.groupby(["team_id", "tm_player_id"], as_index=False)
             .agg(player=("player", "first"), position=("position", "first"),
                  reason=("reason", lambda r: " / ".join(dict.fromkeys(r))), until=("until", "max")))


def crest_ids(con: sqlite3.Connection) -> dict[str, int]:
    """team_id -> Transfermarkt-Vereins-ID (für das Wappen)."""
    return dict(con.execute("""SELECT team_id, tm_club_id FROM squad_value_snapshots
                               WHERE team_id IS NOT NULL GROUP BY team_id""").fetchall())


def model_comparison() -> pd.DataFrame | None:
    f = config.REPORT_DIR / "model_comparison.csv"
    return pd.read_csv(f).set_index("rolle") if f.exists() else None


def describe_model(model: str | None) -> str:
    """'dc+xg xi=0.002 reg=1.0 w_xg=0.5 xi_xg=0.005' -> lesbare Beschreibung."""
    if not model:
        return "–"
    p = dict(kv.split("=") for kv in model.split() if "=" in kv)
    half = lambda xi: f"{np.log(2) / float(xi) / 30.4:.0f} Monate"
    if "xi" not in p:
        return model
    if "w_xg" not in p:
        return f"Dixon-Coles auf Toren (Halbwertszeit {half(p['xi'])})"
    desc = (f"Dixon-Coles aus Toren und xG (xG-Gewicht {float(p['w_xg']):.0%}, Halbwertszeit "
            f"Tore {half(p['xi'])}, xG {half(p['xi_xg'])})").replace("%", " %")
    ens = re.search(r"ens(\d+)", model)
    if ens:
        desc += f", gemittelt über {ens.group(1)} Parametervarianten"
    if "+lgbm" in model:
        desc += " mit LightGBM-Korrektur (Saisonphase, xG-Form, Tabelle, Ruhetage)"
    return desc


# --- Formatierung -------------------------------------------------------------------------

def _num(x: float, digits: int = 1) -> str:
    """Deutsches Zahlenformat: 1.234,5"""
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _pct(p: float, digits: int = 0) -> str:
    return f"{_num(100 * p, digits)} %"


def _kickoff(utc: str) -> tuple[str, str]:
    dt = datetime.fromisoformat(utc).replace(tzinfo=timezone.utc).astimezone(TZ)
    return f"{WEEKDAYS[dt.weekday()]} {dt:%d.%m.}", f"{dt:%H:%M}"


def _img_css(cls: str, path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f".{cls}{{background-image:url(data:image/png;base64,{data})}}"


def crest(team_id: str, crests: dict[str, int], name: str) -> str:
    if team_id in crests and (IMG_DIR / transfermarkt.crest_name(crests[team_id])).exists():
        return f'<span class="crest c{crests[team_id]}" role="img" aria-label="Wappen {escape(name)}"></span>'
    return f'<span class="crest none">{escape(name[:1])}</span>'


def _cell_style(p: float, pmax: float) -> str:
    """Graustufe nach Wahrscheinlichkeit (relativ zum größten Wert der Matrix)."""
    if p < 0.0005:
        return ""
    a = 0.05 + 0.8 * p / pmax
    fg = "var(--bg)" if a >= 0.5 else "var(--text)"
    return f' style="background:rgb(var(--ink) / {a:.2f});color:{fg}"'


def _cell_label(p: float) -> str:
    if p < 0.005:
        return "·" if p >= 0.0005 else ""
    return f"{100 * p:.0f}"


def matrix_table(m: np.ndarray, home: str, away: str, tip: tuple[int, int]) -> str:
    n = MATRIX_SHOW
    sub = m[:n, :n]
    pmax = sub.max()
    head = "".join(f"<th>{j}</th>" for j in range(n))
    rows = []
    for i in range(n):
        cells = "".join(f"<td{' class=mtip' if (i, j) == tip else ''}{_cell_style(sub[i, j], pmax)}>"
                        f"{_cell_label(sub[i, j])}</td>" for j in range(n))
        rows.append(f"<tr><th>{i}</th>{cells}</tr>")
    return (f'<table class="matrix"><caption>in % · Zeilen {escape(home)}, Spalten {escape(away)}</caption><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def outcome_strip(ph: float, pd_: float, pa: float, pick: int) -> str:
    """1X2 als drei Felder mit Balken; der getippte Ausgang ist hervorgehoben."""
    segs = [(1, ph, "🏠 Heimsieg"), (0, pd_, "🤝 Remis"), (-1, pa, "✈️ Auswärtssieg")]
    bar = "".join(f'<i{" class=on" if k == pick else ""} style="width:{100 * p:.1f}%"></i>' for k, p, _ in segs)
    cells = "".join(f'<div{" class=on" if k == pick else ""}><b>{_pct(p)}</b><span>{lbl}</span></div>'
                    for k, p, lbl in segs)
    aria = ", ".join(f"{lbl[2:]} {_pct(p)}" for _, p, lbl in segs)
    return f'<div class="odds" role="img" aria-label="{aria}">{bar}</div><div class="oddcells">{cells}</div>'


def absence_cell(ctx: Ctx, team_id: str) -> str:
    """Ausfälle eines Teams als Liste; Stammspieler (>= 50 % der Minuten) fett und zuerst."""
    a = ctx.ab[ctx.ab.team_id == team_id]
    if a.empty:
        return "<span class='muted'>keine gemeldet</span>"
    a = a.assign(key=[(team_id, p) in ctx.key_players for p in a.player]).sort_values(
        ["key", "player"], ascending=[False, True])
    items = []
    for r in a.itertuples():
        icon = "🟥 " if "rot" in r.reason.lower() else ("🟨 " if "sperre" in r.reason.lower() else "")
        until = f" · bis {pd.Timestamp(r.until):%d.%m.}" if pd.notna(r.until) else ""
        name = f"<b>{escape(r.player)}</b>" if r.key else escape(r.player)
        items.append(f"<li>{icon}{name}<small>{escape(r.reason)}{until}</small></li>")
    return f"<ul class='abl'>{''.join(items)}</ul>"


def _missing_share(ctx: Ctx, team_id: str, match_date: str) -> str:
    imp = ctx.impact.get((match_date, team_id))
    return f"<small class='muted'>fehlt {_pct(imp['lineup'])} der Stammelf</small>" if imp and imp["lineup"] >= 0.05 else ""


WEEKDAYS_LONG = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
          "Oktober", "November", "Dezember"]


def _tip(r) -> tuple[int, int, float, float]:
    """(Tipp Heim, Tipp Auswärts, P(Tendenz des Tipps), P(Tipp exakt))."""
    m = np.array(json.loads(r.matrix))
    th, ta = (int(r.tip_h), int(r.tip_a)) if pd.notna(r.tip_h) else (int(r.pred_h), int(r.pred_a))
    p_tend = {1: r.p_h, 0: r.p_d, -1: r.p_a}[int(np.sign(th - ta))]
    return th, ta, float(p_tend), float(m[th, ta])


def _local_day(utc: str) -> date:
    return datetime.fromisoformat(utc).replace(tzinfo=timezone.utc).astimezone(TZ).date()


def _day_title(d: date) -> str:
    return f"{WEEKDAYS_LONG[d.weekday()]}, {d.day}. {MONTHS[d.month - 1]}"


def verdict(th: int, ta: int, home: str, away: str) -> tuple[int, str]:
    """(Ausgang 1/0/-1, Text) des Tipps."""
    k = int(np.sign(th - ta))
    return k, {1: f"Sieg {home}", 0: "Unentschieden", -1: f"Sieg {away}"}[k]


BLOWOUT_MIN = 0.12   # ab dieser Wahrscheinlichkeit wird ein möglicher Kantersieg erwähnt


def blowout(m: np.ndarray, home: str, away: str) -> str:
    """Hinweis auf einen möglichen Sieg mit 3+ Toren Unterschied (wird bewusst nicht getippt)."""
    g = np.arange(m.shape[0])
    d = g[:, None] - g[None, :]
    ph, pa = m[d >= 3].sum(), m[d <= -3].sum()
    team, p = (home, ph) if ph >= pa else (away, pa)
    return f"💥 Kantersieg (3+ Tore) für {escape(team)}: {_pct(p)}" if p >= BLOWOUT_MIN else ""


def chips(items: list[tuple[str, str]]) -> str:
    """Sprungnavigation zu den Ligen: (Anker, Liga-Code)."""
    return "<nav class='chips'>" + "".join(
        f'<a href="#{anchor}"><span class="logo lg-{lg}"></span>{escape(config.LEAGUES[lg].name)}</a>'
        for anchor, lg in items) + "</nav>"


def match_row(r, ctx: Ctx, provisional: bool) -> str:
    """Eine Zeile je Spiel (Anstoß, Teams, Tipp, Tendenz-Wahrscheinlichkeit); Details per Antippen."""
    _, time = _kickoff(r.kickoff_utc)
    home, away = ctx.names.get(r.home_id, r.home_id), ctx.names.get(r.away_id, r.away_id)
    m = np.array(json.loads(r.matrix))
    uid = r.understat_match_id
    th, ta, p_tend, p_exact = _tip(r)
    pick, text = verdict(th, ta, home, away)
    hcls, acls = {1: (" win", ""), 0: ("", ""), -1: ("", " win")}[pick]
    trs = [f"<tr><th>⚽ Erwartete Tore</th><td>{_num(r.lam_h)}</td><td>{_num(r.lam_a)}</td></tr>"]
    if ctx.res is not None:
        d = pd.Timestamp(r.date)
        fh, fa = (views.form_letters(insights.form(ctx.res, t, d)) for t in (r.home_id, r.away_id))
        if fh or fa:
            trs.append(f"<tr><th>📈 Form <small>letzte 5</small></th><td>{fh}</td><td>{fa}</td></tr>")
    trs.append(f"<tr><th>🚑 Ausfälle</th><td>{absence_cell(ctx, r.home_id)}{_missing_share(ctx, r.home_id, r.date)}</td>"
               f"<td>{absence_cell(ctx, r.away_id)}{_missing_share(ctx, r.away_id, r.date)}</td></tr>")
    notes = [f"🎯 Tipp {th}:{ta} trifft exakt mit {_pct(p_exact, 1)}", blowout(m, home, away)]
    if pd.notna(getattr(r, "lam_h0", np.nan)) and (abs(r.lam_h0 - r.lam_h) + abs(r.lam_a0 - r.lam_a)) >= 0.02:
        notes.append(f"🩹 Ausfälle eingerechnet (ohne: {_num(r.lam_h0, 2)} : {_num(r.lam_a0, 2)} erwartete Tore)")
    notes += [f"🌍 {n}" for n in (europe_note(ctx, t, nm, r.date) for t, nm in ((r.home_id, home), (r.away_id, away))) if n]
    facts = (insights.match_facts(ctx.res, ctx.players, r.home_id, r.away_id, r.date, r.season, ctx.names,
                                  r.lam_h, r.lam_a, ctx.totals or {}) if ctx.res is not None else [])
    det = (outcome_strip(r.p_h, r.p_d, r.p_a, pick)
           + f"<table class='dt'><thead><tr><th></th><th>{crest(r.home_id, ctx.crests, home)}{escape(home)}</th>"
             f"<th>{crest(r.away_id, ctx.crests, away)}{escape(away)}</th></tr></thead><tbody>{''.join(trs)}</tbody></table>"
           + "<ul class='notes'>" + "".join(f"<li>{n}</li>" for n in notes if n)
           + "".join(f"<li>{escape(f)}</li>" for f in facts) + "</ul>"
           + f'<input type="checkbox" id="x{uid}" class="tg"><label for="x{uid}" class="more">📊 Ergebnismatrix</label>'
             f'<div class="mx">{matrix_table(m, home, away, (th, ta))}</div>')
    return f"""<div class="match" id="g{uid}"><input type="checkbox" id="m{uid}" class="tg">
<label for="m{uid}" class="row" aria-label="{escape(home)} gegen {escape(away)}, Tipp {th}:{ta}, {escape(text)}">
<span class="ko">{time}{'*' if provisional else ''}</span>
<span class="tm h{hcls}"><span class="nm">{escape(home)}</span>{crest(r.home_id, ctx.crests, home)}</span>
<span class="tip">{th}:{ta}</span>
<span class="tm a{acls}">{crest(r.away_id, ctx.crests, away)}<span class="nm">{escape(away)}</span></span>
<span class="pr">{_pct(p_tend)}</span></label>
<div class="det">{det}</div></div>"""


def provisional_leagues(pred: pd.DataFrame) -> set[str]:
    """Ligen, deren Anstoßzeiten noch Platzhalter sind (Understat: viele Spiele zur selben Minute)."""
    return {lg for lg, g in pred.groupby("league") if g.kickoff_utc.value_counts().max() >= 6}


def league_section(lg: str, g: pd.DataFrame, ctx: Ctx, provisional: bool) -> str:
    name = config.LEAGUES[lg].name
    g = g.assign(day=g.kickoff_utc.map(_local_day)).sort_values(["kickoff_utc", "home_id"])
    body = "".join(f"<h3 class='day'>{_day_title(day)}</h3>" + "".join(match_row(r, ctx, provisional)
                                                                      for r in dg.itertuples())
                   for day, dg in g.groupby("day", sort=True))
    note = "<p class='muted small'>* Anstoßzeit steht noch nicht fest</p>" if provisional else ""
    return (f"<section id='{lg}' class='league' style='--lg:{THEMES[lg].primary}'>"
            f"<h2><span class='logo lg-{lg}'></span>{escape(name)}</h2>{body}{note}</section>")


def ab_test(sc_all: pd.DataFrame, model: str | None) -> str:
    """Vergleich mit/ohne Ausfall-Anpassung auf denselben ausgewerteten Spielen."""
    if sc_all.empty or not model or not model.endswith(" +ausfaelle"):
        return ""
    base = model.removesuffix(" +ausfaelle")
    a, b = sc_all[sc_all.model == model], sc_all[sc_all.model == base]
    common = set(a.understat_match_id) & set(b.understat_match_id)
    if not common:
        return ""
    rows = []
    for label, d in (("mit Ausfällen", a), ("ohne", b)):
        d = d[d.understat_match_id.isin(common)]
        rows.append(f"<tr><td>{label}</td><td>{_pct(d.tip_exact.mean(), 1)}</td>"
                    f"<td>{_pct(d.tip_tendency.mean(), 1)}</td><td>{int(d.points.sum())}</td>"
                    f"<td>{_num(d.logloss.mean(), 3)}</td></tr>")
    return (f"<h3>🧪 Ausfall-Anpassung im Test</h3><table class='stats'><thead><tr><th>{len(common)} Spiele</th>"
            f"<th>Exakt</th><th>Tendenz</th><th>Punkte</th><th>Log-Loss</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def live_summary(sc: pd.DataFrame) -> str:
    """Bilanz der echten, vorab gespeicherten Tipps."""
    if sc.empty:
        return ""
    first = pd.Timestamp(sc.kickoff_utc.min())
    return (f"<p class='lead'>🔴 Live seit {first:%d.%m.}: {len(sc)} Spiele · Tendenz {_pct(sc.tip_tendency.mean())}"
            f" · exakt {_pct(sc.tip_exact.mean(), 1)} · {int(sc.points.sum())} Punkte</p>")


def model_section(cmp: pd.DataFrame | None, model: str | None) -> str:
    if cmp is None or "aktuell" not in cmp.index or "pts_tip" not in cmp.columns:
        return ""
    a = cmp.loc["aktuell"]
    label = {"aktuell": "aktuell (+ LightGBM)"}
    rows = "".join(f"<tr><td>{escape(label.get(role, role))}</td><td>{_num(r['tip_exact_%'])} %</td>"
                   f"<td>{_num(r['tip_tend_%'])} %</td><td>{_num(r['pts_tip'], 2)}</td></tr>"
                   for role, r in cmp.iterrows())
    return f"""<section class="plain"><h2>🧠 Modell</h2>
<p>{escape(describe_model(model))}.</p>
<table class="stats"><caption>Backtest seit 2019/20 · {_num(a['n'], 0)} Spiele</caption><thead><tr><th>Variante</th><th>Exakt</th><th>Tendenz</th><th>Punkte/Spiel</th></tr></thead>
<tbody>{rows}</tbody></table>
</section>"""


FONT_FILE = config.SEED_DIR / "fonts" / "plus-jakarta-sans-latin-wght-normal.woff2"  # SIL OFL 1.1


def font_css() -> str:
    """Plus Jakarta Sans (variable, 200–800) eingebettet – die Datei lädt nichts nach."""
    if not FONT_FILE.exists():
        return ""
    data = base64.b64encode(FONT_FILE.read_bytes()).decode()
    return ("@font-face{font-family:'Plus Jakarta Sans';font-weight:200 800;font-style:normal;"
            f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")


CSS = """
:root{--bg:#fbfaf7;--surface:#fff;--text:#1b1b19;--muted:#77756d;--line:#e5e3dc;--ink:27 27 25;--good:#2e7d4f;
--bad:#b4473c;--title:#2a5d9f;--down:#b4473c;--accent:#1f5fbf}
@media (prefers-color-scheme:dark){:root{--bg:#131312;--surface:#1c1c1a;--text:#ecebe6;--muted:#99978e;--line:#2e2e2b;
--ink:236 235 230;--good:#6fbf8d;--bad:#e07a6e;--title:#79a7e0;--down:#e07a6e;--accent:#7fb0ef}
.crest{filter:drop-shadow(0 0 1px rgba(255,255,255,.6))}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 "Plus Jakarta Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
-webkit-font-smoothing:antialiased}
main{max-width:680px;margin:0 auto;padding:0 16px 48px}
h1,h2{font-weight:800;letter-spacing:-.02em}
.muted{color:var(--muted)}.small{font-size:12px}
a{color:inherit}
/* Kopf */
.top{padding:26px 0 4px}
.top h1{margin:0;font-size:26px;line-height:1.15}
.top p{margin:4px 0 0;font-size:14px;color:var(--muted)}
.top p b{color:var(--text);font-weight:600}
.headrow{display:flex;align-items:center;justify-content:space-between;gap:8px 12px;flex-wrap:wrap}
.refresh{flex:none;font-size:12px;font-weight:600;color:var(--accent);text-decoration:none;
border:1px solid var(--line);background:var(--surface);border-radius:999px;padding:5px 11px}
.refresh:hover{border-color:var(--accent)}
@media (hover:none),(pointer:coarse),(max-width:560px){.refresh{display:none}}
/* Reiter */
.vt,.tg{position:absolute;opacity:0;pointer-events:none;width:0;height:0}
.tabs{display:flex;gap:6px;position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;margin-top:12px;
border-bottom:1px solid var(--line)}
.tabs label{flex:1;text-align:center;padding:8px 4px;font-size:13px;font-weight:600;border-radius:10px;cursor:pointer;
border:1px solid var(--line);background:var(--surface);color:var(--text);white-space:nowrap}
#t-spiele:checked~.tabs label[for=t-spiele],#t-erg:checked~.tabs label[for=t-erg],
#t-tab:checked~.tabs label[for=t-tab],#t-fakt:checked~.tabs label[for=t-fakt]{background:var(--text);color:var(--bg);
border-color:var(--text)}
.view{display:none}
#t-spiele:checked~.views .v-spiele,#t-erg:checked~.views .v-erg,#t-tab:checked~.views .v-tab,
#t-fakt:checked~.views .v-fakt{display:block}
/* Liga-Navigation */
.chips{display:flex;gap:8px;overflow-x:auto;padding:12px 0 2px;scrollbar-width:none}
.chips::-webkit-scrollbar{display:none}
.chips a{flex:none;display:inline-flex;align-items:center;gap:7px;padding:6px 12px 6px 6px;border-radius:999px;
border:1px solid var(--line);background:var(--surface);color:var(--text);text-decoration:none;font-size:13px;
font-weight:600}
.chips a::after{content:"↓";color:var(--accent);font-size:12px;margin-left:2px}
.chips .logo{width:22px;height:22px;border-radius:50%;margin:0}
.intro{margin:10px 0 0;font-size:12px;color:var(--muted)}
section{scroll-margin-top:64px}
.logo{display:inline-block;width:22px;height:22px;border-radius:5px;background:#fff center/84% no-repeat;flex:none;
vertical-align:-5px;margin-right:8px}
/* Liga und Spiele */
.league h2{font-size:20px;margin:30px 0 0;padding-bottom:6px;border-bottom:2px solid color-mix(in srgb,var(--lg) 70%,var(--muted))}
h3.day{font-size:12px;font-weight:600;color:var(--muted);margin:16px 0 2px}
.match{border-bottom:1px solid var(--line)}
.row{display:grid;grid-template-columns:40px 1fr 44px 1fr 38px;align-items:center;gap:6px;padding:9px 0;cursor:pointer;
-webkit-user-select:none;user-select:none}
.ko{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.tm{display:flex;align-items:center;gap:7px;min-width:0;font-size:14px}
.tm.h{justify-content:flex-end;text-align:right}
.nm{overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.tm.win{font-weight:700}
.crest{width:20px;height:20px;background:center/contain no-repeat;flex:none;display:inline-flex;align-items:center;
justify-content:center}
.crest.none{border-radius:50%;background:var(--line);color:var(--muted);font-size:10px;font-weight:600}
.tip{text-align:center;font-weight:700;font-size:16px;font-variant-numeric:tabular-nums}
.pr{text-align:right;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.pr::after{content:" ▾";font-size:9px}
.match>.tg:checked+.row .pr::after{content:" ▴"}
/* Details */
.det{display:none;margin:0 0 14px;padding:12px;border-radius:12px;background:var(--surface);border:1px solid var(--line);
font-size:13px}
.match>.tg:checked+.row+.det{display:block}
.odds{display:flex;height:6px;gap:2px;border-radius:3px;overflow:hidden}
.odds i{background:rgb(var(--ink) / .14)}.odds i.on{background:var(--accent)}
.oddcells{display:grid;grid-template-columns:repeat(3,1fr);margin-top:6px;font-variant-numeric:tabular-nums}
.oddcells div{display:flex;flex-direction:column;color:var(--muted);font-size:11px}
.oddcells div:nth-child(2){text-align:center}.oddcells div:nth-child(3){text-align:right}
.oddcells b{font-size:16px;color:var(--text);font-weight:600}.oddcells .on b{color:var(--accent);font-weight:700}
.dt{width:100%;border-collapse:collapse;margin-top:12px;table-layout:fixed}
.dt th,.dt td{text-align:left;vertical-align:top;padding:7px 4px;border-top:1px solid var(--line)}
.dt thead th{border-top:0;font-size:12px;font-weight:700;padding-bottom:4px}
.dt thead th .crest{width:16px;height:16px;vertical-align:-3px;margin-right:5px;display:inline-block}
.dt tbody th{width:28%;font-weight:600;font-size:12px;color:var(--muted)}
.dt tbody th small{display:block;font-weight:400}
.dt td{font-variant-numeric:tabular-nums}
.abl{list-style:none;margin:0;padding:0}.abl li{margin:0 0 5px;line-height:1.3}
.abl small{display:block;color:var(--muted);font-size:11px}
.dt td{overflow-wrap:anywhere;-webkit-hyphens:auto;hyphens:auto}
.frm{letter-spacing:.08em;font-size:12px;font-weight:600}
.frm .fS{color:var(--good)}.frm .fU{color:var(--muted)}.frm .fN{color:var(--bad)}
.notes{list-style:none;margin:10px 0 0;padding:10px 0 0;border-top:1px solid var(--line)}
.notes li{margin:4px 0;line-height:1.4}
.more{display:inline-block;margin-top:10px;font-size:12px;font-weight:600;color:var(--accent);cursor:pointer}
.more::after{content:" ▾"}
.mx{display:none;margin-top:8px}.det>.tg:checked+.more+.mx{display:block}
.matrix{border-collapse:separate;border-spacing:2px;font-size:11px;font-variant-numeric:tabular-nums;width:100%;
max-width:360px;table-layout:fixed}
.matrix caption{caption-side:bottom;text-align:left;font-size:11px;color:var(--muted);padding-top:4px}
.matrix th{color:var(--muted);font-weight:500;font-size:10px}
.matrix td{text-align:center;padding:6px 0;border-radius:3px}
.matrix td.mtip{box-shadow:inset 0 0 0 1.5px var(--text);font-weight:700}
/* Textseiten */
.plain h2{font-size:20px;margin:28px 0 8px}
h3{font-size:14px;margin:20px 0 6px}
.lead{font-size:15px;margin:16px 0 4px}
.stats{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;margin-top:6px}
.stats th,.stats td{text-align:right;padding:7px 4px;border-bottom:1px solid var(--line)}
.stats th:first-child,.stats td:first-child{text-align:left}
.stats th{font-size:11px;color:var(--muted);font-weight:500}
.stats caption{caption-side:bottom;text-align:left;font-size:11px;color:var(--muted);padding-top:6px}
footer{margin-top:40px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:11px;line-height:1.5}
"""


def played_matches(con: sqlite3.Connection, sc_all: pd.DataFrame, model: str | None, season_: int) -> pd.DataFrame:
    """Gespielte Spiele der Saison mit Tipp: Live-Tipps, sonst Walk-forward-Tipps aus dem Backtest."""
    cols = ["league", "season", "date", "home_id", "away_id", "hg", "ag", "tip_h", "tip_a", "source"]
    parts = []
    f = config.REPORT_DIR / "backtest_predictions.csv"
    if f.exists():
        bt = pd.read_csv(f)
        bt = bt[bt.season == season_].rename(columns={"home": "home_id", "away": "away_id"})
        parts.append(bt.assign(source="backtest")[cols])
    if len(sc_all) and model:
        live = sc_all[sc_all.model == model].assign(source="live")
        parts.append(live[cols])
    if not parts:
        return pd.DataFrame(columns=cols)
    p = pd.concat(parts, ignore_index=True)
    p = p.sort_values("source", ascending=False).drop_duplicates(["season", "home_id", "away_id"])  # live vor backtest
    return p.dropna(subset=["tip_h", "hg"]).astype({"tip_h": int, "tip_a": int, "hg": int, "ag": int})


def render(con: sqlite3.Connection, today: date | None = None) -> str:
    today = today or date.today()
    season_ = config.current_season(today)
    model = current_model(con)
    pred = upcoming_predictions(con, today, model)
    names = teams.display_names(con)
    crests = crest_ids(con)
    impact, key = absence_impact(con, pred) if len(pred) else ({}, set())
    eu = {t: np.sort(g.date.to_numpy()) for t, g in europe.team_dates(con).groupby("team")}
    res = insights.results_frame(predict.live_fit_data(con))
    players = insights.load_players(con, season_)
    totals = ({"max_total": float((pred.lam_h + pred.lam_a).max()), "min_total": float((pred.lam_h + pred.lam_a).min())}
              if len(pred) else {})
    ctx = Ctx(names, crests, absences(con), impact, key, eu, res, players, totals)
    sc_all = predict.scorecard(con)
    sc = sc_all[sc_all.model == model] if len(sc_all) else sc_all
    snap = con.execute("SELECT MAX(snapshot_date) FROM availability_snapshots").fetchone()[0]
    proj = season.latest(con)
    played = played_matches(con, sc_all, model, season_)
    bt = pd.read_csv(config.REPORT_DIR / "backtest_predictions.csv") if (config.REPORT_DIR / "backtest_predictions.csv").exists() else None
    facts = insights.weekly_facts(res, players, bt, names, season_, today)

    shown_teams = (set(pred.home_id) | set(pred.away_id) | set(proj.team_id) | set(played.home_id)
                   | set(played.away_id) | {k for f in facts if f.kind == "team" for k, _ in f.ranking})
    img_css = [_img_css(f"c{crests[t]}", IMG_DIR / transfermarkt.crest_name(crests[t]))
               for t in sorted(shown_teams) if t in crests]
    img_css += [_img_css(f"lg-{lg}", IMG_DIR / transfermarkt.logo_name(lg)) for lg in config.TOP_LEAGUES]

    leagues = [lg for lg in config.TOP_LEAGUES if (pred.league == lg).any()]
    if leagues:
        prov = provisional_leagues(pred)
        games = (chips([(lg, lg) for lg in leagues]) + "<p class='intro'>% = Chance, dass die Tendenz stimmt</p>"
                 + "".join(league_section(lg, pred[pred.league == lg], ctx, lg in prov) for lg in leagues))
    else:
        games = "<p class='muted'>Keine Vorhersagen vorhanden – erst <code>fussball predict</code> ausführen.</p>"
    results = views.results_view(played, ctx, live_summary(sc) + ab_test(sc_all, model))
    fakten = views.facts_view(facts, ctx) + model_section(model_comparison(), model)

    if len(pred):
        days = sorted(pred.kickoff_utc.map(_local_day))
        d0, d1 = days[0], days[-1]
        title = (f"{d0.day}.–{d1.day}. {MONTHS[d1.month - 1]}" if d0.month == d1.month
                 else f"{d0.day}. {MONTHS[d0.month - 1]} – {d1.day}. {MONTHS[d1.month - 1]}")
        created = datetime.fromisoformat(pred.created_at.max()).astimezone(TZ)
        stamp = f"Stand {WEEKDAYS[created.weekday()]} {created:%d.%m., %H:%M}"
        results_until = (pd.Timestamp(pred.cutoff_date.max()) - pd.Timedelta(days=1)).strftime("%d.%m.")
    else:
        title, stamp, results_until = "– noch keiner geplant", "", "–"
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">
<title>Dein Tippspiel-Assistent</title><style>{font_css()}{CSS}{views.CSS}{''.join(img_css)}</style></head><body><main>
<header class="top"><div class="headrow"><h1>Dein Tippspiel-Assistent</h1>
<a class="refresh" href="tippspiel://refresh" title="Startet den Lauf auf dem Mac (ca. 10 Min.)">↻ Jetzt aktualisieren</a></div><p>⚽ <b>Spieltag {escape(title)}</b>{' · ' + stamp if stamp else ''}</p></header>
<input type="radio" name="view" id="t-spiele" class="vt" checked><input type="radio" name="view" id="t-erg" class="vt">
<input type="radio" name="view" id="t-tab" class="vt"><input type="radio" name="view" id="t-fakt" class="vt">
<div class="tabs"><label for="t-spiele">📅 Spiele</label><label for="t-erg">✅ Ergebnisse</label>
<label for="t-tab">🏆 Tabelle</label><label for="t-fakt">💡 Fakten</label></div>
<div class="views"><div class="view v-spiele">{games}</div><div class="view v-erg">{results}</div>
<div class="view v-tab">{views.table_view(proj, ctx)}</div><div class="view v-fakt">{fakten}</div></div>
<footer>Ergebnisse bis {results_until} · Ausfälle vom {pd.Timestamp(snap).strftime('%d.%m.') if snap else '–'} ·
ohne Wettquoten · Wappen und Logos: Transfermarkt, nur privat</footer>
</main></body></html>"""


def build(con: sqlite3.Connection, out: Path | None = None, today: date | None = None) -> Path:
    out = Path(out or OUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(con, today), encoding="utf-8")
    return out
