"""Statisches Dashboard (eine HTML-Datei, ohne JavaScript, mobil-first).

Im Zentrum steht der nächste Spieltag je Liga im Farbschema der Liga: Tipp, Wahrscheinlichkeit
des Tipps, 1X2, erwartete Tore, Ergebnismatrix und Ausfälle (beides eingeklappt). Darunter
Live-Trefferbilanz und Backtest-Einordnung.

Aufklappen per CSS (Checkbox + Label) statt <details>: startet in jedem Viewer eingeklappt,
auch in der Vorschau der iOS-Dateien-App. Wappen und Logos werden einmal als CSS-Klassen
eingebettet (base64), die Datei lädt also nichts nach.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fussball import config, db, predict, teams
from fussball.features import availability
from fussball.ingest import europe, transfermarkt

OUT_FILE = config.DATA_DIR / "dashboard" / "index.html"
IMG_DIR = config.RAW_DIR / transfermarkt.SOURCE
TZ = ZoneInfo("Europe/Berlin")
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
MATRIX_SHOW = 6  # 0..5 Tore je Team
# Sequentielle Blau-Rampe (hell -> dunkel) für die Matrix
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


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


def europe_note(ctx: Ctx, team_id: str, match_date: str) -> str:
    """'Europapokal vor 3 Tagen' (Belastung wirkt nach dem Spiel; nur wenn nah dran)."""
    dates = ctx.europe.get(team_id)
    if dates is None:
        return ""
    d = np.datetime64(match_date)
    i = np.searchsorted(dates, d)
    notes = []
    if i > 0 and (d - dates[i - 1]) / np.timedelta64(1, "D") <= 4:
        n = int((d - dates[i - 1]) / np.timedelta64(1, "D"))
        notes.append(f"Europapokal vor {n} Tag{'en' if n != 1 else ''}")
    return f"<span class='eu'>{' · '.join(notes)}</span>" if notes else ""


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
    if p < 0.0005:
        return ""
    k = min(len(RAMP) - 1, int(round((len(RAMP) - 1) * p / pmax)))
    fg = "#ffffff" if k >= 6 else "#0b0b0b"
    return f' style="background:{RAMP[k]};color:{fg}"'


def matrix_table(m: np.ndarray, home: str, away: str, tip: tuple[int, int],
                 likely: tuple[int, int] | None = None) -> str:
    n = MATRIX_SHOW
    sub = m[:n, :n]
    pmax = sub.max()
    head = "".join(f"<th>{j}</th>" for j in range(n))
    rows = []
    for i in range(n):
        cells = "".join(
            f"<td{' class=tip' if (i, j) == tip else (' class=likely' if (i, j) == likely else '')}"
            f"{_cell_style(sub[i, j], pmax)}>"
            f"{_num(100 * sub[i, j]) if sub[i, j] >= 0.0005 else ''}</td>" for j in range(n))
        rows.append(f"<tr><th>{i}</th>{cells}</tr>")
    return (f'<table class="matrix"><thead><tr><th class="corner">↓ {escape(home)}<br>→ {escape(away)}</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f'<p class="muted small">Wahrscheinlichkeit je Ergebnis in %. Umrandet = Tipp'
            f'{", gestrichelt = wahrscheinlichstes Einzelergebnis" if likely and likely != tip else ""}</p>')


def outcome_bar(ph: float, pd_: float, pa: float) -> str:
    return (f'<div class="bar" role="img" aria-label="Heimsieg {_pct(ph)}, Remis {_pct(pd_)}, '
            f'Auswärtssieg {_pct(pa)}"><span class="h" style="width:{100 * ph:.1f}%"></span>'
            f'<span class="d" style="width:{100 * pd_:.1f}%"></span>'
            f'<span class="a" style="width:{100 * pa:.1f}%"></span></div>'
            f'<div class="barlabels"><span>Heim {_pct(ph)}</span><span>Remis {_pct(pd_)}</span>'
            f'<span>Auswärts {_pct(pa)}</span></div>')


def absence_list(ctx: Ctx, team_id: str, team: str, match_date: str) -> str:
    a = ctx.ab[ctx.ab.team_id == team_id]
    if a.empty:
        return f"<div><p class='abs-team'>{escape(team)}</p><p class='muted small'>keine gemeldeten Ausfälle</p></div>"
    imp = ctx.impact.get((match_date, team_id))
    summary = (f"<p class='muted small'>fehlt: {_pct(imp['lineup'])} der Stammelf · "
               f"{_pct(imp['att'])} der Offensive</p>" if imp else "")
    items = []
    for r in a.sort_values("player").itertuples():
        pos = f"<span class='muted'> · {escape(r.position)}</span>" if pd.notna(r.position) else ""
        until = f" <span class='muted'>(bis {pd.Timestamp(r.until):%d.%m.})</span>" if pd.notna(r.until) else ""
        badge = "<span class='badge'>Stammspieler</span>" if (team_id, r.player) in ctx.key_players else ""
        items.append(f"<li><strong>{escape(r.player)}</strong>{badge}{pos}<br>{escape(r.reason)}{until}</li>")
    return (f"<div><p class='abs-team'>{escape(team)} <span class='muted'>({len(a)})</span></p>{summary}"
            f"<ul>{''.join(items)}</ul></div>")


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


def match_card(r, ctx: Ctx, provisional: bool, prefix: str = "l", show_league: bool = False) -> str:
    day, time = _kickoff(r.kickoff_utc)
    home, away = ctx.names.get(r.home_id, r.home_id), ctx.names.get(r.away_id, r.away_id)
    crests = ctx.crests
    m = np.array(json.loads(r.matrix))
    n_home = int((ctx.ab.team_id == r.home_id).sum())
    n_away = int((ctx.ab.team_id == r.away_id).sum())
    shift = ""
    if pd.notna(getattr(r, "lam_h0", np.nan)) and (abs(r.lam_h0 - r.lam_h) + abs(r.lam_a0 - r.lam_a)) >= 0.02:
        shift = (f"<p class='muted small'>Ausfälle eingerechnet: erwartete Tore {_num(r.lam_h0, 2)} : "
                 f"{_num(r.lam_a0, 2)} → {_num(r.lam_h, 2)} : {_num(r.lam_a, 2)}</p>")
    uid = f"{prefix}{r.understat_match_id}"
    th, ta, p_tend, _ = _tip(r)
    lg_tag = (f"<span class='lgtag'><span class='logo lg-{r.league}'></span>{escape(config.LEAGUES[r.league].name)}</span>"
              if show_league else "")
    theme = THEMES[r.league]
    likely = ("" if (th, ta) == (r.pred_h, r.pred_a) else
              f"<span class='alt'>wahrscheinlichstes Einzelergebnis {r.pred_h}:{r.pred_a}</span>")
    return f"""
<article class="card" style="--lg:{theme.primary}">
  {lg_tag}<div class="meta"><span>{day} · {time}{'*' if provisional else ''}</span>
    <span class="chance">Tendenz {_pct(p_tend)} · exakt {_pct(m[th, ta], 1)}</span></div>
  <div class="fixture">
    <div class="side">{crest(r.home_id, crests, home)}<span class="team">{escape(home)}</span>{europe_note(ctx, r.home_id, r.date)}</div>
    <div class="score"><span class="tip">{th}:{ta}</span>
      <span class="xg">erw. Tore {_num(r.lam_h)} : {_num(r.lam_a)}</span>{likely}</div>
    <div class="side">{crest(r.away_id, crests, away)}<span class="team">{escape(away)}</span>{europe_note(ctx, r.away_id, r.date)}</div>
  </div>
  {outcome_bar(r.p_h, r.p_d, r.p_a)}
  <input type="checkbox" id="m{uid}" class="tg tg-m"><input type="checkbox" id="a{uid}" class="tg tg-a">
  <div class="toggles"><label for="m{uid}" class="lm">Ergebnismatrix</label>
    <label for="a{uid}" class="la">Ausfälle {n_home} · {n_away}</label></div>
  <div class="panel pm">{matrix_table(m, home, away, (th, ta), (r.pred_h, r.pred_a))}</div>
  <div class="panel pa">{shift}<div class="abs">{absence_list(ctx, r.home_id, home, r.date)}{absence_list(ctx, r.away_id, away, r.date)}</div></div>
</article>"""


def league_section(lg: str, g: pd.DataFrame, ctx: Ctx) -> str:
    th = THEMES[lg]
    name = config.LEAGUES[lg].name
    # Understat setzt nicht terminierte Spiele auf eine gemeinsame Platzhalter-Uhrzeit
    provisional = g.kickoff_utc.value_counts().max() >= 6
    days = list(dict.fromkeys(_kickoff(k)[0] for k in sorted(g.kickoff_utc)))
    span = days[0] if len(days) == 1 else f"{days[0]} – {days[-1]}"
    note = "<p class='muted small note'>* Anstoßzeiten noch nicht terminiert</p>" if provisional else ""
    cards = "".join(match_card(r, ctx, provisional, "l") for r in g.itertuples())
    return f"""<section id="{lg}" class="league" style="--lg:{th.primary};--lg-accent:{th.accent};--lg-ink:{th.ink}">
<header class="lg-head"><span class="logo lg-{lg}" role="img" aria-label="Logo {escape(name)}"></span>
<div><h2>{escape(name)}</h2><p>{span} · {len(g)} Spiele</p></div></header>{note}{cards}</section>"""


def provisional_leagues(pred: pd.DataFrame) -> set[str]:
    """Ligen, deren Anstoßzeiten noch Platzhalter sind (viele Spiele zur selben Minute)."""
    return {lg for lg, g in pred.groupby("league") if g.kickoff_utc.value_counts().max() >= 6}


def day_view(pred: pd.DataFrame, ctx: Ctx) -> str:
    prov = provisional_leagues(pred)
    pred = pred.assign(day=pred.kickoff_utc.map(_local_day)).sort_values(["day", "kickoff_utc", "league"])
    out = []
    for day, g in pred.groupby("day", sort=True):
        title = f"{WEEKDAYS_LONG[day.weekday()]}, {day.day}. {MONTHS[day.month - 1]}"
        cards = "".join(match_card(r, ctx, r.league in prov, "d", show_league=True) for r in g.itertuples())
        count = f"{len(g)} Spiel{'e' if len(g) != 1 else ''}"
        out.append(f"<section class='day'><h2 class='dayhead'>{title} <small>{count}</small></h2>{cards}</section>")
    note = ("<p class='muted small note'>* Anstoßzeit noch nicht terminiert – der Tag kann sich noch ändern</p>"
            if prov else "")
    return note + "".join(out)


def top_view(pred: pd.DataFrame, ctx: Ctx, n: int = 10) -> str:
    rows = []
    for r in pred.itertuples():
        th, ta, p_tend, p_exact = _tip(r)
        rows.append((r, th, ta, p_tend, p_exact))

    def line(r, th, ta, value) -> str:
        home, away = ctx.names.get(r.home_id, r.home_id), ctx.names.get(r.away_id, r.away_id)
        day, _ = _kickoff(r.kickoff_utc)
        return (f"<li class='topline' style='--lg:{THEMES[r.league].primary}'>"
                f"<span class='logo lg-{r.league}' title='{escape(config.LEAGUES[r.league].name)}'></span>"
                f"<span class='tt'><span class='tm'>{crest(r.home_id, ctx.crests, home)}{escape(home)}</span>"
                f"<span class='tm'>{crest(r.away_id, ctx.crests, away)}{escape(away)}</span></span>"
                f"<span class='tscore'><strong>{th}:{ta}</strong><span class='muted'>{day}</span></span>"
                f"<span class='tval'>{_pct(value, 0 if value >= 0.2 else 1)}</span></li>")

    tend = sorted(rows, key=lambda x: -x[3])[:n]
    exact = sorted(rows, key=lambda x: -x[4])[:n]
    return f"""<section class="plain"><h2>Sicherste Tendenzen</h2>
<p class="muted small">Die {n} Tipps, bei denen Sieg/Remis/Niederlage am wahrscheinlichsten stimmt.</p>
<ol class="toplist">{''.join(line(r, th, ta, pt) for r, th, ta, pt, _ in tend)}</ol>
<h2>Wahrscheinlichste exakte Treffer</h2>
<p class="muted small">Die {n} Tipps mit der höchsten Chance auf das exakte Ergebnis. Selbst der beste
Tipp trifft nur mit {_pct(exact[0][4])} – ungefähr jedes {round(1 / exact[0][4])}. Mal.</p>
<ol class="toplist">{''.join(line(r, th, ta, pe) for r, th, ta, _, pe in exact)}</ol></section>"""


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
    for label, d in (("mit Ausfällen", a), ("ohne Ausfälle", b)):
        d = d[d.understat_match_id.isin(common)]
        rows.append(f"<tr><td>{label}</td><td>{_pct(d.tip_exact.mean(), 1)}</td>"
                    f"<td>{_pct(d.tip_tendency.mean(), 1)}</td><td>{int(d.points.sum())}</td>"
                    f"<td>{_num(d.logloss.mean(), 3)}</td></tr>")
    return (f"<h3 class='sub'>Test: Ausfall-Anpassung ({len(common)} Spiele)</h3><table class='stats'><thead><tr>"
            f"<th>Variante</th><th>Exakt</th><th>Tendenz</th><th>Punkte</th><th>Log-Loss</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table><p class='muted small'>Die Ausfall-Anpassung ist eine "
            f"vorsichtige Schätzung (Ausfall-Historie gibt es erst seit 29.09.2026). Dieser Vergleich zeigt "
            f"mit der Zeit, ob sie hilft.</p>")


def scorecard_section(sc: pd.DataFrame, names: dict, ab_html: str = "") -> str:
    if sc.empty:
        return ("<section id='bilanz' class='plain'><h2>Live-Bilanz</h2><p class='muted'>Noch keine "
                "ausgewerteten Spiele – die Bilanz füllt sich, sobald vorhergesagte Spiele gespielt sind.</p></section>")
    rows = []
    for lg, g in [("Alle Ligen", sc)] + [(config.LEAGUES[k].name, v) for k, v in sc.groupby("league")]:
        rows.append(f"<tr><td>{escape(lg)}</td><td>{len(g)}</td><td>{_pct(g.tip_exact.mean(), 1)}</td>"
                    f"<td>{_pct(g.tip_tendency.mean(), 1)}</td><td>{int(g.points.sum())}</td></tr>")
    recent = "".join(
        f"<li><span class='mark {'ok' if r.tip_exact else ('half' if r.tip_tendency else 'no')}'>"
        f"{'✓' if r.tip_exact else ('≈' if r.tip_tendency else '✗')}</span>"
        f"{escape(names.get(r.home_id, r.home_id))} – {escape(names.get(r.away_id, r.away_id))}"
        f"<span class='muted'> Tipp {r.tip_h}:{r.tip_a} · Ergebnis {r.hg}:{r.ag} · {r.points} P.</span></li>"
        for r in sc.sort_values("kickoff_utc").tail(10).iloc[::-1].itertuples())
    return f"""<section id="bilanz" class="plain"><h2>Live-Bilanz</h2>
<table class="stats"><thead><tr><th>Liga</th><th>Spiele</th><th>Exakt</th><th>Tendenz</th><th>Punkte</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<input type="checkbox" id="recent" class="tg tg-r"><div class="toggles"><label for="recent" class="lr">Letzte 10 Spiele</label></div>
<div class="panel pr"><ul class="recent">{recent}</ul><p class="muted small">✓ exakt · ≈ Tendenz richtig · ✗ falsch ·
Punkte: 4 exakt, 3 Tordifferenz, 2 Tendenz</p></div>{ab_html}</section>"""


def backtest_section(cmp: pd.DataFrame | None, model: str | None) -> str:
    if cmp is None or "aktuell" not in cmp.index or "pts_tip" not in cmp.columns:
        return ""
    a = cmp.loc["aktuell"]
    label = {"aktuell": "Tore + xG + LightGBM (aktuell)"}
    steps = "".join(
        f"<tr><td>{escape(label.get(role, role))}</td><td>{_num(r['tip_exact_%'])} %</td>"
        f"<td>{_num(r['tip_tend_%'])} %</td><td>{_num(r['pts_tip'], 2)}</td></tr>" for role, r in cmp.iterrows())
    return f"""<section id="modell" class="plain"><h2>Einordnung</h2>
<p class="muted small">{escape(describe_model(model))}. Backtest ab 2019/20 · {_num(a['n'], 0)} Spiele,
jeweils nur mit Daten bis zum Vortag.</p>
<div class="kpis">
  <div><strong>{_num(a['tip_tend_%'])} %</strong><span>Tendenz richtig</span></div>
  <div><strong>{_num(a['tip_exact_%'])} %</strong><span>exakte Treffer</span></div>
  <div><strong>{_num(a['pts_tip'], 2)}</strong><span>Tippspiel-Punkte pro Spiel (4/3/2)</span></div>
  <div><strong>{_num(a['tip_draw_%'], 0)} %</strong><span>Remis-Tipps (real ca. 25 %)</span></div>
</div>
<p class="muted small"><strong>Tipp-Strategie:</strong> erst der Ausgang (Sieg/Remis/Niederlage), dann das
wahrscheinlichste Ergebnis dazu – so verteilen sich die Tipps wie echte Ergebnisse. Nur das wahrscheinlichste
Einzelergebnis zu tippen brächte {_num(a['exact_%'])} % exakte Treffer, aber nur {_num(a['tend_%'])} % richtige
Tendenzen und {_num(a['pts_argmax'], 2)} Punkte pro Spiel (meist 1:1).</p>
<table class="stats"><thead><tr><th>Modell</th><th>Exakt</th><th>Tendenz</th><th>Punkte</th></tr></thead>
<tbody>{steps}</tbody></table>
<p class="muted small">Log-Loss {_num(a['logloss'], 3)} · RPS {_num(a['rps'], 3)} (Wettmarkt {_num(a['rps_odds'], 3)})
– jeweils kleiner = besser.</p></section>"""


CSS = """
:root{--bg:#f3f3f1;--surface:#fff;--text:#0b0b0b;--muted:#5d5c57;--line:#e3e2dc;--chip:#f3f3f1;
--home:#2a78d6;--draw:#a3a29c;--away:#eb6834;--accent:#2a78d6;--ok:#0ca30c;--no:#d03b3b;--half:#c98500}
@media (prefers-color-scheme:dark){:root{--bg:#0f0f0e;--surface:#1a1a19;--text:#fff;--muted:#b9b8ae;
--line:#302f2b;--chip:#252523;--home:#3987e5;--draw:#7c7b75;--away:#d95926;--accent:#6da7ec}
.crest{filter:drop-shadow(0 0 1px rgba(255,255,255,.9)) drop-shadow(0 0 2px rgba(255,255,255,.35))}
.lg-head{box-shadow:inset 0 0 0 1px rgba(255,255,255,.14)}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:0 16px 48px}
.top{padding:20px 0 4px}.top h1{font-size:26px;letter-spacing:-.02em;margin:0}.top p{margin:4px 0 0;color:var(--muted);font-size:13px}
.vt{position:absolute;opacity:0;pointer-events:none}
.tabs{display:flex;gap:4px;position:sticky;top:0;z-index:3;background:var(--bg);padding:10px 0 8px}
.tabs label{flex:1;text-align:center;padding:9px 4px;border-radius:10px;font-weight:700;font-size:14px;
cursor:pointer;background:var(--surface);border:1px solid var(--line);color:var(--muted)}
#t-liga:checked~.tabs label[for=t-liga],#t-tag:checked~.tabs label[for=t-tag],#t-top:checked~.tabs label[for=t-top]
{background:var(--text);color:var(--bg);border-color:var(--text)}
.view{display:none}
#t-liga:checked~.views .v-liga,#t-tag:checked~.views .v-tag,#t-top:checked~.views .v-top{display:block}
nav{display:flex;gap:8px;overflow-x:auto;padding:4px 0 8px;background:var(--bg);scrollbar-width:none}
nav::-webkit-scrollbar{display:none}
nav a{flex:none;display:flex;align-items:center;gap:6px;padding:5px 12px 5px 5px;border-radius:999px;
color:var(--text);text-decoration:none;font-size:13px;font-weight:600;background:var(--surface);border:1px solid var(--line)}
nav a .logo{width:24px;height:24px;border-radius:50%}nav a.plain{padding-left:12px}
.logo{display:inline-block;background:#fff center/80% no-repeat;flex:none}
.league{margin-top:18px}section{scroll-margin-top:64px}.dayhead{font-size:18px;margin:22px 0 4px}.dayhead small{color:var(--muted);font-weight:400;font-size:13px}.lgtag{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:700;color:var(--muted);margin-bottom:4px}.lgtag .logo{width:18px;height:18px;border-radius:4px}.toplist{list-style:none;padding:0;margin:8px 0 0;counter-reset:top}.topline{display:grid;grid-template-columns:28px 1fr auto auto;align-items:center;gap:10px;background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--lg);border-radius:12px;padding:8px 10px;margin:6px 0}.topline>.logo{width:26px;height:26px;border-radius:6px}.tt{display:flex;flex-direction:column;gap:3px;min-width:0;font-size:13px;font-weight:600}.tm{display:flex;align-items:center;gap:6px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.tm .crest{width:18px;height:18px;flex:none}.tscore{display:flex;flex-direction:column;align-items:center;font-size:11px}.tscore strong{font-size:20px}.tval{font-weight:800;font-size:16px;font-variant-numeric:tabular-nums;min-width:48px;text-align:right}
.lg-head{display:flex;align-items:center;gap:12px;background:var(--lg);color:var(--lg-ink);border-radius:14px;
padding:12px 14px;border-bottom:4px solid var(--lg-accent)}
.lg-head .logo{width:48px;height:48px;border-radius:10px}.lg-head h2{margin:0;font-size:19px;letter-spacing:-.01em}
.lg-head p{margin:0;font-size:13px;opacity:.85}
.note{margin:6px 2px 0}
.card{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:12px 14px 10px;
margin:10px 0;border-top:3px solid var(--lg)}
.meta{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px}
.chance{background:var(--chip);border-radius:999px;padding:2px 8px;font-variant-numeric:tabular-nums}
.fixture{display:grid;grid-template-columns:1fr auto 1fr;align-items:start;gap:8px;margin:10px 0 4px}
.side{display:flex;flex-direction:column;align-items:center;gap:6px;text-align:center;min-width:0}
.crest{width:44px;height:44px;background:center/contain no-repeat;display:flex;align-items:center;justify-content:center}
.crest.none{border-radius:50%;background:var(--chip);color:var(--muted);font-weight:700}
.team{font-weight:600;font-size:14px;line-height:1.25;overflow-wrap:break-word;-webkit-hyphens:auto;hyphens:auto}
.score{display:flex;flex-direction:column;align-items:center;padding-top:4px}
.tip{font-size:30px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:.02em;line-height:1.1;
background:var(--chip);border-radius:10px;padding:4px 14px}
.xg{font-size:11px;color:var(--muted);margin-top:5px;white-space:nowrap}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:12px;gap:2px}
.bar .h{background:var(--home)}.bar .d{background:var(--draw)}.bar .a{background:var(--away)}
.barlabels{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:3px;font-variant-numeric:tabular-nums}
.tg{position:absolute;opacity:0;pointer-events:none;width:0;height:0}
.toggles{display:flex;gap:8px;margin-top:10px}
.toggles label{flex:1;text-align:center;font-size:13px;font-weight:600;color:var(--accent);background:var(--chip);
border-radius:10px;padding:8px 6px;cursor:pointer;user-select:none;-webkit-user-select:none}
.toggles label::after{content:" ▾";font-size:11px}
.panel{display:none;margin-top:10px}
.tg-m:checked~.pm,.tg-a:checked~.pa,.tg-r:checked~.pr{display:block}
.tg-m:checked~.toggles .lm,.tg-a:checked~.toggles .la,.tg-r:checked~.toggles .lr{background:var(--accent);color:#fff}
.tg-m:checked~.toggles .lm::after,.tg-a:checked~.toggles .la::after,.tg-r:checked~.toggles .lr::after{content:" ▴"}
.matrix{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;width:100%;table-layout:fixed}
.matrix th{color:var(--muted);font-weight:600;padding:3px;font-size:11px}
.matrix th.corner{text-align:left;font-weight:400;font-size:10px;line-height:1.2;width:28%;overflow-wrap:anywhere}
.matrix td{text-align:center;padding:6px 1px;border:2px solid var(--surface);border-radius:4px}
.matrix td.tip{outline:2px solid var(--text);outline-offset:-2px;font-weight:800}.matrix td.likely{outline:2px dashed var(--muted);outline-offset:-2px}.alt{font-size:10px;color:var(--muted);margin-top:2px;text-align:center;line-height:1.2;max-width:110px}
.abs{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.abs-team{font-weight:700;margin:0 0 4px;font-size:13px}.badge{display:inline-block;font-size:10px;font-weight:700;color:#fff;background:var(--away);border-radius:4px;padding:0 4px;margin-left:4px;vertical-align:1px}.eu{font-size:10px;color:var(--muted);background:var(--chip);border-radius:999px;padding:1px 6px}h3.sub{font-size:15px;margin:18px 0 6px}
.abs ul{margin:0;padding:0;list-style:none;font-size:12px}.abs li{padding:4px 0;border-top:1px solid var(--line)}
.muted{color:var(--muted)}.small{font-size:12px}
.plain h2{font-size:19px;margin:30px 0 8px}
.stats{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;background:var(--surface);
border-radius:12px;overflow:hidden}
.stats th,.stats td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line)}
.stats th:first-child,.stats td:first-child{text-align:left}.stats td:not(:first-child){white-space:nowrap}.stats th{font-size:12px;color:var(--muted);font-weight:600}
.recent{list-style:none;padding:0;margin:0;font-size:13px}.recent li{padding:6px 0;border-bottom:1px solid var(--line)}
.mark{display:inline-block;width:1.5em;font-weight:800}.mark.ok{color:var(--ok)}.mark.no{color:var(--no)}.mark.half{color:var(--half)}
.kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:10px 0}
.kpis div{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:12px}
.kpis strong{display:block;font-size:22px;letter-spacing:-.01em}.kpis span{color:var(--muted);font-size:12px}
footer{margin-top:32px;color:var(--muted);font-size:12px}
"""


def render(con: sqlite3.Connection, today: date | None = None) -> str:
    today = today or date.today()
    model = current_model(con)
    pred = upcoming_predictions(con, today, model)
    names = teams.display_names(con)
    crests = crest_ids(con)
    impact, key = absence_impact(con, pred) if len(pred) else ({}, set())
    eu = {t: np.sort(g.date.to_numpy()) for t, g in europe.team_dates(con).groupby("team")}
    ctx = Ctx(names, crests, absences(con), impact, key, eu)
    sc_all = predict.scorecard(con)
    sc = sc_all[sc_all.model == model] if len(sc_all) else sc_all
    snap = con.execute("SELECT MAX(snapshot_date) FROM availability_snapshots").fetchone()[0]

    shown_teams = set(pred.home_id) | set(pred.away_id)
    img_css = [_img_css(f"c{crests[t]}", IMG_DIR / transfermarkt.crest_name(crests[t]))
               for t in sorted(shown_teams) if t in crests]
    img_css += [_img_css(f"lg-{lg}", IMG_DIR / transfermarkt.logo_name(lg)) for lg in config.TOP_LEAGUES]

    nav, sections = [], []
    for lg in config.TOP_LEAGUES:
        g = pred[pred.league == lg]
        if g.empty:
            continue
        nav.append(f'<a href="#{lg}"><span class="logo lg-{lg}"></span>{escape(config.LEAGUES[lg].name)}</a>')
        sections.append(league_section(lg, g, ctx))
    nav += ['<a class="plain" href="#bilanz">Bilanz</a>', '<a class="plain" href="#modell">Einordnung</a>']
    if sections:
        views = (f'<div class="view v-liga"><nav class="chips">{"".join(nav)}</nav>{"".join(sections)}</div>'
                 f'<div class="view v-tag">{day_view(pred, ctx)}</div>'
                 f'<div class="view v-top">{top_view(pred, ctx)}</div>')
    else:
        views = "<p class='muted'>Keine Vorhersagen vorhanden – erst <code>fussball predict</code> ausführen.</p>"

    created = (datetime.fromisoformat(pred.created_at.max()).astimezone(TZ).strftime("%d.%m.%Y, %H:%M")
               if len(pred) else "–")
    results_until = ((pd.Timestamp(pred.cutoff_date.max()) - pd.Timedelta(days=1)).strftime("%d.%m.")
                     if len(pred) else "–")
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">
<title>Nächster Spieltag</title><style>{CSS}{''.join(img_css)}</style></head><body><main>
<div class="top"><h1>Nächster Spieltag</h1><p>Stand {created} · Ergebnisse bis {results_until} ·
Ausfälle vom {pd.Timestamp(snap).strftime('%d.%m.') if snap else '–'}</p></div>
<input type="radio" name="view" id="t-liga" class="vt" checked><input type="radio" name="view" id="t-tag" class="vt">
<input type="radio" name="view" id="t-top" class="vt">
<div class="tabs"><label for="t-liga">Ligen</label><label for="t-tag">Tage</label><label for="t-top">Top-Tipps</label></div>
<div class="views">{views}</div>
{scorecard_section(sc, names, ab_test(sc_all, model))}
{backtest_section(model_comparison(), model)}
<footer>Tipp = realistischer Tipp: erst Sieg/Remis/Niederlage, dann das wahrscheinlichste Ergebnis
dazu. Modell: {escape(describe_model(model))}. Wettquoten fließen nicht ein.
Wappen und Logos: Transfermarkt, nur zur privaten Nutzung.</footer>
</main></body></html>"""


def build(con: sqlite3.Connection, out: Path | None = None, today: date | None = None) -> Path:
    out = Path(out or OUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(con, today), encoding="utf-8")
    return out
