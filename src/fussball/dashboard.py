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
from fussball.ingest import transfermarkt

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


# --- Daten ------------------------------------------------------------------------------

def current_model(con: sqlite3.Connection) -> str | None:
    """Modell des jüngsten Predict-Laufs."""
    row = con.execute("SELECT model FROM predictions ORDER BY created_at DESC LIMIT 1").fetchone()
    return row[0] if row else None


def upcoming_predictions(con: sqlite3.Connection, today: date, model: str | None) -> pd.DataFrame:
    """Jüngste Vorhersage des aktuellen Modells je noch nicht gespieltem Spiel."""
    p = db.read(con, """SELECT * FROM predictions WHERE model = :m AND date >= :d
        AND cutoff_date = (SELECT MAX(cutoff_date) FROM predictions WHERE model = :m)""",
                {"m": model, "d": today.isoformat()})
    return p.sort_values(["league", "kickoff_utc", "home_id"])


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
    return (f"Dixon-Coles aus Toren und xG (xG-Gewicht {float(p['w_xg']):.0%}, Halbwertszeit "
            f"Tore {half(p['xi'])}, xG {half(p['xi_xg'])})").replace("%", " %")


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


def matrix_table(m: np.ndarray, home: str, away: str, tip: tuple[int, int]) -> str:
    n = MATRIX_SHOW
    sub = m[:n, :n]
    pmax = sub.max()
    head = "".join(f"<th>{j}</th>" for j in range(n))
    rows = []
    for i in range(n):
        cells = "".join(
            f"<td{' class=tip' if (i, j) == tip else ''}{_cell_style(sub[i, j], pmax)}>"
            f"{_num(100 * sub[i, j]) if sub[i, j] >= 0.0005 else ''}</td>" for j in range(n))
        rows.append(f"<tr><th>{i}</th>{cells}</tr>")
    return (f'<table class="matrix"><thead><tr><th class="corner">↓ {escape(home)}<br>→ {escape(away)}</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f'<p class="muted small">Wahrscheinlichkeit je Ergebnis in %, umrandet = Tipp</p>')


def outcome_bar(ph: float, pd_: float, pa: float) -> str:
    return (f'<div class="bar" role="img" aria-label="Heimsieg {_pct(ph)}, Remis {_pct(pd_)}, '
            f'Auswärtssieg {_pct(pa)}"><span class="h" style="width:{100 * ph:.1f}%"></span>'
            f'<span class="d" style="width:{100 * pd_:.1f}%"></span>'
            f'<span class="a" style="width:{100 * pa:.1f}%"></span></div>'
            f'<div class="barlabels"><span>Heim {_pct(ph)}</span><span>Remis {_pct(pd_)}</span>'
            f'<span>Auswärts {_pct(pa)}</span></div>')


def absence_list(ab: pd.DataFrame, team_id: str, team: str) -> str:
    a = ab[ab.team_id == team_id]
    if a.empty:
        return f"<div><p class='abs-team'>{escape(team)}</p><p class='muted small'>keine gemeldeten Ausfälle</p></div>"
    items = []
    for r in a.itertuples():
        pos = f"<span class='muted'> · {escape(r.position)}</span>" if pd.notna(r.position) else ""
        until = f" <span class='muted'>(bis {pd.Timestamp(r.until):%d.%m.})</span>" if pd.notna(r.until) else ""
        items.append(f"<li><strong>{escape(r.player)}</strong>{pos}<br>{escape(r.reason)}{until}</li>")
    return (f"<div><p class='abs-team'>{escape(team)} <span class='muted'>({len(a)})</span></p>"
            f"<ul>{''.join(items)}</ul></div>")


def match_card(r, names: dict, crests: dict, ab: pd.DataFrame, provisional: bool) -> str:
    day, time = _kickoff(r.kickoff_utc)
    home, away = names.get(r.home_id, r.home_id), names.get(r.away_id, r.away_id)
    m = np.array(json.loads(r.matrix))
    n_home = int((ab.team_id == r.home_id).sum())
    n_away = int((ab.team_id == r.away_id).sum())
    uid = r.understat_match_id
    return f"""
<article class="card">
  <div class="meta"><span>{day} · {time}{'*' if provisional else ''}</span>
    <span class="chance">Tipp trifft zu {_pct(r.p_pred, 1)}</span></div>
  <div class="fixture">
    <div class="side">{crest(r.home_id, crests, home)}<span class="team">{escape(home)}</span></div>
    <div class="score"><span class="tip">{r.pred_h}:{r.pred_a}</span>
      <span class="xg">erw. Tore {_num(r.lam_h)} : {_num(r.lam_a)}</span></div>
    <div class="side">{crest(r.away_id, crests, away)}<span class="team">{escape(away)}</span></div>
  </div>
  {outcome_bar(r.p_h, r.p_d, r.p_a)}
  <input type="checkbox" id="m{uid}" class="tg tg-m"><input type="checkbox" id="a{uid}" class="tg tg-a">
  <div class="toggles"><label for="m{uid}" class="lm">Ergebnismatrix</label>
    <label for="a{uid}" class="la">Ausfälle {n_home} · {n_away}</label></div>
  <div class="panel pm">{matrix_table(m, home, away, (r.pred_h, r.pred_a))}</div>
  <div class="panel pa"><div class="abs">{absence_list(ab, r.home_id, home)}{absence_list(ab, r.away_id, away)}</div></div>
</article>"""


def league_section(lg: str, g: pd.DataFrame, names, crests, ab) -> str:
    th = THEMES[lg]
    name = config.LEAGUES[lg].name
    # Understat setzt nicht terminierte Spiele auf eine gemeinsame Platzhalter-Uhrzeit
    provisional = g.kickoff_utc.value_counts().max() >= 6
    days = list(dict.fromkeys(_kickoff(k)[0] for k in sorted(g.kickoff_utc)))
    span = days[0] if len(days) == 1 else f"{days[0]} – {days[-1]}"
    note = "<p class='muted small note'>* Anstoßzeiten noch nicht terminiert</p>" if provisional else ""
    cards = "".join(match_card(r, names, crests, ab, provisional) for r in g.itertuples())
    return f"""<section id="{lg}" class="league" style="--lg:{th.primary};--lg-accent:{th.accent};--lg-ink:{th.ink}">
<header class="lg-head"><span class="logo lg-{lg}" role="img" aria-label="Logo {escape(name)}"></span>
<div><h2>{escape(name)}</h2><p>{span} · {len(g)} Spiele</p></div></header>{note}{cards}</section>"""


def scorecard_section(sc: pd.DataFrame, names: dict) -> str:
    if sc.empty:
        return ("<section id='bilanz' class='plain'><h2>Live-Bilanz</h2><p class='muted'>Noch keine "
                "ausgewerteten Spiele – die Bilanz füllt sich, sobald vorhergesagte Spiele gespielt sind.</p></section>")
    rows = []
    for lg, g in [("Alle Ligen", sc)] + [(config.LEAGUES[k].name, v) for k, v in sc.groupby("league")]:
        rows.append(f"<tr><td>{escape(lg)}</td><td>{len(g)}</td><td>{_pct(g.exact.mean(), 1)}</td>"
                    f"<td>{_pct(g['diff'].mean(), 1)}</td><td>{_pct(g.tendency.mean(), 1)}</td></tr>")
    recent = "".join(
        f"<li><span class='mark {'ok' if r.exact else ('half' if r.tendency else 'no')}'>"
        f"{'✓' if r.exact else ('≈' if r.tendency else '✗')}</span>"
        f"{escape(names.get(r.home_id, r.home_id))} – {escape(names.get(r.away_id, r.away_id))}"
        f"<span class='muted'> Tipp {r.pred_h}:{r.pred_a} · Ergebnis {r.hg}:{r.ag}</span></li>"
        for r in sc.sort_values("kickoff_utc").tail(10).iloc[::-1].itertuples())
    return f"""<section id="bilanz" class="plain"><h2>Live-Bilanz</h2>
<table class="stats"><thead><tr><th>Liga</th><th>Spiele</th><th>Exakt</th><th>Tordiff.</th><th>Tendenz</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<input type="checkbox" id="recent" class="tg tg-r"><div class="toggles"><label for="recent" class="lr">Letzte 10 Spiele</label></div>
<div class="panel pr"><ul class="recent">{recent}</ul><p class="muted small">✓ exakt · ≈ Tendenz richtig · ✗ falsch</p></div></section>"""


def backtest_section(cmp: pd.DataFrame | None, model: str | None) -> str:
    if cmp is None or "aktuell" not in cmp.index:
        return ""
    a, b = cmp.loc["aktuell"], cmp.loc["Baseline (nur Tore)"]
    return f"""<section id="modell" class="plain"><h2>Einordnung</h2>
<p class="muted small">{escape(describe_model(model))}. Backtest ab 2019/20 · {_num(a['n'], 0)} Spiele,
jeweils nur mit Daten bis zum Vortag.</p>
<div class="kpis">
  <div><strong>{_num(a['exact_%'])} %</strong><span>exakte Treffer</span></div>
  <div><strong>{_num(a['tend_%'])} %</strong><span>Tendenz richtig</span></div>
  <div><strong>{_num(b['exact_%'])} %</strong><span>bisheriges Modell (nur Tore)</span></div>
  <div><strong>{_num(a['exact_1:1_%'])} %</strong><span>zum Vergleich: immer 1:1</span></div>
</div>
<p class="muted small">Log-Loss {_num(a['logloss'], 3)} (vorher {_num(b['logloss'], 3)}) · RPS {_num(a['rps'], 3)}
(vorher {_num(b['rps'], 3)}, Wettmarkt {_num(a['rps_odds'], 3)}) – jeweils kleiner = besser.</p></section>"""


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
nav{display:flex;gap:8px;overflow-x:auto;padding:12px 0;position:sticky;top:0;background:var(--bg);z-index:2;scrollbar-width:none}
nav::-webkit-scrollbar{display:none}
nav a{flex:none;display:flex;align-items:center;gap:6px;padding:5px 12px 5px 5px;border-radius:999px;
color:var(--text);text-decoration:none;font-size:13px;font-weight:600;background:var(--surface);border:1px solid var(--line)}
nav a .logo{width:24px;height:24px;border-radius:50%}nav a.plain{padding-left:12px}
.logo{display:inline-block;background:#fff center/80% no-repeat;flex:none}
.league{margin-top:18px}section{scroll-margin-top:60px}
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
.matrix td.tip{outline:2px solid var(--text);outline-offset:-2px;font-weight:800}
.abs{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.abs-team{font-weight:700;margin:0 0 4px;font-size:13px}
.abs ul{margin:0;padding:0;list-style:none;font-size:12px}.abs li{padding:4px 0;border-top:1px solid var(--line)}
.muted{color:var(--muted)}.small{font-size:12px}
.plain h2{font-size:19px;margin:30px 0 8px}
.stats{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;background:var(--surface);
border-radius:12px;overflow:hidden}
.stats th,.stats td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line)}
.stats th:first-child,.stats td:first-child{text-align:left}.stats th{font-size:12px;color:var(--muted);font-weight:600}
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
    ab = absences(con)
    sc = predict.scorecard(con)
    sc = sc[sc.model == model] if len(sc) else sc
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
        sections.append(league_section(lg, g, names, crests, ab))
    nav += ['<a class="plain" href="#bilanz">Bilanz</a>', '<a class="plain" href="#modell">Einordnung</a>']
    if not sections:
        sections.append("<p class='muted'>Keine Vorhersagen vorhanden – erst <code>fussball predict</code> ausführen.</p>")

    created = (datetime.fromisoformat(pred.created_at.max()).astimezone(TZ).strftime("%d.%m.%Y, %H:%M")
               if len(pred) else "–")
    results_until = ((pd.Timestamp(pred.cutoff_date.max()) - pd.Timedelta(days=1)).strftime("%d.%m.")
                     if len(pred) else "–")
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">
<title>Nächster Spieltag</title><style>{CSS}{''.join(img_css)}</style></head><body><main>
<div class="top"><h1>Nächster Spieltag</h1><p>Stand {created} · Ergebnisse bis {results_until} ·
Ausfälle vom {pd.Timestamp(snap).strftime('%d.%m.') if snap else '–'}</p></div>
<nav>{''.join(nav)}</nav>
{''.join(sections)}
{scorecard_section(sc, names)}
{backtest_section(model_comparison(), model)}
<footer>Tipp = wahrscheinlichstes Ergebnis laut Modell: {escape(describe_model(model))}.
Wettquoten fließen nicht ein.
Wappen und Logos: Transfermarkt, nur zur privaten Nutzung.</footer>
</main></body></html>"""


def build(con: sqlite3.Connection, out: Path | None = None, today: date | None = None) -> Path:
    out = Path(out or OUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(con, today), encoding="utf-8")
    return out
