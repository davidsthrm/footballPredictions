"""Statisches Dashboard (eine HTML-Datei, ohne JavaScript, mobil-first).

Im Zentrum steht der nächste Spieltag je Liga: Tipp, Wahrscheinlichkeit des Tipps, 1X2,
erwartete Tore und aufklappbare Ergebnismatrix. Dazu Ausfälle aus dem letzten
Transfermarkt-Snapshot, die Live-Trefferbilanz und Backtest-Kennzahlen als Einordnung.

Ohne JavaScript, damit die Datei auch in der Vorschau der iOS-Dateien-App (iCloud Drive)
funktioniert. Aufklappen über <details>.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fussball import config, db, predict

OUT_FILE = config.DATA_DIR / "dashboard" / "index.html"
TZ = ZoneInfo("Europe/Berlin")
WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
MATRIX_SHOW = 6  # 0..5 Tore je Team in der Matrix
# Sequentielle Blau-Rampe (hell -> dunkel) für die Matrix
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


# --- Daten ------------------------------------------------------------------------------

def upcoming_predictions(con: sqlite3.Connection, today: date) -> pd.DataFrame:
    """Jüngste Vorhersage je noch nicht gespieltem Spiel aus dem letzten Predict-Lauf."""
    p = db.read(con, """
        SELECT p.*, th.name AS home, ta.name AS away FROM predictions p
        JOIN teams th ON th.team_id = p.home_id JOIN teams ta ON ta.team_id = p.away_id
        WHERE p.cutoff_date = (SELECT MAX(cutoff_date) FROM predictions) AND p.date >= ?""",
                (today.isoformat(),))
    return p.sort_values(["league", "kickoff_utc", "home"])


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


def backtest_headline() -> pd.DataFrame | None:
    f = config.REPORT_DIR / "backtest_summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f, dtype={"season": str})
    return s[s.season == "TEST"].set_index("league")


# --- Formatierung -------------------------------------------------------------------------

def _kickoff(utc: str) -> tuple[str, str]:
    dt = datetime.fromisoformat(utc).replace(tzinfo=timezone.utc).astimezone(TZ)
    return f"{WEEKDAYS[dt.weekday()]} {dt:%d.%m.}", f"{dt:%H:%M}"


def _num(x: float, digits: int = 1) -> str:
    """Deutsches Zahlenformat: 1.234,5"""
    return f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _pct(p: float, digits: int = 0) -> str:
    return f"{_num(100 * p, digits)} %"


def _cell_style(p: float, pmax: float) -> str:
    if p < 0.005:
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
        cells = []
        for j in range(n):
            cls = ' class="tip"' if (i, j) == tip else ""
            label = _num(100 * sub[i, j]) if sub[i, j] >= 0.0005 else ""
            cells.append(f"<td{cls}{_cell_style(sub[i, j], pmax)}>{label}</td>")
        rows.append(f"<tr><th>{i}</th>{''.join(cells)}</tr>")
    return (f'<table class="matrix"><caption>Wahrscheinlichkeit in % · Zeilen: Tore {escape(home)}, '
            f'Spalten: Tore {escape(away)}</caption><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def outcome_bar(ph: float, pd_: float, pa: float) -> str:
    return (f'<div class="bar" role="img" aria-label="Heimsieg {_pct(ph)}, Remis {_pct(pd_)}, '
            f'Auswärtssieg {_pct(pa)}"><span class="h" style="width:{100 * ph:.1f}%"></span>'
            f'<span class="d" style="width:{100 * pd_:.1f}%"></span>'
            f'<span class="a" style="width:{100 * pa:.1f}%"></span></div>'
            f'<div class="barlabels"><span>1 · {_pct(ph)}</span><span>X · {_pct(pd_)}</span>'
            f'<span>2 · {_pct(pa)}</span></div>')


def absence_list(ab: pd.DataFrame, team_id: str, team: str) -> str:
    a = ab[ab.team_id == team_id]
    if a.empty:
        return f"<p class='muted'>{escape(team)}: keine gemeldeten Ausfälle</p>"
    items = []
    for r in a.itertuples():
        pos = f" · {escape(r.position)}" if pd.notna(r.position) else ""
        reason = escape(r.reason)
        until = f" (bis {pd.Timestamp(r.until):%d.%m.})" if pd.notna(r.until) else ""
        items.append(f"<li>{escape(r.player)}{pos} – {reason}{until}</li>")
    items = "".join(items)
    return f"<p><strong>{escape(team)}</strong> ({len(a)})</p><ul>{items}</ul>"


def match_card(r, ab: pd.DataFrame, provisional: bool) -> str:
    day, time = _kickoff(r.kickoff_utc)
    m = np.array(json.loads(r.matrix))
    n_home = int((ab.team_id == r.home_id).sum())
    n_away = int((ab.team_id == r.away_id).sum())
    time_label = f"{time}*" if provisional else time
    return f"""
<article class="card">
  <div class="meta"><span>{day} {time_label}</span><span>Ausfälle {n_home} | {n_away}</span></div>
  <div class="fixture">
    <span class="team home">{escape(r.home)}</span>
    <span class="tip">{r.pred_h}:{r.pred_a}</span>
    <span class="team away">{escape(r.away)}</span>
  </div>
  <div class="sub"><span>Tipp-Wahrscheinlichkeit {_pct(r.p_pred, 1)}</span>
    <span>xTore {_num(r.lam_h, 2)} : {_num(r.lam_a, 2)}</span></div>
  {outcome_bar(r.p_h, r.p_d, r.p_a)}
  <details><summary>Ergebnismatrix</summary>{matrix_table(m, r.home, r.away, (r.pred_h, r.pred_a))}</details>
  <details><summary>Ausfälle</summary>{absence_list(ab, r.home_id, r.home)}{absence_list(ab, r.away_id, r.away)}</details>
</article>"""


def scorecard_section(sc: pd.DataFrame, names: dict[str, str]) -> str:
    if sc.empty:
        return ("<section id='bilanz'><h2>Live-Bilanz</h2><p class='muted'>Noch keine ausgewerteten "
                "Spiele. Die Bilanz füllt sich, sobald vorhergesagte Spiele gespielt sind.</p></section>")
    rows = []
    for lg, g in [("Alle", sc)] + [(config.LEAGUES[k].name, v) for k, v in sc.groupby("league")]:
        rows.append(f"<tr><td>{escape(lg)}</td><td>{len(g)}</td><td>{_pct(g.exact.mean(), 1)}</td>"
                    f"<td>{_pct(g['diff'].mean(), 1)}</td><td>{_pct(g.tendency.mean(), 1)}</td></tr>")
    last = sc.sort_values("kickoff_utc").tail(10).iloc[::-1]
    recent = "".join(
        f"<li><span>{'✓' if r.exact else ('≈' if r.tendency else '✗')}</span> "
        f"{escape(names.get(r.home_id, r.home_id))} – {escape(names.get(r.away_id, r.away_id))}: Tipp {r.pred_h}:{r.pred_a}, Ergebnis {r.hg}:{r.ag}</li>"
        for r in last.itertuples())
    return f"""<section id="bilanz"><h2>Live-Bilanz</h2>
<table class="stats"><thead><tr><th>Liga</th><th>Spiele</th><th>Exakt</th><th>Tordiff.</th><th>Tendenz</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<details><summary>Letzte 10 Spiele</summary><ul class="recent">{recent}</ul>
<p class="muted">✓ exakt · ≈ Tendenz richtig · ✗ falsch</p></details></section>"""


def backtest_section(bt: pd.DataFrame | None) -> str:
    if bt is None or "ALL" not in bt.index:
        return ""
    a = bt.loc["ALL"]
    return f"""<section id="modell"><h2>Einordnung (Backtest ab 2019/20)</h2>
<div class="kpis">
  <div><strong>{_num(a['exact_%'])} %</strong><span>exakte Treffer Modell</span></div>
  <div><strong>{_num(a['exact_1:1_%'])} %</strong><span>„immer 1:1“</span></div>
  <div><strong>{_num(a['tend_%'])} %</strong><span>Tendenz richtig</span></div>
  <div><strong>{_num(a['rps'], 3)}</strong><span>RPS Modell (Markt {_num(a['rps_odds'], 3)}, kleiner = besser)</span></div>
</div><p class="muted">{_num(a['n'], 0)} Spiele, Walk-forward ohne Blick in die Zukunft.</p></section>"""


CSS = """
:root{--bg:#f6f6f4;--surface:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--line:#e3e2dc;
--home:#2a78d6;--draw:#9a9993;--away:#eb6834;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--surface:#1a1a19;--text:#fff;--muted:#c3c2b7;
--line:#34332f;--home:#3987e5;--draw:#7c7b75;--away:#d95926;--accent:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px 16px 48px}
header h1{font-size:22px;margin:4px 0}header p{margin:0;color:var(--muted);font-size:13px}
nav{display:flex;gap:8px;overflow-x:auto;padding:12px 0;position:sticky;top:0;background:var(--bg);z-index:1}
nav a{flex:none;padding:6px 12px;border:1px solid var(--line);border-radius:999px;color:var(--text);
text-decoration:none;font-size:13px;background:var(--surface)}
nav{scrollbar-width:none}nav::-webkit-scrollbar{display:none}
h2{font-size:18px;margin:28px 0 4px}h2 small{color:var(--muted);font-weight:400;font-size:13px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0}
.meta,.sub{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}
.fixture{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;margin:6px 0}
.team{font-weight:600;overflow-wrap:anywhere}.home{text-align:right}
.fixture .tip{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;padding:2px 10px;
border-radius:8px;background:var(--bg)}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:10px;gap:2px}
.bar .h{background:var(--home)}.bar .d{background:var(--draw)}.bar .a{background:var(--away)}
.barlabels{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:2px}
details{margin-top:8px}summary{cursor:pointer;color:var(--accent);font-size:13px}
.matrix{border-collapse:collapse;margin-top:8px;font-size:12px;font-variant-numeric:tabular-nums;width:100%}
.matrix caption{caption-side:bottom;color:var(--muted);font-size:11px;text-align:left;padding-top:4px}
.matrix th{color:var(--muted);font-weight:500;padding:3px}.matrix td{text-align:center;padding:5px 2px;
border:2px solid var(--surface)}.matrix td.tip{outline:2px solid var(--text);outline-offset:-2px;font-weight:700}
ul{margin:4px 0 8px;padding-left:18px;font-size:13px}.muted{color:var(--muted);font-size:13px}
.stats{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
.stats th,.stats td{text-align:right;padding:6px 4px;border-bottom:1px solid var(--line)}
.stats th:first-child,.stats td:first-child{text-align:left}
.recent{list-style:none;padding:0}.recent span{display:inline-block;width:1.2em}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}
.kpis div{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.kpis strong{display:block;font-size:20px}.kpis span{color:var(--muted);font-size:12px}
footer{margin-top:32px;color:var(--muted);font-size:12px}
"""


def render(con: sqlite3.Connection, today: date | None = None) -> str:
    today = today or date.today()
    pred = upcoming_predictions(con, today)
    ab = absences(con)
    sc = predict.scorecard(con)
    snap = con.execute("SELECT MAX(snapshot_date) FROM availability_snapshots").fetchone()[0]
    nav, sections = [], []
    for lg in config.TOP_LEAGUES:
        g = pred[pred.league == lg]
        if g.empty:
            continue
        name = config.LEAGUES[lg].name
        # Understat setzt nicht terminierte Spiele auf eine gemeinsame Platzhalter-Uhrzeit
        provisional = g.kickoff_utc.value_counts().max() >= 6
        d0, d1 = g.date.min(), g.date.max()
        span = pd.Timestamp(d0).strftime("%d.%m.") + ("" if d0 == d1 else "–" + pd.Timestamp(d1).strftime("%d.%m."))
        nav.append(f'<a href="#{lg}">{escape(name)}</a>')
        cards = "".join(match_card(r, ab, provisional) for r in g.itertuples())
        note = "<p class='muted'>* Anstoßzeiten noch nicht terminiert</p>" if provisional else ""
        sections.append(f'<section id="{lg}"><h2>{escape(name)} <small>{span}</small></h2>{note}{cards}</section>')
    nav += ['<a href="#bilanz">Bilanz</a>', '<a href="#modell">Modell</a>']
    if not sections:
        sections.append("<p class='muted'>Keine Vorhersagen vorhanden – erst <code>fussball predict</code> ausführen.</p>")
    model = pred.model.iloc[0] if len(pred) else "–"
    created = (datetime.fromisoformat(pred.created_at.max()).astimezone(TZ).strftime("%d.%m.%Y %H:%M")
               if len(pred) else "–")
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">
<title>Nächster Spieltag</title><style>{CSS}</style></head><body><main>
<header><h1>Nächster Spieltag</h1><p>Vorhersage vom {created} · Ergebnisse bis {(pd.Timestamp(pred.cutoff_date.max()) - pd.Timedelta(days=1)).strftime('%d.%m.') if len(pred) else '–'}
 · Ausfälle Stand {pd.Timestamp(snap).strftime('%d.%m.') if snap else '–'}</p></header>
<nav>{''.join(nav)}</nav>
{''.join(sections)}
{scorecard_section(sc, dict(con.execute("SELECT team_id, name FROM teams").fetchall()))}
{backtest_section(backtest_headline())}
<footer>Modell: {escape(model)} · Tipp = wahrscheinlichstes Ergebnis der Dixon-Coles-Matrix ·
Quoten fließen nicht ein.</footer>
</main></body></html>"""


def build(con: sqlite3.Connection, out: Path | None = None, today: date | None = None) -> Path:
    out = Path(out or OUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(con, today), encoding="utf-8")
    return out
