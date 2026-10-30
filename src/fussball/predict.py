"""Live-Vorhersage des nächsten Spieltags jeder Top-5-Liga.

Ablauf: Daten der laufenden Saison auffrischen -> je Liga nächsten Spieltag aus dem
Spielplan bestimmen -> Dixon-Coles mit allen Ergebnissen vor dem Cutoff (= heute) fitten
-> Ergebnismatrix, argmax-Tipp und 1X2 speichern (Tabelle `predictions`) und als JSON
für das Dashboard exportieren.

Ergebnisse: football-data.co.uk hinkt einige Tage hinterher; Spiele, die dort noch
fehlen, werden aus Understat ergänzt (dort am selben Tag verfügbar).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from fussball import config, db
from fussball.backtest import metrics, walkforward
from fussball.backtest.run import LGBM_FILE, TIP_WEIGHTS_FILE, load_params, model_name
from fussball.features import availability, form
from fussball.ingest import europe
from fussball.models import lgbm, tipping
from fussball.models.blend import EnsembleDixonColes
from fussball.models.dixon_coles import score_matrix
from fussball.models.lgbm import LGBMCorrector

MAX_SPAN_DAYS = 4  # ein Spieltag erstreckt sich höchstens über Fr–Di
EXPORT_FILE = config.REPORT_DIR / "next_matchday.json"


def next_matchday(fixtures: pd.DataFrame, today: date) -> pd.DataFrame:
    """Nächster Spieltag einer Liga: ab dem ersten Spiel am/nach `today` alle folgenden
    Spiele, solange kein Team doppelt vorkommt und der Spieltag ≤ MAX_SPAN_DAYS dauert."""
    fx = fixtures[fixtures.date >= today.isoformat()].sort_values(["kickoff_utc", "understat_match_id"])
    if fx.empty:
        return fx
    first = pd.Timestamp(fx.date.iloc[0])
    seen, keep = set(), []
    for r in fx.itertuples():
        if (pd.Timestamp(r.date) - first).days > MAX_SPAN_DAYS or {r.home_id, r.away_id} & seen:
            break
        seen |= {r.home_id, r.away_id}
        keep.append(r.Index)
    return fx.loc[keep]


def live_fit_data(con: sqlite3.Connection) -> pd.DataFrame:
    """Ergebnisse aus `matches` plus neuere Top-5-Ergebnisse, die nur Understat schon hat."""
    data = walkforward.load_fit_data(con)
    last = data.groupby("league").date.max()
    extra = db.read(con, "SELECT league, season, date, team_id AS home, opponent_id AS away, "
                         "goals AS hg, goals_against AS ag, xg AS xg_h, xga AS xg_a, "
                         f"{walkforward.STYLE_SQL} FROM team_xg xh "
                         "WHERE is_home = 1 AND match_id IS NULL")
    if extra.empty:
        return data
    extra["date"] = pd.to_datetime(extra.date)
    newest = pd.to_datetime(extra.league.map(last))
    extra = extra[newest.isna() | (extra.date > newest)]
    extra["tier"] = extra.league.map(lambda c: config.LEAGUES[c].tier)
    extra["country"] = extra.league.map(lambda c: config.LEAGUES[c].country)
    return pd.concat([data, extra.assign(match_id=pd.NA)], ignore_index=True)


def refresh(con: sqlite3.Connection, max_age_h: float = 3.0) -> list[str]:
    """Holt die Dateien der laufenden Saison neu, falls älter als `max_age_h` Stunden.

    Fehler einzelner Quellen (z.B. kein Netz) brechen nicht ab – dann wird mit dem Cache
    vorhergesagt. Liefert die Liste der fehlgeschlagenen Quellen.
    """
    import logging
    from fussball import teams
    from fussball.ingest import football_data, other_leagues, understat, understat_matches
    season = [config.current_season()]
    failed = []
    for name, step in (("football-data", lambda: football_data.ingest(con, seasons=season, max_age_h=max_age_h)),
                       ("understat", lambda: (teams.sync_seed_map(con),
                                              understat.ingest(con, seasons=season, max_age_h=max_age_h))),
                       ("europapokal", lambda: (europe.ingest(con, seasons=season),
                                                europe.fetch_images(con, season[0]))),
                       ("weitere Ligen", lambda: other_leagues.ingest(con, seasons=season)),
                       ("Spielberichte", lambda: understat_matches.ingest(con))):
        try:
            step()
        except Exception as e:  # noqa: BLE001 – Netz/Quelle nicht erreichbar: mit Cache weiter
            logging.getLogger(__name__).warning("Auffrischen %s fehlgeschlagen: %s", name, e)
            failed.append(name)
    return failed


def lgbm_correct(data: pd.DataFrame, base: pd.DataFrame, corrector: LGBMCorrector) -> pd.DataFrame:
    """Wendet die LightGBM-Korrektur auf die DC-Raten der anstehenden Spiele an.

    Die Spiele werden als Zeilen ohne Ergebnis an die Historie gehängt; die Features
    entstehen mit derselben Funktion wie im Backtest (nur Spiele vor dem Spieltag).
    """
    pending = pd.DataFrame({
        "match_id": -base.understat_match_id.to_numpy(), "league": base.league.to_numpy(),
        "season": base.season.to_numpy(), "date": pd.to_datetime(base.date).to_numpy(),
        "home": base.home_id.to_numpy(), "away": base.away_id.to_numpy(),
        "hg": np.nan, "ag": np.nan, "xg_h": np.nan, "xg_a": np.nan, "tier": 1,
        "country": base.league.map(lambda c: config.LEAGUES[c].country).to_numpy()})
    feats = form.build(pd.concat([data, pending], ignore_index=True))
    rows = lgbm.to_team_rows(feats[feats.match_id < 0],
                             base.assign(match_id=-base.understat_match_id)[["match_id", "lam_h", "lam_a"]])
    rates = lgbm.match_rates(rows, corrector.rates(rows)).set_index("match_id")
    ids = -base.understat_match_id
    return base.assign(lam_h=ids.map(rates.lam_h).to_numpy(), lam_a=ids.map(rates.lam_a).to_numpy())


def predict(con: sqlite3.Connection, today: date | None = None,
            leagues: list[str] | None = None) -> pd.DataFrame:
    today = today or date.today()
    params = load_params()
    corrector = LGBMCorrector.load(LGBM_FILE)
    name = model_name(params, with_lgbm=corrector is not None)
    cutoff = pd.Timestamp(today)  # Fit nutzt nur Ergebnisse mit date < heute
    data = live_fit_data(con)
    fixtures = db.read(con, "SELECT * FROM fixtures")
    names = dict(con.execute("SELECT team_id, name FROM teams").fetchall())
    now = datetime.now(timezone.utc)
    created = now.isoformat(timespec="seconds")
    # Live-Lauf: angepfiffene Spiele nicht neu tippen – sonst überschreibt ein später Lauf am
    # Spieltag (verspäteter Job, manueller Refresh) den Tipp von vor dem Anpfiff.
    started = (lambda ko: pd.Timestamp(ko, tz="UTC") <= now) if today == date.today() else (lambda ko: False)

    base = []
    for league in leagues or config.TOP_LEAGUES:
        md = next_matchday(fixtures[fixtures.league == league], today)
        if md.empty:
            continue
        country = config.LEAGUES[league].country
        model = EnsembleDixonColes.from_params(params).fit(data[data.country == country], cutoff)
        for f in md.itertuples():
            if started(f.kickoff_utc):
                continue
            lam, mu = model.rates(f.home_id, f.away_id)
            base.append({
                "understat_match_id": f.understat_match_id, "league": league, "season": f.season,
                "kickoff_utc": f.kickoff_utc, "date": f.date, "home_id": f.home_id, "away_id": f.away_id,
                "lam_h": lam, "lam_a": mu, "rho": model.rho,
                "unknown_team": not {f.home_id, f.away_id} <= set(model.params_["attack"]),
            })
    if not base:
        return pd.DataFrame()
    out = pd.DataFrame(base)
    if corrector is not None:
        out = lgbm_correct(data, out, corrector)
    variants = [(name, out), (name + AVAIL_SUFFIX, with_absences(con, out))]
    stored = [finish(con, v, n, created, today) for n, v in variants]
    from fussball import season
    season.project(con, data.dropna(subset=["hg"]), data, params, today)
    result = stored[-1]
    result["home"] = result.home_id.map(names)
    result["away"] = result.away_id.map(names)
    export(result, today, params)
    return result


AVAIL_SUFFIX = " +ausfaelle"


def with_absences(con: sqlite3.Connection, out: pd.DataFrame) -> pd.DataFrame:
    """Passt die Raten an die Ausfälle aus dem letzten Snapshot vor dem jeweiligen Spieltag an."""
    adj = out.copy()
    for (d, season), idx in adj.groupby(["date", "season"]).groups.items():
        per_team, _ = availability.team_absences(con, d, int(season))
        info = per_team.set_index("team_id").to_dict("index")
        for i in idx:
            r = adj.loc[i]
            adj.loc[i, ["lam_h", "lam_a"]] = availability.adjust_rates(
                r.lam_h, r.lam_a, info.get(r.home_id, {}), info.get(r.away_id, {}))
    return adj


def finish(con: sqlite3.Connection, out: pd.DataFrame, name: str, created: str, today: date) -> pd.DataFrame:
    """Matrix, 1X2 und Tipps berechnen und als Modell `name` speichern."""
    weights = tipping.load_weights(TIP_WEIGHTS_FILE)
    rows = []
    for r in out.itertuples():
        m = score_matrix(r.lam_h, r.lam_a, r.rho)
        ph, pd_, pa = metrics.outcome_probs(m)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        ti, tj = tipping.tip(m, weights)
        rows.append({"p_h": ph, "p_d": pd_, "p_a": pa, "pred_h": int(i), "pred_a": int(j),
                     "p_pred": float(m[i, j]), "tip_h": ti, "tip_a": tj, "p_tip": float(m[ti, tj]),
                     "matrix": json.dumps(np.round(m, 6).tolist())})
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1).assign(
        created_at=created, model=name, cutoff_date=today.isoformat())
    db.upsert(con, "predictions", out.drop(columns=["unknown_team"]),
              key=["model", "understat_match_id", "cutoff_date"])
    return out


def export(pred: pd.DataFrame, today: date, params: dict) -> None:
    """JSON für das Dashboard: pro Liga der nächste Spieltag mit Matrix und 1X2."""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pred.created_at.iloc[0], "cutoff_date": today.isoformat(),
        "model": pred.model.iloc[0], "params": params,
        "leagues": {
            lg: {"name": config.LEAGUES[lg].name, "matches": [
                {k: (json.loads(r["matrix"]) if k == "matrix" else r[k]) for k in (
                    "understat_match_id", "kickoff_utc", "home", "away", "tip_h", "tip_a", "p_tip",
                    "pred_h", "pred_a", "p_pred",
                    "p_h", "p_d", "p_a", "lam_h", "lam_a", "matrix", "unknown_team")}
                for r in g.sort_values("kickoff_utc").to_dict("records")]}
            for lg, g in pred.groupby("league")},
    }
    EXPORT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def scorecard(con: sqlite3.Connection) -> pd.DataFrame:
    """Echte Trefferbilanz: je Spiel die letzte Vorhersage vor dem Spieltag gegen das Ergebnis."""
    p = db.read(con, """
        SELECT p.*, x.goals AS hg, x.goals_against AS ag FROM predictions p
        JOIN team_xg x ON x.understat_match_id = p.understat_match_id AND x.is_home = 1
        WHERE p.cutoff_date <= p.date""")
    if p.empty:
        return p
    p = p.sort_values("cutoff_date").drop_duplicates(["model", "understat_match_id"], keep="last")
    for k, v in metrics.hits(p.pred_h, p.pred_a, p.hg, p.ag).items():
        p[k] = v
    # realistischer Tipp (ältere Vorhersagen ohne Tipp: wahrscheinlichstes Ergebnis)
    p["tip_h"] = p.tip_h.fillna(p.pred_h).astype(int)
    p["tip_a"] = p.tip_a.fillna(p.pred_a).astype(int)
    for k, v in metrics.hits(p.tip_h, p.tip_a, p.hg, p.ag).items():
        p[f"tip_{k}"] = v
    p["points"] = metrics.tippspiel_points(p.tip_h, p.tip_a, p.hg, p.ag)
    p["logloss"] = [-np.log(max(json.loads(m)[min(h, 9)][min(a, 9)], 1e-12))
                    for m, h, a in zip(p.matrix, p.hg, p.ag)]
    return p.drop(columns=["matrix"])
