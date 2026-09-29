from datetime import date

import pandas as pd

from fussball import db, predict
from fussball.ingest import football_data as fd
from tests.conftest import make_csv


def fixtures(rows):
    return pd.DataFrame(rows, columns=["understat_match_id", "kickoff_utc", "date", "home_id", "away_id"])


def test_next_matchday_stops_at_repeated_team_and_span():
    fx = fixtures([
        (1, "2026-10-10 13:30:00", "2026-10-10", "a", "b"),
        (2, "2026-10-10 13:30:00", "2026-10-10", "c", "d"),
        (3, "2026-10-11 15:30:00", "2026-10-11", "e", "f"),
        (4, "2026-10-17 13:30:00", "2026-10-17", "g", "h"),   # > MAX_SPAN_DAYS
        (5, "2026-10-12 19:00:00", "2026-10-12", "a", "c"),   # nächste Runde (a spielt schon)
        (0, "2026-09-01 13:30:00", "2026-09-01", "x", "y"),   # Vergangenheit
    ])
    assert predict.next_matchday(fx, date(2026, 9, 29)).understat_match_id.tolist() == [1, 2, 3]


def test_live_fit_data_adds_only_newer_understat_results(con):
    m, o = fd.parse(make_csv(year=2024), "D1", 2024)
    fd.store(con, m, o)
    last = pd.Timestamp(m.date.max())
    rows = []
    for i, d in enumerate([last - pd.Timedelta(days=3), last + pd.Timedelta(days=2)]):
        rows.append({"understat_match_id": i, "team_id": "alpha", "match_id": None, "league": "D1",
                     "season": 2024, "date": d.date().isoformat(), "is_home": 1,
                     "opponent_id": "beta", "goals": 5, "goals_against": 0})
    db.upsert(con, "team_xg", rows, key=["understat_match_id", "team_id"])
    data = predict.live_fit_data(con)
    extra = data[data.match_id.isna()]
    assert len(data) == len(m) + 1
    assert extra.date.tolist() == [last + pd.Timedelta(days=2)]
    assert (extra.tier == 1).all() and (extra.country == "GER").all()


def test_scorecard_uses_last_prediction_before_match(con):
    import json

    import numpy as np
    from fussball import teams
    teams.register_canonical(con, {"Alpha": "GER", "Beta": "GER"})
    m = np.zeros((10, 10)); m[1, 1] = 0.5; m[2, 1] = 0.5
    base = {"model": "dc", "understat_match_id": 7, "league": "D1", "season": 2026,
            "kickoff_utc": "2026-10-10 13:30:00", "date": "2026-10-10", "home_id": "alpha",
            "away_id": "beta", "created_at": "x", "matrix": json.dumps(m.tolist())}
    db.upsert(con, "predictions", [
        {**base, "cutoff_date": "2026-10-01", "pred_h": 1, "pred_a": 1},
        {**base, "cutoff_date": "2026-10-10", "pred_h": 2, "pred_a": 1},   # letzte vor Anstoß
        {**base, "cutoff_date": "2026-10-11", "pred_h": 0, "pred_a": 0},   # nach dem Spiel: ignorieren
    ], key=["model", "understat_match_id", "cutoff_date"])
    db.upsert(con, "team_xg", [{"understat_match_id": 7, "team_id": "alpha", "league": "D1", "season": 2026,
                                "date": "2026-10-10", "is_home": 1, "opponent_id": "beta",
                                "goals": 2, "goals_against": 1}], key=["understat_match_id", "team_id"])
    sc = predict.scorecard(con)
    assert len(sc) == 1 and sc.cutoff_date.iloc[0] == "2026-10-10"
    assert bool(sc.exact.iloc[0]) and abs(sc.logloss.iloc[0] - np.log(2)) < 1e-9
