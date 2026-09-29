import json

import numpy as np

from fussball import dashboard, db, teams
from fussball.models.dixon_coles import score_matrix


def test_render_contains_matchday_and_no_script(con):
    teams.register_canonical(con, {"Alpha <FC>": "GER", "Beta": "GER"})
    m = score_matrix(1.6, 1.0, -0.05)
    db.upsert(con, "predictions", [{
        "created_at": "2026-10-09T06:00:00+00:00", "model": "dc", "cutoff_date": "2026-10-09",
        "understat_match_id": 1, "league": "D1", "season": 2026, "kickoff_utc": "2026-10-10 13:30:00",
        "date": "2026-10-10", "home_id": "alpha-fc", "away_id": "beta", "lam_h": 1.6, "lam_a": 1.0,
        "rho": -0.05, "p_h": 0.5, "p_d": 0.25, "p_a": 0.25, "pred_h": 1, "pred_a": 0, "p_pred": 0.12,
        "matrix": json.dumps(m.tolist())}], key=["model", "understat_match_id", "cutoff_date"])
    db.upsert(con, "availability_snapshots", [
        {"snapshot_date": "2026-10-09", "snapshot_ts": "x", "league": "D1", "tm_club_id": 1,
         "team_id": "beta", "tm_player_id": 7, "player": "Kim", "category": c, "reason": r}
        for c, r in (("injury", "Muskelfaserriss"), ("suspension", "Rotsperre"))],
        key=["snapshot_date", "tm_club_id", "tm_player_id", "category"])
    from datetime import date
    html = dashboard.render(con, today=date(2026, 10, 9))
    assert "Alpha &lt;FC&gt;" in html and "<FC>" not in html          # HTML-escaped
    assert "<script" not in html                                        # läuft ohne JavaScript
    assert "Sa 10.10. 15:30" in html                                    # UTC -> Europe/Berlin
    assert "Muskelfaserriss / Rotsperre" in html and "Ausfälle 0 | 1" in html
    assert np.isclose(m.sum(), 1)
