import numpy as np
import pandas as pd
import pytest

from fussball import db, teams
from fussball.features import availability as av
from fussball.features import form
from fussball.ingest import europe

EUROPE_HTML = """<html><body><div class="box"><table><tbody>
<tr class="bg_blau_20"><td>Di. 17.09.2024 18:45</td></tr>
<tr><td>17.09.2024</td><td><a href="/juventus-turin/startseite/verein/506/saison_id/2024">Juventus</a></td>
<td></td><td>3:1</td><td></td><td><a href="/psv-eindhoven/startseite/verein/383/saison_id/2024">PSV</a></td></tr>
<tr class="bg_blau_20"><td>Mi. 18.09.2024 21:00</td></tr>
<tr><td></td><td><a href="/manchester-city/startseite/verein/281/saison_id/2024">Man City</a></td>
<td></td><td>-:-</td><td></td><td><a href="/inter-mailand/startseite/verein/46/saison_id/2024">Inter</a></td></tr>
<tr><td>1</td><td></td><td><a href="/fc-liverpool/spielplan/verein/31/saison_id/2024">Liverpool</a></td><td>8</td><td>21</td></tr>
</tbody></table></div></body></html>"""


def test_europe_parser_dates_and_clubs():
    df = europe.parse(EUROPE_HTML)
    assert df[["date", "tm_home_id", "tm_away_id"]].values.tolist() == [
        ["2024-09-17", 506, 383], ["2024-09-18", 281, 46]]   # Tabellenzeile (1 Verein) ignoriert


def test_europe_features_point_in_time():
    data = pd.DataFrame({"match_id": [1, 2], "league": "D1", "season": 2024, "tier": 1, "country": "GER",
                         "date": pd.to_datetime(["2024-09-14", "2024-09-21"]), "home": ["A", "B"],
                         "away": ["B", "A"], "hg": [1, 0], "ag": [0, 0], "xg_h": np.nan, "xg_a": np.nan})
    eu = pd.DataFrame({"team": ["A"], "date": pd.to_datetime(["2024-09-18"])})
    f = form.build(data, eu).set_index(["match_id", "team"])
    assert f.loc[(1, "A"), "eu_until"] == 4 and f.loc[(1, "A"), "eu_since"] == form.MAX_REST
    assert f.loc[(2, "A"), "eu_since"] == 3 and f.loc[(2, "A"), "eu_14d"] == 1
    assert f.loc[(2, "A"), "rest_all"] == 3 and f.loc[(2, "B"), "eu_14d"] == 0


@pytest.mark.parametrize("tm,expected", [("Vinícius Júnior", 0), ("Christian Nørgaard", 1),
                                         ("Kai Havertz", 2), ("Unbekannter Spieler", None)])
def test_match_player(tm, expected):
    cand = pd.Series(["vinicius junior", "christian norgaard", "kai havertz", "kai hansen"]).map(av._norm)
    assert av.match_player(tm, cand) == expected


def test_adjust_rates_caps_and_direction():
    lh, la = av.adjust_rates(1.6, 1.2, {"missing_att": 0.3, "missing_def": 0.0},
                             {"missing_att": 0.0, "missing_def": 0.5})
    assert lh == pytest.approx(1.6 * (1 - 0.15) * (1 + 0.15))
    assert la == pytest.approx(1.2)
    lh, _ = av.adjust_rates(1.6, 1.2, {"missing_att": 2.0}, {})
    assert lh == pytest.approx(1.6 * (1 - av.MAX_ATTACK_CUT))


def test_team_absences_uses_last_snapshot_before_match(con):
    teams.register_canonical(con, {"Alpha": "GER"})
    db.upsert(con, "player_season_stats", [
        {"season": 2026, "league": "D1", "understat_player_id": i, "player": n, "team_id": "alpha",
         "position": pos, "games": 10, "minutes": mins, "goals": 0, "assists": 0, "xg": xg, "xa": 0.0, "npxg": xg}
        for i, (n, pos, mins, xg) in enumerate([("Max Stürmer", "F S", 900, 6.0), ("Tom Keeper", "GK", 900, 0.0),
                                                 ("Ben Bank", "M S", 100, 0.5)] + [(f"P{k}", "D", 900, 0.5) for k in range(9)])],
        key=["season", "league", "understat_player_id"])
    snap = lambda d, pid, name, until: {"snapshot_date": d, "snapshot_ts": "x", "league": "D1", "tm_club_id": 1,
                                         "team_id": "alpha", "tm_player_id": pid, "player": name,
                                         "category": "injury", "reason": "Verletzung", "until": until}
    db.upsert(con, "availability_snapshots", [
        snap("2026-10-01", 1, "Max Stürmer", None),
        snap("2026-10-01", 2, "Ben Bank", "2026-10-03"),          # vor dem Spiel zurück
        snap("2026-10-12", 3, "Tom Keeper", None)],               # Snapshot nach dem Spieltag
        key=["snapshot_date", "tm_club_id", "tm_player_id", "category"])
    per_team, players = av.team_absences(con, "2026-10-10", 2026)
    assert players.player.tolist() == ["Max Stürmer"]
    assert per_team.missing_att.iloc[0] == pytest.approx(6.0 / 11.0)


def test_squad_values_use_previous_season_only(con):
    from fussball.ingest import squad_values
    teams.register_canonical(con, {"Alpha": "GER", "Beta": "GER"})
    con.executescript(squad_values.SCHEMA)
    db.upsert(con, "team_name_map", [{"source": "transfermarkt_id", "source_name": "1", "team_id": "alpha"},
                                     {"source": "transfermarkt_id", "source_name": "2", "team_id": "beta"}],
              key=["source", "source_name"])
    db.upsert(con, "squad_values_history", [
        {"season": 2023, "league": "D1", "tm_club_id": 1, "total_value_eur": 400e6},
        {"season": 2023, "league": "D1", "tm_club_id": 2, "total_value_eur": 100e6},
        {"season": 2024, "league": "D1", "tm_club_id": 1, "total_value_eur": 900e6}],   # nach Saisonverlauf
        key=["season", "league", "tm_club_id"])
    tv = squad_values.team_values(con).set_index(["team_id", "season"])
    assert tv.loc[("alpha", 2024), "mv_log"] == pytest.approx(np.log(400))     # Wert der Vorsaison
    assert tv.loc[("alpha", 2024), "mv_rel"] == pytest.approx(np.log(400) - (np.log(400) + np.log(100)) / 2)
    assert ("alpha", 2023) not in tv.index
