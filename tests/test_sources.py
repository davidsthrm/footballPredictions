"""Parser-Tests für Understat, ClubElo und Transfermarkt mit minimalen synthetischen Fixtures."""

import pandas as pd
import pytest

from fussball import db, teams
from fussball.ingest import clubelo, understat
from fussball.ingest import football_data as fd
from fussball.ingest import transfermarkt as tm
from tests.conftest import make_csv


def understat_json(fd_matches: pd.DataFrame, names: dict[str, str]) -> dict:
    dates, hist = [], {}
    for i, r in enumerate(fd_matches.itertuples()):
        dt = f"{r.date} 14:30:00"
        dates.append({"id": str(1000 + i), "isResult": True,
                      "h": {"id": "1", "title": names[r.home]}, "a": {"id": "2", "title": names[r.away]},
                      "goals": {"h": str(r.fthg), "a": str(r.ftag)}, "xG": {"h": "1.5", "a": "0.7"},
                      "datetime": dt})
        for team, side in ((names[r.home], "h"), (names[r.away], "a")):
            hist.setdefault(team, []).append({"h_a": side, "date": dt, "npxG": 1.2, "npxGA": 0.6,
                                              "ppda": {"att": 200, "def": 20}, "deep": 5})
    dates.append({"id": "9", "isResult": False, "h": {"title": "X"}, "a": {"title": "Y"},
                  "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None},
                  "datetime": "2025-05-01 12:00:00"})
    return {"dates": dates, "players": [],
            "teams": {str(k): {"title": t, "history": h} for k, (t, h) in enumerate(hist.items())}}


def test_understat_parse_suggest_and_link(con):
    m, o = fd.parse(make_csv(), "D1", 2024)
    fd.store(con, m, o)
    names = {"Alpha": "Alpha FC", "Beta": "Beta 04", "Gamma": "Gamma", "Delta": "Delta United"}
    xg = understat.parse(understat_json(m, names), "D1", 2024)
    assert len(xg) == 2 * len(m)  # nicht gespieltes Spiel ignoriert
    assert xg.npxg.notna().all()

    sugg = {s["source_name"]: s["team_id"] for s in understat.suggest_names(con, xg)}
    assert sugg == {"Alpha FC": "alpha", "Beta 04": "beta", "Gamma": "gamma", "Delta United": "delta"}

    db.upsert(con, "team_name_map", [{"source": "understat", "source_name": k, "team_id": v}
                                     for k, v in sugg.items()], key=["source", "source_name"])
    n, unmapped = understat.store(con, xg)
    assert n == len(xg) and not unmapped
    assert con.execute("SELECT COUNT(*) FROM team_xg WHERE match_id IS NULL").fetchone()[0] == 0


def test_clubelo_parse_and_store(con):
    raw = (b"Rank,Club,Country,Level,Elo,From,To\n"
           b"None,Bayern,GER,1,1800.5,2012-05-01,2012-05-10\n"
           b"None,Bayern,GER,1,1850.0,2014-08-01,2014-08-23\n"
           b"None,Bayern,GER,1,1861.2,2014-08-24,2014-08-30\n")
    df = clubelo.parse(raw)
    assert list(df.columns) == ["Rank", "club", "country", "level", "elo", "valid_from", "valid_to"]
    teams.register_canonical(con, {"Bayern Munich": "GER"})
    assert clubelo.store(con, df, "bayern-munich") == 2  # Eintrag vor MIN_DATE fällt weg


LEAGUE_HTML = """<html><body><table class="items"><thead><tr><th>Verein</th><th>name</th><th>Kader</th>
<th>ø-Alter</th><th>Legionäre</th><th>ø-Marktwert</th><th>Gesamtmarktwert</th></tr></thead><tbody>
<tr class="odd"><td></td><td class="hauptlink"><a href="/fc-test/startseite/verein/42/saison_id/2026"
 title="FC Test">FC Test</a></td><td>25</td><td>26,1</td><td>10</td><td>5,00 Mio. €</td><td>125,50 Mio. €</td></tr>
</tbody></table></body></html>"""


def _player(pid, name, pos=None):
    pos_row = f"<tr><td>{pos}</td></tr>" if pos else ""
    return (f'<td><table class="inline-table"><tr><td class="hauptlink"><a href="/x/profil/spieler/{pid}" '
            f'title="{name}">{name}</a></td></tr>{pos_row}</table></td>')


CLUB_HTML = f"""<html><body><h2>Sperren und Verletzungen</h2>
<table class="items"><thead><tr><th>Spieler</th><th>Alter</th><th>Grund</th><th>seit</th>
<th>bis voraussichtlich</th><th>Verpasste Spiele</th></tr></thead><tbody>
<tr><td colspan="6">Verletzungen</td></tr>
<tr class="odd">{_player(111, "Max Muster", "Torwart")}<td>27</td><td>Kreuzbandriss</td>
<td>01.08.2026</td><td>01.03.2027</td><td>7</td></tr>
<tr><td colspan="6">Sperren</td></tr>
<tr class="even">{_player(222, "Eva Beispiel", "Innenverteidiger")}<td>24</td><td>Rotsperre</td>
<td>20.09.2026</td><td>-</td><td>1</td></tr>
</tbody></table>
<h2>Sperre droht</h2>
<table class="items"><thead><tr><th>Spieler</th><th>Gelbe Karten</th></tr></thead><tbody>
<tr class="odd">{_player(333, "Kim K")}<td>4</td></tr>
</tbody></table></body></html>"""


def test_transfermarkt_league_parser():
    (club,) = tm.parse_league(LEAGUE_HTML)
    assert (club.tm_club_id, club.name, club.slug, club.squad_size) == (42, "FC Test", "fc-test", 25)
    assert club.avg_age == pytest.approx(26.1)
    assert club.total_value_eur == pytest.approx(125.5e6)


def test_transfermarkt_club_parser():
    recs = {r["tm_player_id"]: r for r in tm.parse_club(CLUB_HTML)}
    assert recs[111] == {"tm_player_id": 111, "player": "Max Muster", "position": "Torwart",
                         "category": "injury", "age": 27, "reason": "Kreuzbandriss",
                         "since": "2026-08-01", "until": "2027-03-01", "missed_games": 7}
    assert recs[222]["category"] == "suspension" and recs[222]["until"] is None
    assert recs[333]["category"] == "suspension_threat"
    assert "gelbe karten: 4" in recs[333]["reason"]


@pytest.mark.parametrize("text,value", [("1,04 Mrd. €", 1.04e9), ("45,11 Mio. €", 45.11e6),
                                        ("500 Tsd. €", 5e5), ("-", None), ("", None)])
def test_parse_value(text, value):
    assert tm.parse_value(text) == (pytest.approx(value) if value else None)


def test_registration_is_not_a_league_absence():
    html = CLUB_HTML.replace("Rotsperre", "keine Spielberechtigung")
    recs = {r["tm_player_id"]: r for r in tm.parse_club(html)}
    assert recs[222]["category"] == "registration"
    assert recs[111]["category"] == "injury"
