from fussball import db
from fussball.ingest import football_data as fd
from tests.conftest import make_csv


def test_odds_triplets_detects_pre_and_closing():
    cols = ["HomeTeam", "HS", "B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
            "BbAvH", "BbAvD", "BbAvA", "BFDH", "BFDD", "BFDA", "BFDCH", "BFDCD", "BFDCA",
            "B365AHH", "B365AHA", "AHh", "HTHG"]
    t = fd.odds_triplets(cols)
    assert t[("PS", "closing")] == ("PSCH", "PSCD", "PSCA")
    assert t[("PS", "pre")] == ("PSH", "PSD", "PSA")
    assert t[("Avg", "pre")] == ("BbAvH", "BbAvD", "BbAvA")
    assert t[("BFD", "closing")] == ("BFDCH", "BFDCD", "BFDCA")
    assert ("B365", "pre") in t and ("BFD", "pre") in t
    assert len(t) == 6  # Asian-Handicap-Spalten (B365AHH/-A) sind kein 1X2-Tripel


def test_parse_handles_bom_latin1_and_two_digit_years():
    utf = make_csv(bom=True)
    latin = make_csv(bom=False).decode().replace("/2024,", "/24,").replace("Alpha", "Mönch").encode("latin-1")
    m1, _ = fd.parse(utf, "D1", 2024)
    m2, _ = fd.parse(latin, "D1", 2024)
    assert len(m1) == len(m2) == 24  # 4 Teams, Doppelrunde; Leerzeile am Ende ignoriert
    assert m1.date.iloc[0] == m2.date.iloc[0] == "2024-08-01"
    assert "Mönch" in set(m2.home)


def test_store_is_idempotent_and_ids_stable(con):
    m, o = fd.parse(make_csv(), "D1", 2024)
    fd.store(con, m, o)
    ids1 = db.read(con, "SELECT match_id, home_id, away_id, date FROM matches ORDER BY match_id")
    n_odds1 = con.execute("SELECT COUNT(*) FROM odds_benchmark").fetchone()[0]
    fd.store(con, m, o)
    ids2 = db.read(con, "SELECT match_id, home_id, away_id, date FROM matches ORDER BY match_id")
    assert ids1.equals(ids2)
    assert con.execute("SELECT COUNT(*) FROM odds_benchmark").fetchone()[0] == n_odds1 == 24 * 4
    assert con.execute("SELECT COUNT(*) FROM team_name_map WHERE source='football-data'").fetchone()[0] == 4
