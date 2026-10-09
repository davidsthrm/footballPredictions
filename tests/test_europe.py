from datetime import date

import numpy as np
import pandas as pd

from fussball import tournament
from fussball.dashboard_europe import round_label
from fussball.ingest import europe, other_leagues
from fussball.models.europe import EU_TIER, EuropeModel
from fussball.predict_europe import next_round

ROW = """<tr><td class="hide-for-small">{d}</td>
<td><a href="/{hs}/startseite/verein/{hi}/saison_id/2026" title="{hn}">{hn}</a></td>
<td><a href="/{hs}/startseite/verein/{hi}/saison_id/2026" title="{hn}"><img/></a></td>
<td><a class="ergebnis-link" href="/spielbericht/index/spielbericht/{mid}">{score}</a></td>
<td><a href="/{as_}/startseite/verein/{ai}/saison_id/2026" title="{an}"><img/></a></td>
<td><a href="/{as_}/startseite/verein/{ai}/saison_id/2026" title="{an}">{an}</a></td></tr>"""


def test_europe_parse_reads_scores_names_and_open_fixtures():
    rows = (ROW.format(d="17.09.2026", hs="juventus-turin", hi=506, hn="Juventus Turin", mid=11, score="3:1",
                       as_="psv-eindhoven", ai=383, an="PSV Eindhoven")
            + ROW.format(d="", hs="aek-athen", hi=2441, hn="AEK Athen", mid=12, score="1:0",
                         as_="lask", ai=413, an="LASK")
            + ROW.format(d="15.10.2026", hs="olympique-lyon", hi=1041, hn="Olympique Lyon", mid=13, score="-:-",
                         as_="crystal-palace", ai=873, an="Crystal Palace"))
    df = europe.parse(f'<div class="box"><table><tbody>{rows}</tbody></table></div>')
    assert df.tm_match_id.tolist() == [11, 12, 13]
    assert df.date.tolist() == ["2026-09-17", "2026-09-17", "2026-10-15"]     # Datum gilt für Folgezeilen
    assert (df.hg.tolist()[:2], df.ag.tolist()[:2]) == ([3, 1], [1, 0])
    assert pd.isna(df.hg.iloc[2]) and df.home_name.iloc[2] == "Olympique Lyon"


def test_other_leagues_parse_keeps_only_results():
    csv = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
           "N1,10/08/2024,PSV Eindhoven,Ajax,2,1,1.5,4.0,6.0\n"
           "N1,17/08/2024,Ajax,Feyenoord,,,2.0,3.5,3.5\n"
           "N1,10/08/2013,Ajax,Twente,1,1,1.9,3.3,4.0\n")      # vor der ersten Projektsaison
    df = other_leagues.parse(csv.encode(), "N1")
    assert list(df.columns) == ["date", "home", "away", "hg", "ag", "league"]
    assert df[["home", "away", "hg", "ag"]].values.tolist() == [["PSV Eindhoven", "Ajax", 2, 1]]
    new = "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH\nAustria,Bundesliga,2024/2025,03/08/2024,17:00,Salzburg,Rapid,3,0,H,1.4\n"
    assert other_leagues.parse(new.encode(), "AUT")[["home", "hg"]].values.tolist() == [["Salzburg", 3]]


def test_next_round_takes_first_date_per_competition_plus_two_days():
    d = pd.to_datetime
    fx = pd.DataFrame({"competition": ["CL", "CL", "CL", "EL", "EL"],
                       "date": d(["2026-10-13", "2026-10-14", "2026-10-21", "2026-10-15", "2026-10-08"]),
                       "hg": [np.nan, np.nan, np.nan, np.nan, 1.0]})
    out = next_round(fx, date(2026, 10, 9))
    assert sorted(out.date.dt.strftime("%m-%d")) == ["10-13", "10-14", "10-15"]


def test_europe_model_links_leagues_through_european_games():
    rng = np.random.default_rng(0)
    rows, day = [], pd.Timestamp("2024-08-01")
    strong, weak = [f"S{i}" for i in range(4)], [f"W{i}" for i in range(4)]
    for k in range(240):     # zwei Ligen gleicher innerer Struktur ...
        for lg, tm in (("A", strong), ("B", weak)):
            h, a = rng.choice(tm, 2, replace=False)
            rows.append({"date": day, "home": h, "away": a, "hg": rng.poisson(1.4), "ag": rng.poisson(1.1), "tier": lg})
        if k % 4 == 0:       # ... aber im Europapokal gewinnt Liga A klar
            rows.append({"date": day, "home": rng.choice(strong), "away": rng.choice(weak),
                         "hg": rng.poisson(2.4), "ag": rng.poisson(0.6), "tier": EU_TIER})
        day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows).astype({"hg": float, "ag": float}).assign(eu=lambda x: x.tier == EU_TIER)
    m = EuropeModel(xi=0.001, reg=1.0, w_eu=2.0).fit(df, day)
    lam, mu = m.rates("W0", "S0")      # schwaches Team zu Hause gegen starkes
    assert mu > lam


def test_europe_parse_reads_round_time_and_matchday():
    hdr = lambda t: f"<tr><td colspan='6'>{t}</td></tr>"
    match = lambda mid, h, a: ROW.format(d="", hs=f"c{h}", hi=h, hn=f"C{h}", mid=mid, score="-:-",
                                          as_=f"c{a}", ai=a, an=f"C{a}")
    html = (f'<div class="box"><h2>Gruppe GP</h2><table><tbody>{hdr("Di. 16.09.2025 21:00")}{match(1, 1, 2)}'
            f'{hdr("Di. 30.09.2025 18:45")}{match(2, 3, 4)}</tbody></table></div>'
            f'<div class="box"><h2>KO-Runde</h2><table><tbody>{hdr("Achtelfinale - Hinspiele")}'
            f'{hdr("Mi. 11.03.2026 21:00")}{match(3, 5, 6)}{hdr("Finale")}{hdr("Sa. 30.05.2026 18:00")}'
            f'{match(4, 5, 7)}</tbody></table></div>')
    df = europe.parse(html).set_index("tm_match_id")
    assert df.loc[1, ["round", "kickoff", "matchday"]].tolist() == ["Ligaphase", "21:00", 1]
    assert df.loc[2, "matchday"] == 2 and df.loc[2, "kickoff"] == "18:45"
    assert df.loc[3, ["round", "leg"]].tolist() == ["Achtelfinale", "Hinspiel"]
    assert df.loc[4, "round"] == "Finale" and pd.isna(df.loc[4, "leg"])


def test_round_labels():
    assert round_label("Ligaphase", None, 3, 2026) == "Ligaphase, 3. Spieltag"
    assert round_label("Zwischenrunde", "Rückspiel", None, 2026) == "Play-offs, Rückspiel"
    assert round_label("Zwischenrunde", "Hinspiel", None, 2018) == "Zwischenrunde, Hinspiel"


def test_tournament_forecast_is_consistent():
    rng = np.random.default_rng(3)
    teams = [f"T{i:02d}" for i in range(36)]
    rows, day = [], pd.Timestamp("2025-01-01")
    for k in range(600):                        # Historie: Stärke steigt mit der Nummer
        h, a = rng.choice(36, 2, replace=False)
        rows.append({"date": day, "home": teams[h], "away": teams[a], "hg": rng.poisson(0.6 + h / 25),
                     "ag": rng.poisson(0.5 + a / 25), "tier": EU_TIER})
        day += pd.Timedelta(hours=12)
    df = pd.DataFrame(rows).astype({"hg": float, "ag": float}).assign(eu=True)
    model = EuropeModel(xi=0.001, reg=1.0, w_eu=1.0).fit(df, day)
    fixtures = pd.DataFrame([{"home": teams[i], "away": teams[(i + k) % 36], "hg": np.nan, "ag": np.nan}
                             for k in (1, 2, 3, 4) for i in range(36)])
    f = tournament.simulate(model, fixtures, n=2000)
    t = f.table
    assert np.isclose(t.p_win.sum(), 1) and np.isclose(t.p_top8.sum(), 8) and np.isclose(t.p_r16.sum(), 16)
    assert np.isclose(t.p_final.sum(), 2) and np.allclose(t.p_top8 + t.p_playoff + t.p_out, 1)
    assert t.set_index("team").p_win["T35"] > t.set_index("team").p_win["T00"]
    assert [len(r) for r in f.bracket] == [8, 8, 4, 2, 1]
