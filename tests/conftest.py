import numpy as np
import pytest

from fussball import db

HEADER = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,"
          "HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,PSH,PSD,PSA,AvgH,AvgD,AvgA,PSCH,PSCD,PSCA,"
          "B365>2.5,B365<2.5,AHh,B365AHH,B365AHA")


def make_csv(league="D1", teams=("Alpha", "Beta", "Gamma", "Delta"), rounds=2, start_day=1,
             year=2024, seed=0, bom=True) -> bytes:
    """Synthetische football-data.co.uk-CSV mit Doppelrunde und Quoten-Spalten."""
    rng = np.random.default_rng(seed)
    lines = [HEADER]
    day = start_day
    for r in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg, ag = rng.poisson(1.5), rng.poisson(1.1)
                ftr = "H" if hg > ag else "D" if hg == ag else "A"
                month, d = 8 + (day - 1) // 28, (day - 1) % 28 + 1
                lines.append(f"{league},{d:02d}/{month:02d}/{year},15:30,{h},{a},{hg},{ag},{ftr},0,0,D,Ref,"
                             f"10,8,4,3,11,12,5,4,1,2,0,0,2.1,3.4,3.5,2.15,3.45,3.6,2.05,3.3,3.4,2.2,3.5,3.4,"
                             f"1.9,1.9,-0.25,1.95,1.95")
                day += 1
    text = "\n".join(lines) + "\n,,,,,,,,\n"
    return (("﻿" if bom else "") + text).encode("utf-8")


@pytest.fixture
def con():
    c = db.connect(":memory:")
    yield c
    c.close()
