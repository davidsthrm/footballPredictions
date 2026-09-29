"""Harte Regel: Wettquoten nie als Features / Fit-Daten.

1. Statisch: `odds_benchmark` taucht nur in erlaubten Modulen auf, und nur backtest/run.py
   importiert das Benchmark-Modul.
2. Laufzeit: Der Match-Frame des Parsers enthält keine Quoten-Spalten, und die
   Modell-Vorhersagen ändern sich nicht, wenn man die Quoten beliebig verändert.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fussball.backtest import walkforward
from fussball.ingest import football_data as fd
from tests.conftest import make_csv

SRC = Path(__file__).resolve().parents[1] / "src" / "fussball"
ALLOWED_ODDS_TABLE = {"db.py", "ingest/football_data.py", "backtest/benchmark.py", "checks.py"}
ALLOWED_BENCHMARK_IMPORT = {"backtest/run.py"}
ODDS_COLUMN = re.compile(r"^(B365|BW|IW|LB|PS|WH|SJ|VC|GB|BS|SB|SO|Bb|Max|Avg|BF|1XB|PP|SKB|BV)")


def _modules():
    for p in SRC.rglob("*.py"):
        yield p.relative_to(SRC).as_posix(), p.read_text()


def test_odds_table_only_in_allowed_modules():
    offenders = [name for name, text in _modules()
                 if "odds_benchmark" in text and name not in ALLOWED_ODDS_TABLE]
    assert not offenders, f"odds_benchmark wird außerhalb der Whitelist benutzt: {offenders}"


def test_benchmark_module_only_imported_by_report_runner():
    pat = re.compile(r"^\s*(from\s+fussball\.backtest\s+import\s+[^\n]*\bbenchmark\b"
                     r"|(from|import)\s+fussball\.backtest\.benchmark)", re.MULTILINE)
    offenders = [name for name, text in _modules()
                 if pat.search(text) and name not in ALLOWED_BENCHMARK_IMPORT]
    assert not offenders, f"Benchmark-Modul importiert von: {offenders}"


def test_features_and_models_never_mention_odds():
    for name, text in _modules():
        if name.startswith(("features/", "models/")) or name == "backtest/walkforward.py":
            assert not re.search(r"odds", text, re.I), f"{name} erwähnt odds"


def test_parsed_matches_contain_no_odds_columns():
    matches, odds = fd.parse(make_csv(), "D1", 2024)
    leaked = [c for c in matches.columns if ODDS_COLUMN.match(c) or c.startswith(("odds", "o_"))]
    assert not leaked, leaked
    assert set(odds.bookmaker) >= {"B365", "PS", "Avg"}


def test_predictions_invariant_to_odds(con):
    for season, year in ((2023, 2023), (2024, 2024)):
        m, o = fd.parse(make_csv(year=year, seed=season), "D1", season)
        fd.store(con, m, o)

    def predict():
        data = walkforward.load_fit_data(con)
        return walkforward.predict_league(data, "D1", [2024], xi=0.002)

    before = predict()
    with con:
        con.execute("UPDATE odds_benchmark SET odds_h = 1.01, odds_d = 50, odds_a = 99")
        con.execute("INSERT INTO odds_benchmark VALUES (999999, 'X', 'pre', 1.5, 4, 6)")
    after = predict()
    pd.testing.assert_frame_equal(before, after)
    assert np.isfinite(before.p_actual).all()
