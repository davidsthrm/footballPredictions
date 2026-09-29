"""Zentrale Konstanten: Pfade, Ligen, Saisons, HTTP-Einstellungen."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FUSSBALL_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = DATA_DIR / "reports"
DB_PATH = Path(os.environ.get("FUSSBALL_DB", DATA_DIR / "fussball.sqlite"))
SEED_DIR = ROOT / "data_seed"

FIRST_SEASON = 2014  # Saison 2014/15; Saisons werden über ihr Startjahr identifiziert

USER_AGENT = os.environ.get(
    "FUSSBALL_USER_AGENT",
    "fussball-research/0.1 (private non-commercial project; rate-limited, cached)",
)


@dataclass(frozen=True)
class League:
    code: str  # football-data.co.uk-Code, dient als interne Liga-ID
    name: str
    tier: int
    country: str  # ClubElo-Ländercode
    understat: str | None = None
    transfermarkt: str | None = None


LEAGUES: dict[str, League] = {
    lg.code: lg
    for lg in [
        League("E0", "Premier League", 1, "ENG", "EPL", "GB1"),
        League("SP1", "La Liga", 1, "ESP", "La_liga", "ES1"),
        League("I1", "Serie A", 1, "ITA", "Serie_A", "IT1"),
        League("F1", "Ligue 1", 1, "FRA", "Ligue_1", "FR1"),
        League("D1", "Bundesliga", 1, "GER", "Bundesliga", "L1"),
        # 2. Ligen: nur Fit-Daten (Aufsteiger-Stärke), keine Vorhersage
        League("E1", "Championship", 2, "ENG"),
        League("SP2", "Segunda División", 2, "ESP"),
        League("I2", "Serie B", 2, "ITA"),
        League("F2", "Ligue 2", 2, "FRA"),
        League("D2", "2. Bundesliga", 2, "GER"),
    ]
}
TOP_LEAGUES = [c for c, lg in LEAGUES.items() if lg.tier == 1]
SECOND_TIER = {lg.country: c for c, lg in LEAGUES.items() if lg.tier == 2}


def season_of(d: date) -> int:
    """Startjahr der Saison, zu der ein Datum gehört (Saisonwechsel am 1. Juli)."""
    return d.year if d.month >= 7 else d.year - 1


def current_season(today: date | None = None) -> int:
    return season_of(today or date.today())


def seasons(until: int | None = None) -> list[int]:
    return list(range(FIRST_SEASON, (until or current_season()) + 1))


def season_label(season: int) -> str:
    return f"{season}/{(season + 1) % 100:02d}"
