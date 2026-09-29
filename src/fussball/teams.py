"""Teamnamen-Vereinheitlichung.

Kanonisch ist der football-data.co.uk-Name (team_id = slug davon). Namen anderer
Quellen werden über data_seed/team_name_map.csv gemappt. Die Datei ist committet
und wird manuell kuratiert; `fussball teams suggest` schlägt Einträge vor.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

from fussball import config, db

SEED_MAP = config.SEED_DIR / "team_name_map.csv"
CANONICAL_SOURCE = "football-data"


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def register_canonical(con: sqlite3.Connection, names_by_country: dict[str, str]) -> None:
    """Legt Teams aus football-data an (Name -> Land) und mappt sie auf sich selbst."""
    existing = dict(con.execute("SELECT team_id, country FROM teams").fetchall())
    rows = []
    for name, country in names_by_country.items():
        tid = slug(name)
        if tid in existing and existing[tid] != country:
            raise ValueError(f"team_id-Kollision: {tid} ({existing[tid]} vs. {country})")
        rows.append({"team_id": tid, "name": name, "country": country})
    db.upsert(con, "teams", rows, key=["team_id"])
    db.upsert(con, "team_name_map",
              [{"source": CANONICAL_SOURCE, "source_name": r["name"], "team_id": r["team_id"]} for r in rows],
              key=["source", "source_name"])


def load_seed_map(path: Path = SEED_MAP) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["source", "source_name", "team_id"])
    return pd.read_csv(path, dtype=str, comment="#").fillna("")


def sync_seed_map(con: sqlite3.Connection, path: Path = SEED_MAP) -> list[str]:
    """Schreibt die kuratierte Map in die DB. Liefert Einträge mit unbekannter team_id."""
    seed = load_seed_map(path)
    known = {r[0] for r in con.execute("SELECT team_id FROM teams")}
    bad = seed[~seed.team_id.isin(known)]
    db.upsert(con, "team_name_map", seed[seed.team_id.isin(known)][["source", "source_name", "team_id"]],
              key=["source", "source_name"])
    return [f"{r.source}:{r.source_name} -> {r.team_id}" for r in bad.itertuples()]


def name_map(con: sqlite3.Connection, source: str) -> dict[str, str]:
    return dict(con.execute(
        "SELECT source_name, team_id FROM team_name_map WHERE source = ?", (source,)).fetchall())


def append_seed(rows: list[dict], path: Path = SEED_MAP) -> int:
    """Hängt neue Mapping-Einträge an die Seed-Datei an (bestehende bleiben unverändert)."""
    seed = load_seed_map(path)
    have = set(zip(seed.source, seed.source_name))
    new = []
    for r in rows:
        if (r["source"], r["source_name"]) not in have:
            have.add((r["source"], r["source_name"]))
            new.append(r)
    if new:
        out = pd.concat([seed, pd.DataFrame(new)[["source", "source_name", "team_id"]]])
        out = out.sort_values(["source", "team_id", "source_name"])
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return len(new)


def backfill_snapshots(con: sqlite3.Connection) -> None:
    """Setzt team_id in Transfermarkt-Snapshots nachträglich, sobald ein Mapping existiert."""
    with con:
        con.execute("""
            UPDATE squad_value_snapshots SET team_id = (
                SELECT m.team_id FROM team_name_map m
                WHERE m.source = 'transfermarkt' AND m.source_name = squad_value_snapshots.tm_club_name)
            WHERE team_id IS NULL""")
        con.execute("""
            UPDATE availability_snapshots SET team_id = (
                SELECT s.team_id FROM squad_value_snapshots s
                WHERE s.tm_club_id = availability_snapshots.tm_club_id AND s.team_id IS NOT NULL
                ORDER BY s.snapshot_date DESC LIMIT 1)
            WHERE team_id IS NULL""")
