"""Kommandozeile: `uv run fussball <befehl>`."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from fussball import config, db, teams

log = logging.getLogger("fussball")


def _seasons(arg: str | None) -> list[int] | None:
    """'2014-2016' oder '2019,2021' -> Liste von Startjahren."""
    if not arg:
        return None
    if "-" in arg:
        a, b = arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in arg.split(",")]


def _leagues(arg: str | None) -> list[str] | None:
    return arg.split(",") if arg else None


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(df.to_string(index=False) if len(df) else "(leer)")


def cmd_ingest(args) -> int:
    con = db.connect()
    if args.source == "football-data":
        from fussball.ingest import football_data
        _print(football_data.ingest(con, _leagues(args.leagues), _seasons(args.seasons)))
        bad = teams.sync_seed_map(con)
    elif args.source == "understat":
        from fussball.ingest import understat
        teams.sync_seed_map(con)
        summary, sugg = understat.ingest(con, _leagues(args.leagues), _seasons(args.seasons),
                                         suggest=args.suggest)
        _print(summary)
        if sugg:
            n = teams.append_seed(sugg)
            print(f"{n} neue Namensvorschläge in {teams.SEED_MAP} – prüfen und erneut laden.")
        bad = []
    elif args.source == "clubelo":
        from fussball.ingest import clubelo
        teams.sync_seed_map(con)
        if args.suggest:
            from fussball.http import Fetcher
            clubs = clubelo.discover_clubs(Fetcher("clubelo"), config.seasons())
            sugg = clubelo.suggest_names(con, clubs)
            for s in sugg:
                if s.pop("fuzzy", False):
                    print(f"  fuzzy: {s['source_name']!r} -> {s['team_id']}")
            print(f"{teams.append_seed(sugg)} neue Vorschläge in {teams.SEED_MAP}")
            teams.sync_seed_map(con)
        _print(clubelo.ingest(con, refresh=args.refresh))
        bad = []
    elif args.source == "europe":
        from fussball.ingest import europe
        teams.sync_seed_map(con)
        _print(europe.ingest(con, _seasons(args.seasons)))
        if args.suggest:
            sugg = europe.suggest_ids(con)
            for x in sugg:
                if "slug" in x:
                    print(f"  {x['slug']:<32} -> {x['team_id']:<18} ({x['score']})")
            print(f"{teams.append_seed([{k: x[k] for k in ('source', 'source_name', 'team_id')} for x in sugg])} "
                  f"neue Vorschläge in {teams.SEED_MAP}")
            teams.sync_seed_map(con)
        bad = []
    elif args.source == "squad-values":
        from fussball.ingest import squad_values
        teams.sync_seed_map(con)
        _print(squad_values.ingest(con, _seasons(args.seasons)).groupby("league").clubs.sum().reset_index())
        if args.suggest:
            sugg = squad_values.suggest_ids(con)
            for x in sugg:
                print(f"  {x['slug']:<32} -> {x['team_id']:<18} ({x['score']})")
            print(f"{teams.append_seed([{k: x[k] for k in ('source', 'source_name', 'team_id')} for x in sugg])} "
                  f"neue Vorschläge in {teams.SEED_MAP}")
            teams.sync_seed_map(con)
        bad = []
    else:
        raise ValueError(args.source)
    for b in bad:
        print("WARNUNG Seed-Map:", b)
    return 0


def cmd_snapshot(args) -> int:
    from fussball.http import BlockedError
    from fussball.ingest import transfermarkt
    con = db.connect()
    teams.sync_seed_map(con)
    try:
        _print(transfermarkt.snapshot(con, _leagues(args.leagues)))
    except BlockedError as e:
        log.error("%s", e)
        return 2
    teams.backfill_snapshots(con)
    return 0


def cmd_teams(args) -> int:
    con = db.connect()
    bad = teams.sync_seed_map(con)
    teams.backfill_snapshots(con)
    for b in bad:
        print("Unbekannte team_id in Seed-Map:", b)
    print("Seed-Map synchronisiert.")
    return 1 if bad else 0


def cmd_check(args) -> int:
    from fussball import checks
    con = db.connect()
    teams.sync_seed_map(con)
    return checks.run(con, verbose=args.verbose)


def cmd_backtest(args) -> int:
    from fussball.backtest import run as bt
    con = db.connect()
    return bt.main(con, leagues=_leagues(args.leagues), xi=args.xi, tune_params=args.tune,
                   test_from=args.test_from)


def cmd_predict(args) -> int:
    from datetime import date
    from fussball import predict
    con = db.connect()
    teams.sync_seed_map(con)
    if not args.no_refresh:
        predict.refresh(con)
    today = date.fromisoformat(args.date) if args.date else None
    out = predict.predict(con, today=today, leagues=_leagues(args.leagues))
    if out.empty:
        print("Keine anstehenden Spiele im Spielplan.")
        return 1
    for lg, g in out.groupby("league", sort=False):
        print(f"\n{config.LEAGUES[lg].name} – {g.date.min()} bis {g.date.max()}")
        for r in g.sort_values("kickoff_utc").itertuples():
            flag = "  (Team ohne Historie!)" if r.unknown_team else ""
            print(f"  {r.kickoff_utc[5:16]} UTC  {r.home:>22} {r.tip_h}:{r.tip_a} {r.away:<22}"
                  f"  (wahrsch. {r.pred_h}:{r.pred_a})  1/X/2 {r.p_h:4.0%} {r.p_d:4.0%} {r.p_a:4.0%}{flag}")
    print(f"\n{len(out)} Vorhersagen gespeichert (Modell: {out.model.iloc[0]}, Cutoff {out.cutoff_date.iloc[0]}).")
    return 0


def cmd_dashboard(args) -> int:
    from fussball import dashboard
    con = db.connect()
    targets = args.out or [dashboard.OUT_FILE] + ([config.DASHBOARD_COPY] if config.DASHBOARD_COPY else [])
    for out in targets:
        print("Dashboard:", dashboard.build(con, out))
    return 0


def cmd_scorecard(args) -> int:
    from fussball import predict
    sc = predict.scorecard(db.connect())
    if sc.empty:
        print("Noch keine ausgewerteten Live-Vorhersagen.")
        return 0
    agg = sc.groupby("league").agg(n=("tip_exact", "size"), exact=("tip_exact", "mean"),
                                   diff=("tip_diff", "mean"), tendency=("tip_tendency", "mean"),
                                   points=("points", "mean"), logloss=("logloss", "mean"))
    agg.loc["ALL"] = [len(sc), sc.tip_exact.mean(), sc.tip_diff.mean(), sc.tip_tendency.mean(),
                      sc.points.mean(), sc.logloss.mean()]
    for c in ("exact", "diff", "tendency"):
        agg[c] = (100 * agg[c]).round(1)
    _print(agg.reset_index())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fussball", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="Daten laden (idempotent, mit Cache)")
    s.add_argument("source", choices=["football-data", "understat", "clubelo", "europe", "squad-values"])
    s.add_argument("--leagues", help="z.B. E0,D1 (Default: alle)")
    s.add_argument("--seasons", help="Startjahre, z.B. 2014-2026 oder 2024,2025")
    s.add_argument("--suggest", action="store_true",
                   help="Teamnamen-Vorschläge in data_seed/team_name_map.csv schreiben")
    s.add_argument("--refresh", action="store_true", help="clubelo: Historien neu abrufen")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("snapshot", help="Täglicher Transfermarkt-Snapshot (Verletzungen/Sperren)")
    s.add_argument("source", choices=["transfermarkt"])
    s.add_argument("--leagues", help="z.B. D1,E0 (Default: Top 5)")
    s.set_defaults(func=cmd_snapshot)

    s = sub.add_parser("predict", help="Nächsten Spieltag vorhersagen und speichern")
    s.add_argument("--leagues", help="z.B. D1,E0 (Default: Top 5)")
    s.add_argument("--date", help="Stichtag YYYY-MM-DD (Default: heute)")
    s.add_argument("--no-refresh", action="store_true", help="laufende Saison nicht neu abrufen")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("dashboard", help="Statisches HTML-Dashboard erzeugen")
    s.add_argument("--out", action="append",
                   help="Zieldatei (mehrfach möglich, z.B. zusätzlich in iCloud Drive)")
    s.set_defaults(func=cmd_dashboard)

    s = sub.add_parser("scorecard", help="Echte Trefferbilanz der gespeicherten Vorhersagen")
    s.set_defaults(func=cmd_scorecard)

    s = sub.add_parser("teams", help="Seed-Map in DB übernehmen und Snapshots nachmappen")
    s.set_defaults(func=cmd_teams)

    s = sub.add_parser("check", help="Datenqualitäts-Checks")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("backtest", help="Walk-forward-Backtest der Dixon-Coles-Baseline")
    s.add_argument("--leagues", help="z.B. E0,D1 (Default: Top 5)")
    s.add_argument("--xi", type=float, help="Time-Decay pro Tag (Default: getunter Wert)")
    s.add_argument("--tune", action="store_true", help="ξ auf den Validierungssaisons tunen")
    s.add_argument("--test-from", type=int, default=2019, help="erste Test-Saison (Default 2019)")
    s.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
