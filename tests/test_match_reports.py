from fussball.ingest import understat_matches as um


def _shot(i, minute, player, result, side, assist=None, situation="OpenPlay"):
    return {"id": str(i), "minute": str(minute), "player": player, "result": result, "h_a": side,
            "player_assisted": assist, "situation": situation, "xG": "0.3"}


def _player(pid, name, side, goals=0, assists=0):
    return {"player_id": str(pid), "player": name, "position": "FW", "time": "90", "goals": str(goals),
            "own_goals": "0", "assists": str(assists), "shots": "2", "key_passes": "1", "xG": "0.5", "xA": "0.1",
            "yellow_card": "0", "red_card": "0"}


def test_parse_goals_including_own_goals_and_penalties():
    data = {"shots": {"h": [_shot(1, 3, "A", "MissedShots", "h"), _shot(2, 62, "Gegner", "OwnGoal", "h")],
                      "a": [_shot(3, 88, "B", "Goal", "a", assist="C"), _shot(4, 90, "B", "Goal", "a", situation="Penalty")]},
            "rosters": {"h": {"1": _player(10, "A", "h")}, "a": {"2": _player(20, "B", "a", goals=2)}}}
    goals, players = um.parse(data, 99)
    assert goals.sort_values("minute")[["minute", "h_a", "player", "own_goal"]].values.tolist() == [
        [62, "h", "Gegner", 1], [88, "a", "B", 0], [90, "a", "B", 0]]
    assert goals.set_index("shot_id").loc[3, "assist"] == "C"
    assert goals.set_index("shot_id").loc[4, "situation"] == "Penalty"
    assert players.set_index("player").loc["B", "goals"] == 2 and len(players) == 2
