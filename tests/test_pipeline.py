from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulate import play_group_stage, run_simulations
from prepare_data import prepare_training_data
from fetch_data import build_matches_url, load_env, matches_to_frame
from process_live_data import process_live_data
from simulate_live import current_group_points, fallback_round_of_32_pairs, run_live_simulations
from train import train_model
from utils import group_stage_pairs, load_matches, load_teams


def test_sample_data_loads():
    matches = load_matches(ROOT / "data" / "matches_sample.csv")
    teams = load_teams(ROOT / "data" / "teams_2026_sample.csv")

    assert len(matches) >= 40
    assert len(teams) == 48
    assert teams["group"].nunique() == 12


def test_prepare_data_preserves_teams_and_uses_prior_rankings():
    results = pd.DataFrame(
        [
            {
                "date": "2024-01-10",
                "home_team": "United States",
                "away_team": "South Korea",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "Test City",
                "country": "United States",
                "neutral": False,
            }
        ]
    )
    rankings = pd.DataFrame(
        [
            {"rank_date": "2024-01-01", "country_full": "USA", "rank": 11, "total_points": 1670},
            {"rank_date": "2024-01-01", "country_full": "Korea Republic", "rank": 23, "total_points": 1550},
        ]
    )

    prepared = prepare_training_data(results, rankings, start_date="2024-01-01")

    assert prepared.loc[0, "team_a"] == "United States"
    assert prepared.loc[0, "team_b"] == "South Korea"
    assert prepared.loc[0, "rank_diff"] == -12
    assert prepared.loc[0, "target"] == "team_a_win"


def test_football_data_url_and_response_conversion():
    url = build_matches_url("wc", season=2026, status="SCHEDULED")
    assert url.endswith("/competitions/WC/matches?season=2026&status=SCHEDULED")

    payload = {
        "competition": {"code": "WC", "name": "FIFA World Cup"},
        "matches": [
            {
                "id": 123,
                "utcDate": "2026-06-11T19:00:00Z",
                "status": "SCHEDULED",
                "stage": "GROUP_STAGE",
                "group": "GROUP_A",
                "matchday": 1,
                "homeTeam": {"id": 1, "name": "Mexico"},
                "awayTeam": {"id": 2, "name": "South Africa"},
                "score": {"winner": None, "duration": "REGULAR", "fullTime": {"home": None, "away": None}},
                "season": {"startDate": "2026-06-11"},
                "lastUpdated": "2026-01-01T00:00:00Z",
            }
        ],
    }
    frame = matches_to_frame(payload)

    assert len(frame) == 1
    assert frame.loc[0, "competition_code"] == "WC"
    assert frame.loc[0, "home_team"] == "Mexico"
    assert frame.loc[0, "away_team"] == "South Africa"


def test_env_loader_does_not_overwrite_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOOTBALL_DATA_API_KEY=file-token\n", encoding="utf-8")
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "existing-token")

    load_env(env_file)

    assert __import__("os").environ["FOOTBALL_DATA_API_KEY"] == "existing-token"


def test_live_processing_creates_48_teams_and_locks_only_finished_results():
    match_rows = []
    ranking_rows = []
    match_id = 1
    for group_index, group_letter in enumerate("ABCDEFGHIJKL"):
        names = [f"Team {group_letter}{index}" for index in range(4)]
        for index, name in enumerate(names):
            ranking_rows.append(
                {
                    "rank_date": "2024-06-20",
                    "country_full": name,
                    "rank": group_index * 4 + index + 1,
                    "total_points": 1700 - group_index * 4 - index,
                    "confederation": "TEST",
                }
            )
        for home_index in range(4):
            for away_index in range(home_index + 1, 4):
                finished = home_index == 0 and away_index == 1
                match_rows.append(
                    {
                        "match_id": match_id,
                        "utc_date": f"2026-06-{11 + group_index:02d}T12:00:00Z",
                        "status": "FINISHED" if finished else "TIMED",
                        "stage": "GROUP_STAGE",
                        "group": f"GROUP_{group_letter}",
                        "home_team": names[home_index],
                        "away_team": names[away_index],
                        "home_score": 1 if finished else None,
                        "away_score": 0 if finished else None,
                    }
                )
                match_id += 1

    results, schedule, teams = process_live_data(pd.DataFrame(match_rows), pd.DataFrame(ranking_rows))

    assert len(teams) == 48
    assert len(schedule) == 72
    assert len(results) == 12
    assert teams["confederation"].eq("TEST").all()


def test_live_points_and_round_of_32_fallback_are_well_formed():
    teams = pd.DataFrame(
        [
            {"team": f"Team {group}{index}", "group": f"GROUP_{group}", "fifa_rank": rank + 1}
            for rank, (group, index) in enumerate(
                (group, index) for group in "ABCDEFGHIJKL" for index in range(4)
            )
        ]
    )
    tables = {}
    for group, group_teams in teams.groupby("group"):
        table = group_teams[["team"]].copy()
        table["points"] = [9, 6, 3, 0]
        table["goal_difference"] = [5, 2, 0, -7]
        table["goals_for"] = [8, 5, 3, 0]
        table["goals_against"] = [3, 3, 3, 7]
        tables[group] = table

    pairs = fallback_round_of_32_pairs(tables, teams)
    entrants = [team for pair in pairs for team in pair]
    schedule = pd.DataFrame(
        [
            {
                "stage": "GROUP_STAGE",
                "status": "FINISHED",
                "home_team": "Team A0",
                "away_team": "Team A1",
                "home_score": 2,
                "away_score": 0,
            }
        ]
    )
    points = current_group_points(schedule, teams)

    assert len(pairs) == 16
    assert len(set(entrants)) == 32
    assert points["Team A0"] == 3
    assert points["Team A1"] == 0


def test_run_live_simulations_locks_finished_matches_and_produces_probabilities():
    matches = load_matches(ROOT / "data" / "matches_sample.csv")
    model, _ = train_model(matches)

    teams = pd.DataFrame(
        [
            {"team": f"Team {group}{index}", "group": f"GROUP_{group}", "fifa_rank": rank + 1}
            for rank, (group, index) in enumerate(
                (group, index) for group in "ABCDEFGHIJKL" for index in range(4)
            )
        ]
    )

    schedule_rows = []
    match_id = 1
    for group_letter, group_teams in teams.groupby("group"):
        names = group_teams["team"].tolist()
        for pair_index, (home, away) in enumerate(group_stage_pairs(group_teams)):
            finished = home == "Team A0" and away == "Team A1"
            schedule_rows.append(
                {
                    "match_id": match_id,
                    "utc_date": f"2026-06-{11 + pair_index:02d}T12:00:00Z",
                    "status": "FINISHED" if finished else "TIMED",
                    "stage": "GROUP_STAGE",
                    "group": group_letter,
                    "home_team": home,
                    "away_team": away,
                    "home_score": 2 if finished else None,
                    "away_score": 0 if finished else None,
                }
            )
            match_id += 1
    for stage in ("LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"):
        schedule_rows.append(
            {
                "match_id": None,
                "utc_date": None,
                "status": "SCHEDULED",
                "stage": stage,
                "group": None,
                "home_team": None,
                "away_team": None,
                "home_score": None,
                "away_score": None,
            }
        )
    schedule = pd.DataFrame(schedule_rows)

    results = run_live_simulations(model, schedule, teams, simulations=20, seed=7)

    assert len(results) == 48
    assert results["current_points"].eq(0).sum() == 47
    assert results.set_index("team").loc["Team A0", "current_points"] == 3
    assert results.set_index("team").loc["Team A1", "current_points"] == 0

    probability_columns = [
        "round_32_probability",
        "last_16_probability",
        "quarterfinal_probability",
        "semifinal_probability",
        "final_probability",
        "title_probability",
    ]
    for column in probability_columns:
        assert results[column].between(0, 1).all()
    assert abs(results["round_32_probability"].sum() - 32.0) < 1e-9
    assert abs(results["last_16_probability"].sum() - 16.0) < 1e-9
    assert abs(results["title_probability"].sum() - 1.0) < 1e-9


def test_model_trains_and_predicts_probabilities():
    matches = load_matches(ROOT / "data" / "matches_sample.csv")
    model, metrics = train_model(matches)

    assert metrics["rows"] == len(matches)
    assert set(metrics["classes"]) == {"draw", "team_a_win", "team_b_win"}
    assert hasattr(model, "predict_proba")


def test_group_stage_returns_32_qualifiers():
    matches = load_matches(ROOT / "data" / "matches_sample.csv")
    teams = load_teams(ROOT / "data" / "teams_2026_sample.csv")
    model, _ = train_model(matches)

    qualifiers, group_tables = play_group_stage(model, teams, np.random.default_rng(7))

    assert len(qualifiers) == 32
    assert len(set(qualifiers)) == 32
    assert len(group_tables) == 12


def test_simulation_outputs_probabilities_for_all_teams():
    matches = load_matches(ROOT / "data" / "matches_sample.csv")
    teams = load_teams(ROOT / "data" / "teams_2026_sample.csv")
    model, _ = train_model(matches)

    results = run_simulations(model, teams, simulations=20, seed=7)

    assert len(results) == 48
    assert results["title_probability"].between(0, 1).all()
    assert results["knockout_probability"].between(0, 1).all()
    assert abs(results["title_probability"].sum() - 1.0) < 1e-9
