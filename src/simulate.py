"""Run Monte Carlo simulations for a 48-team FIFA World Cup."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train import train_model
from utils import (
    MODEL_DIR,
    OUTPUT_DIR,
    best_third_place_teams,
    choose_match_winner,
    group_stage_pairs,
    load_matches,
    load_teams,
    probability_lookup,
    sample_match_score,
)


def _team_index(teams: pd.DataFrame) -> dict[str, pd.Series]:
    return {row["team"]: row for _, row in teams.iterrows()}


def _blank_group_table(group: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": group["team"].tolist(),
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
        }
    ).set_index("team")


def play_group_stage(model, teams: pd.DataFrame, rng: np.random.Generator) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Simulate group-stage matches and return the 32 teams that advance."""

    lookup = _team_index(teams)
    group_tables: dict[str, pd.DataFrame] = {}
    for group_name, group in teams.groupby("group", sort=True):
        table = _blank_group_table(group)
        for team_a_name, team_b_name in group_stage_pairs(group):
            prediction = probability_lookup(model, lookup[team_a_name], lookup[team_b_name])
            goals_a, goals_b = sample_match_score(prediction, rng)
            table.loc[team_a_name, "goals_for"] += goals_a
            table.loc[team_a_name, "goals_against"] += goals_b
            table.loc[team_b_name, "goals_for"] += goals_b
            table.loc[team_b_name, "goals_against"] += goals_a
            if goals_a > goals_b:
                table.loc[team_a_name, "points"] += 3
            elif goals_b > goals_a:
                table.loc[team_b_name, "points"] += 3
            else:
                table.loc[[team_a_name, team_b_name], "points"] += 1
        table["goal_difference"] = table["goals_for"] - table["goals_against"]
        ordered = table.reset_index().sort_values(
            ["points", "goal_difference", "goals_for", "team"],
            ascending=[False, False, False, True],
        )
        group_tables[group_name] = ordered

    automatic_qualifiers = []
    for table in group_tables.values():
        automatic_qualifiers.extend(table.head(2)["team"].tolist())
    third_place = best_third_place_teams(group_tables.values(), count=8)
    return automatic_qualifiers + third_place, group_tables


def play_knockout(model, teams: pd.DataFrame, qualifiers: list[str], rng: np.random.Generator) -> str:
    """Simulate a seeded knockout bracket from 32 qualifiers."""

    lookup = _team_index(teams)
    ordered = sorted(qualifiers, key=lambda name: (lookup[name]["fifa_rank"], name))
    bracket = []
    for i in range(16):
        bracket.append((ordered[i], ordered[-(i + 1)]))

    current_pairs = bracket
    winners: list[str] = []
    while current_pairs:
        winners = []
        for team_a_name, team_b_name in current_pairs:
            prediction = probability_lookup(model, lookup[team_a_name], lookup[team_b_name])
            winners.append(choose_match_winner(prediction, rng, allow_draw=False))
        if len(winners) == 1:
            return winners[0]
        current_pairs = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
    raise RuntimeError("knockout simulation ended without a champion")


def simulate_tournament(model, teams: pd.DataFrame, rng: np.random.Generator) -> tuple[str, list[str]]:
    """Run one full tournament simulation."""

    qualifiers, _ = play_group_stage(model, teams, rng)
    champion = play_knockout(model, teams, qualifiers, rng)
    return champion, qualifiers


def run_simulations(
    model,
    teams: pd.DataFrame,
    simulations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run many tournaments and summarize champion and knockout probabilities."""

    rng = np.random.default_rng(seed)
    champion_counts: Counter[str] = Counter()
    knockout_counts: Counter[str] = Counter()

    for _ in range(simulations):
        champion, qualifiers = simulate_tournament(model, teams, rng)
        champion_counts[champion] += 1
        knockout_counts.update(qualifiers)

    rows = []
    for team in teams["team"]:
        rows.append(
            {
                "team": team,
                "group": teams.loc[teams["team"] == team, "group"].iloc[0],
                "fifa_rank": int(teams.loc[teams["team"] == team, "fifa_rank"].iloc[0]),
                "knockout_probability": knockout_counts[team] / simulations,
                "title_probability": champion_counts[team] / simulations,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["title_probability", "knockout_probability", "fifa_rank"],
        ascending=[False, False, True],
    )


def load_or_train_model(model_path: str | Path, matches_path: str | Path):
    """Load a saved model, or train one if no artifact exists yet."""

    path = Path(model_path)
    if path.exists():
        return joblib.load(path)
    matches = load_matches(matches_path)
    model, _ = train_model(matches)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the 2026 FIFA World Cup.")
    parser.add_argument("--teams", default=str(Path("data") / "teams_2026_sample.csv"), help="48-team CSV path.")
    parser.add_argument("--matches", default=str(Path("data") / "matches_sample.csv"), help="Training CSV path.")
    parser.add_argument("--model", default=str(MODEL_DIR / "match_model.joblib"), help="Saved model path.")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of tournaments to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random seed.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "predictions.csv"), help="Predictions CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teams = load_teams(args.teams)
    model = load_or_train_model(args.model, args.matches)
    results = run_simulations(model, teams, simulations=args.simulations, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.head(12).to_string(index=False, formatters={
        "knockout_probability": "{:.1%}".format,
        "title_probability": "{:.1%}".format,
    }))
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

