"""Simulate the unfinished 2026 World Cup around official live results."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils import (
    OUTPUT_DIR,
    MatchPrediction,
    best_third_place_teams,
    build_match_features,
    load_teams,
    probability_lookup,
    sample_match_score,
)


STAGE_ORDER = ["LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]


def team_index(teams: pd.DataFrame) -> dict[str, pd.Series]:
    return {row["team"]: row for _, row in teams.iterrows()}


class PredictionCache:
    """Cache model probabilities for repeated tournament matchups."""

    def __init__(self, model, teams: pd.DataFrame):
        self.model = model
        self.teams = team_index(teams)
        self.cache = {}

    def get(self, home: str, away: str):
        key = (home, away)
        if key not in self.cache:
            self.cache[key] = probability_lookup(self.model, self.teams[home], self.teams[away])
        return self.cache[key]

    def precompute_all(self) -> None:
        """Predict every ordered matchup in one model call."""

        names = list(self.teams)
        pairs = [(home, away) for home in names for away in names if home != away]
        features = pd.concat(
            [build_match_features(self.teams[home], self.teams[away], model=self.model) for home, away in pairs],
            ignore_index=True,
        )
        raw = self.model.predict_proba(features)
        class_lookup = {label: index for index, label in enumerate(self.model.classes_)}
        for row_index, (home, away) in enumerate(pairs):
            home_probability = float(raw[row_index, class_lookup["team_a_win"]])
            draw_probability = float(raw[row_index, class_lookup["draw"]])
            away_probability = float(raw[row_index, class_lookup["team_b_win"]])
            total = home_probability + draw_probability + away_probability
            self.cache[(home, away)] = MatchPrediction(
                home,
                away,
                home_probability / total,
                draw_probability / total,
                away_probability / total,
            )


def fallback_round_of_32_pairs(
    group_tables: dict[str, pd.DataFrame],
    teams: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Create a deterministic bracket until the API publishes official R32 teams."""

    ranks = teams.set_index("team")["fifa_rank"].to_dict()
    winners = [table.iloc[0]["team"] for table in group_tables.values()]
    runners = [table.iloc[1]["team"] for table in group_tables.values()]
    thirds = best_third_place_teams(group_tables.values(), count=8)
    winners.sort(key=lambda team: (ranks[team], team))
    runners.sort(key=lambda team: (ranks[team], team))
    thirds.sort(key=lambda team: (ranks[team], team), reverse=True)

    pairs = list(zip(winners[:8], thirds))
    pairs.extend(zip(winners[8:], list(reversed(runners[-4:]))))
    remaining_runners = runners[:-4]
    pairs.extend(
        (remaining_runners[index], remaining_runners[-(index + 1)])
        for index in range(len(remaining_runners) // 2)
    )
    return pairs


def official_stage_pairs(schedule: pd.DataFrame, stage: str) -> list[tuple[str, str]] | None:
    fixtures = schedule.loc[schedule["stage"] == stage].sort_values(["utc_date", "match_id"])
    if fixtures.empty or fixtures[["home_team", "away_team"]].isna().any().any():
        return None
    return list(fixtures[["home_team", "away_team"]].itertuples(index=False, name=None))


def actual_knockout_winner(match: pd.Series) -> str:
    if match["winner"] == "HOME_TEAM":
        return match["home_team"]
    if match["winner"] == "AWAY_TEAM":
        return match["away_team"]
    if match["home_score"] > match["away_score"]:
        return match["home_team"]
    if match["away_score"] > match["home_score"]:
        return match["away_team"]
    raise ValueError(f"Finished knockout match {match['match_id']} has no winner")


def current_group_points(schedule: pd.DataFrame, teams: pd.DataFrame) -> dict[str, int]:
    points = {team: 0 for team in teams["team"]}
    finished = schedule.loc[(schedule["stage"] == "GROUP_STAGE") & (schedule["status"] == "FINISHED")]
    for match in finished.itertuples(index=False):
        if match.home_score > match.away_score:
            points[match.home_team] += 3
        elif match.away_score > match.home_score:
            points[match.away_team] += 3
        else:
            points[match.home_team] += 1
            points[match.away_team] += 1
    return points


def run_live_simulations(
    model,
    schedule: pd.DataFrame,
    teams: pd.DataFrame,
    simulations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the live tournament with array-based state suitable for 10,000+ runs."""

    rng = np.random.default_rng(seed)
    predictions = PredictionCache(model, teams)
    predictions.precompute_all()
    names = teams["team"].tolist()
    name_to_index = {name: index for index, name in enumerate(names)}
    ranks = teams["fifa_rank"].to_numpy(dtype=int)
    team_count = len(names)

    base_points = np.zeros(team_count, dtype=np.int16)
    base_goals_for = np.zeros(team_count, dtype=np.int16)
    base_goals_against = np.zeros(team_count, dtype=np.int16)
    unfinished_group_matches = []
    group_schedule = schedule.loc[schedule["stage"] == "GROUP_STAGE"]
    for match in group_schedule.itertuples(index=False):
        home, away = name_to_index[match.home_team], name_to_index[match.away_team]
        if match.status == "FINISHED":
            home_score, away_score = int(match.home_score), int(match.away_score)
            base_goals_for[home] += home_score
            base_goals_against[home] += away_score
            base_goals_for[away] += away_score
            base_goals_against[away] += home_score
            if home_score > away_score:
                base_points[home] += 3
            elif away_score > home_score:
                base_points[away] += 3
            else:
                base_points[[home, away]] += 1
        else:
            unfinished_group_matches.append((home, away, predictions.get(match.home_team, match.away_team)))

    group_members = {
        group: [name_to_index[name] for name in group_frame["team"]]
        for group, group_frame in teams.groupby("group", sort=True)
    }
    official_pairs = {}
    for stage in STAGE_ORDER:
        pairs = official_stage_pairs(schedule, stage)
        official_pairs[stage] = (
            [(name_to_index[home], name_to_index[away]) for home, away in pairs] if pairs is not None else None
        )
    actual_winners = {}
    finished_knockout = schedule.loc[
        (schedule["stage"].isin(STAGE_ORDER))
        & (schedule["status"] == "FINISHED")
    ].dropna(subset=["home_team", "away_team"])
    for _, match in finished_knockout.iterrows():
        winner = actual_knockout_winner(match)
        pair_key = frozenset((name_to_index[match["home_team"]], name_to_index[match["away_team"]]))
        actual_winners[pair_key] = name_to_index[winner]

    round_names = ["round_32", "last_16", "quarterfinal", "semifinal", "final"]
    counters = {round_name: np.zeros(team_count, dtype=np.int32) for round_name in round_names}
    champions = np.zeros(team_count, dtype=np.int32)

    def knockout_winner(home: int, away: int) -> int:
        fixed = actual_winners.get(frozenset((home, away)))
        if fixed is not None:
            return fixed
        prediction = predictions.get(names[home], names[away])
        home_weight = prediction.p_team_a_win + prediction.p_draw / 2
        return home if rng.random() < home_weight else away

    for _ in range(simulations):
        points = base_points.copy()
        goals_for = base_goals_for.copy()
        goals_against = base_goals_against.copy()
        for home, away, prediction in unfinished_group_matches:
            home_score, away_score = sample_match_score(prediction, rng)
            goals_for[home] += home_score
            goals_against[home] += away_score
            goals_for[away] += away_score
            goals_against[away] += home_score
            if home_score > away_score:
                points[home] += 3
            elif away_score > home_score:
                points[away] += 3
            else:
                points[[home, away]] += 1

        group_orders = {}
        for group, members in group_members.items():
            group_orders[group] = sorted(
                members,
                key=lambda index: (
                    -int(points[index]),
                    -int(goals_for[index] - goals_against[index]),
                    -int(goals_for[index]),
                    names[index],
                ),
            )
        winners = [order[0] for order in group_orders.values()]
        runners = [order[1] for order in group_orders.values()]
        third_candidates = [order[2] for order in group_orders.values()]
        thirds = sorted(
            third_candidates,
            key=lambda index: (
                -int(points[index]),
                -int(goals_for[index] - goals_against[index]),
                -int(goals_for[index]),
                names[index],
            ),
        )[:8]
        current = winners + runners + thirds
        counters["round_32"][current] += 1

        for stage in STAGE_ORDER:
            pairs = official_pairs[stage]
            if pairs is None and stage == "LAST_32":
                winners_by_rank = sorted(winners, key=lambda index: (ranks[index], names[index]))
                runners_by_rank = sorted(runners, key=lambda index: (ranks[index], names[index]))
                thirds_by_rank = sorted(thirds, key=lambda index: (ranks[index], names[index]), reverse=True)
                pairs = list(zip(winners_by_rank[:8], thirds_by_rank))
                pairs.extend(zip(winners_by_rank[8:], list(reversed(runners_by_rank[-4:]))))
                remaining_runners = runners_by_rank[:-4]
                pairs.extend(
                    (remaining_runners[index], remaining_runners[-(index + 1)])
                    for index in range(len(remaining_runners) // 2)
                )
            elif pairs is None:
                pairs = [(current[index], current[index + 1]) for index in range(0, len(current), 2)]
            current = [knockout_winner(home, away) for home, away in pairs]
            reached_name = {
                "LAST_32": "last_16",
                "LAST_16": "quarterfinal",
                "QUARTER_FINALS": "semifinal",
                "SEMI_FINALS": "final",
            }.get(stage)
            if reached_name:
                counters[reached_name][current] += 1
        champions[current[0]] += 1

    current_points_map = {names[index]: int(base_points[index]) for index in range(team_count)}
    rows = []
    for team in teams.itertuples(index=False):
        index = name_to_index[team.team]
        rows.append(
            {
                "team": team.team,
                "group": team.group,
                "fifa_rank": int(team.fifa_rank),
                "current_points": current_points_map[team.team],
                "round_32_probability": counters["round_32"][index] / simulations,
                "last_16_probability": counters["last_16"][index] / simulations,
                "quarterfinal_probability": counters["quarterfinal"][index] / simulations,
                "semifinal_probability": counters["semifinal"][index] / simulations,
                "final_probability": counters["final"][index] / simulations,
                "title_probability": champions[index] / simulations,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["title_probability", "final_probability", "round_32_probability", "fifa_rank"],
        ascending=[False, False, False, True],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live World Cup simulations around official results.")
    parser.add_argument("--schedule", default="data/processed/worldcup_2026_schedule.csv")
    parser.add_argument("--teams", default="data/processed/teams_2026.csv")
    parser.add_argument("--model", default="models/match_model.joblib")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "live_predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = pd.read_csv(args.schedule)
    teams = load_teams(args.teams)
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"No trained model found at {model_path}. Run 'python src/prepare_data.py' "
            "and 'python src/train.py' first, or enable retraining in the dashboard."
        )
    model = joblib.load(model_path)
    results = run_live_simulations(model, schedule, teams, args.simulations, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    probability_columns = [column for column in results.columns if column.endswith("_probability")]
    formatters = {column: "{:.1%}".format for column in probability_columns}
    print(results.head(12).to_string(index=False, formatters=formatters))
    print(f"Saved live predictions to {output}")


if __name__ == "__main__":
    main()
