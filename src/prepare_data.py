"""Build a leakage-safe training dataset from raw results and FIFA rankings."""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


ALIASES = {
    "united states": "usa",
    "south korea": "korea republic",
    "north korea": "korea dpr",
    "iran": "ir iran",
    "ivory coast": "cote divoire",
    "cape verde": "cabo verde",
    "cape verde islands": "cabo verde",
    "bosnia herzegovina": "bosnia and herzegovina",
    "czech republic": "czechia",
    "china": "china pr",
    "congo": "congo dr",
    "curacao": "curacao",
    "turkiye": "turkey",
}

FEATURE_COLUMNS = [
    "rank_diff",
    "fifa_points_diff",
    "form_points_diff",
    "form_win_rate_diff",
    "form_goal_diff",
    "matches_played_diff",
    "home_advantage",
    "neutral",
    "tournament_importance",
]


def normalize_team_name(value: str) -> str:
    """Create a stable join key across result and ranking naming conventions."""

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    compact = text.replace(" ", "")
    alias = ALIASES.get(text, text)
    if alias == text:
        alias = ALIASES.get(compact, text)
    return alias.replace(" ", "")


def tournament_importance(name: str) -> int:
    """Map competitions to a simple match-importance scale."""

    value = str(name).lower()
    if "fifa world cup" in value and "qualification" not in value:
        return 4
    if any(token in value for token in ("euro", "copa america", "african cup", "asian cup", "gold cup")):
        return 3 if "qualification" not in value else 2
    if "qualification" in value or "nations league" in value:
        return 2
    return 1


def attach_rankings(matches: pd.DataFrame, rankings: pd.DataFrame, side: str) -> pd.DataFrame:
    """Attach the latest ranking published on or before each match date."""

    team_col = f"{side}_team"
    left = matches.reset_index()
    left["team_name"] = left[team_col]
    left["team_key"] = left["team_name"].map(normalize_team_name)
    right = rankings[["rank_date", "team_key", "rank", "total_points"]].copy()

    # merge_asof requires the timestamp to be globally sorted even when using `by`.
    left = left.sort_values(["date", "team_key"])
    right = right.sort_values(["rank_date", "team_key"])
    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="rank_date",
        by="team_key",
        direction="backward",
        tolerance=pd.Timedelta(days=550),
    )
    return (
        merged.rename(columns={"rank": f"{side}_rank", "total_points": f"{side}_fifa_points"})
        .drop(columns=["team_name", "team_key", "rank_date"])
        .set_index("index")
        .sort_index()
    )


def add_recent_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add pre-match rolling form features without using the current result."""

    history: dict[str, deque[tuple[int, int, int]]] = defaultdict(lambda: deque(maxlen=window))
    rows: list[dict[str, float]] = []

    def summary(team: str) -> tuple[float, float, float, int]:
        games = history[team]
        if not games:
            return 1.0, 1 / 3, 0.0, 0
        points = sum(game[0] for game in games) / len(games)
        win_rate = sum(game[0] == 3 for game in games) / len(games)
        goal_diff = sum(game[1] - game[2] for game in games) / len(games)
        return points, win_rate, goal_diff, len(games)

    for match in matches.itertuples(index=False):
        home = summary(match.home_team)
        away = summary(match.away_team)
        rows.append(
            {
                "home_form_points": home[0],
                "away_form_points": away[0],
                "home_form_win_rate": home[1],
                "away_form_win_rate": away[1],
                "home_form_goal_diff": home[2],
                "away_form_goal_diff": away[2],
                "home_matches_played": home[3],
                "away_matches_played": away[3],
            }
        )

        home_score, away_score = int(match.home_score), int(match.away_score)
        if home_score > away_score:
            home_points, away_points = 3, 0
        elif home_score < away_score:
            home_points, away_points = 0, 3
        else:
            home_points = away_points = 1
        history[match.home_team].append((home_points, home_score, away_score))
        history[match.away_team].append((away_points, away_score, home_score))

    return pd.concat([matches.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def prepare_training_data(
    results: pd.DataFrame,
    rankings: pd.DataFrame,
    start_date: str = "1993-01-01",
) -> pd.DataFrame:
    """Create the processed match-level training table."""

    matches = results.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    matches = matches.loc[matches["date"] >= pd.Timestamp(start_date)].sort_values("date").reset_index(drop=True)
    matches["neutral"] = matches["neutral"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    ranking_frame = rankings.copy()
    ranking_frame["rank_date"] = pd.to_datetime(ranking_frame["rank_date"], errors="coerce")
    ranking_frame["team_key"] = ranking_frame["country_full"].map(normalize_team_name)
    ranking_frame = ranking_frame.dropna(subset=["rank_date", "rank", "total_points"])

    matches = attach_rankings(matches, ranking_frame, "home")
    matches = attach_rankings(matches, ranking_frame, "away")
    matches = matches.dropna(subset=["home_rank", "away_rank"]).sort_values("date").reset_index(drop=True)
    matches = add_recent_form(matches)

    matches["rank_diff"] = matches["home_rank"] - matches["away_rank"]
    matches["fifa_points_diff"] = matches["home_fifa_points"] - matches["away_fifa_points"]
    matches["form_points_diff"] = matches["home_form_points"] - matches["away_form_points"]
    matches["form_win_rate_diff"] = matches["home_form_win_rate"] - matches["away_form_win_rate"]
    matches["form_goal_diff"] = matches["home_form_goal_diff"] - matches["away_form_goal_diff"]
    matches["matches_played_diff"] = matches["home_matches_played"] - matches["away_matches_played"]
    matches["home_advantage"] = (matches["neutral"] == 0).astype(int)
    matches["tournament_importance"] = matches["tournament"].map(tournament_importance)
    matches["target"] = np.select(
        [matches["home_score"] > matches["away_score"], matches["home_score"] < matches["away_score"]],
        ["team_a_win", "team_b_win"],
        default="draw",
    )
    matches = matches.rename(columns={"home_team": "team_a", "away_team": "team_b"})

    output_columns = [
        "date", "team_a", "team_b", "home_score", "away_score", "tournament", "target",
        "home_rank", "away_rank", "home_fifa_points", "away_fifa_points",
        "home_form_points", "away_form_points", "home_form_win_rate", "away_form_win_rate",
        "home_form_goal_diff", "away_form_goal_diff", "home_matches_played", "away_matches_played",
        *FEATURE_COLUMNS,
    ]
    return matches[output_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare historical matches for model training.")
    parser.add_argument("--results", default="data/raw/results.csv")
    parser.add_argument("--rankings", default="data/raw/fifa_ranking-2024-06-20.csv")
    parser.add_argument("--output", default="data/processed/matches_training.csv")
    parser.add_argument("--start-date", default="1993-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    rankings = pd.read_csv(args.rankings)
    prepared = prepare_training_data(results, rankings, start_date=args.start_date)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    print(f"Prepared {len(prepared):,} ranked matches from {prepared['date'].min()} to {prepared['date'].max()}")
    print(f"Class distribution:\n{prepared['target'].value_counts(normalize=True).round(3)}")
    print(f"Saved training data to {output}")


if __name__ == "__main__":
    main()
