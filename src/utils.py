"""Shared helpers for World Cup model training and tournament simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MATCH_COLUMNS = {
    "date",
    "team_a",
    "team_b",
    "team_a_score",
    "team_b_score",
    "team_a_rank",
    "team_b_rank",
}
TEAM_COLUMNS = {"team", "confederation", "fifa_rank", "elo_rating", "group"}


@dataclass(frozen=True)
class MatchPrediction:
    """Probability estimates for a single match from team A's perspective."""

    team_a: str
    team_b: str
    p_team_a_win: float
    p_draw: float
    p_team_b_win: float


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file and raise a clear error if it is missing."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")
    return pd.read_csv(csv_path)


def validate_columns(frame: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    """Validate that a dataframe contains the columns expected by the pipeline."""

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    """Load historical match data used for training."""

    frame = read_csv(path or DATA_DIR / "matches_sample.csv")
    validate_columns(frame, MATCH_COLUMNS, "match data")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("match data contains invalid dates")
    return frame


def load_teams(path: str | Path | None = None) -> pd.DataFrame:
    """Load the tournament team field used for simulation."""

    frame = read_csv(path or DATA_DIR / "teams_2026_sample.csv")
    validate_columns(frame, TEAM_COLUMNS, "team data")
    if frame["team"].duplicated().any():
        duplicates = sorted(frame.loc[frame["team"].duplicated(), "team"].unique())
        raise ValueError(f"team data contains duplicate teams: {', '.join(duplicates)}")
    if len(frame) != 48:
        raise ValueError(f"team data must contain 48 teams for the 2026 format, found {len(frame)}")
    return frame.copy()


def build_training_frame(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model features and a three-class target from historical matches."""

    validate_columns(matches, MATCH_COLUMNS, "match data")
    frame = matches.copy()
    frame["rank_diff"] = frame["team_a_rank"] - frame["team_b_rank"]
    frame["score_diff"] = frame["team_a_score"] - frame["team_b_score"]
    frame["abs_rank_diff"] = frame["rank_diff"].abs()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    frame["is_recent"] = (frame["year"] >= frame["year"].max() - 4).astype(int)
    frame["target"] = np.select(
        [frame["score_diff"] > 0, frame["score_diff"] < 0],
        ["team_a_win", "team_b_win"],
        default="draw",
    )
    features = frame[["rank_diff", "abs_rank_diff", "team_a_rank", "team_b_rank", "is_recent"]]
    return features, frame["target"]


def build_match_features(team_a: pd.Series, team_b: pd.Series, model=None) -> pd.DataFrame:
    """Build the feature row expected by either the legacy or enriched model."""

    team_a_rank = float(team_a["fifa_rank"])
    team_b_rank = float(team_b["fifa_rank"])
    rank_diff = team_a_rank - team_b_rank
    profiles = getattr(model, "team_profiles_", {}) if model is not None else {}
    profile_a = profiles.get(team_a["team"], {})
    profile_b = profiles.get(team_b["team"], {})

    def difference(key: str) -> float:
        value_a = profile_a.get(key, np.nan)
        value_b = profile_b.get(key, np.nan)
        return float(value_a - value_b) if pd.notna(value_a) and pd.notna(value_b) else np.nan

    values = {
        "rank_diff": rank_diff,
        "abs_rank_diff": abs(rank_diff),
        "team_a_rank": team_a_rank,
        "team_b_rank": team_b_rank,
        "is_recent": 1,
        "fifa_points_diff": difference("fifa_points"),
        "form_points_diff": difference("form_points"),
        "form_win_rate_diff": difference("form_win_rate"),
        "form_goal_diff": difference("form_goal_diff"),
        "matches_played_diff": difference("matches_played"),
        "home_advantage": 0,
        "neutral": 1,
        "tournament_importance": 4,
    }
    columns = getattr(model, "feature_columns_", list(values)) if model is not None else list(values)
    defaults = getattr(model, "feature_defaults_", {}) if model is not None else {}
    row = {column: values.get(column, defaults.get(column, 0.0)) for column in columns}
    row = {column: defaults.get(column, 0.0) if pd.isna(value) else value for column, value in row.items()}
    return pd.DataFrame([row], columns=columns)


def probability_lookup(model, team_a: pd.Series, team_b: pd.Series) -> MatchPrediction:
    """Predict win/draw/loss probabilities for team A against team B."""

    features = build_match_features(team_a, team_b, model=model)
    probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))
    p_a = float(probabilities.get("team_a_win", 0.0))
    p_draw = float(probabilities.get("draw", 0.0))
    p_b = float(probabilities.get("team_b_win", 0.0))
    total = p_a + p_draw + p_b
    if total <= 0:
        return MatchPrediction(team_a["team"], team_b["team"], 1 / 3, 1 / 3, 1 / 3)
    return MatchPrediction(team_a["team"], team_b["team"], p_a / total, p_draw / total, p_b / total)


def sample_match_score(prediction: MatchPrediction, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a plausible scoreline from outcome probabilities via independent Poisson draws.

    The 0.75/1.7/0.45 coefficients are hand-tuned to produce realistic low-scoring
    international-football scorelines; they are not fit from data.
    """

    expected_a = 0.75 + 1.7 * prediction.p_team_a_win + 0.45 * prediction.p_draw
    expected_b = 0.75 + 1.7 * prediction.p_team_b_win + 0.45 * prediction.p_draw
    return int(rng.poisson(expected_a)), int(rng.poisson(expected_b))


def choose_match_winner(
    prediction: MatchPrediction,
    rng: np.random.Generator,
    allow_draw: bool = False,
) -> str | None:
    """Sample a match outcome. Knockout matches convert draw mass into a winner."""

    if allow_draw:
        outcome = rng.choice(
            ["team_a", "draw", "team_b"],
            p=[prediction.p_team_a_win, prediction.p_draw, prediction.p_team_b_win],
        )
        if outcome == "draw":
            return None
        return prediction.team_a if outcome == "team_a" else prediction.team_b

    team_a_weight = prediction.p_team_a_win + prediction.p_draw / 2
    team_b_weight = prediction.p_team_b_win + prediction.p_draw / 2
    total = team_a_weight + team_b_weight
    return rng.choice([prediction.team_a, prediction.team_b], p=[team_a_weight / total, team_b_weight / total])


def group_stage_pairs(group: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the round-robin pairings for one four-team group."""

    teams = group["team"].tolist()
    return [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i + 1, len(teams))]


def best_third_place_teams(group_tables: Iterable[pd.DataFrame], count: int = 8) -> list[str]:
    """Select the best third-place teams using points and goal difference."""

    thirds = []
    for table in group_tables:
        ordered = table.sort_values(
            ["points", "goal_difference", "goals_for", "team"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        thirds.append(ordered.iloc[2])
    third_frame = pd.DataFrame(thirds)
    return (
        third_frame.sort_values(
            ["points", "goal_difference", "goals_for", "team"],
            ascending=[False, False, False, True],
        )
        .head(count)["team"]
        .tolist()
    )
