"""Convert the football-data.org feed into live tournament inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import normalize_team_name


API_COLUMNS = {
    "match_id",
    "utc_date",
    "status",
    "stage",
    "group",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


def validate_api_matches(matches: pd.DataFrame) -> None:
    missing = sorted(API_COLUMNS.difference(matches.columns))
    if missing:
        raise ValueError(f"API match data is missing columns: {', '.join(missing)}")


def latest_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    """Return one latest available ranking row per normalized country name."""

    frame = rankings.copy()
    frame["rank_date"] = pd.to_datetime(frame["rank_date"], errors="coerce")
    frame["team_key"] = frame["country_full"].map(normalize_team_name)
    frame = frame.dropna(subset=["rank_date", "rank"]).sort_values("rank_date")
    return frame.drop_duplicates("team_key", keep="last").set_index("team_key")


def build_team_table(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Build the official 48-team field from group-stage fixtures."""

    group_matches = matches.loc[matches["stage"] == "GROUP_STAGE"].copy()
    home = group_matches[["home_team", "group"]].rename(columns={"home_team": "team"})
    away = group_matches[["away_team", "group"]].rename(columns={"away_team": "team"})
    teams = pd.concat([home, away], ignore_index=True).dropna().drop_duplicates("team")
    if len(teams) != 48:
        raise ValueError(f"Expected 48 unique group-stage teams, found {len(teams)}")

    rank_lookup = latest_rankings(rankings)
    teams["team_key"] = teams["team"].map(normalize_team_name)
    teams["fifa_rank"] = teams["team_key"].map(rank_lookup["rank"])
    teams["confederation"] = teams["team_key"].map(rank_lookup["confederation"])
    missing_rank = teams["fifa_rank"].isna()
    fallback_start = int(rank_lookup["rank"].max()) + 1
    teams.loc[missing_rank, "fifa_rank"] = np.arange(fallback_start, fallback_start + missing_rank.sum())
    teams["confederation"] = teams["confederation"].fillna("UNKNOWN")
    teams["fifa_rank"] = teams["fifa_rank"].astype(int)
    teams["elo_rating"] = (2050 - teams["fifa_rank"] * 7).clip(lower=1200).astype(int)
    return teams[["team", "confederation", "fifa_rank", "elo_rating", "group"]].sort_values(
        ["group", "fifa_rank", "team"]
    )


def build_finished_results(matches: pd.DataFrame) -> pd.DataFrame:
    """Return only final results; live or future scores must not enter training."""

    finished = matches.loc[matches["status"] == "FINISHED"].copy()
    finished["date"] = pd.to_datetime(finished["utc_date"], utc=True).dt.date.astype(str)
    finished["tournament"] = "FIFA World Cup"
    finished["neutral"] = True
    columns = [
        "match_id",
        "date",
        "stage",
        "group",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    ]
    return finished[columns].sort_values(["date", "match_id"])


def build_schedule(matches: pd.DataFrame) -> pd.DataFrame:
    """Preserve every official fixture while normalizing order and timestamps."""

    schedule = matches.copy()
    schedule["utc_date"] = pd.to_datetime(schedule["utc_date"], utc=True)
    return schedule.sort_values(["utc_date", "match_id"]).reset_index(drop=True)


def process_live_data(matches: pd.DataFrame, rankings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_api_matches(matches)
    schedule = build_schedule(matches)
    results = build_finished_results(schedule)
    teams = build_team_table(schedule, rankings)
    return results, schedule, teams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare live 2026 World Cup inputs.")
    parser.add_argument("--api-matches", default="data/raw/football_data_wc_matches.csv")
    parser.add_argument("--rankings", default="data/raw/fifa_ranking-2024-06-20.csv")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = pd.read_csv(args.api_matches)
    rankings = pd.read_csv(args.rankings)
    results, schedule, teams = process_live_data(matches, rankings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "worldcup_2026_results.csv", index=False)
    schedule.to_csv(output_dir / "worldcup_2026_schedule.csv", index=False)
    teams.to_csv(output_dir / "teams_2026.csv", index=False)
    print(f"Prepared {len(teams)} official teams")
    print(f"Locked {len(results)} finished matches; {len(schedule) - len(results)} fixtures remain live or scheduled")
    print(f"Saved live inputs to {output_dir}")


if __name__ == "__main__":
    main()
