"""Fetch and cache competition matches from football-data.org v4."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


API_BASE_URL = "https://api.football-data.org/v4"


def load_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without adding a third-party dependency."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_matches_url(
    competition: str,
    season: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
) -> str:
    """Build the football-data.org competition matches URL."""

    params = {}
    if season is not None:
        params["season"] = season
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to
    if status:
        params["status"] = status
    query = f"?{urlencode(params)}" if params else ""
    return f"{API_BASE_URL}/competitions/{competition.upper()}/matches{query}"


def fetch_json(url: str, token: str, timeout: int = 30) -> dict:
    """Request JSON with clear messages for common API failures."""

    request = Request(
        url,
        headers={
            "X-Auth-Token": token,
            "Accept": "application/json",
            "User-Agent": "worldcup-2026-prediction/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        messages = {
            401: "The API key was rejected. Check FOOTBALL_DATA_API_KEY in .env.",
            403: "This endpoint or competition is not included in the current API plan.",
            429: "The free-tier request limit was reached. Wait before retrying.",
        }
        raise RuntimeError(messages.get(exc.code, f"football-data.org returned HTTP {exc.code}: {detail}")) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach football-data.org: {exc.reason}") from exc


def cache_is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600
    return age_hours <= max_age_hours


def load_or_fetch(
    url: str,
    token: str,
    cache_path: Path,
    refresh: bool = False,
    max_cache_age_hours: float = 24,
) -> tuple[dict, bool]:
    """Use a fresh local response when possible to protect the free quota."""

    if not refresh and cache_is_fresh(cache_path, max_cache_age_hours):
        return json.loads(cache_path.read_text(encoding="utf-8")), True
    payload = fetch_json(url, token)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, False


def matches_to_frame(payload: dict) -> pd.DataFrame:
    """Flatten API match objects into a stable CSV schema."""

    rows = []
    for match in payload.get("matches", []):
        score = match.get("score") or {}
        full_time = score.get("fullTime") or {}
        home = match.get("homeTeam") or {}
        away = match.get("awayTeam") or {}
        competition = match.get("competition") or payload.get("competition") or {}
        rows.append(
            {
                "match_id": match.get("id"),
                "utc_date": match.get("utcDate"),
                "status": match.get("status"),
                "competition_code": competition.get("code"),
                "competition": competition.get("name"),
                "season_start": (match.get("season") or {}).get("startDate"),
                "stage": match.get("stage"),
                "group": match.get("group"),
                "matchday": match.get("matchday"),
                "home_team_id": home.get("id"),
                "home_team": home.get("name"),
                "away_team_id": away.get("id"),
                "away_team": away.get("name"),
                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),
                "winner": score.get("winner"),
                "duration": score.get("duration"),
                "last_updated": match.get("lastUpdated"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["utc_date", "match_id"], na_position="last").reset_index(drop=True)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch competition matches from football-data.org.")
    parser.add_argument("--competition", default="WC", help="Competition code, for example WC.")
    parser.add_argument("--season", type=int, default=2026, help="Competition starting year.")
    parser.add_argument("--date-from", help="Optional YYYY-MM-DD lower date bound.")
    parser.add_argument("--date-to", help="Optional YYYY-MM-DD upper date bound.")
    parser.add_argument("--status", help="Optional match status filter such as SCHEDULED or FINISHED.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--cache-dir", default="data/api_cache")
    parser.add_argument("--output", default="data/raw/football_data_wc_matches.csv")
    parser.add_argument("--refresh", action="store_true", help="Ignore the local 24-hour cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(args.env_file)
    token = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not token:
        raise SystemExit(
            "Missing FOOTBALL_DATA_API_KEY. Create .env from .env.example and add your key."
        )

    url = build_matches_url(
        args.competition,
        season=args.season,
        date_from=args.date_from,
        date_to=args.date_to,
        status=args.status,
    )
    cache_name = f"{args.competition.lower()}_{args.season}_matches.json"
    payload, used_cache = load_or_fetch(
        url,
        token,
        Path(args.cache_dir) / cache_name,
        refresh=args.refresh,
    )
    matches = matches_to_frame(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output, index=False)
    source = "cache" if used_cache else "football-data.org"
    print(f"Loaded {len(matches):,} matches from {source}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
