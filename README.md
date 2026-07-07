# World Cup 2026 Prediction

A machine learning project for estimating 2026 FIFA World Cup outcomes from historical international results, FIFA rankings, and Monte Carlo tournament simulations.

The repository now contains a complete runnable baseline:

- historical match feature engineering
- a scikit-learn match outcome model
- a 48-team World Cup group-stage and knockout simulator
- sample data files for local development
- tests for the training and simulation pipeline

## Project Structure

```text
.
├── data/
│   ├── raw/               # fetched/reference data (fifa rankings, results, live API cache)
│   ├── processed/         # generated training and live-tournament tables
│   ├── matches_sample.csv
│   └── teams_2026_sample.csv
├── src/
│   ├── __init__.py
│   ├── fetch_data.py      # pulls WC26 fixtures from football-data.org
│   ├── prepare_data.py    # builds the leakage-safe historical training table
│   ├── process_live_data.py  # turns the raw API feed into teams/results/schedule
│   ├── train.py           # trains, calibrates, and compares match outcome models
│   ├── simulate.py        # offline Monte Carlo simulation on sample data
│   ├── simulate_live.py   # live tournament simulation around real results
│   ├── charts.py          # renders model and tournament charts
│   └── utils.py           # shared feature/prediction helpers
├── tests/
│   └── test_pipeline.py
├── app.py                 # Streamlit dashboard
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dashboard UI

Run the browser dashboard from the project folder:

```bash
streamlit run app.py
```

The dashboard shows the live title race, group view, generated charts, and model diagnostics. Use the sidebar button to refresh the API data, process the tournament schedule, run 10,000+ simulations, and rebuild charts without typing every command manually. Keep retraining off for quick forecast updates; turn it on after changing training data or model code.

## Football-data.org API

Create your local environment file in PowerShell:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your football-data.org token:

```text
FOOTBALL_DATA_API_KEY=your_real_token
```

The `.env` file is ignored by Git. Never commit or share it.

Fetch the 2026 World Cup matches:

```bash
python src/fetch_data.py
```

The command writes `data/raw/football_data_wc_matches.csv`. Raw API responses are cached under `data/api_cache/` for 24 hours to conserve the free request quota. Use `--refresh` only when you need a fresh response:

```bash
python src/fetch_data.py --refresh
```

Useful filters include:

```bash
python src/fetch_data.py --status SCHEDULED
python src/fetch_data.py --date-from 2026-06-11 --date-to 2026-07-19
```

## Live Tournament Forecast

Convert the API response into official teams, completed results, and the full schedule:

```bash
python src/process_live_data.py
```

This creates:

```text
data/processed/worldcup_2026_results.csv
data/processed/worldcup_2026_schedule.csv
data/processed/teams_2026.csv
```

Run 10,000 live simulations:

```bash
python src/simulate_live.py --simulations 10000
```

Completed matches are locked to their real scores. Paused and scheduled group matches are simulated, and the output includes each team's current points plus probabilities of reaching every knockout round and winning the title. Results are saved to `outputs/live_predictions.csv`.

The API publishes knockout fixtures before it knows the qualifying teams. While those team fields are empty, the simulator uses a deterministic seeded round-of-32 fallback. It automatically uses official API pairings as soon as both teams are populated.

To update the live forecast as matches finish:

```bash
python src/fetch_data.py --refresh
python src/process_live_data.py
python src/simulate_live.py --simulations 10000
```

Prepare the real historical training dataset:

```bash
python src/prepare_data.py
```

Train, calibrate, and compare the candidate models:

```bash
python src/train.py
```

Training uses chronological 70% training, 15% calibration, and 15% test partitions. It compares:

- majority-class baseline
- FIFA-ranking heuristic baseline
- gradient boosting
- draw-balanced gradient boosting
- balanced random forest
- calibrated versions of every learned model

The selected artifact uses sigmoid probability calibration and a draw threshold tuned only on calibration data. Reports are written to:

```text
outputs/model/model_comparison.csv
outputs/model/feature_importance.csv
outputs/model/draw_calibration.csv
outputs/model/confusion_matrix.csv
outputs/model/metrics.json
outputs/model/classification_report.txt
```

Generate model and tournament charts:

```bash
python src/charts.py
```

Charts are saved under `outputs/charts/`:

- `model_comparison.png`
- `feature_importance.png`
- `draw_calibration.png`
- `confusion_matrix.png`
- `title_probabilities.png`
- `tournament_progression.png`

Run 1,000 tournament simulations:

```bash
python src/simulate.py --simulations 1000
```

The simulator writes results to `outputs/predictions.csv` and prints the top teams by title probability.

## Data

`data/matches_sample.csv` is a compact development dataset with international matches from recent major tournaments.

`data/teams_2026_sample.csv` is a sample 48-team tournament field. It is intentionally not presented as an official final 2026 World Cup field. Replace it with updated teams, FIFA rankings, Elo ratings, and group assignments when official data is available.

Research links:

- https://www.researchgate.net/publication/367796814_Predicting_the_Winner_of_Games_in_World_Cup_Soccer_Matches
- https://www.technologyreview.com/2018/06/12/2659/machine-learning-predicts-world-cup-winner/

Expected team columns:

```text
team, confederation, fifa_rank, elo_rating, group
```

Expected match columns:

```text
date, team_a, team_b, team_a_score, team_b_score, team_a_rank, team_b_rank
```

## How It Works

1. `src/prepare_data.py` matches each result to the latest FIFA ranking available before kickoff and creates rolling five-match form features without future-data leakage.
2. `src/train.py` splits matches chronologically into 70% training, 15% calibration, and 15% test. It compares baselines and calibrated gradient boosting/random forest candidates using calibration-split scores, then reports and ships the selected candidate as-is (fit on the training split, calibrated on the calibration split) so the test-set metrics describe the exact artifact that gets saved.
3. `src/simulate.py` loads the selected model and a 48-team tournament field.
4. Each group plays a four-team round robin.
5. The top two teams in each group plus the eight best third-place teams advance to a 32-team knockout bracket.
6. Repeated simulations estimate each team's knockout and title probabilities.

The enriched model uses FIFA rank and points, rolling points per game, rolling win rate, rolling goal difference, match experience, venue status, and tournament importance. Accuracy, balanced accuracy, macro F1, draw precision/recall, multiclass Brier score, and log loss are reported. Calibrated probability quality matters more to Monte Carlo simulation than winner accuracy alone.

## Testing

Run the test suite:

```bash
pytest
```

The tests verify that the sample data loads, the model trains, the group stage returns 32 qualifiers, and tournament probabilities are well formed.

## Next Improvements

- Add a larger historical dataset from Kaggle, football-data, or an API.
- Add current FIFA rankings and Elo ratings as separate data sources.
- Use rolling team strength features instead of only match-day FIFA rank differences.
- Calibrate predicted probabilities with `CalibratedClassifierCV`.
- Add host-country, travel-distance, rest-day, and confederation features.
- Replace the seeded placeholder knockout bracket with the official FIFA bracket mapping once finalized.
