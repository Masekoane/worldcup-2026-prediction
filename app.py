"""Streamlit dashboard for the World Cup forecast project."""

from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODEL_OUTPUTS = OUTPUTS / "model"
CHARTS = OUTPUTS / "charts"
PREDICTIONS = OUTPUTS / "live_predictions_10000.csv"

PROBABILITY_COLUMNS = [
    "round_32_probability",
    "last_16_probability",
    "quarterfinal_probability",
    "semifinal_probability",
    "final_probability",
    "title_probability",
]


st.set_page_config(
    page_title="World Cup 2026 Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --accent: #3987e5;
        --ink: #ffffff;
        --muted: #c3c2b7;
        --line: rgba(255, 255, 255, 0.10);
        --surface: #1a1a19;
        --good: #0ca30c;
        --warn: #fab219;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3 {
        letter-spacing: 0;
    }

    .stat-tile {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        min-height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.3rem;
    }

    .stat-tile .stat-label {
        color: var(--muted);
        font-size: 0.84rem;
    }

    .stat-tile .stat-value {
        color: var(--ink);
        font-size: 1.7rem;
        font-weight: 600;
        line-height: 1.2;
    }

    .stat-tile .stat-caption {
        font-size: 0.86rem;
        font-weight: 600;
    }

    .stat-caption.good { color: var(--good); }
    .stat-caption.warn { color: var(--warn); }
    .stat-caption.neutral { color: var(--muted); font-weight: 500; }

    .forecast-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.8rem;
    }

    .forecast-header p {
        color: var(--muted);
        margin: 0.25rem 0 0;
        max-width: 720px;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def stat_tile(label: str, value: str, caption: str | None = None, tone: str = "neutral") -> str:
    """Render a stat-tile card: label, value, and an optional muted/status caption."""

    caption_html = ""
    if caption:
        caption_html = f'<div class="stat-caption {html.escape(tone)}">{html.escape(caption)}</div>'
    return (
        '<div class="stat-tile">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        f"{caption_html}"
        "</div>"
    )


def percent(value: float) -> str:
    return f"{value:.1%}"


def file_age(path: Path) -> str:
    if not path.exists():
        return "missing"
    timestamp = pd.Timestamp(path.stat().st_mtime, unit="s")
    return timestamp.strftime("%Y-%m-%d %H:%M")


@st.cache_data(show_spinner=False)
def read_predictions(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    for column in PROBABILITY_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values("title_probability", ascending=False)


@st.cache_data(show_spinner=False)
def read_csv(path: str, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col)


COMMAND_TIMEOUT_SECONDS = 600


def run_command(label: str, args: list[str]) -> tuple[bool, str]:
    with st.spinner(label):
        try:
            completed = subprocess.run(
                [sys.executable, *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {COMMAND_TIMEOUT_SECONDS}s (step may be stuck, e.g. a network stall)."
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part.strip())
    return completed.returncode == 0, output.strip()


def run_pipeline(refresh_api: bool, retrain_model: bool, simulations: int) -> None:
    steps: list[tuple[str, list[str]]] = []
    fetch_args = ["src/fetch_data.py"]
    if refresh_api:
        fetch_args.append("--refresh")
    steps.append(("Fetching football-data.org matches", fetch_args))
    steps.append(("Processing live tournament data", ["src/process_live_data.py"]))
    if retrain_model:
        steps.append(("Preparing historical training data", ["src/prepare_data.py"]))
        steps.append(("Training calibrated model", ["src/train.py"]))
    steps.append(
        (
            f"Running {simulations:,} tournament simulations",
            [
                "src/simulate_live.py",
                "--simulations",
                str(simulations),
                "--output",
                str(PREDICTIONS),
            ],
        )
    )
    steps.append(("Rendering charts", ["src/charts.py", "--predictions", str(PREDICTIONS)]))

    progress = st.progress(0)
    log_box = st.empty()
    logs: list[str] = []
    for index, (label, command) in enumerate(steps, start=1):
        success, output = run_command(label, command)
        logs.append(f"$ python {' '.join(command)}\n{output or '(no output)'}")
        log_box.code("\n\n".join(logs), language="text")
        progress.progress(index / len(steps))
        if not success:
            st.error(f"Stopped at: {label}")
            return

    st.cache_data.clear()
    st.success("Forecast updated.")


def chart_image(name: str) -> None:
    path = CHARTS / name
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"Run the forecast to create {name}.")


with st.sidebar:
    st.title("Controls")
    st.caption("Refresh data, rerun simulations, and rebuild charts from one place.")

    simulations = st.number_input(
        "Simulations",
        min_value=100,
        max_value=100_000,
        value=10_000,
        step=1_000,
    )
    refresh_api = st.toggle("Refresh API data", value=False)
    retrain_model = st.toggle("Retrain model", value=False)

    if st.button("Run Forecast", type="primary", use_container_width=True):
        run_pipeline(refresh_api=refresh_api, retrain_model=retrain_model, simulations=int(simulations))

    st.divider()
    st.caption("Artifacts")
    st.write(f"Predictions: `{file_age(PREDICTIONS)}`")
    st.write(f"Diagnostics: `{file_age(MODEL_OUTPUTS / 'metrics.json')}`")


st.markdown(
    """
    <div class="forecast-header">
      <div>
        <h1>World Cup 2026 Forecast</h1>
        <p>Live tournament probabilities from your calibrated match model and Monte Carlo simulator.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not PREDICTIONS.exists():
    st.warning("No live prediction file found yet. Use Run Forecast in the sidebar.")
    st.stop()

predictions = read_predictions(str(PREDICTIONS))
leader = predictions.iloc[0]
title_sum = predictions["title_probability"].sum()
finished_points = int(predictions["current_points"].sum()) if "current_points" in predictions else 0
teams_count = len(predictions)

probability_balanced = abs(title_sum - 1.0) < 0.01

metric_columns = st.columns(4)
with metric_columns[0]:
    st.markdown(
        stat_tile(
            "Title favorite",
            str(leader["team"]),
            f'{percent(float(leader["title_probability"]))} title probability',
            "good",
        ),
        unsafe_allow_html=True,
    )
with metric_columns[1]:
    st.markdown(stat_tile("Teams in field", f"{teams_count}"), unsafe_allow_html=True)
with metric_columns[2]:
    st.markdown(
        stat_tile(
            "Probability check",
            percent(float(title_sum)),
            "sums to 100% across all teams" if probability_balanced else "does not sum to 100% — check inputs",
            "good" if probability_balanced else "warn",
        ),
        unsafe_allow_html=True,
    )
with metric_columns[3]:
    st.markdown(
        stat_tile("Current group points", f"{finished_points}", "earned in locked results"),
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Forecast", "Groups", "Charts", "Model", "Run Log"])

with tabs[0]:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Title Race")
        top_n = st.slider("Teams shown", 5, min(32, len(predictions)), 16)
        chart_data = predictions.nlargest(top_n, "title_probability")[["team", "title_probability"]]

        bar_height = 26
        bars = (
            alt.Chart(chart_data)
            .mark_bar(size=14, cornerRadiusEnd=4, color="#3987e5")
            .encode(
                x=alt.X(
                    "title_probability:Q",
                    title="Title probability",
                    axis=alt.Axis(format="%", gridColor="#2c2c2a", domainColor="#383835", tickColor="#383835"),
                    scale=alt.Scale(domain=[0, chart_data["title_probability"].max() * 1.18]),
                ),
                y=alt.Y("team:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("title_probability:Q", title="Title probability", format=".1%"),
                ],
            )
        )
        labels = bars.mark_text(align="left", dx=6, color="#c3c2b7", fontSize=11).encode(
            text=alt.Text("title_probability:Q", format=".1%")
        )
        combined = (
            (bars + labels)
            .properties(height=bar_height * len(chart_data) + 40)
            .configure_axis(labelColor="#ffffff", titleColor="#c3c2b7")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(combined, use_container_width=True)

    with right:
        st.subheader("Leaderboard")
        shown = predictions[
            [
                "team",
                "group",
                "fifa_rank",
                "current_points",
                "round_32_probability",
                "quarterfinal_probability",
                "semifinal_probability",
                "final_probability",
                "title_probability",
            ]
        ].copy()
        st.dataframe(
            shown,
            hide_index=True,
            use_container_width=True,
            column_config={
                "round_32_probability": st.column_config.ProgressColumn("R32", format="percent", min_value=0, max_value=1),
                "quarterfinal_probability": st.column_config.ProgressColumn("QF", format="percent", min_value=0, max_value=1),
                "semifinal_probability": st.column_config.ProgressColumn("SF", format="percent", min_value=0, max_value=1),
                "final_probability": st.column_config.ProgressColumn("Final", format="percent", min_value=0, max_value=1),
                "title_probability": st.column_config.ProgressColumn("Title", format="percent", min_value=0, max_value=1),
            },
        )

with tabs[1]:
    st.subheader("Group View")
    groups = ["All"] + sorted(predictions["group"].dropna().unique().tolist())
    selected_group = st.selectbox("Group", groups, index=0)
    group_data = predictions if selected_group == "All" else predictions[predictions["group"] == selected_group]
    st.dataframe(
        group_data.sort_values(["group", "current_points", "title_probability"], ascending=[True, False, False]),
        hide_index=True,
        use_container_width=True,
    )

with tabs[2]:
    st.subheader("Charts")
    chart_tabs = st.tabs(["Title", "Progression", "Model Comparison", "Draws", "Features", "Confusion"])
    with chart_tabs[0]:
        chart_image("title_probabilities.png")
    with chart_tabs[1]:
        chart_image("tournament_progression.png")
    with chart_tabs[2]:
        chart_image("model_comparison.png")
    with chart_tabs[3]:
        chart_image("draw_calibration.png")
    with chart_tabs[4]:
        chart_image("feature_importance.png")
    with chart_tabs[5]:
        chart_image("confusion_matrix.png")

with tabs[3]:
    st.subheader("Model Diagnostics")
    comparison_path = MODEL_OUTPUTS / "model_comparison.csv"
    importance_path = MODEL_OUTPUTS / "feature_importance.csv"
    report_path = MODEL_OUTPUTS / "classification_report.txt"

    if comparison_path.exists():
        comparison = read_csv(str(comparison_path))
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    else:
        st.info("Model comparison is not available yet.")

    col_a, col_b = st.columns(2)
    with col_a:
        if importance_path.exists():
            st.caption("Feature importance")
            st.dataframe(read_csv(str(importance_path)), hide_index=True, use_container_width=True)
    with col_b:
        if report_path.exists():
            st.caption("Classification report")
            st.code(report_path.read_text(encoding="utf-8"), language="text")

with tabs[4]:
    st.subheader("How To Run")
    st.code("streamlit run app.py", language="powershell")
    st.markdown(
        """
        <p class="small-note">
        Use the sidebar button for normal updates. Keep retraining off unless you changed the historical
        training data or model code.
        </p>
        """,
        unsafe_allow_html=True,
    )
