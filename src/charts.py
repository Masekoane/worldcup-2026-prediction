"""Generate model-diagnostic and live-forecast charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "navy": "#17324d",
    "blue": "#2878b5",
    "green": "#2a9d6f",
    "gold": "#e9a23b",
    "red": "#d1495b",
    "gray": "#6b7280",
    "light": "#e8edf2",
}


def _style_axis(axis, title: str, xlabel: str = "") -> None:
    axis.set_title(title, fontsize=14, fontweight="bold", color=COLORS["navy"], pad=12)
    axis.set_xlabel(xlabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_model_comparison(comparison: pd.DataFrame, output: Path) -> None:
    ordered = comparison.sort_values("log_loss", ascending=True).copy()
    labels = ordered["model"].str.replace("_", " ").str.title()
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))

    colors = [COLORS["green"] if value else COLORS["gray"] for value in ordered["calibrated"]]
    axes[0].barh(labels, ordered["log_loss"], color=colors)
    axes[0].invert_yaxis()
    _style_axis(axes[0], "Probability Quality by Model", "Log loss (lower is better)")

    positions = np.arange(len(ordered))
    axes[1].barh(positions - 0.18, ordered["accuracy"], height=0.34, color=COLORS["blue"], label="Accuracy")
    axes[1].barh(positions + 0.18, ordered["draw_recall"], height=0.34, color=COLORS["gold"], label="Draw recall")
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 0.75)
    axes[1].legend(frameon=False)
    _style_axis(axes[1], "Accuracy and Draw Detection", "Score")
    _save(figure, output)


def plot_feature_importance(importance: pd.DataFrame, output: Path) -> None:
    ordered = importance.sort_values("importance_mean").copy()
    labels = ordered["feature"].str.replace("_", " ").str.title()
    colors = [COLORS["green"] if value >= 0 else COLORS["red"] for value in ordered["importance_mean"]]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(
        labels,
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color=colors,
        alpha=0.9,
        capsize=3,
    )
    axis.axvline(0, color=COLORS["navy"], linewidth=0.8)
    _style_axis(axis, "Permutation Feature Importance", "Increase in predictive value")
    _save(figure, output)


def plot_draw_calibration(calibration: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], label="Perfect calibration")
    axis.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_draw_rate"],
        marker="o",
        linewidth=2.5,
        color=COLORS["green"],
        label="Calibrated model",
    )
    axis.set_xlim(0, 0.6)
    axis.set_ylim(0, 0.6)
    axis.set_ylabel("Observed draw rate")
    axis.legend(frameon=False)
    _style_axis(axis, "Draw Probability Calibration", "Predicted draw probability")
    _save(figure, output)


def plot_confusion(confusion: pd.DataFrame, output: Path) -> None:
    labels = [label.replace("team_a_win", "Home win").replace("team_b_win", "Away win").title() for label in confusion.index]
    values = confusion.to_numpy()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="Blues")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] > values.max() * 0.55 else COLORS["navy"]
            axis.text(column, row, f"{values[row, column]:,}", ha="center", va="center", color=color, fontsize=12)
    axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Holdout Confusion Matrix", fontsize=14, fontweight="bold", color=COLORS["navy"], pad=12)
    figure.colorbar(image, ax=axis, shrink=0.8)
    _save(figure, output)


def plot_title_probabilities(predictions: pd.DataFrame, output: Path, top: int = 16) -> None:
    leaders = predictions.nlargest(top, "title_probability").sort_values("title_probability")
    figure, axis = plt.subplots(figsize=(10, 8))
    bars = axis.barh(leaders["team"], leaders["title_probability"], color=COLORS["blue"])
    for bar, value in zip(bars, leaders["title_probability"]):
        axis.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9)
    axis.set_xlim(0, leaders["title_probability"].max() * 1.22)
    _style_axis(axis, "2026 World Cup Title Forecast", "Title probability")
    _save(figure, output)


def plot_progression(predictions: pd.DataFrame, output: Path, top: int = 10) -> None:
    leaders = predictions.nlargest(top, "title_probability")
    columns = [
        "round_32_probability",
        "last_16_probability",
        "quarterfinal_probability",
        "semifinal_probability",
        "final_probability",
        "title_probability",
    ]
    labels = ["R32", "R16", "Quarterfinal", "Semifinal", "Final", "Champion"]
    palette = plt.cm.tab10(np.linspace(0, 1, len(leaders)))
    figure, axis = plt.subplots(figsize=(12, 7))
    for (_, team), color in zip(leaders.iterrows(), palette):
        axis.plot(labels, team[columns].to_numpy(dtype=float), marker="o", linewidth=2, label=team["team"], color=color)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Probability")
    axis.legend(ncol=2, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _style_axis(axis, "Tournament Progression Forecast")
    _save(figure, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project charts.")
    parser.add_argument("--model-dir", default="outputs/model")
    parser.add_argument("--predictions", default="outputs/live_predictions_10000.csv")
    parser.add_argument("--output-dir", default="outputs/charts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    predictions = pd.read_csv(args.predictions)
    plot_model_comparison(pd.read_csv(model_dir / "model_comparison.csv"), output_dir / "model_comparison.png")
    plot_feature_importance(pd.read_csv(model_dir / "feature_importance.csv"), output_dir / "feature_importance.png")
    plot_draw_calibration(pd.read_csv(model_dir / "draw_calibration.csv"), output_dir / "draw_calibration.png")
    confusion = pd.read_csv(model_dir / "confusion_matrix.csv", index_col=0)
    plot_confusion(confusion, output_dir / "confusion_matrix.png")
    plot_title_probabilities(predictions, output_dir / "title_probabilities.png")
    plot_progression(predictions, output_dir / "tournament_progression.png")
    print(f"Saved 6 charts to {output_dir}")


if __name__ == "__main__":
    main()
