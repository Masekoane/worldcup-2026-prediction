"""Train, calibrate, compare, and diagnose match-outcome models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from prepare_data import FEATURE_COLUMNS
from utils import MODEL_DIR, OUTPUT_DIR, build_training_frame, load_matches


CLASS_ORDER = ["draw", "team_a_win", "team_b_win"]


def _candidate_models(random_state: int) -> dict[str, Pipeline]:
    common = dict(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=20,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=random_state,
    )
    return {
        "gradient_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("classifier", HistGradientBoostingClassifier(**common)),
            ]
        ),
        "draw_balanced_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        **common,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "balanced_random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _feature_frame(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if set(FEATURE_COLUMNS).issubset(matches.columns) and "target" in matches.columns:
        return matches[FEATURE_COLUMNS].copy(), matches["target"].copy(), FEATURE_COLUMNS.copy()
    features, target = build_training_frame(matches)
    return features, target, features.columns.tolist()


def _aligned_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    lookup = {label: index for index, label in enumerate(model.classes_)}
    return np.column_stack([raw[:, lookup[label]] if label in lookup else np.zeros(len(features)) for label in CLASS_ORDER])


def _labels_from_probabilities(probabilities: np.ndarray, draw_threshold: float) -> np.ndarray:
    labels = np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
    labels[probabilities[:, 0] >= draw_threshold] = "draw"
    return labels


def _tune_draw_threshold(probabilities: np.ndarray, target: pd.Series) -> float:
    best_threshold = 1 / 3
    best_score = -np.inf
    for threshold in np.linspace(0.18, 0.40, 45):
        predictions = _labels_from_probabilities(probabilities, float(threshold))
        score = f1_score(target, predictions, labels=CLASS_ORDER, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _metrics(target: pd.Series, probabilities: np.ndarray, draw_threshold: float) -> tuple[dict[str, float], np.ndarray]:
    predictions = _labels_from_probabilities(probabilities, draw_threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        predictions,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    one_hot = pd.get_dummies(pd.Categorical(target, categories=CLASS_ORDER)).to_numpy(dtype=float)
    scores = {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "log_loss": float(log_loss(target, probabilities, labels=CLASS_ORDER)),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "macro_f1": float(f1_score(target, predictions, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "draw_precision": float(precision[0]),
        "draw_recall": float(recall[0]),
        "draw_f1": float(f1[0]),
        "draw_threshold": float(draw_threshold),
    }
    return scores, predictions


def _fit_calibrator(model: Pipeline, x_calibration: pd.DataFrame, y_calibration: pd.Series):
    if set(y_calibration.unique()) != set(model.classes_):
        return model
    minimum_class_count = int(y_calibration.value_counts().min())
    if minimum_class_count < 2:
        return model
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(model),
        method="sigmoid",
        cv=min(5, minimum_class_count),
        ensemble=False,
    )
    calibrated.fit(x_calibration, y_calibration)
    return calibrated


def _negative_multiclass_log_loss(estimator, features: pd.DataFrame, target: pd.Series) -> float:
    probabilities = _aligned_probabilities(estimator, features)
    return -float(log_loss(target, probabilities, labels=CLASS_ORDER))


def _majority_baseline(y_train: pd.Series, rows: int) -> np.ndarray:
    distribution = y_train.value_counts(normalize=True)
    values = np.array([distribution.get(label, 0.0) for label in CLASS_ORDER], dtype=float)
    return np.tile(values / values.sum(), (rows, 1))


def _ranking_baseline(features: pd.DataFrame) -> np.ndarray:
    rank_diff = features["rank_diff"].to_numpy(dtype=float)
    home_advantage = features.get("home_advantage", pd.Series(np.zeros(len(features)))).to_numpy(dtype=float)
    draw = 0.16 + 0.18 * np.exp(-np.abs(rank_diff) / 28)
    conditional_home = 1 / (1 + np.exp(np.clip(rank_diff / 18 - home_advantage * 0.25, -20, 20)))
    home = (1 - draw) * conditional_home
    away = (1 - draw) * (1 - conditional_home)
    return np.column_stack([draw, home, away])


def _attach_prediction_metadata(model, matches: pd.DataFrame, feature_columns: list[str]) -> None:
    model.feature_columns_ = feature_columns
    features, _, _ = _feature_frame(matches)
    model.feature_defaults_ = features.median(numeric_only=True).to_dict()
    model.team_profiles_ = {}

    if {"team_a", "team_b", "date"}.issubset(matches.columns):
        for row in matches.sort_values("date").itertuples(index=False):
            for side, team in (("home", row.team_a), ("away", row.team_b)):
                model.team_profiles_[team] = {
                    "rank": getattr(row, f"{side}_rank", np.nan),
                    "fifa_points": getattr(row, f"{side}_fifa_points", np.nan),
                    "form_points": getattr(row, f"{side}_form_points", np.nan),
                    "form_win_rate": getattr(row, f"{side}_form_win_rate", np.nan),
                    "form_goal_diff": getattr(row, f"{side}_form_goal_diff", np.nan),
                    "matches_played": getattr(row, f"{side}_matches_played", np.nan),
                }


def train_model(matches: pd.DataFrame, random_state: int = 42) -> tuple[object, dict[str, object]]:
    """Evaluate baselines and calibrated candidates using chronological splits."""

    frame = matches.sort_values("date").reset_index(drop=True) if "date" in matches.columns else matches.copy()
    features, target, feature_columns = _feature_frame(frame)
    train_end = max(3, int(len(frame) * 0.70))
    calibration_end = max(train_end + 1, int(len(frame) * 0.85))
    calibration_end = min(calibration_end, len(frame) - 1)
    x_train, y_train = features.iloc[:train_end], target.iloc[:train_end]
    x_calibration, y_calibration = features.iloc[train_end:calibration_end], target.iloc[train_end:calibration_end]
    x_test, y_test = features.iloc[calibration_end:], target.iloc[calibration_end:]
    if y_test.empty or y_calibration.empty:
        raise ValueError("At least ten matches are required for train/calibration/test splits")

    comparison_rows = []
    baseline_probabilities_calibration = {
        "majority_baseline": _majority_baseline(y_train, len(x_calibration)),
        "ranking_baseline": _ranking_baseline(x_calibration),
    }
    baseline_probabilities_test = {
        "majority_baseline": _majority_baseline(y_train, len(x_test)),
        "ranking_baseline": _ranking_baseline(x_test),
    }
    for name, test_probabilities in baseline_probabilities_test.items():
        # Threshold is tuned on the calibration split so the baseline never sees
        # test labels before its held-out score is computed, same as the candidates below.
        threshold = (
            _tune_draw_threshold(baseline_probabilities_calibration[name], y_calibration)
            if name == "ranking_baseline"
            else 1.0
        )
        scores, _ = _metrics(y_test, test_probabilities, threshold)
        comparison_rows.append({"model": name, "calibrated": False, **scores})

    evaluated_models = {}
    for name, candidate in _candidate_models(random_state).items():
        candidate.fit(x_train, y_train)
        raw_calibration = _aligned_probabilities(candidate, x_calibration)
        raw_threshold = _tune_draw_threshold(raw_calibration, y_calibration)
        raw_test = _aligned_probabilities(candidate, x_test)
        raw_scores, _ = _metrics(y_test, raw_test, raw_threshold)
        comparison_rows.append({"model": name, "calibrated": False, **raw_scores})

        calibrated = _fit_calibrator(candidate, x_calibration, y_calibration)
        calibrated_calibration_probabilities = _aligned_probabilities(calibrated, x_calibration)
        draw_threshold = _tune_draw_threshold(calibrated_calibration_probabilities, y_calibration)
        # Selection score is computed on the calibration split only, so architecture
        # choice never depends on the test set. Test metrics below are reported,
        # not competed on.
        selection_scores, _ = _metrics(y_calibration, calibrated_calibration_probabilities, draw_threshold)

        test_probabilities = _aligned_probabilities(calibrated, x_test)
        test_scores, predictions = _metrics(y_test, test_probabilities, draw_threshold)
        comparison_rows.append({"model": f"{name}_calibrated", "calibrated": True, **test_scores})
        evaluated_models[name] = {
            "model": calibrated,
            "selection_scores": selection_scores,
            "scores": test_scores,
            "probabilities": test_probabilities,
            "predictions": predictions,
            "threshold": draw_threshold,
        }

    # Reward calibrated probability quality while requiring useful draw discrimination.
    # Selection uses calibration-split scores only; the test set is reserved for the
    # one-time held-out report below.
    selected_name = min(
        evaluated_models,
        key=lambda name: evaluated_models[name]["selection_scores"]["log_loss"]
        - 0.08 * evaluated_models[name]["selection_scores"]["draw_f1"],
    )
    selected = evaluated_models[selected_name]
    selected_model = selected["model"]

    importance = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring=_negative_multiclass_log_loss,
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    draw_true = (y_test.to_numpy() == "draw").astype(int)
    fraction_positive, mean_predicted = calibration_curve(
        draw_true,
        selected["probabilities"][:, 0],
        n_bins=10,
        strategy="quantile",
    )
    calibration_frame = pd.DataFrame(
        {"mean_predicted_probability": mean_predicted, "observed_draw_rate": fraction_positive}
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_test, selected["predictions"], labels=CLASS_ORDER),
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    )

    # Ship the exact model that was scored above (trained on x_train, calibrated on
    # x_calibration) so metrics.json describes the artifact that gets saved, not a
    # separately refit model. Refitting on the test split before shipping would also
    # make the "held-out" test score reported below invalid for the shipped model.
    selected_model.draw_threshold_ = selected["threshold"]
    _attach_prediction_metadata(selected_model, frame, feature_columns)

    report = classification_report(y_test, selected["predictions"], labels=CLASS_ORDER, zero_division=0)
    comparisons = pd.DataFrame(comparison_rows).sort_values(["log_loss", "draw_f1"], ascending=[True, False])
    metrics = {
        "selected_model": f"{selected_name}_calibrated",
        "accuracy": selected["scores"]["accuracy"],
        "balanced_accuracy": selected["scores"]["balanced_accuracy"],
        "log_loss": selected["scores"]["log_loss"],
        "brier_score": selected["scores"]["brier_score"],
        "draw_precision": selected["scores"]["draw_precision"],
        "draw_recall": selected["scores"]["draw_recall"],
        "draw_f1": selected["scores"]["draw_f1"],
        "draw_threshold": selected["threshold"],
        "report": report,
        "rows": len(frame),
        "train_rows": len(x_train),
        "calibration_rows": len(x_calibration),
        "test_rows": len(x_test),
        "classes": CLASS_ORDER,
        "features": feature_columns,
        "_comparisons": comparisons,
        "_feature_importance": importance_frame,
        "_calibration": calibration_frame,
        "_confusion": confusion,
    }
    return selected_model, metrics


def save_model(model, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_diagnostics(metrics: dict[str, object], report_dir: str | Path) -> Path:
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metrics["_comparisons"].to_csv(directory / "model_comparison.csv", index=False)
    metrics["_feature_importance"].to_csv(directory / "feature_importance.csv", index=False)
    metrics["_calibration"].to_csv(directory / "draw_calibration.csv", index=False)
    metrics["_confusion"].to_csv(directory / "confusion_matrix.csv")
    public_metrics = {key: value for key, value in metrics.items() if not key.startswith("_") and key != "report"}
    (directory / "metrics.json").write_text(json.dumps(public_metrics, indent=2), encoding="utf-8")
    (directory / "classification_report.txt").write_text(str(metrics["report"]), encoding="utf-8")
    return directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train calibrated World Cup match-outcome models.")
    parser.add_argument("--matches", default=str(Path("data") / "processed" / "matches_training.csv"))
    parser.add_argument("--output", default=str(MODEL_DIR / "match_model.joblib"))
    parser.add_argument("--report-dir", default=str(OUTPUT_DIR / "model"))
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.matches)
    matches = pd.read_csv(path, parse_dates=["date"]) if path.exists() else load_matches("data/matches_sample.csv")
    model, metrics = train_model(matches, random_state=args.random_state)
    output_path = save_model(model, args.output)
    report_dir = save_diagnostics(metrics, args.report_dir)
    print(
        f"Trained on {metrics['rows']:,} matches "
        f"({metrics['train_rows']:,} train / {metrics['calibration_rows']:,} calibration / {metrics['test_rows']:,} test)"
    )
    print(metrics["_comparisons"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Selected model: {metrics['selected_model']}")
    print(
        f"Draw precision={metrics['draw_precision']:.3f}, recall={metrics['draw_recall']:.3f}, "
        f"F1={metrics['draw_f1']:.3f}, threshold={metrics['draw_threshold']:.3f}"
    )
    print(metrics["report"])
    print(f"Saved model to {output_path}")
    print(f"Saved diagnostics to {report_dir}")


if __name__ == "__main__":
    main()
