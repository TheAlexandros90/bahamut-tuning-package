from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import get_scorer

logger = logging.getLogger(__name__)

_CLASSIFICATION_SCORING_TOKENS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "jaccard",
    "roc_auc",
    "average_precision",
    "neg_log_loss",
)
_REGRESSION_SCORING_TOKENS = (
    "neg_mean",
    "neg_root_mean",
    "neg_median_absolute_error",
    "r2",
    "explained_variance",
    "max_error",
    "d2_",
)
_TASK_TYPE_ALIASES = {
    "classification": "classification",
    "classifier": "classification",
    "binary": "classification",
    "multiclass": "classification",
    "regression": "regression",
    "regressor": "regression",
}


def normalize_targets(
    target: str | list[str] | tuple[str, ...] | None = None,
    targets: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if targets:
        return list(targets)
    if isinstance(target, str):
        return [target]
    if target:
        return list(target)
    return []


def normalize_task_type(task_type: str | None) -> str | None:
    if task_type is None:
        return None
    normalized = task_type.strip().lower()
    if normalized not in _TASK_TYPE_ALIASES:
        raise ValueError(f"Unsupported task_type: {task_type}")
    return _TASK_TYPE_ALIASES[normalized]


def infer_default_scoring(task_type: str) -> str:
    if task_type == "classification":
        return "accuracy"
    if task_type == "regression":
        return "neg_root_mean_squared_error"
    raise ValueError(f"Unsupported task_type: {task_type}")


def validate_scoring_for_task(task_type: str, scoring: str) -> None:
    try:
        get_scorer(scoring)
    except Exception as err:
        raise ValueError(f"Unknown sklearn scorer: {scoring}") from err

    lowered = scoring.lower()
    looks_classification = any(token in lowered for token in _CLASSIFICATION_SCORING_TOKENS)
    looks_regression = any(token in lowered for token in _REGRESSION_SCORING_TOKENS)

    if task_type == "classification" and looks_regression and not looks_classification:
        raise ValueError(f"Scoring '{scoring}' looks incompatible with classification.")
    if task_type == "regression" and looks_classification and not looks_regression:
        raise ValueError(f"Scoring '{scoring}' looks incompatible with regression.")


def build_cv_summary(cv: Any) -> str:
    if cv is None:
        return "No upstream CV splitter; fallback cv=5."

    name = type(cv).__name__
    n_splits = getattr(cv, "n_splits", None)
    if n_splits is None:
        splitter_getter = getattr(cv, "get_n_splits", None)
        if callable(splitter_getter):
            try:
                n_splits = splitter_getter()
            except Exception:
                n_splits = None

    if n_splits is None:
        return f"Using upstream CV splitter: {name}."
    return f"Using upstream CV splitter: {name}(n_splits={n_splits})."


def summarize_dataframe_context(
    df: pd.DataFrame | None,
    targets: list[str] | None = None,
    max_rows: int = 12,
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame({"info": ["No raw df provided; using upstream split artifacts."]})

    target_set = set(targets or [])
    rows: list[dict[str, Any]] = []
    for column in df.columns[:max_rows]:
        series = df[column]
        sample = series.dropna().iloc[0] if not series.dropna().empty else None
        rows.append(
            {
                "column": column,
                "role": "target" if column in target_set else "feature",
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2),
                "n_unique": int(series.nunique(dropna=True)),
                "sample": sample,
            }
        )
    return pd.DataFrame(rows)


def resolve_algorithm_source(
    selection_algorithm: str | None,
    context_algorithm: str | None,
    recommended_algorithm: str | None,
) -> str:
    if selection_algorithm:
        return "user"
    if context_algorithm:
        return "upstream_selected"
    if recommended_algorithm:
        return "upstream_recommendation"
    return "unknown"


def top_configs_from_cv_results(results: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if results.empty:
        return []
    sorted_results = results.sort_values("rank_test_score").head(limit)
    return [
        {
            "rank": int(row.rank_test_score),
            "score": float(row.mean_test_score),
            "params": row.params,
        }
        for row in sorted_results.itertuples()
    ]


def infer_algorithm_recommendations(
    *,
    task_type: str,
    df: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    available_algorithms: list[str] | tuple[str, ...] | None = None,
    top_k: int = 3,
) -> tuple[list[str], str]:
    available = list(available_algorithms or [])
    defaults = {
        "classification": [
            "RandomForestClassifier",
            "LogisticRegression",
            "GradientBoostingClassifier",
            "ExtraTreesClassifier",
            "SVC",
        ],
        "regression": [
            "RandomForestRegressor",
            "ElasticNet",
            "GradientBoostingRegressor",
            "ExtraTreesRegressor",
            "Ridge",
        ],
    }

    if task_type not in defaults:
        raise ValueError(f"Unsupported task_type: {task_type}")

    def _filter_candidates(candidates: list[str]) -> list[str]:
        if not available:
            return candidates[:top_k]
        ordered = [name for name in candidates if name in available]
        ordered.extend(name for name in available if name not in ordered)
        return ordered[:top_k]

    resolved_targets = list(targets or [])
    if df is None or not resolved_targets or any(target not in df.columns for target in resolved_targets):
        recommendations = _filter_candidates(defaults[task_type])
        reason = (
            "No upstream recommendation was provided. Using generic task-level defaults because the dataset "
            "profile was not fully available to analyze."
        )
        return recommendations, reason

    X = df.drop(columns=[target for target in resolved_targets if target in df.columns], errors="ignore")
    n_rows, n_features = X.shape
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    categorical_ratio = len(categorical_columns) / max(1, n_features)

    if task_type == "classification":
        class_count: int | None = None
        if len(resolved_targets) == 1 and resolved_targets[0] in df.columns:
            class_count = int(df[resolved_targets[0]].nunique(dropna=True))

        if n_features >= 40 and not categorical_columns:
            preferred = ["LogisticRegression", "RandomForestClassifier", "SVC"]
            heuristic = "wide mostly numeric classification dataset"
        elif n_rows < 1500 and not categorical_columns:
            preferred = ["LogisticRegression", "SVC", "RandomForestClassifier"]
            heuristic = "small mostly numeric classification dataset"
        elif categorical_ratio > 0.25 or n_rows >= 5000:
            preferred = ["RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier"]
            heuristic = "mixed-feature or larger classification dataset"
        else:
            preferred = ["RandomForestClassifier", "LogisticRegression", "GradientBoostingClassifier"]
            heuristic = "general-purpose classification dataset"

        recommendations = _filter_candidates(preferred)
        reason = (
            f"No upstream recommendation was provided. Generated fallback recommendations from the dataset profile: "
            f"{n_rows} rows, {n_features} features, {len(numeric_columns)} numeric, {len(categorical_columns)} categorical"
            + (f", {class_count} target classes" if class_count is not None else "")
            + f". Heuristic used: {heuristic}."
        )
        return recommendations, reason

    if n_features >= 40 and not categorical_columns:
        preferred = ["ElasticNet", "Ridge", "RandomForestRegressor"]
        heuristic = "wide mostly numeric regression dataset"
    elif n_rows < 1500 and not categorical_columns:
        preferred = ["ElasticNet", "RandomForestRegressor", "GradientBoostingRegressor"]
        heuristic = "small mostly numeric regression dataset"
    elif categorical_ratio > 0.25 or n_rows >= 5000:
        preferred = ["RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor"]
        heuristic = "mixed-feature or larger regression dataset"
    else:
        preferred = ["RandomForestRegressor", "GradientBoostingRegressor", "ElasticNet"]
        heuristic = "general-purpose regression dataset"

    recommendations = _filter_candidates(preferred)
    reason = (
        f"No upstream recommendation was provided. Generated fallback recommendations from the dataset profile: "
        f"{n_rows} rows, {n_features} features, {len(numeric_columns)} numeric, {len(categorical_columns)} categorical. "
        f"Heuristic used: {heuristic}."
    )
    return recommendations, reason
