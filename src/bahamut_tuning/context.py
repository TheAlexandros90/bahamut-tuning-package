from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .helpers import normalize_targets, normalize_task_type, validate_scoring_for_task
from .types import TuningContext

logger = logging.getLogger(__name__)


class TuningContextAdapter:
    """Normalize upstream artifacts while preserving upstream decisions."""

    def normalize(
        self,
        *,
        df: pd.DataFrame | None = None,
        target: str | list[str] | None = None,
        targets: list[str] | None = None,
        task_type: str | None = None,
        selected_algorithm: str | None = None,
        recommended_algorithms: list[str] | None = None,
        recommender_reason: str | None = None,
        X_train: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
        y_train: pd.Series | pd.DataFrame | None = None,
        y_test: pd.Series | pd.DataFrame | None = None,
        cv: Any = None,
        scorer: str | None = None,
        preprocessor: Any = None,
        metadata: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        feature_lists: dict[str, list[str]] | None = None,
    ) -> TuningContext:
        resolved_targets = normalize_targets(target=target, targets=targets)
        normalized_task_type = normalize_task_type(task_type)

        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if X_train is not None and not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if X_test is not None and not isinstance(X_test, pd.DataFrame):
            raise TypeError("X_test must be a pandas DataFrame.")
        if (X_train is None) != (y_train is None):
            raise ValueError("X_train and y_train must be provided together.")
        if (X_test is None) != (y_test is None):
            raise ValueError("X_test and y_test must be provided together.")

        resolved_metadata = dict(metadata or {})
        resolved_diagnostics = dict(diagnostics or {})
        resolved_feature_lists = dict(feature_lists or {})

        if df is not None and resolved_targets:
            missing = [column for column in resolved_targets if column not in df.columns]
            if missing and y_train is None:
                raise ValueError(
                    f"Target columns missing from df and no upstream y_train provided: {missing}"
                )
            if missing:
                resolved_diagnostics.setdefault(
                    "target_resolution_note",
                    "Targets are not present in df, but upstream y_train/y_test artifacts were provided.",
                )

        if normalized_task_type and scorer:
            validate_scoring_for_task(normalized_task_type, scorer)

        if X_train is not None:
            resolved_diagnostics.setdefault("upstream_split", "provided")
        if cv is not None:
            resolved_diagnostics.setdefault("upstream_cv", type(cv).__name__)
        if preprocessor is not None:
            resolved_metadata.setdefault("preprocessor_source", "upstream")
        if recommended_algorithms and selected_algorithm and selected_algorithm not in recommended_algorithms:
            resolved_diagnostics.setdefault(
                "algorithm_note",
                "Selected algorithm differs from the first upstream recommendation.",
            )

        logger.info(
            "Normalized tuning context: task_type=%s selected_algorithm=%s upstream_split=%s upstream_cv=%s",
            normalized_task_type,
            selected_algorithm,
            X_train is not None,
            cv is not None,
        )

        return TuningContext(
            df=df,
            target=resolved_targets,
            task_type=normalized_task_type,
            selected_algorithm=selected_algorithm,
            recommended_algorithms=recommended_algorithms or [],
            recommender_reason=recommender_reason,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv=cv,
            scorer=scorer,
            preprocessor=preprocessor,
            metadata=resolved_metadata,
            diagnostics=resolved_diagnostics,
            feature_lists=resolved_feature_lists,
        )


def normalize_context_inputs(**kwargs: Any) -> TuningContext:
    """Helper function for unit-testable context normalization."""

    return TuningContextAdapter().normalize(**kwargs)
