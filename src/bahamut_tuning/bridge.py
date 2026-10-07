"""Bahamut integration without importing or installing Bahamut itself."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit


class IndexedGroupKFold(GroupKFold):
    """Resolve group labels by row index, including subsets in nested CV.

    SearchCV does not forward outer group labels to its inner CV by default.
    Binding labels to the training index keeps both levels disjoint without
    introducing identifiers as model features. X must remain a DataFrame.
    """

    def __init__(self, groups: pd.Series, n_splits: int = 5):
        super().__init__(n_splits=n_splits)
        if not isinstance(groups, pd.Series) or not groups.index.is_unique:
            raise ValueError("Group-aware CV requires a Series with a unique training index.")
        if groups.isna().any():
            raise ValueError("Group labels must not be missing.")
        self.indexed_groups = groups.copy()

    def split(self, X, y=None, groups=None):
        if not isinstance(X, pd.DataFrame) or not X.index.is_unique:
            raise ValueError("Group-aware CV requires a DataFrame with a unique index.")
        if not X.index.isin(self.indexed_groups.index).all():
            raise ValueError("CV received rows outside the bound training partition.")
        resolved = self.indexed_groups.reindex(X.index)
        if groups is not None and list(groups) != resolved.tolist():
            raise ValueError("Provided group labels disagree with the training groups.")
        yield from super().split(X, y, groups=resolved.to_numpy())


def _target_series(value, name: str):
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("The tuning bridge currently requires a single target column.")
        value = value.iloc[:, 0]
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series or a single-column DataFrame.")
    return value.copy()


def context_from_bahamut(
    segments: Mapping[str, Any],
    *,
    task_type: str | None = None,
    temporal: bool | None = None,
    cv=None,
    n_splits: int = 5,
    gap: int = 0,
    random_state: int = 42,
    **kwargs,
) -> dict[str, Any]:
    if not isinstance(segments, Mapping):
        raise TypeError("Pass the dictionary returned by BahamutSplit.ejecutar_segmentacion().")
    config = segments.get("split_config") or {}
    if not isinstance(config, Mapping):
        raise TypeError("split_config must be a mapping.")
    problem = config.get("problema")
    if task_type is None:
        task_type = {"clasificacion": "classification", "regresion": "regression"}.get(problem)
    if task_type is None:
        raise ValueError("Provide task_type='classification' or 'regression' for this bundle.")
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    resolved = {}
    for split in ("train", "test"):
        X, y = segments.get(f"X_{split}"), segments.get(f"y_{split}")
        if X is None and y is None and split == "test":
            continue
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X_{split} must be a pandas DataFrame.")
        y = _target_series(y, f"y_{split}")
        if not X.index.equals(y.index):
            raise ValueError(f"X_{split} and y_{split} must have identical indices in the same order.")
        if y.name in X.columns:
            raise ValueError(f"X_{split} contains the target column {y.name!r}.")
        resolved[f"X_{split}"] = X.copy()
        resolved[f"y_{split}"] = y
    X_train = resolved["X_train"]
    if not X_train.index.is_unique:
        raise ValueError("The bridge requires a unique source index; reset it before splitting in Bahamut.")
    if "X_test" in resolved:
        X_test = resolved["X_test"]
        if not X_test.index.is_unique or not X_train.index.intersection(X_test.index).empty:
            raise ValueError("Train and test indices must be unique and disjoint.")
        if not X_train.columns.equals(X_test.columns):
            raise ValueError("Train and test must have the same predictor columns in the same order.")
    groups = segments.get("groups_train")
    if (config.get("use_group_split") or segments.get("groups_test") is not None) and groups is None:
        raise ValueError("This Bahamut bundle uses groups but groups_train is missing.")
    if groups is not None and (not isinstance(groups, pd.Series) or not groups.index.equals(X_train.index)):
        raise ValueError("groups_train must be a Series aligned with X_train.")
    if groups is not None and segments.get("groups_test") is not None:
        test_groups = segments["groups_test"]
        if not isinstance(test_groups, pd.Series) or "X_test" not in resolved or not test_groups.index.equals(resolved["X_test"].index):
            raise ValueError("groups_test must be a Series aligned with X_test.")
        if set(groups).intersection(test_groups):
            raise ValueError("Train and test groups must be disjoint.")
    known_temporal = problem == "series_temporales"
    if known_temporal and temporal is False:
        raise ValueError("A temporal Bahamut bundle cannot be declared non-temporal.")
    if temporal is None:
        if not config and groups is None and cv is None:
            raise ValueError("This older bundle has no split_config. Specify temporal=True/False or an explicit cv.")
        temporal = known_temporal
    if temporal and groups is not None:
        raise ValueError("The bridge cannot combine temporal and group splitting.")
    if cv is not None and (temporal or groups is not None):
        raise ValueError("For temporal/group bundles use n_splits and gap; an arbitrary cv could override their split policy.")
    if cv is None:
        if temporal:
            cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        elif groups is not None:
            cv = IndexedGroupKFold(groups, n_splits=n_splits)
        elif task_type == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    forbidden = set(kwargs) & {"df", "target", "targets", "X_train", "y_train", "X_test", "y_test"}
    if forbidden:
        raise ValueError(f"Do not override Bahamut artifacts through {sorted(forbidden)}.")
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata["bahamut_split_config"] = dict(config)
    metadata["validation_rows_reserved"] = len(segments["X_validation"]) if segments.get("X_validation") is not None else 0
    return dict(resolved, task_type=task_type, target=resolved["y_train"].name or "target", cv=cv,
                random_state=random_state, metadata=metadata, **kwargs)
