from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _get_xgboost_estimators() -> tuple[type[Any] | None, type[Any] | None]:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception:
        return None, None
    return XGBClassifier, XGBRegressor


def available_estimators(task_type: str, random_state: int = 42) -> dict[str, BaseEstimator]:
    xgb_classifier, xgb_regressor = _get_xgboost_estimators()

    if task_type == "classification":
        models: dict[str, BaseEstimator] = {
            "LogisticRegression": LogisticRegression(max_iter=3000),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
            "RandomForestClassifier": RandomForestClassifier(random_state=random_state),
            "ExtraTreesClassifier": ExtraTreesClassifier(random_state=random_state),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "SVC": SVC(probability=True),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        }
        if xgb_classifier is not None:
            models["XGBoostClassifier"] = xgb_classifier(
                random_state=random_state,
                eval_metric="logloss",
            )
        return models

    if task_type == "regression":
        models = {
            "Ridge": Ridge(),
            "Lasso": Lasso(max_iter=5000),
            "ElasticNet": ElasticNet(max_iter=5000),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state),
            "RandomForestRegressor": RandomForestRegressor(random_state=random_state),
            "ExtraTreesRegressor": ExtraTreesRegressor(random_state=random_state),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        }
        if xgb_regressor is not None:
            models["XGBoostRegressor"] = xgb_regressor(random_state=random_state)
        return models

    raise ValueError(f"Unsupported task_type: {task_type}")


def build_search_space(algorithm: str) -> dict[str, list[Any]]:
    spaces: dict[str, dict[str, list[Any]]] = {
        "LogisticRegression": {
            "model__C": np.logspace(-4, 2, 9).tolist(),
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear", "saga"],
        },
        "DecisionTreeClassifier": {
            "model__max_depth": [None, 3, 5, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__criterion": ["gini", "entropy", "log_loss"],
        },
        "RandomForestClassifier": {
            "model__n_estimators": [100, 200, 400, 700],
            "model__max_depth": [None, 5, 10, 20, 35],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__bootstrap": [True, False],
        },
        "ExtraTreesClassifier": {
            "model__n_estimators": [100, 200, 400, 700],
            "model__max_depth": [None, 5, 10, 20, 35],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__bootstrap": [True, False],
        },
        "KNeighborsClassifier": {
            "model__n_neighbors": [3, 5, 7, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
            "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        },
        "SVC": {
            "model__C": np.logspace(-3, 2, 8).tolist(),
            "model__gamma": ["scale", "auto"] + np.logspace(-4, 0, 5).tolist(),
            "model__kernel": ["rbf", "linear", "poly"],
        },
        "GradientBoostingClassifier": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__max_depth": [2, 3, 5],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },
        "XGBoostClassifier": {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [3, 5, 7, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5, 8],
            "model__reg_alpha": [0, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
        "Ridge": {"model__alpha": np.logspace(-4, 3, 12).tolist()},
        "Lasso": {"model__alpha": np.logspace(-5, 1, 10).tolist()},
        "ElasticNet": {
            "model__alpha": np.logspace(-5, 1, 10).tolist(),
            "model__l1_ratio": [0.05, 0.1, 0.25, 0.5, 0.7, 0.9, 0.98],
        },
        "DecisionTreeRegressor": {
            "model__max_depth": [None, 3, 5, 8, 12, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__criterion": ["squared_error", "friedman_mse", "absolute_error"],
        },
        "RandomForestRegressor": {
            "model__n_estimators": [100, 200, 400, 700],
            "model__max_depth": [None, 5, 10, 20, 35],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__bootstrap": [True, False],
        },
        "ExtraTreesRegressor": {
            "model__n_estimators": [100, 200, 400, 700],
            "model__max_depth": [None, 5, 10, 20, 35],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__bootstrap": [True, False],
        },
        "KNeighborsRegressor": {
            "model__n_neighbors": [3, 5, 7, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
            "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        },
        "SVR": {
            "model__C": np.logspace(-3, 2, 8).tolist(),
            "model__gamma": ["scale", "auto"] + np.logspace(-4, 0, 5).tolist(),
            "model__kernel": ["rbf", "linear"],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.4],
        },
        "GradientBoostingRegressor": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__max_depth": [2, 3, 5],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },
        "XGBoostRegressor": {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [3, 5, 7, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5, 8],
            "model__reg_alpha": [0, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
    }
    if algorithm not in spaces:
        raise ValueError(f"No search space defined for algorithm: {algorithm}")
    return spaces[algorithm]


# Hyperparameters whose candidate grids stand in for a continuous range.
# Matching is done on the *leaf* name (after the last "__"), never as a
# substring: the old substring check treated every name containing the
# letter "c" (e.g. "model__criterion") as continuous.
CONTINUOUS_PARAM_NAMES = frozenset(
    {
        "alpha",
        "c",
        "coef0",
        "epsilon",
        "gamma",
        "l1_ratio",
        "learning_rate",
        "min_child_weight",
        "nu",
        "reg_alpha",
        "reg_lambda",
        "tol",
    }
)


def leaf_param_name(param_name: str) -> str:
    return str(param_name).split("__")[-1].strip().lower()


def has_continuous_like_params(space: dict[str, list[Any]]) -> bool:
    return any(leaf_param_name(name) in CONTINUOUS_PARAM_NAMES for name in space)


def space_size(space: dict[str, list[Any]]) -> int:
    size = 1
    for values in space.values():
        size *= max(1, len(values))
    return size
