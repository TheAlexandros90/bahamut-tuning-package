from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

_VALID_TUNING_STRATEGIES = {"auto", "grid", "random", "optuna", "guide_only"}
_VALID_MODES = {"plan_only", "generate_code", "run_tuning"}
_VALID_OVERFIT_POLICIES = {"warn", "block_high", "block_moderate"}


@dataclass(slots=True)
class TuningContext:
    """Normalized artifacts coming from the broader ML library."""

    df: pd.DataFrame | None = None
    target: list[str] = field(default_factory=list)
    task_type: str | None = None
    selected_algorithm: str | None = None
    recommended_algorithms: list[str] = field(default_factory=list)
    recommender_reason: str | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | pd.DataFrame | None = None
    y_test: pd.Series | pd.DataFrame | None = None
    cv: Any = None
    scorer: str | None = None
    preprocessor: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    feature_lists: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TuningSelection:
    """User-confirmed selections before building a tuning plan."""

    target: list[str] = field(default_factory=list)
    selected_algorithm: str | None = None
    tuning_strategy: str = "auto"
    scoring: str | None = None
    iterations: int = 30
    timeout_seconds: int | None = None
    overfit_policy: str = "warn"
    nested_cv: bool = False
    nested_cv_folds: int = 3
    mode: str = "plan_only"

    def __post_init__(self) -> None:
        if self.tuning_strategy not in _VALID_TUNING_STRATEGIES:
            raise ValueError(f"Unsupported tuning_strategy: {self.tuning_strategy}")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.overfit_policy not in _VALID_OVERFIT_POLICIES:
            raise ValueError(f"Unsupported overfit_policy: {self.overfit_policy}")
        self.iterations = max(5, int(self.iterations))
        self.nested_cv_folds = max(3, int(self.nested_cv_folds))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TuningPlan:
    """Plan produced by the planner and consumed by the executor."""

    task_type: str
    selected_algorithm: str
    recommended_algorithm: str | None
    algorithm_source: str
    tuning_strategy: str
    tuning_strategy_reason: str
    scoring: str
    cv_summary: str
    search_space: dict[str, list[Any]]
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    iterations: int = 30
    timeout_seconds: int | None = None
    overfit_policy: str = "warn"
    nested_cv: bool = False
    nested_cv_folds: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TuningResult:
    """Structured execution output returned by the executor."""

    selected_algorithm: str
    tuning_strategy_used: str
    best_params: dict[str, Any]
    model_best_params: dict[str, Any]
    best_cv_score: float | None
    final_score: float | None
    top_trials: list[dict[str, Any]]
    runtime: float
    warnings: list[str]
    explanation: str
    reproducible_code_snippet: str
    generalization: dict[str, Any] = field(default_factory=dict)
    nested_validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
