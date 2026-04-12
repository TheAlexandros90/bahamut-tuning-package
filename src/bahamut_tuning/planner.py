from __future__ import annotations

import logging

from .helpers import (
    build_cv_summary,
    infer_default_scoring,
    infer_algorithm_recommendations,
    validate_scoring_for_task,
)
from .spaces import available_estimators as resolve_available_estimators
from .spaces import build_search_space, space_size
from .types import TuningContext, TuningPlan, TuningSelection

logger = logging.getLogger(__name__)


def _has_optuna() -> bool:
    try:
        import optuna  # noqa: F401
    except Exception:
        return False
    return True


class HyperparameterPlanner:
    """Build a deterministic and inspectable tuning plan."""

    def __init__(self, random_state: int = 42, small_space_threshold: int = 120) -> None:
        self.random_state = random_state
        self.small_space_threshold = small_space_threshold

    def available_estimators(self, task_type: str):
        return resolve_available_estimators(task_type=task_type, random_state=self.random_state)

    def build_search_space(self, algorithm: str):
        return build_search_space(algorithm)

    def choose_strategy(
        self,
        *,
        requested: str,
        algorithm: str,
        space: dict[str, list[object]],
        upstream_policy: str | None = None,
    ) -> tuple[str, str]:
        allowed = {"grid", "random", "optuna", "guide_only"}
        if requested in allowed:
            return requested, "Chosen by user override."
        if requested != "auto":
            raise ValueError(f"Unsupported strategy: {requested}")

        if upstream_policy in allowed:
            return upstream_policy, "Applied upstream policy preference from metadata."

        total_size = space_size(space)
        lower_algorithm = algorithm.lower()
        expensive = any(token in lower_algorithm for token in ("svc", "svr", "xgboost", "gradientboosting"))
        continuous_like = any(
            token in param_name.lower()
            for param_name in space
            for token in ("alpha", "c", "learning_rate", "gamma", "epsilon", "lambda", "min_child_weight")
        )

        if total_size <= self.small_space_threshold:
            return "grid", f"Small discrete space ({total_size} configs) -> GridSearchCV."
        if expensive or continuous_like:
            return "optuna", "Expensive or continuous-like search space -> Optuna for efficiency."
        return "random", f"Moderate-large space ({total_size} configs) -> RandomizedSearchCV."

    def build_plan(self, context: TuningContext, selection: TuningSelection) -> TuningPlan:
        if not context.task_type:
            raise ValueError("task_type is required from upstream library context.")

        warnings: list[str] = []
        assumptions: list[str] = []
        estimators = self.available_estimators(context.task_type)
        valid_upstream_recommendations = [name for name in context.recommended_algorithms if name in estimators]
        fallback_recommendations: list[str] = []
        fallback_reason: str | None = None
        if not valid_upstream_recommendations:
            fallback_recommendations, fallback_reason = infer_algorithm_recommendations(
                task_type=context.task_type,
                df=context.df,
                targets=context.target,
                available_algorithms=sorted(estimators.keys()),
                top_k=3,
            )

        recommended_algorithm = (
            valid_upstream_recommendations[0]
            if valid_upstream_recommendations
            else fallback_recommendations[0] if fallback_recommendations else None
        )
        selected_algorithm = selection.selected_algorithm or context.selected_algorithm or recommended_algorithm

        if not selected_algorithm:
            raise ValueError("No algorithm selected or recommended.")
        if selected_algorithm not in estimators:
            raise ValueError(
                f"Unsupported algorithm '{selected_algorithm}' for task '{context.task_type}'."
            )

        scoring = selection.scoring or context.scorer
        if not scoring:
            scoring = infer_default_scoring(context.task_type)
            assumptions.append(f"Scoring not provided upstream; defaulted to '{scoring}'.")
        validate_scoring_for_task(context.task_type, scoring)

        search_space = self.build_search_space(selected_algorithm)
        upstream_policy = context.metadata.get("tuning_policy") or context.metadata.get("BahamutTuning_policy")
        strategy, strategy_reason = self.choose_strategy(
            requested=selection.tuning_strategy,
            algorithm=selected_algorithm,
            space=search_space,
            upstream_policy=upstream_policy,
        )

        if context.cv is None:
            assumptions.append("Fallback cv=5 will be used because upstream CV was not provided.")
        else:
            assumptions.append("Upstream CV splitter will be reused during execution.")

        if selection.nested_cv:
            assumptions.append(
                f"Nested CV is enabled with {selection.nested_cv_folds} outer folds for a stricter generalization estimate."
            )

        if selection.overfit_policy == "block_high":
            assumptions.append("Execution will fail if the detected generalization risk is high.")
        elif selection.overfit_policy == "block_moderate":
            assumptions.append("Execution will fail if the detected generalization risk is moderate or high.")
        else:
            assumptions.append("Generalization risk will be reported as warnings without blocking execution.")

        if context.preprocessor is None:
            assumptions.append(
                "No upstream preprocessing artifact was provided; a default numeric/categorical preprocessor will be built."
            )
        else:
            assumptions.append("Upstream preprocessing artifact will be reused during execution.")

        if valid_upstream_recommendations and context.recommender_reason:
            assumptions.append(f"Upstream recommender reason: {context.recommender_reason}")
        elif fallback_reason:
            assumptions.append(f"Fallback recommendation reason: {fallback_reason}")

        if len(context.target) > 1:
            warnings.append(
                "Multi-target execution depends on estimator support and upstream validation; review the chosen estimator carefully."
            )

        if strategy == "optuna" and not _has_optuna():
            warnings.append("Optuna requested or recommended but not installed. Execution will fall back to random search.")

        logger.info(
            "Built tuning plan: algorithm=%s source=%s strategy=%s scoring=%s",
            selected_algorithm,
            (
                "user"
                if selection.selected_algorithm
                else "upstream_selected"
                if context.selected_algorithm
                else "upstream_recommendation"
                if valid_upstream_recommendations
                else "fallback_analysis"
            ),
            strategy,
            scoring,
        )

        return TuningPlan(
            task_type=context.task_type,
            selected_algorithm=selected_algorithm,
            recommended_algorithm=recommended_algorithm,
            algorithm_source=(
                "user"
                if selection.selected_algorithm
                else "upstream_selected"
                if context.selected_algorithm
                else "upstream_recommendation"
                if valid_upstream_recommendations
                else "fallback_analysis"
            ),
            tuning_strategy=strategy,
            tuning_strategy_reason=strategy_reason,
            scoring=scoring,
            cv_summary=build_cv_summary(context.cv),
            search_space=search_space,
            warnings=warnings,
            assumptions=assumptions,
            iterations=selection.iterations,
            timeout_seconds=selection.timeout_seconds,
            overfit_policy=selection.overfit_policy,
            nested_cv=selection.nested_cv,
            nested_cv_folds=selection.nested_cv_folds,
        )
