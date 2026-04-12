from __future__ import annotations

from typing import Any

from .context import TuningContextAdapter
from .executor import HyperparameterExecutor
from .helpers import (
    infer_algorithm_recommendations,
    normalize_targets,
    normalize_task_type,
    summarize_dataframe_context,
    validate_scoring_for_task,
)
from .planner import HyperparameterPlanner
from .types import TuningPlan, TuningResult, TuningSelection
from .ui import InteractiveTuningUI


class HyperTuneWorkbench:
    """Facade for notebook and programmatic tuning workflows."""

    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        self.random_state = random_state
        self.adapter = TuningContextAdapter()
        self.context = self.adapter.normalize(**kwargs)
        self.planner = HyperparameterPlanner(random_state=random_state)
        self.executor = HyperparameterExecutor(self.planner, random_state=random_state)
        self._last_plan: TuningPlan | None = None
        self._last_result: TuningResult | None = None

    def update_context(self, **kwargs: Any) -> None:
        self.context = self.adapter.normalize(**kwargs)
        self._last_plan = None
        self._last_result = None

    def configure_problem(
        self,
        *,
        task_type: str | None = None,
        target: str | list[str] | tuple[str, ...] | None = None,
        selected_algorithm: str | None = None,
        scorer: str | None = None,
    ) -> None:
        if task_type is not None:
            self.context.task_type = normalize_task_type(task_type)
        if target is not None:
            self.context.target = normalize_targets(target=target)
        if selected_algorithm is not None:
            self.context.selected_algorithm = selected_algorithm
        if scorer is not None:
            if self.context.task_type:
                validate_scoring_for_task(self.context.task_type, scorer)
            self.context.scorer = scorer
        self._last_plan = None
        self._last_result = None

    def available_algorithms(self) -> list[str]:
        if not self.context.task_type:
            return []
        return sorted(self.planner.available_estimators(self.context.task_type).keys())

    def recommend_algorithms(
        self,
        top_k: int = 3,
        *,
        task_type: str | None = None,
        targets: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        resolved_task_type = normalize_task_type(task_type) if task_type is not None else self.context.task_type
        resolved_targets = list(targets) if targets is not None else list(self.context.target)

        if not resolved_task_type:
            return {
                "algorithms": [],
                "reason": "task_type is required before recommendations can be generated.",
                "source": "unavailable",
            }

        available = sorted(self.planner.available_estimators(resolved_task_type).keys())
        upstream = [name for name in self.context.recommended_algorithms if name in available]
        if upstream:
            return {
                "algorithms": upstream[:top_k],
                "reason": self.context.recommender_reason or "Using upstream recommendations from the parent library.",
                "source": "upstream",
            }

        algorithms, reason = infer_algorithm_recommendations(
            task_type=resolved_task_type,
            df=self.context.df,
            targets=resolved_targets,
            available_algorithms=available,
            top_k=top_k,
        )
        return {
            "algorithms": algorithms,
            "reason": reason,
            "source": "fallback_analysis",
        }

    def context_summary(self):
        return summarize_dataframe_context(self.context.df, targets=self.context.target)

    def validate_execution_readiness(self) -> dict[str, Any]:
        blockers: list[str] = []
        notes: list[str] = []
        recommendation_payload = self.recommend_algorithms()

        if not self.context.task_type:
            blockers.append("task_type is required.")
        if not self.context.selected_algorithm:
            if recommendation_payload["algorithms"]:
                notes.append(
                    f"No selected algorithm was provided. '{recommendation_payload['algorithms'][0]}' is the current recommendation."
                )
            else:
                blockers.append("A selected or recommended algorithm is required.")
        elif self.context.task_type and self.context.selected_algorithm:
            available = self.available_algorithms()
            if self.context.selected_algorithm not in available:
                blockers.append(
                    f"Selected algorithm '{self.context.selected_algorithm}' is not valid for task_type '{self.context.task_type}'."
                )
        if self.context.X_train is None or self.context.y_train is None:
            if self.context.df is None:
                blockers.append("Execution needs either upstream X_train/y_train or df plus target.")
            elif not self.context.target:
                blockers.append("Execution from df requires target columns.")
            else:
                notes.append("No upstream split provided; execution will rely on CV over the full dataframe.")
        if self.context.preprocessor is None:
            notes.append("No upstream preprocessor provided; default preprocessing fallback will be built.")

        return {
            "ready": not blockers,
            "blockers": blockers,
            "notes": notes,
        }

    def nested_cv_preview(self, folds: int = 3) -> dict[str, Any]:
        return {
            "enabled": True,
            "outer_folds": max(3, int(folds)),
            "note": "Nested CV will use the current tuning plan inside each outer fold to produce a stricter generalization estimate.",
        }

    def plan(self, selection: TuningSelection | None = None) -> TuningPlan:
        if selection is None:
            selection = TuningSelection(
                target=self.context.target,
                selected_algorithm=self.context.selected_algorithm,
                scoring=self.context.scorer,
            )
        if selection.target:
            self.context.target = list(selection.target)
        self._last_plan = self.planner.build_plan(self.context, selection)
        self._last_result = None
        return self._last_plan

    def execute(self, plan: TuningPlan | None = None, run: bool = True) -> TuningResult:
        active_plan = plan or self._last_plan or self.plan()
        self._last_result = self.executor.execute(context=self.context, plan=active_plan, run=run)
        return self._last_result

    def generate_code(self, plan: TuningPlan | None = None, run: bool = False) -> str:
        active_plan = plan or self._last_plan or self.plan()
        return self.executor.generate_code(context=self.context, plan=active_plan, run=run)

    def last_result(self) -> TuningResult | None:
        return self._last_result

    def render_ui(self) -> Any:
        return InteractiveTuningUI(self).render()
