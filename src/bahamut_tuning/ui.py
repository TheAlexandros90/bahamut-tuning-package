from __future__ import annotations

import html
from typing import Any

import pandas as pd

from .helpers import infer_default_scoring, summarize_dataframe_context
from .types import TuningSelection

# The notebook stack is an optional extra (`bahamut-tuning[notebook]`).
# Importing the package must never require IPython/ipywidgets to be installed.
try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:  # pragma: no cover - exercised only without the notebook extra
    widgets = None
    display = None


_SCORING_OPTIONS = {
    "classification": [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "roc_auc",
        "neg_log_loss",
    ],
    "regression": [
        "neg_root_mean_squared_error",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "r2",
        "explained_variance",
        "max_error",
    ],
}

_OVERFIT_POLICY_LABELS = {
    "warn": "Warn only",
    "block_high": "Block if high",
    "block_moderate": "Block if moderate/high",
}

_RISK_STYLES = {
    "low": {"accent": "#0f766e", "background": "#ecfdf5", "title": "Low risk"},
    "moderate": {"accent": "#b45309", "background": "#fff7ed", "title": "Moderate risk"},
    "high": {"accent": "#b91c1c", "background": "#fef2f2", "title": "High risk"},
    "unknown": {"accent": "#475569", "background": "#f8fafc", "title": "Risk unknown"},
}


class InteractiveTuningUI:
    """Notebook UI for confirmation, overrides, planning, and execution."""

    def __init__(self, workbench: Any) -> None:
        if widgets is None or display is None:
            raise ImportError(
                "render_ui() requires the notebook extra. Install it with "
                "`pip install bahamut-tuning[notebook]` (ipywidgets + IPython)."
            )

        self.workbench = workbench
        context = self.workbench.context
        task_type = context.task_type or "classification"
        algorithms = sorted(self.workbench.planner.available_estimators(task_type).keys())
        target_options = list(context.df.columns) if context.df is not None else list(context.target)
        recommendation_payload = self.workbench.recommend_algorithms(task_type=task_type, targets=list(context.target))
        selected_algorithm = context.selected_algorithm or (
            recommendation_payload["algorithms"][0] if recommendation_payload["algorithms"] else algorithms[0]
        )

        self.header = widgets.HTML(self._header_html())
        self.status_card = widgets.HTML()
        self.dataset_card = widgets.HTML()
        self.recommendation_card = widgets.HTML()
        self.status_card.layout = widgets.Layout(flex="1 1 0")
        self.dataset_card.layout = widgets.Layout(flex="1 1 0")
        self.recommendation_card.layout = widgets.Layout(flex="1 1 0")
        self.plan_output = widgets.Output()
        self.code_output = widgets.Output()
        self.result_output = widgets.Output()
        self.schema_output = widgets.Output()

        self.task_widget = widgets.ToggleButtons(
            options=[("Classification", "classification"), ("Regression", "regression")],
            value=task_type,
            description="Problem",
            style={"button_width": "160px"},
        )
        self.target_widget = widgets.SelectMultiple(
            options=target_options,
            value=tuple(context.target),
            description="Targets",
            rows=min(8, max(4, len(target_options) or 4)),
            layout=widgets.Layout(width="100%", height="170px"),
        )
        self.algorithm_widget = widgets.Dropdown(
            options=algorithms,
            value=selected_algorithm,
            description="Algorithm",
            layout=widgets.Layout(width="100%"),
        )
        self.strategy_widget = widgets.Dropdown(
            options=["auto", "grid", "random", "optuna", "guide_only"],
            value="auto",
            description="Strategy",
            layout=widgets.Layout(width="100%"),
        )
        self.metric_widget = widgets.Combobox(
            options=self._metric_options(task_type),
            value=context.scorer or infer_default_scoring(task_type),
            description="Scoring",
            placeholder="e.g. f1_macro or neg_root_mean_squared_error",
            ensure_option=False,
            layout=widgets.Layout(width="100%"),
        )
        self.iter_widget = widgets.IntSlider(
            value=30,
            min=5,
            max=200,
            step=5,
            description="Iterations",
            continuous_update=False,
            readout=True,
            layout=widgets.Layout(width="100%"),
        )
        self.timeout_widget = widgets.BoundedIntText(value=0, min=0, description="Timeout s")
        self.overfit_policy_widget = widgets.Dropdown(
            options=[
                ("Warn only", "warn"),
                ("Block if high", "block_high"),
                ("Block if moderate/high", "block_moderate"),
            ],
            value="warn",
            description="Risk policy",
            layout=widgets.Layout(width="100%"),
        )
        self.nested_cv_widget = widgets.Checkbox(
            value=False,
            description="Enable nested CV",
            indent=False,
            layout=widgets.Layout(width="50%"),
        )
        self.nested_cv_folds_widget = widgets.BoundedIntText(value=3, min=3, max=10, description="Outer folds")
        self.nested_cv_folds_widget.disabled = True
        self.mode_widget = widgets.ToggleButtons(
            options=[
                ("Plan only", "plan_only"),
                ("Generate code", "generate_code"),
                ("Run tuning", "run_tuning"),
            ],
            description="Mode",
        )
        self.use_recommended_button = widgets.Button(
            description="Use recommended",
            icon="check",
            layout=widgets.Layout(width="170px"),
        )
        self.go_button = widgets.Button(
            description="Build plan",
            icon="sitemap",
            button_style="success",
            layout=widgets.Layout(width="170px"),
        )

        self.go_button.on_click(self._on_apply)
        self.use_recommended_button.on_click(self._on_use_recommended)
        self.task_widget.observe(self._on_task_change, names="value")
        self.mode_widget.observe(self._on_mode_change, names="value")
        self.target_widget.observe(self._on_target_change, names="value")
        self.algorithm_widget.observe(self._on_controls_change, names="value")
        self.strategy_widget.observe(self._on_controls_change, names="value")
        self.metric_widget.observe(self._on_controls_change, names="value")
        self.overfit_policy_widget.observe(self._on_controls_change, names="value")
        self.nested_cv_widget.observe(self._on_nested_cv_toggle, names="value")
        self.nested_cv_folds_widget.observe(self._on_controls_change, names="value")

        self.tabs = widgets.Tab(
            children=[self.schema_output, self.plan_output, self.code_output, self.result_output],
            layout=widgets.Layout(width="100%"),
        )
        self.tabs.set_title(0, "Schema")
        self.tabs.set_title(1, "Plan")
        self.tabs.set_title(2, "Code")
        self.tabs.set_title(3, "Result")

        self._refresh_overview()
        self._render_schema()
        self._on_mode_change(None)

    def _build_selection(self) -> TuningSelection:
        timeout = self.timeout_widget.value if self.timeout_widget.value > 0 else None
        return TuningSelection(
            target=list(self.target_widget.value),
            selected_algorithm=self.algorithm_widget.value,
            tuning_strategy=self.strategy_widget.value,
            scoring=(self.metric_widget.value or infer_default_scoring(self.task_widget.value)).strip(),
            iterations=self.iter_widget.value,
            timeout_seconds=timeout,
            overfit_policy=self.overfit_policy_widget.value,
            nested_cv=self.nested_cv_widget.value,
            nested_cv_folds=self.nested_cv_folds_widget.value,
            mode=self.mode_widget.value,
        )

    def _metric_options(self, task_type: str) -> list[str]:
        return list(_SCORING_OPTIONS.get(task_type, []))

    def _current_dataset_shape(self) -> tuple[str, str]:
        df = self.workbench.context.df
        if df is None:
            return "Upstream", "split artifacts"
        return str(df.shape[0]), str(df.shape[1])

    def _valid_recommendations(self, task_type: str) -> list[str]:
        return self.workbench.recommend_algorithms(
            task_type=task_type,
            targets=list(self.target_widget.value),
        )["algorithms"]

    def _preview_readiness(self) -> dict[str, Any]:
        blockers: list[str] = []
        notes: list[str] = []
        task_type = self.task_widget.value
        targets = list(self.target_widget.value)
        algorithm = self.algorithm_widget.value

        if not task_type:
            blockers.append("Select a problem type.")
        if not algorithm:
            blockers.append("Select an algorithm.")
        if self.workbench.context.X_train is None or self.workbench.context.y_train is None:
            if self.workbench.context.df is None:
                blockers.append("Provide upstream split artifacts or a dataframe with target columns.")
            elif not targets:
                blockers.append("Choose at least one target column.")
            else:
                notes.append("No upstream split detected. Execution would rely on CV over the full dataframe.")
        if self.workbench.context.preprocessor is None:
            notes.append("No upstream preprocessor detected. A default fallback would be created.")
        if self.nested_cv_widget.value:
            notes.append(f"Nested CV enabled with {self.nested_cv_folds_widget.value} outer folds. Execution will be slower.")
        return {"ready": not blockers, "blockers": blockers, "notes": notes}

    def _header_html(self) -> str:
        return """
<div style="padding:18px 20px;border:1px solid #d7e3f4;border-radius:18px;
background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%);color:white;margin-bottom:12px;">
  <div style="font-size:22px;font-weight:700;letter-spacing:0.2px;">Bahamut Tuning Control Center</div>
  <div style="margin-top:6px;font-size:13px;opacity:0.92;">
    Configure the problem, confirm targets, choose the estimator, inspect the plan, generate code, or run tuning.
  </div>
</div>
"""

    def _card_html(self, title: str, lines: list[str], accent: str = "#2563eb") -> str:
        content = "".join(f"<div style='margin-top:6px;font-size:13px;line-height:1.45;'>{line}</div>" for line in lines)
        return f"""
<div style="padding:14px 16px;border:1px solid #dde5ef;border-radius:16px;background:#fbfdff;min-height:124px;">
  <div style="font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{accent};">{html.escape(title)}</div>
  {content}
</div>
"""

    def _result_callout_html(self, title: str, lines: list[str], *, accent: str, background: str) -> str:
        content = "".join(f"<div style='margin-top:6px;font-size:13px;line-height:1.5;'>{line}</div>" for line in lines)
        return f"""
<div style="padding:14px 16px;border:1px solid {accent};border-left:6px solid {accent};border-radius:16px;background:{background};margin-bottom:12px;">
  <div style="font-size:12px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:{accent};">{html.escape(title)}</div>
  {content}
</div>
"""

    def _fmt_metric(self, value: Any) -> str:
        if value is None:
            return "not available"
        if isinstance(value, (int, float)):
            return f"{float(value):.4f}"
        return html.escape(str(value))

    def _generalization_callout(self, result: Any) -> str | None:
        generalization = getattr(result, "generalization", None)
        if not generalization:
            return None

        risk_level = str(generalization.get("risk_level", "unknown"))
        risk_style = _RISK_STYLES.get(risk_level, _RISK_STYLES["unknown"])
        lines = [
            html.escape(str(generalization.get("assessment", "No assessment available."))),
            (
                f"<strong>Training vs CV gap:</strong> {self._fmt_metric(generalization.get('train_vs_cv_gap'))}"
                " If this is positive and large, the model is fitting training data better than validation data."
            ),
            (
                f"<strong>CV vs holdout gap:</strong> {self._fmt_metric(generalization.get('cv_vs_holdout_gap'))}"
                " Positive values mean cross-validation is more optimistic than the holdout split."
            ),
            (
                f"<strong>CV vs nested CV gap:</strong> {self._fmt_metric(generalization.get('cv_vs_nested_gap'))}"
                " Positive values mean the simpler CV estimate is more optimistic than nested CV."
            ),
            (
                f"<strong>Scores:</strong> train={self._fmt_metric(generalization.get('training_score'))}, "
                f"cv={self._fmt_metric(generalization.get('best_cv_score'))}, "
                f"holdout={self._fmt_metric(generalization.get('holdout_score'))}, "
                f"nested={self._fmt_metric(generalization.get('nested_cv_score'))}"
            ),
        ]
        warnings = generalization.get("warnings") or []
        lines.extend(f"<strong>Warning:</strong> {html.escape(str(item))}" for item in warnings)
        return self._result_callout_html(
            f"Generalization Check: {risk_style['title']}",
            lines,
            accent=risk_style["accent"],
            background=risk_style["background"],
        )

    def _nested_validation_callout(self, result: Any) -> str | None:
        nested_validation = getattr(result, "nested_validation", None)
        if not nested_validation or not nested_validation.get("enabled"):
            return None

        lines = [
            (
                f"<strong>Outer folds:</strong> {html.escape(str(nested_validation.get('outer_folds')))}. "
                f"<strong>Strategy used inside outer folds:</strong> {html.escape(str(nested_validation.get('strategy_used')))}."
            ),
            (
                f"<strong>Nested mean +/- std:</strong> {self._fmt_metric(nested_validation.get('score_mean'))} +/- "
                f"{self._fmt_metric(nested_validation.get('score_std'))}"
            ),
            (
                f"<strong>Range:</strong> min={self._fmt_metric(nested_validation.get('score_min'))}, "
                f"max={self._fmt_metric(nested_validation.get('score_max'))}"
            ),
            "Nested CV is the stricter estimate because each outer fold repeats model selection before scoring on unseen data.",
        ]
        warnings = nested_validation.get("warnings") or []
        lines.extend(f"<strong>Note:</strong> {html.escape(str(item))}" for item in warnings)
        return self._result_callout_html(
            "Nested CV Interpretation",
            lines,
            accent="#1d4ed8",
            background="#eff6ff",
        )

    def _refresh_overview(self) -> None:
        rows, cols = self._current_dataset_shape()
        readiness = self._preview_readiness()
        task_type = self.task_widget.value
        recommendation_payload = self.workbench.recommend_algorithms(
            task_type=task_type,
            targets=list(self.target_widget.value),
        )
        recommendations = recommendation_payload["algorithms"]
        target_preview = ", ".join(self.target_widget.value) if self.target_widget.value else "No target selected"

        status_lines = [
            f"<strong>Status:</strong> {'Ready to run' if readiness['ready'] else 'Needs review'}",
            f"<strong>Algorithm:</strong> {html.escape(str(self.algorithm_widget.value or 'None'))}",
            f"<strong>Mode:</strong> {html.escape(self._mode_label(self.mode_widget.value))}",
            f"<strong>Risk policy:</strong> {html.escape(_OVERFIT_POLICY_LABELS.get(self.overfit_policy_widget.value, str(self.overfit_policy_widget.value)))}",
        ]
        if readiness["blockers"]:
            status_lines.extend(f"<span style='color:#b45309;'>• {html.escape(item)}</span>" for item in readiness["blockers"])
        else:
            status_lines.extend(f"<span style='color:#0f766e;'>• {html.escape(item)}</span>" for item in readiness["notes"])

        dataset_lines = [
            f"<strong>Problem:</strong> {html.escape(task_type.title())}",
            f"<strong>Rows / columns:</strong> {html.escape(rows)} / {html.escape(cols)}",
            f"<strong>Targets:</strong> {html.escape(target_preview)}",
        ]
        if self.workbench.context.scorer:
            dataset_lines.append(f"<strong>Upstream scorer:</strong> {html.escape(self.workbench.context.scorer)}")

        recommendation_lines = [
            f"<strong>Recommended:</strong> {html.escape(', '.join(recommendations) if recommendations else 'No upstream recommendation')}",
        ]
        recommendation_lines.append(f"<strong>Source:</strong> {html.escape(recommendation_payload['source'])}")
        recommendation_lines.append(html.escape(recommendation_payload["reason"]))
        recommendation_lines.append(
            f"<strong>Strategy hint:</strong> {html.escape(self.strategy_widget.value)}"
        )
        if self.nested_cv_widget.value:
            recommendation_lines.append(
                f"<strong>Nested CV:</strong> enabled with {self.nested_cv_folds_widget.value} outer folds"
            )

        self.status_card.value = self._card_html(
            "Execution readiness",
            status_lines,
            accent="#0f766e" if readiness["ready"] else "#b45309",
        )
        self.dataset_card.value = self._card_html("Dataset context", dataset_lines, accent="#2563eb")
        self.recommendation_card.value = self._card_html("Recommendation", recommendation_lines, accent="#7c3aed")

    def _mode_label(self, mode: str) -> str:
        labels = {
            "plan_only": "Plan only",
            "generate_code": "Generate code",
            "run_tuning": "Run tuning",
        }
        return labels.get(mode, mode)

    def _on_task_change(self, change: Any) -> None:
        task_type = self.task_widget.value
        algorithms = sorted(self.workbench.planner.available_estimators(task_type).keys())
        self.algorithm_widget.options = algorithms

        recommended = self._valid_recommendations(task_type)
        if recommended:
            self.algorithm_widget.value = recommended[0]
        elif self.algorithm_widget.value not in algorithms:
            self.algorithm_widget.value = algorithms[0]

        self.metric_widget.options = self._metric_options(task_type)
        self.metric_widget.value = infer_default_scoring(task_type)
        self._refresh_overview()

    def _on_target_change(self, change: Any) -> None:
        self._render_schema()
        self._refresh_overview()

    def _on_controls_change(self, change: Any) -> None:
        self._refresh_overview()

    def _on_nested_cv_toggle(self, change: Any) -> None:
        self.nested_cv_folds_widget.disabled = not self.nested_cv_widget.value
        self._refresh_overview()

    def _on_mode_change(self, change: Any) -> None:
        mode = self.mode_widget.value
        if mode == "plan_only":
            self.go_button.description = "Build plan"
            self.go_button.icon = "sitemap"
        elif mode == "generate_code":
            self.go_button.description = "Generate code"
            self.go_button.icon = "code"
        else:
            self.go_button.description = "Run tuning"
            self.go_button.icon = "play"
        self._refresh_overview()

    def _on_use_recommended(self, _: Any) -> None:
        recommendations = self._valid_recommendations(self.task_widget.value)
        if recommendations:
            self.algorithm_widget.value = recommendations[0]
        self._refresh_overview()

    def _render_schema(self) -> None:
        summary = summarize_dataframe_context(self.workbench.context.df, targets=list(self.target_widget.value))
        with self.schema_output:
            self.schema_output.clear_output(wait=True)
            if "role" in summary.columns:
                def highlight_targets(row: pd.Series) -> list[str]:
                    if row.get("role") == "target":
                        return ["background-color: #eff6ff; font-weight: 600;"] * len(row)
                    return [""] * len(row)

                display(
                    summary.style.hide(axis="index").apply(highlight_targets, axis=1).set_table_styles(
                        [
                            {"selector": "th", "props": [("background-color", "#f8fafc"), ("font-weight", "600")]},
                            {"selector": "td", "props": [("padding", "6px 10px")]},
                        ]
                    )
                )
            else:
                display(summary)

    def _render_plan(self, plan: Any) -> None:
        plan_df = pd.DataFrame(
            [
                {"field": "task_type", "value": plan.task_type},
                {"field": "selected_algorithm", "value": plan.selected_algorithm},
                {"field": "algorithm_source", "value": plan.algorithm_source},
                {"field": "tuning_strategy", "value": plan.tuning_strategy},
                {"field": "scoring", "value": plan.scoring},
                {"field": "cv_summary", "value": plan.cv_summary},
                {"field": "iterations", "value": plan.iterations},
                {"field": "timeout_seconds", "value": plan.timeout_seconds},
                {"field": "overfit_policy", "value": plan.overfit_policy},
                {"field": "nested_cv", "value": plan.nested_cv},
                {"field": "nested_cv_folds", "value": plan.nested_cv_folds},
            ]
        )
        search_df = pd.DataFrame(
            [
                {
                    "hyperparameter": key,
                    "candidates": len(values),
                    "preview": ", ".join(map(str, values[:5])) + (" ..." if len(values) > 5 else ""),
                }
                for key, values in plan.search_space.items()
            ]
        )
        with self.plan_output:
            self.plan_output.clear_output(wait=True)
            display(plan_df)
            if not search_df.empty:
                display(search_df)
            if plan.assumptions:
                display(pd.DataFrame({"assumptions": plan.assumptions}))
            if plan.warnings:
                display(pd.DataFrame({"warnings": plan.warnings}))

    def _render_code(self, code: str) -> None:
        with self.code_output:
            self.code_output.clear_output(wait=True)
            display(
                widgets.Textarea(
                    value=code,
                    layout=widgets.Layout(width="100%", height="280px"),
                    disabled=True,
                )
            )

    def _render_result(self, result: Any) -> None:
        result_df = pd.DataFrame(
            [
                {"field": "selected_algorithm", "value": result.selected_algorithm},
                {"field": "tuning_strategy_used", "value": result.tuning_strategy_used},
                {"field": "best_cv_score", "value": result.best_cv_score},
                {"field": "final_score", "value": result.final_score},
                {"field": "runtime", "value": round(result.runtime, 4)},
            ]
        )
        with self.result_output:
            self.result_output.clear_output(wait=True)
            generalization_callout = self._generalization_callout(result)
            if generalization_callout:
                display(widgets.HTML(value=generalization_callout))
            nested_callout = self._nested_validation_callout(result)
            if nested_callout:
                display(widgets.HTML(value=nested_callout))
            display(result_df)
            if result.top_trials:
                display(pd.DataFrame(result.top_trials))
            if result.generalization:
                display(pd.DataFrame([result.generalization]))
            if result.nested_validation:
                display(pd.DataFrame([result.nested_validation]))
            if result.warnings:
                display(pd.DataFrame({"warnings": result.warnings}))

    def _on_apply(self, _: Any) -> None:
        try:
            self.workbench.configure_problem(
                task_type=self.task_widget.value,
                target=list(self.target_widget.value),
                selected_algorithm=self.algorithm_widget.value,
                scorer=(self.metric_widget.value or infer_default_scoring(self.task_widget.value)).strip(),
            )
            selection = self._build_selection()
            self._render_schema()
            plan = self.workbench.plan(selection=selection)
            self._render_plan(plan)
            self._render_code(self.workbench.generate_code(plan=plan))
            self.result_output.clear_output(wait=True)
            if selection.mode == "run_tuning":
                result = self.workbench.execute(plan=plan, run=True)
                self._render_result(result)
                self.tabs.selected_index = 3
            elif selection.mode == "generate_code":
                self.tabs.selected_index = 2
            else:
                self.tabs.selected_index = 1
        except Exception as err:
            with self.result_output:
                self.result_output.clear_output(wait=True)
                display(pd.DataFrame({"error": [str(err)]}))
            self.tabs.selected_index = 3
        finally:
            self._refresh_overview()

    def render(self) -> Any:
        controls_left = widgets.VBox(
            [
                self.task_widget,
                self.target_widget,
            ],
            layout=widgets.Layout(width="40%", gap="10px"),
        )
        controls_right = widgets.VBox(
            [
                self.algorithm_widget,
                self.strategy_widget,
                self.metric_widget,
                widgets.HBox([self.iter_widget, self.timeout_widget]),
                self.overfit_policy_widget,
                widgets.HBox([self.nested_cv_widget, self.nested_cv_folds_widget], layout=widgets.Layout(gap="10px")),
                self.mode_widget,
                widgets.HBox([self.use_recommended_button, self.go_button], layout=widgets.Layout(gap="10px")),
            ],
            layout=widgets.Layout(width="60%", gap="10px"),
        )
        cards = widgets.HBox(
            [self.status_card, self.dataset_card, self.recommendation_card],
            layout=widgets.Layout(width="100%", gap="12px"),
        )
        box = widgets.VBox(
            [
                self.header,
                cards,
                widgets.HBox([controls_left, controls_right], layout=widgets.Layout(width="100%", gap="14px")),
                self.tabs,
            ],
            layout=widgets.Layout(width="100%", gap="14px"),
        )
        return box


