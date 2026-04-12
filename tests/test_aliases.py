from __future__ import annotations

import pandas as pd

from bahamut_tuning import HyperTuneWorkbench, TuningSelection


def test_final_selection_api_and_workbench_helpers() -> None:
    selection = TuningSelection(mode="run_tuning", tuning_strategy="grid")
    assert selection.mode == "run_tuning"
    assert selection.tuning_strategy == "grid"

    workbench = HyperTuneWorkbench(task_type="regression", selected_algorithm="Ridge")
    workbench.configure_problem(task_type="classification", target=["y"], selected_algorithm="LogisticRegression", scorer="accuracy")
    readiness = workbench.validate_execution_readiness()

    assert "LogisticRegression" in workbench.available_algorithms()
    assert workbench.context.task_type == "classification"
    assert workbench.context.target == ["y"]
    assert readiness["ready"] is False
    assert readiness["blockers"]


def test_workbench_fallback_recommendations_are_profile_based() -> None:
    df = pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "x2": [10, 20, 15, 18, 22],
            "target": [100.0, 110.0, 120.0, 115.0, 130.0],
        }
    )
    workbench = HyperTuneWorkbench(
        df=df,
        target=["target"],
        task_type="regression",
        scorer="neg_root_mean_squared_error",
    )

    recommendation = workbench.recommend_algorithms()

    assert recommendation["source"] == "fallback_analysis"
    assert recommendation["algorithms"]
    assert recommendation["algorithms"][0] in workbench.available_algorithms()
    assert "dataset profile" in recommendation["reason"].lower()
