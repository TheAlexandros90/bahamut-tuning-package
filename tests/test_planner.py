from __future__ import annotations

import pandas as pd

from bahamut_tuning import HyperparameterPlanner, TuningContextAdapter, TuningSelection


def test_planner_respects_upstream_policy() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [10.0, 11.0, 12.0, 13.0]})
    context = TuningContextAdapter().normalize(
        df=df,
        target=["target"],
        task_type="regression",
        selected_algorithm="RandomForestRegressor",
        scorer="neg_root_mean_squared_error",
        metadata={"tuning_policy": "random"},
    )
    selection = TuningSelection(selected_algorithm="RandomForestRegressor", tuning_strategy="auto")

    plan = HyperparameterPlanner(random_state=42).build_plan(context, selection)

    assert plan.selected_algorithm == "RandomForestRegressor"
    assert plan.tuning_strategy == "random"
    assert plan.algorithm_source == "user"
    assert plan.scoring == "neg_root_mean_squared_error"
