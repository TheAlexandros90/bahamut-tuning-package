from __future__ import annotations

import pandas as pd
import pytest

from bahamut_tuning import TuningContextAdapter


def test_context_adapter_rejects_missing_target_without_upstream_y() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3]})
    adapter = TuningContextAdapter()

    with pytest.raises(ValueError):
        adapter.normalize(df=df, target=["target"], task_type="regression")


def test_context_adapter_keeps_upstream_split_artifacts() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    X_train = df[["feature"]].iloc[:2].copy()
    y_train = df["target"].iloc[:2].copy()

    context = TuningContextAdapter().normalize(
        df=df,
        target=["target"],
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        scorer="accuracy",
    )

    assert context.X_train is X_train
    assert context.y_train is y_train
    assert context.diagnostics["upstream_split"] == "provided"
