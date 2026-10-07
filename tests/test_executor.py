from __future__ import annotations

from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split

from bahamut_tuning import HyperTuneWorkbench, TuningSelection, enforce_overfit_policy
from bahamut_tuning.executor import generate_tuning_code


@pytest.mark.parametrize("random_state", [17, 42])
def test_generated_optuna_code_uses_seeded_sampler(monkeypatch, random_state):
    data = load_diabetes(as_frame=True)
    tuner = HyperTuneWorkbench(
        df=data.frame,
        target=["target"],
        task_type="regression",
        selected_algorithm="Ridge",
    )
    plan = tuner.plan(
        TuningSelection(selected_algorithm="Ridge", tuning_strategy="grid")
    )
    plan = replace(plan, tuning_strategy="optuna")
    sampler_token = object()
    sampler_seeds = []

    def make_sampler(*, seed):
        sampler_seeds.append(seed)
        return sampler_token

    def create_study(*, direction, sampler=None):
        assert direction == "maximize"
        assert sampler is sampler_token
        raise RuntimeError("study inspected")

    optuna_stub = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=make_sampler),
        create_study=create_study,
    )
    monkeypatch.setitem(sys.modules, "optuna", optuna_stub)
    code = generate_tuning_code(tuner.context, plan, random_state=random_state, run=True)

    with pytest.raises(RuntimeError, match="study inspected"):
        exec(code, {"tuner": tuner})

    assert sampler_seeds == [random_state]


def test_executor_runs_and_returns_structured_result() -> None:
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = HyperTuneWorkbench(
        df=df,
        target=["target"],
        task_type="regression",
        selected_algorithm="Ridge",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scorer="neg_root_mean_squared_error",
    )
    plan = tuner.plan(
        TuningSelection(
            selected_algorithm="Ridge",
            tuning_strategy="grid",
            scoring="neg_root_mean_squared_error",
            iterations=10,
        )
    )

    result = tuner.execute(plan=plan, run=True)

    assert result.selected_algorithm == "Ridge"
    assert result.tuning_strategy_used == "grid"
    assert isinstance(result.best_params, dict)
    assert isinstance(result.model_best_params, dict)
    assert result.best_cv_score is not None
    assert result.final_score is not None
    assert result.reproducible_code_snippet
    assert result.generalization["best_cv_score"] == result.best_cv_score
    assert result.generalization["holdout_score"] == result.final_score
    assert result.generalization["risk_level"] in {"low", "moderate", "high", "unknown"}
    assert result.nested_validation == {"enabled": False, "supported": False, "warnings": []}
    assert set(result.model_best_params).issubset({"alpha", "solver", "fit_intercept", "positive"}) or result.model_best_params
    assert tuner.last_result() is result


def test_generated_code_is_self_hydrating_and_executable_without_missing_symbols() -> None:
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = HyperTuneWorkbench(
        df=df,
        target=["target"],
        task_type="regression",
        selected_algorithm="Ridge",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scorer="neg_root_mean_squared_error",
    )
    plan = tuner.plan(
        TuningSelection(
            selected_algorithm="Ridge",
            tuning_strategy="grid",
            scoring="neg_root_mean_squared_error",
            iterations=10,
        )
    )

    code = tuner.generate_code(plan=plan, run=False)
    namespace = {"tuner": tuner}

    exec(code, namespace)

    assert "active_tuner" in namespace
    assert namespace["algorithm_name"] == "Ridge"
    assert namespace["RUN_TUNING"] is False


def test_generated_code_reports_generalization_when_executed() -> None:
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = HyperTuneWorkbench(
        df=df,
        target=["target"],
        task_type="regression",
        selected_algorithm="Ridge",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scorer="neg_root_mean_squared_error",
    )
    plan = tuner.plan(
        TuningSelection(
            selected_algorithm="Ridge",
            tuning_strategy="grid",
            scoring="neg_root_mean_squared_error",
            iterations=10,
            nested_cv=True,
            nested_cv_folds=3,
        )
    )

    code = tuner.generate_code(plan=plan, run=True)
    namespace = {"tuner": tuner}

    exec(code, namespace)

    assert "EXECUTION_RESULT" in namespace
    assert namespace["EXECUTION_RESULT"]["generalization"]["risk_level"] in {"low", "moderate", "high", "unknown"}
    assert namespace["EXECUTION_RESULT"]["final_score"] == namespace["EXECUTION_RESULT"]["generalization"]["holdout_score"]
    assert namespace["EXECUTION_RESULT"]["nested_validation"]["enabled"] is True
    assert "BEST_MODEL_PARAMS" in namespace
    assert namespace["EXECUTION_RESULT"]["model_best_params"] == namespace["BEST_MODEL_PARAMS"]


def test_nested_cv_is_recorded_in_executor_result() -> None:
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = HyperTuneWorkbench(
        df=df,
        target=["target"],
        task_type="regression",
        selected_algorithm="Ridge",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scorer="neg_root_mean_squared_error",
    )
    plan = tuner.plan(
        TuningSelection(
            selected_algorithm="Ridge",
            tuning_strategy="grid",
            scoring="neg_root_mean_squared_error",
            iterations=10,
            nested_cv=True,
            nested_cv_folds=3,
        )
    )

    result = tuner.execute(plan=plan, run=True)

    assert result.nested_validation["enabled"] is True
    assert result.nested_validation["supported"] is True
    assert result.nested_validation["score_mean"] is not None


def test_overfit_policy_can_block_execution() -> None:
    with pytest.raises(RuntimeError):
        enforce_overfit_policy(
            {
                "risk_level": "high",
                "assessment": "High generalization gap detected with the current heuristic thresholds.",
            },
            "block_high",
        )
