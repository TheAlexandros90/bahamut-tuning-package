"""Regression tests for the defects found during the 2026 audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from bahamut_tuning import HyperparameterPlanner
from bahamut_tuning.executor import (
    compute_generalization_diagnostics,
    scoring_is_bounded,
)
from bahamut_tuning.spaces import build_search_space, has_continuous_like_params

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def test_package_imports_without_the_notebook_extra() -> None:
    """`import bahamut_tuning` must not require IPython (notebook extra only)."""

    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'IPython' or name.startswith('IPython.'):\n"
        "            return self\n"
        "        return None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('No module named IPython')\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
        "import bahamut_tuning\n"
        "assert bahamut_tuning.InteractiveTuningUI is not None\n"
        "print('ok')\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_render_ui_raises_a_clear_error_without_widgets() -> None:
    from bahamut_tuning import ui as ui_module

    if ui_module.widgets is not None:
        pytest.skip("ipywidgets is installed; the guard cannot be exercised here.")

    with pytest.raises(ImportError, match="notebook extra"):
        ui_module.InteractiveTuningUI(workbench=None)


@pytest.mark.parametrize(
    "algorithm, expected",
    [
        ("DecisionTreeClassifier", False),
        ("DecisionTreeRegressor", False),
        ("RandomForestClassifier", False),
        ("LogisticRegression", True),
        ("SVC", True),
        ("GradientBoostingRegressor", True),
    ],
)
def test_continuous_detection_matches_leaf_names_only(algorithm: str, expected: bool) -> None:
    """`criterion` contains the letter "c" but is a categorical hyperparameter."""

    assert has_continuous_like_params(build_search_space(algorithm)) is expected


def test_decision_tree_auto_strategy_is_random_not_optuna() -> None:
    planner = HyperparameterPlanner()
    space = build_search_space("DecisionTreeClassifier")
    strategy, _ = planner.choose_strategy(requested="auto", algorithm="DecisionTreeClassifier", space=space)
    assert strategy == "random"


@pytest.mark.parametrize(
    "scoring, bounded",
    [
        ("accuracy", True),
        ("f1_macro", True),
        ("r2", True),
        ("roc_auc", True),
        ("neg_root_mean_squared_error", False),
        ("neg_mean_absolute_error", False),
        ("max_error", False),
        ("neg_log_loss", False),
    ],
)
def test_scoring_boundedness_detection(scoring: str, bounded: bool) -> None:
    assert scoring_is_bounded(scoring) is bounded


class _FixedScoreEstimator(Ridge):
    pass


def _diagnostics(scoring, training_score, cv_score, holdout_score=None):
    class _Scorer:
        def __init__(self, values):
            self.values = list(values)

        def __call__(self, estimator, X, y):
            return self.values.pop(0)

    import bahamut_tuning.executor as executor_module

    values = [training_score] + ([holdout_score] if holdout_score is not None else [])
    original = executor_module.get_scorer
    executor_module.get_scorer = lambda name: _Scorer(values)
    try:
        return compute_generalization_diagnostics(
            fitted_estimator=Pipeline([("model", _FixedScoreEstimator())]),
            X_train=None,
            y_train=None,
            scoring=scoring,
            best_cv_score=cv_score,
            X_test=object() if holdout_score is not None else None,
            y_test=object() if holdout_score is not None else None,
        )
    finally:
        executor_module.get_scorer = original


def test_target_scale_metrics_do_not_report_false_high_risk() -> None:
    """A 2-unit RMSE gap on a target of ~55 units is not a high generalization risk."""

    diagnostics = _diagnostics("neg_root_mean_squared_error", -53.76, -55.72, -53.61)
    assert diagnostics["gap_basis"] == "relative"
    assert diagnostics["risk_level"] in {"low", "moderate"}
    assert diagnostics["train_vs_cv_gap"] == pytest.approx(1.96, abs=0.01)
    assert diagnostics["normalized_train_vs_cv_gap"] == pytest.approx(1.96 / 55.72, abs=0.01)


def test_target_scale_metrics_still_flag_real_overfitting() -> None:
    diagnostics = _diagnostics("neg_root_mean_squared_error", -5.0, -55.0, -60.0)
    assert diagnostics["risk_level"] == "high"


def test_bounded_metrics_keep_absolute_thresholds() -> None:
    diagnostics = _diagnostics("accuracy", 0.99, 0.93, 0.92)
    assert diagnostics["gap_basis"] == "absolute"
    assert diagnostics["score_scale"] == 1.0
    assert diagnostics["risk_level"] == "high"
