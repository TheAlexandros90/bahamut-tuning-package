from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit

from bahamut_tuning import HyperTuneWorkbench, TuningSelection
from bahamut_tuning.bridge import IndexedGroupKFold
from bahamut_tuning.executor import _nested_outer_cv


@pytest.fixture
def bundle():
    rng = np.random.default_rng(9)
    X = pd.DataFrame({"x1": rng.normal(size=120), "x2": rng.normal(size=120)})
    y = (2 * X.x1 - X.x2 + rng.normal(scale=0.1, size=120)).rename("y")
    return {
        "X_train": X.iloc[:90], "y_train": y.iloc[:90],
        "X_validation": X.iloc[90:105], "y_validation": y.iloc[90:105],
        "X_test": X.iloc[105:], "y_test": y.iloc[105:],
        "split_config": {"problema": "series_temporales"},
    }


def test_temporal_bridge_never_trains_on_future_and_preserves_holdout(bundle):
    tuner = HyperTuneWorkbench.from_bahamut(bundle, task_type="regression", n_splits=3, gap=2)
    assert isinstance(tuner.context.cv, TimeSeriesSplit)
    for train, test in tuner.context.cv.split(tuner.context.X_train):
        assert max(train) + 2 < min(test)
    pd.testing.assert_frame_equal(tuner.context.X_train, bundle["X_train"])
    pd.testing.assert_frame_equal(tuner.context.X_test, bundle["X_test"])
    assert tuner.context.metadata["validation_rows_reserved"] == 15


def test_nested_temporal_cv_preserves_gap_and_window():
    inner = TimeSeriesSplit(n_splits=2, gap=2, max_train_size=12, test_size=4)
    outer = _nested_outer_cv("regression", 3, 42, inner_cv=inner)
    assert isinstance(outer, TimeSeriesSplit)
    assert (outer.n_splits, outer.gap, outer.max_train_size, outer.test_size) == (3, 2, 12, 4)
    for train, test in outer.split(np.zeros((60, 2))):
        assert max(train) + 2 < min(test)
    assert inner.n_splits == 2


def test_group_cv_keeps_entities_disjoint_at_both_levels(bundle):
    bundle["split_config"] = {"problema": "regresion"}
    groups = pd.Series(np.repeat(np.arange(18), 5), index=bundle["X_train"].index)
    bundle["groups_train"] = groups
    tuner = HyperTuneWorkbench.from_bahamut(bundle, n_splits=2)
    inner = tuner.context.cv
    outer = _nested_outer_cv("regression", 3, 42, inner_cv=inner)
    assert isinstance(outer, IndexedGroupKFold)
    for fit, test in outer.split(tuner.context.X_train):
        assert set(groups.iloc[fit]).isdisjoint(groups.iloc[test])
        subset = tuner.context.X_train.iloc[fit]
        for inner_fit, inner_test in inner.split(subset):
            assert set(groups.loc[subset.iloc[inner_fit].index]).isdisjoint(groups.loc[subset.iloc[inner_test].index])


@pytest.mark.parametrize("grouped", [False, True])
def test_bridge_executes_nested_search_and_generated_code(bundle, grouped):
    if grouped:
        bundle["split_config"] = {"problema": "regresion"}
        bundle["groups_train"] = pd.Series(np.repeat(np.arange(18), 5), index=bundle["X_train"].index)
    tuner = HyperTuneWorkbench.from_bahamut(bundle, task_type="regression", n_splits=2, selected_algorithm="Ridge")
    plan = tuner.plan(TuningSelection(selected_algorithm="Ridge", tuning_strategy="grid", nested_cv=True, nested_cv_folds=3))
    plan.search_space = {"model__alpha": [0.1, 1.0]}
    result = tuner.execute(plan=plan)
    assert result.nested_validation["supported"]
    assert len(result.nested_validation["scores"]) == 3
    assert np.isfinite(result.final_score)
    namespace = {"tuner": tuner}
    exec(tuner.generate_code(plan=plan, run=True), namespace)
    assert namespace["nested_validation"]["supported"]


def test_group_cv_rejects_rows_outside_training(bundle):
    cv = IndexedGroupKFold(pd.Series(np.repeat(np.arange(18), 5), index=bundle["X_train"].index), n_splits=2)
    with pytest.raises(ValueError, match="outside"):
        list(cv.split(bundle["X_test"]))


def test_older_bundle_requires_explicit_split_policy(bundle):
    bundle.pop("split_config")
    with pytest.raises(ValueError, match="older bundle"):
        HyperTuneWorkbench.from_bahamut(bundle, task_type="regression")
    assert isinstance(HyperTuneWorkbench.from_bahamut(bundle, task_type="regression", temporal=True).context.cv, TimeSeriesSplit)


def test_temporal_policy_cannot_be_overridden(bundle):
    with pytest.raises(ValueError, match="cannot be declared"):
        HyperTuneWorkbench.from_bahamut(bundle, task_type="regression", temporal=False)
    with pytest.raises(ValueError, match="arbitrary cv"):
        HyperTuneWorkbench.from_bahamut(bundle, task_type="regression", cv=KFold(3))


def test_bundle_rejects_target_leakage_and_overlapping_partitions(bundle):
    contaminated = dict(bundle)
    contaminated["X_train"] = bundle["X_train"].assign(y=bundle["y_train"])
    with pytest.raises(ValueError, match="target column"):
        HyperTuneWorkbench.from_bahamut(contaminated, task_type="regression")
    overlapping = dict(bundle, X_test=bundle["X_train"].iloc[:5], y_test=bundle["y_train"].iloc[:5])
    with pytest.raises(ValueError, match="disjoint"):
        HyperTuneWorkbench.from_bahamut(overlapping, task_type="regression")


def test_bahamut_generated_bundle_contains_policy():
    bahamut = pytest.importorskip("bahamut")
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=60), "x": np.arange(60), "y": np.arange(60) * 2})
    segments = (bahamut.BahamutSplit(df).definir_problema("series_temporales").definir_objetivo("y")
                .definir_predictoras(exclude_cols=["date"]).definir_orden_temporal("date")
                .configurar_split(test_size=0.2, shuffle=False).ejecutar_segmentacion())
    tuner = HyperTuneWorkbench.from_bahamut(segments, task_type="regression", n_splits=2)
    assert isinstance(tuner.context.cv, TimeSeriesSplit)
    assert tuner.context.X_train.columns.tolist() == ["x"]


def test_missing_group_labels_cannot_fall_back_to_random_cv(bundle):
    bundle["split_config"] = {"problema": "regresion", "use_group_split": True}
    with pytest.raises(ValueError, match="groups_train is missing"):
        HyperTuneWorkbench.from_bahamut(bundle)


def test_train_and_test_cannot_share_groups(bundle):
    bundle["split_config"] = {"problema": "regresion"}
    bundle["groups_train"] = pd.Series(np.repeat(np.arange(18), 5), index=bundle["X_train"].index)
    bundle["groups_test"] = pd.Series(0, index=bundle["X_test"].index)
    with pytest.raises(ValueError, match="groups must be disjoint"):
        HyperTuneWorkbench.from_bahamut(bundle)
