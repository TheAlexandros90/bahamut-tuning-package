# Bahamut Tuning

Bahamut Tuning is a notebook-friendly hyperparameter-tuning subsystem designed to live inside a broader machine-learning library.

It is intentionally not a standalone AutoML framework. The package assumes that upstream components may already define the problem, the recommended algorithm, the split strategy, the CV splitter, the scorer, and the preprocessing pipeline.

## Design goals

- Reuse upstream artifacts instead of silently duplicating them.
- Keep planning and execution as separate stages.
- Support notebook interaction and direct programmatic usage.
- Return structured Python objects at every important step.
- Remain modular and easy to embed into a larger library.

## Main components

- `TuningContextAdapter`: normalizes upstream artifacts.
- `InteractiveTuningUI`: notebook UI for confirmation and overrides.
- `HyperparameterPlanner`: chooses algorithm, strategy, search space, and assumptions.
- `HyperparameterExecutor`: runs tuning or generates reusable code.
- `HyperTuneWorkbench`: facade for notebook and programmatic workflows.

## Public API

```python
from bahamut_tuning import HyperTuneWorkbench, TuningSelection

workbench = HyperTuneWorkbench(
    df=df,
    target=["y"],
    task_type="regression",
    selected_algorithm="RandomForestRegressor",
    cv=cv_object,
    scorer="neg_root_mean_squared_error",
    preprocessor=preprocessor,
    metadata=upstream_metadata,
)

plan = workbench.plan()
result = workbench.execute(run=True)
code = workbench.generate_code()
workbench.render_ui()
```

## Installation

### Reusing Bahamut partitions

```python
tuner = HyperTuneWorkbench.from_bahamut(
    segmentos,  # BahamutSplit.ejecutar_segmentacion()
    task_type="regression",
    selected_algorithm="Ridge",
    n_splits=3,
    gap=1,  # rows between temporal training and validation folds
)
plan = tuner.plan(TuningSelection(nested_cv=True, nested_cv_folds=3))
result = tuner.execute(plan=plan)
```

The bridge keeps `X_train` predictors and uses `X_test` only for the final
holdout score. The upstream validation partition stays reserved; it is not
merged into training. Task type is inferred for classification/regression
bundles and must be supplied for temporal bundles.

Temporal bundles use `TimeSeriesSplit`. Group bundles use `IndexedGroupKFold`,
which binds group labels to the original training index and keeps groups
separate in both inner and outer folds. Both nested validation and generated
code preserve the CV policy. Random `KFold`/`StratifiedKFold` retain their
configured shuffle policy in nested validation too. Unsupported custom
splitters are rejected for nested validation rather than replaced silently.

Source indices must be unique and disjoint between train and test. Reset an
ambiguous index **before** splitting in Bahamut. A bridge bundle cannot override
upstream X/y artifacts through keyword arguments. For older bundles without
`split_config`, specify `task_type` and `temporal=True/False` explicitly. Group
bundles require their `groups_train` labels. To change the split policy, create
new upstream segments; an arbitrary `cv=` cannot override temporal/group bundles.

`gap` is measured in rows, not elapsed time. Choose it for your prediction
horizon and feature construction; the bridge cannot detect leakage already
introduced by upstream feature engineering. Small datasets must contain enough
rows/classes/groups for each nested fold. `examples/bahamut_bridge_demo.py`
demonstrates both policies with synthetic data and no network access.

Install both sibling clones before running that example. From the tuning clone:

```bash
pip install -e "../bahamut" -e "."
python examples/bahamut_bridge_demo.py
```

Base package:

```bash
pip install .
```

Notebook support:

```bash
pip install .[notebook]
```

Optuna and XGBoost support:

```bash
pip install .[optuna,xgboost]
```

Everything:

```bash
pip install .[all]
```

## Deliverables included in this project

- Package code in `src/bahamut_tuning`
- Notebook example in `examples/BahamutTuning_notebook_example.ipynb`
- Integrated upstream usage example in `examples/integrated_usage.py`
- Tests in `tests/`

## Notes

- `render_ui()` requires `ipywidgets`.
- Optuna is optional and only required when the chosen strategy is `optuna`.
- XGBoost estimators are exposed only when `xgboost` is installed.
- The production API uses the neutral `Tuning*` names and `HyperTuneWorkbench` as the subsystem facade.
