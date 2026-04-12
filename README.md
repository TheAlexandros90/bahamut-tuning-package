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
