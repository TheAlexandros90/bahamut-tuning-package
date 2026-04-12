from __future__ import annotations

from copy import deepcopy
import logging
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .helpers import top_configs_from_cv_results
from .types import TuningContext, TuningPlan, TuningResult

logger = logging.getLogger(__name__)


def _get_optuna_module(required: bool = False) -> Any:
    try:
        import optuna
    except Exception as err:
        if required:
            raise RuntimeError(
                "Optuna strategy requires the optional dependency `optuna`. Install bahamut-tuning[optuna]."
            ) from err
        return None
    return optuna


def default_preprocessor_from_df(X: pd.DataFrame) -> ColumnTransformer:
    """Build a minimal fallback preprocessor only when upstream preprocessing is absent."""

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def resolve_training_data(
    context: TuningContext,
) -> tuple[pd.DataFrame, Any, pd.DataFrame | None, Any | None, list[str]]:
    """Resolve training artifacts without redefining upstream split logic."""

    warnings: list[str] = []
    if context.X_train is not None and context.y_train is not None:
        return context.X_train, context.y_train, context.X_test, context.y_test, warnings

    if context.df is None:
        raise ValueError("Execution needs either upstream X_train/y_train or raw df + target.")
    if not context.target:
        raise ValueError("Execution from df requires target columns in context.target.")

    X = context.df.drop(columns=context.target)
    y_df = context.df[context.target]
    y = y_df.iloc[:, 0] if y_df.shape[1] == 1 else y_df
    warnings.append("No upstream split provided; execution will use full df with CV only and no holdout final score.")
    return X, y, None, None, warnings


def _generalization_risk_level(max_gap: float | None) -> str:
    if max_gap is None:
        return "unknown"
    if max_gap >= 0.05:
        return "high"
    if max_gap >= 0.02:
        return "moderate"
    return "low"


def compute_generalization_diagnostics(
    fitted_estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: Any,
    scoring: str,
    best_cv_score: float | None,
    X_test: pd.DataFrame | None = None,
    y_test: Any | None = None,
    nested_cv_score: float | None = None,
) -> dict[str, Any]:
    """Estimate generalization risk from train, CV, and optional holdout scores."""

    scorer = get_scorer(scoring)
    training_score = float(scorer(fitted_estimator, X_train, y_train))
    holdout_score = None
    if X_test is not None and y_test is not None:
        holdout_score = float(scorer(fitted_estimator, X_test, y_test))

    train_vs_cv_gap = None
    if best_cv_score is not None:
        train_vs_cv_gap = float(training_score - best_cv_score)

    cv_vs_holdout_gap = None
    if best_cv_score is not None and holdout_score is not None:
        cv_vs_holdout_gap = float(best_cv_score - holdout_score)

    cv_vs_nested_gap = None
    if best_cv_score is not None and nested_cv_score is not None:
        cv_vs_nested_gap = float(best_cv_score - nested_cv_score)

    comparable_gaps = [
        gap
        for gap in [train_vs_cv_gap, cv_vs_holdout_gap, cv_vs_nested_gap]
        if gap is not None and gap > 0
    ]
    max_gap = max(comparable_gaps) if comparable_gaps else 0.0
    risk_level = _generalization_risk_level(max_gap)

    warnings: list[str] = []
    if holdout_score is None:
        warnings.append("No holdout split available; overfitting is being checked only against the CV score.")
    if train_vs_cv_gap is not None and train_vs_cv_gap >= 0.05:
        warnings.append("Training score is materially higher than the CV score; the tuned model may be overfitting.")
    if cv_vs_holdout_gap is not None and cv_vs_holdout_gap >= 0.05:
        warnings.append("CV score is materially higher than the holdout score; the tuned model may be overfitting to cross-validation.")
    if cv_vs_nested_gap is not None and cv_vs_nested_gap >= 0.05:
        warnings.append("CV score is materially higher than the nested CV estimate; model selection may be optimistic.")

    if risk_level == "high":
        assessment = "High generalization gap detected with the current heuristic thresholds."
    elif risk_level == "moderate":
        assessment = "Moderate generalization gap detected; inspect features, search space, and split strategy."
    elif risk_level == "low":
        assessment = "Generalization gap is currently low under the built-in heuristic thresholds."
    else:
        assessment = "Generalization risk could not be fully assessed from the available artifacts."

    return {
        "scoring": scoring,
        "training_score": training_score,
        "best_cv_score": best_cv_score,
        "holdout_score": holdout_score,
        "nested_cv_score": nested_cv_score,
        "train_vs_cv_gap": train_vs_cv_gap,
        "cv_vs_holdout_gap": cv_vs_holdout_gap,
        "cv_vs_nested_gap": cv_vs_nested_gap,
        "risk_level": risk_level,
        "assessment": assessment,
        "thresholds": {"moderate_gap": 0.02, "high_gap": 0.05},
        "warnings": warnings,
    }


def enforce_overfit_policy(generalization: dict[str, Any], overfit_policy: str) -> None:
    """Block execution when the configured generalization-risk policy is violated."""

    if overfit_policy == "warn":
        return

    risk_level = str(generalization.get("risk_level", "unknown"))
    order = {"unknown": 0, "low": 1, "moderate": 2, "high": 3}
    threshold = {"block_high": 3, "block_moderate": 2}.get(overfit_policy)
    if threshold is None:
        raise ValueError(f"Unsupported overfit policy: {overfit_policy}")

    if order.get(risk_level, 0) >= threshold:
        raise RuntimeError(
            "Execution blocked by overfit policy "
            f"'{overfit_policy}'. Risk={risk_level}. {generalization.get('assessment', '')}"
        )


def _nested_outer_cv(task_type: str, outer_folds: int, random_state: int) -> Any:
    if task_type == "classification":
        return StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)


def _clone_cv(cv: Any) -> Any:
    if cv is None or isinstance(cv, int):
        return cv
    return deepcopy(cv)


def _search_estimator_for_strategy(
    pipeline: Pipeline,
    *,
    search_space: dict[str, list[Any]],
    strategy: str,
    scoring: str,
    cv: Any,
    iterations: int,
    timeout_seconds: int | None,
    random_state: int,
) -> tuple[Any | None, str, list[str]]:
    warnings: list[str] = []

    if strategy == "grid":
        return (
            GridSearchCV(
                estimator=clone(pipeline),
                param_grid=search_space,
                scoring=scoring,
                cv=_clone_cv(cv),
                n_jobs=-1,
                refit=True,
            ),
            "grid",
            warnings,
        )
    if strategy == "random":
        return (
            RandomizedSearchCV(
                estimator=clone(pipeline),
                param_distributions=search_space,
                n_iter=iterations,
                scoring=scoring,
                cv=_clone_cv(cv),
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            ),
            "random",
            warnings,
        )
    if strategy == "optuna":
        warnings.append(
            "Nested CV cannot wrap Optuna directly; using RandomizedSearchCV as an approximation for outer validation."
        )
        return (
            RandomizedSearchCV(
                estimator=clone(pipeline),
                param_distributions=search_space,
                n_iter=iterations,
                scoring=scoring,
                cv=_clone_cv(cv),
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            ),
            "random_approximation",
            warnings,
        )
    return None, strategy, warnings


def evaluate_nested_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: Any,
    *,
    task_type: str,
    search_space: dict[str, list[Any]],
    strategy: str,
    scoring: str,
    inner_cv: Any,
    iterations: int,
    timeout_seconds: int | None,
    outer_folds: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run optional nested CV to get a stricter estimate of tuned-model generalization."""

    outer_cv = _nested_outer_cv(task_type=task_type, outer_folds=outer_folds, random_state=random_state)
    search_estimator, strategy_used, warnings = _search_estimator_for_strategy(
        pipeline=pipeline,
        search_space=search_space,
        strategy=strategy,
        scoring=scoring,
        cv=inner_cv,
        iterations=iterations,
        timeout_seconds=timeout_seconds,
        random_state=random_state,
    )
    if search_estimator is None:
        return {
            "enabled": True,
            "supported": False,
            "strategy_used": strategy_used,
            "outer_folds": outer_folds,
            "score_mean": None,
            "score_std": None,
            "score_min": None,
            "score_max": None,
            "scores": [],
            "warnings": warnings + [f"Nested CV is not supported for strategy '{strategy}'."],
        }

    scores = cross_val_score(
        search_estimator,
        X_train,
        y_train,
        scoring=scoring,
        cv=outer_cv,
        n_jobs=-1,
    )
    numeric_scores = [float(score) for score in scores]
    return {
        "enabled": True,
        "supported": True,
        "strategy_used": strategy_used,
        "outer_folds": outer_folds,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "scores": numeric_scores,
        "warnings": warnings,
    }


def extract_model_best_params(best_params: dict[str, Any], model_prefix: str = "model__") -> dict[str, Any]:
    """Strip pipeline prefixes so tuned estimator params can be reused directly on a model instance."""

    return {
        key[len(model_prefix):]: value
        for key, value in best_params.items()
        if key.startswith(model_prefix)
    }


def generate_tuning_code(
    context: TuningContext,
    plan: TuningPlan,
    random_state: int = 42,
    run: bool = False,
) -> str:
    """Generate self-contained code that plugs into an active HyperTuneWorkbench."""

    lines = [
        "# Reproducible hyperparameter tuning snippet",
        "from sklearn.base import clone",
        "from sklearn.model_selection import GridSearchCV, RandomizedSearchCV",
        "from bahamut_tuning import (",
        "    compute_generalization_diagnostics,",
        "    enforce_overfit_policy,",
        "    extract_model_best_params,",
        "    evaluate_nested_validation,",
        "    resolve_training_data,",
        ")",
        "",
        "if 'tuner' in globals():",
        "    active_tuner = tuner",
        "elif 'example_cls' in globals() and isinstance(example_cls, dict) and 'tuner' in example_cls:",
        "    active_tuner = example_cls['tuner']",
        "elif 'tuner_reg' in globals():",
        "    active_tuner = tuner_reg",
        "else:",
        "    raise NameError('Expected a HyperTuneWorkbench instance in `tuner`, `tuner_reg`, or `example_cls[\"tuner\"]`.')",
        "",
        "context = active_tuner.context",
        "planner = active_tuner.planner",
        "executor = active_tuner.executor",
        "X_train, y_train, X_test, y_test, execution_notes = resolve_training_data(context)",
        f"algorithm_name = {plan.selected_algorithm!r}",
        f"scoring = {plan.scoring!r}",
        f"param_space = {plan.search_space!r}",
        "cv = context.cv if context.cv is not None else 5",
        f"overfit_policy = {plan.overfit_policy!r}",
        f"nested_cv_enabled = {str(plan.nested_cv)}",
        f"nested_cv_folds = {plan.nested_cv_folds}",
        "estimator = clone(planner.available_estimators(context.task_type)[algorithm_name])",
        "pipeline = executor.build_estimator_pipeline(",
        "    context=context,",
        "    estimator=estimator,",
        "    X_train=X_train,",
        ")",
        f"RUN_TUNING = {str(run)}",
        "",
    ]

    if plan.tuning_strategy == "grid":
        lines += [
            "search = GridSearchCV(",
            "    estimator=pipeline,",
            "    param_grid=param_space,",
            "    scoring=scoring,",
            "    cv=cv,",
            "    n_jobs=-1,",
            "    refit=True,",
            ")",
        ]
    elif plan.tuning_strategy == "random":
        lines += [
            "search = RandomizedSearchCV(",
            "    estimator=pipeline,",
            "    param_distributions=param_space,",
            f"    n_iter={plan.iterations},",
            "    scoring=scoring,",
            "    cv=cv,",
            "    n_jobs=-1,",
            f"    random_state={random_state},",
            "    refit=True,",
            ")",
        ]
    elif plan.tuning_strategy == "optuna":
        lines += [
            "import optuna",
            "import numpy as np",
            "from sklearn.model_selection import cross_val_score",
            "",
            "def objective(trial):",
            "    params = {name: trial.suggest_categorical(name, values) for name, values in param_space.items()}",
            "    candidate = clone(pipeline).set_params(**params)",
            "    scores = cross_val_score(candidate, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)",
            "    return float(np.mean(scores))",
            "",
            "if RUN_TUNING:",
            "    study = optuna.create_study(direction='maximize')",
            f"    study.optimize(objective, n_trials={plan.iterations}, timeout={plan.timeout_seconds})",
            "    best_params = study.best_trial.params",
            "    best_cv_score = float(study.best_value)",
            "    best_pipeline = clone(pipeline).set_params(**best_params)",
            "    best_pipeline.fit(X_train, y_train)",
            "    nested_validation = {'enabled': False, 'supported': False, 'warnings': []}",
            "    if nested_cv_enabled:",
            "        nested_validation = evaluate_nested_validation(",
            "            pipeline=pipeline,",
            "            X_train=X_train,",
            "            y_train=y_train,",
            "            task_type=context.task_type,",
            "            search_space=param_space,",
            "            strategy='optuna',",
            "            scoring=scoring,",
            "            inner_cv=cv,",
            f"            iterations={plan.iterations},",
            f"            timeout_seconds={plan.timeout_seconds!r},",
            "            outer_folds=nested_cv_folds,",
            f"            random_state={random_state},",
            "        )",
            "    generalization = compute_generalization_diagnostics(",
            "        fitted_estimator=best_pipeline,",
            "        X_train=X_train,",
            "        y_train=y_train,",
            "        scoring=scoring,",
            "        best_cv_score=best_cv_score,",
            "        X_test=X_test,",
            "        y_test=y_test,",
            "        nested_cv_score=nested_validation.get('score_mean'),",
            "    )",
            "    enforce_overfit_policy(generalization, overfit_policy)",
            "    BEST_PIPELINE = best_pipeline",
            "    BEST_PIPELINE_PARAMS = dict(best_params)",
            "    BEST_MODEL_PARAMS = extract_model_best_params(BEST_PIPELINE_PARAMS)",
            "    BEST_MODEL = clone(estimator).set_params(**BEST_MODEL_PARAMS)",
            "    EXECUTION_RESULT = {",
            "        'selected_algorithm': algorithm_name,",
            "        'best_params': BEST_PIPELINE_PARAMS,",
            "        'model_best_params': BEST_MODEL_PARAMS,",
            "        'best_cv_score': best_cv_score,",
            "        'final_score': generalization['holdout_score'],",
            "        'generalization': generalization,",
            "        'nested_validation': nested_validation,",
            "        'notes': execution_notes,",
            "    }",
            "    print(EXECUTION_RESULT)",
            "else:",
            "    print('Snippet prepared. Set RUN_TUNING=True to execute Optuna tuning.')",
        ]
    else:
        lines += [
            "# guide_only mode: inspect the plan, assumptions, and search space before wiring execution."
        ]

    if plan.tuning_strategy in {"grid", "random"}:
        lines += [
            "if RUN_TUNING:",
            "    search.fit(X_train, y_train)",
            "    nested_validation = {'enabled': False, 'supported': False, 'warnings': []}",
            "    if nested_cv_enabled:",
            "        nested_validation = evaluate_nested_validation(",
            "            pipeline=pipeline,",
            "            X_train=X_train,",
            "            y_train=y_train,",
            "            task_type=context.task_type,",
            "            search_space=param_space,",
            f"            strategy={plan.tuning_strategy!r},",
            "            scoring=scoring,",
            "            inner_cv=cv,",
            f"            iterations={plan.iterations},",
            f"            timeout_seconds={plan.timeout_seconds!r},",
            "            outer_folds=nested_cv_folds,",
            f"            random_state={random_state},",
            "        )",
            "    generalization = compute_generalization_diagnostics(",
            "        fitted_estimator=search.best_estimator_,",
            "        X_train=X_train,",
            "        y_train=y_train,",
            "        scoring=scoring,",
            "        best_cv_score=float(search.best_score_),",
            "        X_test=X_test,",
            "        y_test=y_test,",
            "        nested_cv_score=nested_validation.get('score_mean'),",
            "    )",
            "    enforce_overfit_policy(generalization, overfit_policy)",
            "    BEST_PIPELINE = search.best_estimator_",
            "    BEST_PIPELINE_PARAMS = dict(search.best_params_)",
            "    BEST_MODEL_PARAMS = extract_model_best_params(BEST_PIPELINE_PARAMS)",
            "    BEST_MODEL = clone(estimator).set_params(**BEST_MODEL_PARAMS)",
            "    EXECUTION_RESULT = {",
            "        'selected_algorithm': algorithm_name,",
            "        'best_params': BEST_PIPELINE_PARAMS,",
            "        'model_best_params': BEST_MODEL_PARAMS,",
            "        'best_cv_score': float(search.best_score_),",
            "        'final_score': generalization['holdout_score'],",
            "        'generalization': generalization,",
            "        'nested_validation': nested_validation,",
            "        'notes': execution_notes,",
            "    }",
            "    print(EXECUTION_RESULT)",
            "else:",
            "    print('Snippet prepared. Set RUN_TUNING=True to execute the search.')",
        ]
    return "\n".join(lines)


def execute_tuning_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: Any,
    plan: TuningPlan,
    cv: Any,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execution helper used by the executor and unit tests."""

    strategy = plan.tuning_strategy
    warnings: list[str] = []
    top_trials: list[dict[str, Any]] = []

    optuna_module = _get_optuna_module(required=False)

    if strategy == "optuna" and optuna_module is None:
        strategy = "random"
        warnings.append("Optuna is not installed; execution fell back to RandomizedSearchCV.")

    if strategy == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=plan.search_space,
            scoring=plan.scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_cv = float(search.best_score_)
        fitted = search.best_estimator_
        top_trials = [{"rank": 1, "score": best_cv, "params": best_params}]
    elif strategy == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=plan.search_space,
            n_iter=plan.iterations,
            scoring=plan.scoring,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_cv = float(search.best_score_)
        fitted = search.best_estimator_
        top_trials = top_configs_from_cv_results(
            pd.DataFrame(search.cv_results_),
            limit=5,
        )
    elif strategy == "optuna":
        optuna_module = _get_optuna_module(required=True)
        sampler = optuna_module.samplers.TPESampler(seed=random_state)
        study = optuna_module.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            params = {
                key: trial.suggest_categorical(key, values)
                for key, values in plan.search_space.items()
            }
            candidate = clone(pipeline).set_params(**params)
            scores = cross_val_score(
                candidate,
                X_train,
                y_train,
                scoring=plan.scoring,
                cv=cv,
                n_jobs=-1,
            )
            return float(np.mean(scores))

        study.optimize(objective, n_trials=plan.iterations, timeout=plan.timeout_seconds)
        best_params = study.best_trial.params
        fitted = clone(pipeline).set_params(**best_params)
        fitted.fit(X_train, y_train)
        best_cv = float(study.best_value)
        ranked_trials = sorted(
            study.trials,
            key=lambda trial: trial.value if trial.value is not None else float("-inf"),
            reverse=True,
        )[:5]
        top_trials = [
            {"rank": index + 1, "score": trial.value, "params": trial.params}
            for index, trial in enumerate(ranked_trials)
        ]
    else:
        raise ValueError(f"Unsupported execution strategy: {strategy}")

    return {
        "strategy": strategy,
        "warnings": warnings,
        "best_params": best_params,
        "best_cv_score": best_cv,
        "top_trials": top_trials,
        "fitted_estimator": fitted,
    }


class HyperparameterExecutor:
    """Execute a tuning plan or generate reproducible starter code."""

    def __init__(self, planner: Any, random_state: int = 42) -> None:
        self.planner = planner
        self.random_state = random_state

    def build_estimator_pipeline(
        self,
        context: TuningContext,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
    ) -> Pipeline:
        if context.preprocessor is not None and isinstance(context.preprocessor, Pipeline):
            upstream_pipeline = clone(context.preprocessor)
            if "model" in upstream_pipeline.named_steps:
                upstream_pipeline.set_params(model=estimator)
                return upstream_pipeline
            return Pipeline(upstream_pipeline.steps + [("model", estimator)])

        if context.preprocessor is not None:
            return Pipeline([("preprocess", clone(context.preprocessor)), ("model", estimator)])

        return Pipeline([("preprocess", default_preprocessor_from_df(X_train)), ("model", estimator)])

    def generate_code(self, context: TuningContext, plan: TuningPlan, run: bool = False) -> str:
        return generate_tuning_code(
            context=context,
            plan=plan,
            random_state=self.random_state,
            run=run,
        )

    def execute(
        self,
        context: TuningContext,
        plan: TuningPlan,
        run: bool = True,
    ) -> TuningResult:
        code = self.generate_code(context=context, plan=plan, run=run)
        if not run or plan.tuning_strategy == "guide_only":
            return TuningResult(
                selected_algorithm=plan.selected_algorithm,
                tuning_strategy_used=plan.tuning_strategy,
                best_params={},
                model_best_params={},
                best_cv_score=None,
                final_score=None,
                top_trials=[],
                runtime=0.0,
                warnings=plan.warnings.copy(),
                explanation=f"Execution skipped (run={run}). {plan.tuning_strategy_reason}",
                reproducible_code_snippet=code,
                generalization={},
                nested_validation={},
            )

        X_train, y_train, X_test, y_test, warnings = resolve_training_data(context)
        estimators = self.planner.available_estimators(plan.task_type)
        estimator = clone(estimators[plan.selected_algorithm])
        pipeline = self.build_estimator_pipeline(context=context, estimator=estimator, X_train=X_train)
        cv = context.cv if context.cv is not None else 5

        logger.info(
            "Executing tuning plan: algorithm=%s strategy=%s has_holdout=%s",
            plan.selected_algorithm,
            plan.tuning_strategy,
            X_test is not None and y_test is not None,
        )

        start = perf_counter()
        search_output = execute_tuning_search(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            plan=plan,
            cv=cv,
            random_state=self.random_state,
        )
        runtime = perf_counter() - start

        nested_validation = {"enabled": False, "supported": False, "warnings": []}
        if plan.nested_cv:
            nested_validation = evaluate_nested_validation(
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                task_type=plan.task_type,
                search_space=plan.search_space,
                strategy=plan.tuning_strategy,
                scoring=plan.scoring,
                inner_cv=cv,
                iterations=plan.iterations,
                timeout_seconds=plan.timeout_seconds,
                outer_folds=plan.nested_cv_folds,
                random_state=self.random_state,
            )

        generalization = compute_generalization_diagnostics(
            fitted_estimator=search_output["fitted_estimator"],
            X_train=X_train,
            y_train=y_train,
            scoring=plan.scoring,
            best_cv_score=search_output["best_cv_score"],
            X_test=X_test,
            y_test=y_test,
            nested_cv_score=nested_validation.get("score_mean"),
        )
        enforce_overfit_policy(generalization, plan.overfit_policy)
        model_best_params = extract_model_best_params(search_output["best_params"])

        return TuningResult(
            selected_algorithm=plan.selected_algorithm,
            tuning_strategy_used=search_output["strategy"],
            best_params=search_output["best_params"],
            model_best_params=model_best_params,
            best_cv_score=search_output["best_cv_score"],
            final_score=generalization["holdout_score"],
            top_trials=search_output["top_trials"],
            runtime=runtime,
            warnings=(
                plan.warnings
                + warnings
                + search_output["warnings"]
                + nested_validation.get("warnings", [])
                + generalization["warnings"]
            ),
            explanation=(
                f"Used {search_output['strategy']} for {plan.selected_algorithm}. "
                f"Reason: {plan.tuning_strategy_reason} "
                f"Generalization risk: {generalization['risk_level']}."
            ),
            reproducible_code_snippet=code,
            generalization=generalization,
            nested_validation=nested_validation,
        )
