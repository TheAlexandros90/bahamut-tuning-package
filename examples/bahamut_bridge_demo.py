"""Offline Bahamut -> tuning demo, including temporal and grouped validation.

Install bahamut and bahamut-tuning first. Run this file from any directory.
"""
import numpy as np
import pandas as pd

from bahamut import BahamutSplit
from bahamut_tuning import HyperTuneWorkbench, TuningSelection


def run_demo():
    rng = np.random.default_rng(17)
    df = pd.DataFrame({
        "fecha": pd.date_range("2024-01-01", periods=180),
        "cliente": np.repeat(np.arange(30), 6),
        "x1": rng.normal(size=180),
        "x2": rng.normal(size=180),
    })
    df["ventas"] = 10 + 2 * df.x1 - df.x2 + rng.normal(scale=0.2, size=180)
    for temporal in (True, False):
        splitter = (BahamutSplit(df)
                    .definir_problema("series_temporales" if temporal else "regresion")
                    .definir_objetivo("ventas")
                    .definir_predictoras(exclude_cols=["fecha", "cliente"]))
        if temporal:
            splitter.definir_orden_temporal("fecha")
        else:
            splitter.definir_grupos("cliente")
        splitter.configurar_split(test_size=0.2, validation_size=0.2, shuffle=not temporal)
        segments = splitter.ejecutar_segmentacion()
        tuner = HyperTuneWorkbench.from_bahamut(
            segments, task_type="regression", n_splits=2,
            gap=1 if temporal else 0, selected_algorithm="Ridge",
        )
        plan = tuner.plan(TuningSelection(
            selected_algorithm="Ridge", tuning_strategy="grid",
            nested_cv=True, nested_cv_folds=3,
        ))
        plan.search_space = {"model__alpha": [0.1, 1.0]}
        result = tuner.execute(plan=plan)
        print("Temporal" if temporal else "Grupos")
        print("CV:", plan.cv_summary)
        print("Train/test:", len(segments["X_train"]), len(segments["X_test"]))
        print("CV anidada:", result.nested_validation["score_mean"])
        print("Test final (neg RMSE):", result.final_score)
    return True


if __name__ == "__main__":
    run_demo()
