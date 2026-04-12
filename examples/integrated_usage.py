from __future__ import annotations

from pathlib import Path
import sys

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bahamut_tuning import HyperTuneWorkbench, TuningSelection


def build_example() -> None:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42,
    )

    upstream_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    tuner = HyperTuneWorkbench(
        df=df,
        target=["target"],
        task_type="classification",
        selected_algorithm="LogisticRegression",
        recommended_algorithms=["LogisticRegression", "RandomForestClassifier"],
        recommender_reason="Linear baseline has strong signal and interpretability.",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scorer="f1_macro",
        preprocessor=upstream_pipeline,
        metadata={"tuning_policy": "random"},
    )

    selection = TuningSelection(
        selected_algorithm="LogisticRegression",
        tuning_strategy="auto",
        scoring="f1_macro",
        iterations=20,
    )
    plan = tuner.plan(selection=selection)
    print("Plan:")
    print(plan.to_dict())

    result = tuner.execute(plan=plan, run=True)
    print("\nBest params:")
    print(result.best_params)
    print("Best CV:", result.best_cv_score)

    print("\nGenerated code:\n")
    print(tuner.generate_code(plan))


if __name__ == "__main__":
    build_example()
