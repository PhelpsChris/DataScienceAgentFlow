"""Train and compare three candidate models with a fixed, explainable procedure.

1. Decide the task: classification (category target, or a number with <= 10
   distinct values) or regression (any other number).
2. Split 60% train / 20% validation / 20% test with a fixed random seed.
3. Fit three candidates, ordered from simplest to most complex.
4. Pick the best validation score. A *tie* (within TIE_TOLERANCE) goes to the
   simpler model, so the result is the same on every run.
5. Refit the winner on train+validation and report final scores on the
   untouched test set, plus permutation feature importance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TIE_TOLERANCE = 0.01
MAX_TRAIN_ROWS = 20_000
LEAKAGE_CORR = 0.8


@dataclass(frozen=True)
class Candidate:
    name: str
    complexity: int  # lower = simpler, easier to explain
    plain_english: str


REGRESSION_CANDIDATES = [
    Candidate("Linear Regression (Ridge)", 1, "a straight-line formula; each feature adds or subtracts a fixed amount"),
    Candidate("Random Forest", 2, "many decision trees that vote; captures non-linear effects"),
    Candidate("Gradient Boosting", 3, "trees built one after another, each fixing the last one's mistakes"),
]
CLASSIFICATION_CANDIDATES = [
    Candidate("Logistic Regression", 1, "a weighted formula that turns features into class probabilities"),
    Candidate("Random Forest", 2, "many decision trees that vote; captures non-linear effects"),
    Candidate("Gradient Boosting", 3, "trees built one after another, each fixing the last one's mistakes"),
]


def detect_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    return "classification" if y.nunique() <= 10 else "regression"


def _estimator(task: str, name: str):
    if task == "regression":
        return {
            "Linear Regression (Ridge)": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        }[name]
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }[name]


def _pipeline(X: pd.DataFrame, task: str, name: str) -> Pipeline:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ]
    )
    return Pipeline([("prep", pre), ("model", _estimator(task, name))])


def _scores(task: str, y_true, y_pred) -> dict:
    if task == "regression":
        return {
            "r2": round(float(r2_score(y_true, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def primary_metric(task: str) -> str:
    return "r2" if task == "regression" else "f1_macro"


def leakage_suspects(df: pd.DataFrame, target: str) -> list[dict]:
    """Numeric features so strongly correlated with a numeric target that they may be
    'answers in disguise' (e.g. G1/G2 for G3). Worth re-running without them."""
    if not pd.api.types.is_numeric_dtype(df[target]):
        return []
    out = []
    for c in df.columns:
        if c == target or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        r = df[[c, target]].corr().iloc[0, 1]
        if pd.notna(r) and abs(r) >= LEAKAGE_CORR:
            out.append({"feature": c, "correlation": round(float(r), 3)})
    return sorted(out, key=lambda d: -abs(d["correlation"]))


def train_and_compare(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    exclude_columns: list[str] | None = None,
) -> dict:
    """Run the full procedure and return a JSON-serialisable result."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is not in the dataset.")
    exclude = [c for c in (exclude_columns or []) if c in df.columns and c != target]
    data = df.drop(columns=exclude)
    notes = []
    if len(data) > MAX_TRAIN_ROWS:
        data = data.sample(MAX_TRAIN_ROWS, random_state=SEED)
        notes.append(f"Dataset has {len(df)} rows; used a random sample of {MAX_TRAIN_ROWS} to keep training fast.")

    y = data[target]
    X = data.drop(columns=target)
    if X.shape[1] == 0:
        raise ValueError("No feature columns are left to train on.")
    task = detect_task(y) if task in (None, "", "auto") else task
    if task == "classification":
        y = y.astype(str)
    candidates = REGRESSION_CANDIDATES if task == "regression" else CLASSIFICATION_CANDIDATES
    metric = primary_metric(task)

    strat = y if task == "classification" and y.value_counts().min() >= 5 else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=strat)
    strat2 = y_tmp if strat is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=SEED, stratify=strat2
    )

    results = []
    for cand in candidates:
        pipe = _pipeline(X_train, task, cand.name).fit(X_train, y_train)
        results.append(
            {
                "model": cand.name,
                "complexity": cand.complexity,
                "how_it_works": cand.plain_english,
                "validation": _scores(task, y_val, pipe.predict(X_val)),
            }
        )

    best_score = max(r["validation"][metric] for r in results)
    tied = [r for r in results if best_score - r["validation"][metric] <= TIE_TOLERANCE]
    winner = min(tied, key=lambda r: r["complexity"])
    if len(tied) > 1:
        why = (
            f"{' and '.join(r['model'] for r in tied)} scored within {TIE_TOLERANCE} of each other on "
            f"validation {metric}; by our tie rule we keep the simplest, {winner['model']}, "
            "because it is easier to explain and less likely to overfit."
        )
    elif winner["validation"][metric] == best_score:
        why = f"{winner['model']} had the best validation {metric} ({best_score}) by more than {TIE_TOLERANCE}."
    else:  # pragma: no cover - unreachable, kept for clarity
        why = f"{winner['model']} selected."

    # Refit on train+validation, evaluate once on the untouched test set.
    final = _pipeline(X_tmp, task, winner["model"]).fit(X_tmp, y_tmp)
    test_pred = final.predict(X_test)
    test_scores = _scores(task, y_test, test_pred)

    if task == "regression":
        base_pred = np.full(len(y_test), float(y_tmp.mean()))
        baseline = {"description": "always predict the training average", **_scores(task, y_test, base_pred)}
    else:
        majority = y_tmp.mode().iloc[0]
        baseline = {
            "description": f"always predict the most common class ('{majority}')",
            **_scores(task, y_test, np.full(len(y_test), majority)),
        }

    scoring = "r2" if task == "regression" else "f1_macro"
    pi = permutation_importance(final, X_test, y_test, n_repeats=5, random_state=SEED, scoring=scoring, n_jobs=-1)
    imp = sorted(
        (
            {"feature": c, "importance": round(float(m), 4), "std": round(float(s), 4)}
            for c, m, s in zip(X.columns, pi.importances_mean, pi.importances_std)
        ),
        key=lambda d: -d["importance"],
    )

    out = {
        "target": target,
        "task": task,
        "primary_metric": metric,
        "excluded_columns": exclude,
        "n_rows": int(len(data)),
        "n_features": int(X.shape[1]),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "candidates": results,
        "selected_model": winner["model"],
        "selection_reason": why,
        "test_scores": test_scores,
        "baseline": baseline,
        "feature_importance": imp[:15],
        "notes": notes,
        "_test_truth": [v.item() if hasattr(v, "item") else v for v in y_test.tolist()],
        "_test_pred": [v.item() if hasattr(v, "item") else v for v in test_pred.tolist()],
    }
    if task == "classification":
        labels = sorted(y.unique().tolist())
        out["class_labels"] = labels
        out["confusion_matrix"] = confusion_matrix(y_test, test_pred, labels=labels).tolist()
    return out
