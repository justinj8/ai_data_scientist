"""Predictive modeling done carefully: baselines, cross-validation, a held-out test set, leakage
checks and permutation importance."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from adsci.analytics.profile import _as_numeric, semantic_type

Task = Literal["classification", "regression"]
MAX_ROWS = 200_000
MAX_CATEGORIES = 50


class ModelResult(BaseModel):
    name: Literal["baseline", "linear", "gradient_boosting"]
    cv_mean: dict[str, float]
    cv_std: dict[str, float]
    test: dict[str, float]


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    std: float


class ModelReport(BaseModel):
    task: Task
    target: str
    primary_metric: str
    n_train: int
    n_test: int
    features_used: list[str]
    baseline: dict[str, float]
    models: list[ModelResult]
    best_model: str
    importances: list[FeatureImportance]
    dropped_features: list[tuple[str, str]]
    class_balance: dict[str, float] | None = None
    warnings: list[str] = []


def _infer_task(y: pd.Series) -> Task:
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 10:
        return "classification"
    return "regression"


def _select_features(df: pd.DataFrame, target: str, features: list[str] | None) -> tuple[list[str], list[tuple[str, str]]]:
    candidates = [c for c in (features or df.columns) if c != target]
    keep: list[str] = []
    dropped: list[tuple[str, str]] = []
    y_num = _as_numeric(df[target]) if not pd.api.types.is_string_dtype(df[target]) else None
    for c in candidates:
        s = df[c]
        sem = semantic_type(s)
        if sem == "id":
            dropped.append((c, "identifier"))
        elif s.nunique(dropna=True) <= 1:
            dropped.append((c, "constant"))
        elif sem == "datetime":
            dropped.append((c, "datetime; derive features (e.g. month, tenure) first"))
        elif sem == "text" or (sem == "categorical" and s.nunique() > MAX_CATEGORIES):
            dropped.append((c, f"high-cardinality text (> {MAX_CATEGORIES} categories)"))
        else:
            if y_num is not None and sem in ("numeric", "boolean"):
                r = _as_numeric(s).corr(y_num)
                if pd.notna(r) and abs(r) > 0.95:
                    dropped.append((c, f"possible target leakage (r={r:.2f})"))
                    continue
            keep.append(c)
    return keep, dropped


def _preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            ),
        ]
    )


def _test_metrics(task: Task, model, X, y, binary: bool) -> dict[str, float]:
    if task == "regression":
        pred = model.predict(X)
        return {
            "r2": float(r2_score(y, pred)),
            "rmse": float(root_mean_squared_error(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
        }
    pred = model.predict(X)
    proba = model.predict_proba(X)
    if binary:
        auc = roc_auc_score(y, proba[:, 1])
        f1 = f1_score(y, pred, pos_label=model.classes_[1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=model.classes_)
        f1 = f1_score(y, pred, average="weighted")
    return {"roc_auc": float(auc), "f1": float(f1), "accuracy": float(accuracy_score(y, pred))}


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    task: Task | None = None,
    seed: int = 0,
) -> ModelReport:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    warns: list[str] = []
    data = df.dropna(subset=[target])
    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=seed)
        warns.append(f"Trained on a random sample of {MAX_ROWS:,} rows.")
    y = data[target]
    task = task or _infer_task(y)
    used, dropped = _select_features(data, target, features)
    if not used:
        raise ValueError("No usable features remain after dropping identifiers, constants and leakage.")
    X = data[used].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)

    balance = None
    binary = False
    if task == "classification":
        y = y.astype(str) if not pd.api.types.is_bool_dtype(y) else y
        counts = y.value_counts(normalize=True)
        balance = {str(k): round(float(v), 6) for k, v in counts.items()}
        binary = len(counts) == 2
        if counts.min() < 0.10:
            warns.append(
                f"Class imbalance: the minority class is {counts.min():.1%} of rows; "
                "prefer ROC-AUC/F1 over accuracy."
            )
        if counts.min() * len(y) < 10:
            raise ValueError("Too few examples of the minority class to train and evaluate a model.")
    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=stratify)

    pre = _preprocessor(X, used)
    if task == "classification":
        estimators = {
            "baseline": DummyClassifier(strategy="most_frequent"),
            "linear": LogisticRegression(max_iter=2000),
            "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        }
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        scoring = (
            {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
            if binary
            else {"roc_auc": "roc_auc_ovr_weighted", "f1": "f1_weighted", "accuracy": "accuracy"}
        )
        if binary and not pd.api.types.is_bool_dtype(y):
            # sklearn's "f1" scorer needs an explicit positive label for string classes.
            from sklearn.metrics import make_scorer

            pos = sorted(y.unique())[1]
            scoring["f1"] = make_scorer(f1_score, pos_label=pos)
        primary = "roc_auc"
    else:
        estimators = {
            "baseline": DummyRegressor(strategy="mean"),
            "linear": Ridge(alpha=1.0),
            "gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
        }
        cv = KFold(5, shuffle=True, random_state=seed)
        scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
        primary = "r2"

    results: list[ModelResult] = []
    fitted: dict[str, Pipeline] = {}
    for name, est in estimators.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        cvr = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring)
        cv_mean, cv_std = {}, {}
        for metric in scoring:
            vals = np.asarray(cvr[f"test_{metric}"], dtype=float)
            if metric in ("rmse", "mae"):
                vals = -vals
            cv_mean[metric] = float(np.nanmean(vals))
            cv_std[metric] = float(np.nanstd(vals))
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        results.append(
            ModelResult(name=name, cv_mean=cv_mean, cv_std=cv_std, test=_test_metrics(task, pipe, X_te, y_te, binary))
        )

    best = max((r for r in results if r.name != "baseline"), key=lambda r: r.cv_mean[primary])
    baseline = next(r for r in results if r.name == "baseline").test
    if best.test[primary] <= baseline[primary] + 0.01:
        warns.append("The best model barely beats the baseline; these features carry little predictive signal.")

    scorer = scoring[primary]
    perm = permutation_importance(
        fitted[best.name], X_te, y_te, n_repeats=10, random_state=seed, scoring=scorer
    )
    importances = sorted(
        (
            FeatureImportance(feature=f, importance=float(m), std=float(s))
            for f, m, s in zip(used, perm.importances_mean, perm.importances_std, strict=True)
        ),
        key=lambda i: -i.importance,
    )
    return ModelReport(
        task=task, target=target, primary_metric=primary, n_train=len(X_tr), n_test=len(X_te),
        features_used=used, baseline=baseline, models=results, best_model=best.name,
        importances=importances, dropped_features=dropped, class_balance=balance, warnings=warns,
    )
