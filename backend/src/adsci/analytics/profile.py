"""Dataset profiling: semantic types, distributions, correlations and data-quality warnings."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

Semantic = Literal["numeric", "categorical", "datetime", "boolean", "text", "id"]
WarningKind = Literal[
    "high_missing", "constant", "id_like", "high_skew", "outliers", "duplicates", "possible_leakage"
]
_ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid)", re.IGNORECASE)
_BOOL_STRINGS = {"true", "false", "yes", "no", "y", "n", "t", "f"}


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    semantic: Semantic
    missing_pct: float
    n_unique: int
    top_values: list[dict] = []
    stats: dict[str, float] | None = None
    outlier_pct: float | None = None


class DataQualityWarning(BaseModel):
    column: str | None
    kind: WarningKind
    message: str


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    correlations: dict[str, dict[str, float]]
    warnings: list[DataQualityWarning]


def _is_bool(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    vals = s.dropna().unique()
    if len(vals) != 2:
        return False
    if pd.api.types.is_numeric_dtype(s):
        return set(np.asarray(vals, dtype=float)) == {0.0, 1.0}
    return {str(v).strip().lower() for v in vals} <= _BOOL_STRINGS


def semantic_type(s: pd.Series) -> Semantic:
    n = int(s.notna().sum())
    nu = int(s.nunique(dropna=True))
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if _is_bool(s):
        return "boolean"
    integerish = pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == object
    if _ID_NAME.search(str(s.name)) and integerish and (nu / max(n, 1) >= 0.5 or pd.api.types.is_integer_dtype(s)):
        return "id"
    if n > 50 and nu == n and integerish and not pd.api.types.is_float_dtype(s):
        return "id"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if nu <= max(50, int(0.05 * n)):
        return "categorical"
    return "text"


def _as_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    lowered = s.astype(str).str.strip().str.lower()
    return lowered.map({"true": 1.0, "yes": 1.0, "y": 1.0, "t": 1.0, "false": 0.0, "no": 0.0, "n": 0.0, "f": 0.0})


def profile_frame(
    df: pd.DataFrame, target: str | None = None, max_categories: int = 10, max_corr_cols: int = 30
) -> DatasetProfile:
    n = len(df)
    columns: list[ColumnProfile] = []
    warns: list[DataQualityWarning] = []
    numeric_cols: list[str] = []

    for name in df.columns:
        s = df[name]
        sem = semantic_type(s)
        missing = float(s.isna().mean() * 100) if n else 0.0
        nu = int(s.nunique(dropna=True))
        cp = ColumnProfile(name=str(name), dtype=str(s.dtype), semantic=sem, missing_pct=round(missing, 4), n_unique=nu)
        if sem in ("categorical", "boolean", "id", "text"):
            vc = s.value_counts(dropna=True).head(max_categories)
            cp.top_values = [
                {"value": v.item() if hasattr(v, "item") else v, "count": int(c), "pct": round(100 * c / max(n, 1), 2)}
                for v, c in vc.items()
            ]
        if sem == "numeric":
            x = s.dropna().astype(float)
            if len(x):
                q1, q3 = np.percentile(x, [25, 75])
                iqr = q3 - q1
                out = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100 if iqr > 0 else 0.0
                cp.stats = {
                    "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "p25": float(q1),
                    "median": float(x.median()), "p75": float(q3), "max": float(x.max()), "skew": float(x.skew()),
                }
                cp.outlier_pct = round(float(out), 4)
                if abs(cp.stats["skew"]) > 2:
                    warns.append(DataQualityWarning(column=str(name), kind="high_skew",
                                                    message=f"{name} is highly skewed (skew {cp.stats['skew']:.1f}); "
                                                            "consider a log transform or median-based statistics."))
                if out > 5:
                    warns.append(DataQualityWarning(column=str(name), kind="outliers",
                                                    message=f"{out:.1f}% of {name} values are outliers (1.5x IQR rule)."))
            numeric_cols.append(str(name))
        elif sem == "boolean":
            numeric_cols.append(str(name))
        if missing > 30:
            warns.append(DataQualityWarning(column=str(name), kind="high_missing",
                                            message=f"{name} is {missing:.0f}% missing."))
        if nu <= 1:
            warns.append(DataQualityWarning(column=str(name), kind="constant",
                                            message=f"{name} has a single value and carries no information."))
        if sem == "id":
            warns.append(DataQualityWarning(column=str(name), kind="id_like",
                                            message=f"{name} looks like an identifier; exclude it from models and stats."))
        columns.append(cp)

    dups = int(df.duplicated().sum()) if n else 0
    if dups:
        warns.append(DataQualityWarning(column=None, kind="duplicates", message=f"{dups} fully duplicated rows."))

    corr_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1][:max_corr_cols]
    correlations: dict[str, dict[str, float]] = {}
    if len(corr_cols) >= 2:
        mat = pd.DataFrame({c: _as_numeric(df[c]) for c in corr_cols}).corr()
        correlations = {
            a: {b: (1.0 if a == b else round(float(mat.loc[a, b]), 6)) for b in corr_cols} for a in corr_cols
        }

    if target is not None and target in df.columns:
        y = _as_numeric(df[target])
        if y.notna().any():
            for c in numeric_cols:
                if c == target:
                    continue
                r = _as_numeric(df[c]).corr(y)
                if pd.notna(r) and abs(r) > 0.95:
                    warns.append(DataQualityWarning(
                        column=c, kind="possible_leakage",
                        message=f"{c} is almost perfectly correlated with {target} (r={r:.2f}); it may leak the target.",
                    ))

    return DatasetProfile(
        n_rows=n, n_cols=df.shape[1], duplicate_rows=dups, columns=columns, correlations=correlations, warnings=warns
    )
