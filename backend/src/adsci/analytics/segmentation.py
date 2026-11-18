"""Customer/record segmentation with k-means, choosing k by silhouette score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SILHOUETTE_SAMPLE = 5000


class ClusterProfile(BaseModel):
    cluster: int
    size: int
    share: float
    means: dict[str, float]
    distinguishing: list[str]


class SegmentReport(BaseModel):
    k: int
    silhouette: float
    silhouette_by_k: dict[int, float]
    features: list[str]
    log_transformed: list[str]
    profiles: list[ClusterProfile]
    labels: list[int]
    warnings: list[str] = []


def segment(
    df: pd.DataFrame,
    features: list[str],
    k: int | None = None,
    k_range: tuple[int, int] = (2, 8),
    seed: int = 0,
) -> SegmentReport:
    X = df[features].astype(float)
    X = X.fillna(X.median())
    # Log-transform heavily right-skewed non-negative features so they don't dominate distances.
    logged = [c for c in features if X[c].min() >= 0 and X[c].skew() > 1]
    Xt = X.copy()
    for c in logged:
        Xt[c] = np.log1p(Xt[c])
    Z = StandardScaler().fit_transform(Xt)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Z), min(len(Z), SILHOUETTE_SAMPLE), replace=False)

    def fit(kk: int):
        return KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z)

    scores: dict[int, float] = {}
    models = {}
    ks = [k] if k else list(range(k_range[0], min(k_range[1], len(Z) - 1) + 1))
    for kk in ks:
        m = fit(kk)
        models[kk] = m
        scores[kk] = float(silhouette_score(Z[sample], m.labels_[sample]))
    best_k = k or max(scores, key=scores.get)
    labels = models[best_k].labels_
    overall_mean, overall_std = X.mean(), X.std().replace(0, 1)
    profiles = []
    for c in range(best_k):
        members = X[labels == c]
        z = ((members.mean() - overall_mean) / overall_std).sort_values(key=np.abs, ascending=False)
        profiles.append(ClusterProfile(
            cluster=c, size=int(len(members)), share=round(len(members) / len(X), 6),
            means={f: float(members[f].mean()) for f in features},
            distinguishing=[f"{f} {'high' if z[f] > 0 else 'low'} ({z[f]:+.1f} sd)" for f in z.index[:3]],
        ))
    warns = []
    if scores[best_k] < 0.25:
        warns.append(f"Weak cluster structure (silhouette {scores[best_k]:.2f}); segments overlap substantially.")
    return SegmentReport(
        k=best_k, silhouette=scores[best_k], silhouette_by_k=scores, features=features, log_transformed=logged,
        profiles=profiles, labels=[int(x) for x in labels], warnings=warns,
    )
