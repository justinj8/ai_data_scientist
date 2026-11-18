"""Assumption-aware hypothesis tests that always report effect sizes.

Test selection:
- Two groups: Shapiro-Wilk per group (skipped for n > 5000, where the CLT applies) and Levene's test choose
  between Student's t, Welch's t and Mann-Whitney U.
- Three or more groups: one-way ANOVA, Welch ANOVA or Kruskal-Wallis, plus Holm-corrected pairwise tests.
- Two categorical variables: chi-square test of independence (Fisher's exact for sparse 2x2 tables).
- Two numeric variables: Pearson (both roughly normal) or Spearman correlation, with a Fisher-z CI.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

SHAPIRO_MAX_N = 5000
SMALL_N = 20


class AssumptionCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class GroupSummary(BaseModel):
    n: int
    mean: float
    median: float
    std: float


class PairwiseResult(BaseModel):
    a: str
    b: str
    p_value: float
    p_adjusted: float
    mean_diff: float


class StatTestResult(BaseModel):
    test: str
    statistic: float
    p_value: float
    effect_size: float | None = None
    effect_size_name: str | None = None
    effect_magnitude: str | None = None
    ci: tuple[float, float] | None = None
    ci_label: str | None = None
    n: int
    groups: dict[str, GroupSummary] | None = None
    posthoc: list[PairwiseResult] = []
    assumptions: list[AssumptionCheck] = []
    warnings: list[str] = []
    interpretation: str = ""


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _magnitude(name: str, value: float) -> str:
    v = abs(value)
    cuts = {
        "Cohen's d": (0.2, 0.5, 0.8),
        "rank-biserial correlation": (0.1, 0.3, 0.5),
        "Cramér's V": (0.1, 0.3, 0.5),
        "correlation": (0.1, 0.3, 0.5),
        "eta squared": (0.01, 0.06, 0.14),
        "epsilon squared": (0.01, 0.06, 0.14),
    }.get(name, (0.1, 0.3, 0.5))
    return "negligible" if v < cuts[0] else "small" if v < cuts[1] else "medium" if v < cuts[2] else "large"


def _normality(x: np.ndarray, label: str) -> AssumptionCheck:
    if len(x) < 3:
        return AssumptionCheck(name="normality", passed=False, detail=f"{label}: too few values to check")
    if len(x) > SHAPIRO_MAX_N:
        return AssumptionCheck(
            name="normality", passed=True, detail=f"{label}: n={len(x)} > {SHAPIRO_MAX_N}; the mean is ~normal (CLT)"
        )
    p = stats.shapiro(x).pvalue
    return AssumptionCheck(name="normality", passed=p >= 0.05, detail=f"{label}: Shapiro-Wilk {_fmt_p(p)}")


def _summaries(groups: dict[str, np.ndarray]) -> dict[str, GroupSummary]:
    return {
        k: GroupSummary(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                        std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        for k, v in groups.items()
    }


def _welch_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t = stats.t.ppf(0.5 + level / 2, df)
    d = a.mean() - b.mean()
    return float(d - t * se), float(d + t * se)


def compare_groups(df: pd.DataFrame, value: str, group: str, alpha: float = 0.05) -> StatTestResult:
    """Compare a numeric ``value`` across the levels of ``group``."""
    data = df[[value, group]].dropna()
    groups = {str(k): v[value].astype(float).to_numpy() for k, v in data.groupby(group, sort=True)}
    if len(groups) < 2:
        raise ValueError(f"'{group}' needs at least two groups with data; found {len(groups)}.")
    warns: list[str] = []
    small = [k for k, v in groups.items() if len(v) < SMALL_N]
    if small:
        warns.append(f"Small sample in group(s) {', '.join(small)} (n < {SMALL_N}); results are less reliable.")
    checks = [_normality(v, f"{group}={k}") for k, v in groups.items()]
    normal = all(c.passed for c in checks)
    lev_p = stats.levene(*groups.values()).pvalue
    checks.append(AssumptionCheck(name="equal variances", passed=lev_p >= 0.05, detail=f"Levene {_fmt_p(lev_p)}"))
    equal_var = lev_p >= 0.05
    summaries = _summaries(groups)
    n = int(sum(len(v) for v in groups.values()))

    if len(groups) == 2:
        (ka, a), (kb, b) = groups.items()
        if normal:
            res = stats.ttest_ind(a, b, equal_var=equal_var)
            name = "Student t-test" if equal_var else "Welch t-test"
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            eff, eff_name = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0, "Cohen's d"
            stat, p = float(res.statistic), float(res.pvalue)
        else:
            res = stats.mannwhitneyu(a, b, alternative="two-sided")
            name = "Mann-Whitney U"
            eff, eff_name = float(2 * res.statistic / (len(a) * len(b)) - 1), "rank-biserial correlation"
            stat, p = float(res.statistic), float(res.pvalue)
        result = StatTestResult(
            test=name, statistic=stat, p_value=p, effect_size=eff, effect_size_name=eff_name,
            effect_magnitude=_magnitude(eff_name, eff), ci=_welch_ci(a, b),
            ci_label=f"95% CI of mean difference ({ka} - {kb})", n=n, groups=summaries, assumptions=checks,
            warnings=warns,
        )
        sig = p < alpha
        result.interpretation = (
            f"The difference in {value} between {group}={ka} (mean {a.mean():.4g}) and {group}={kb} "
            f"(mean {b.mean():.4g}) is {'statistically significant' if sig else 'not statistically significant'} "
            f"({name}, {_fmt_p(p)}; {eff_name} = {eff:.3f}, {result.effect_magnitude} effect). "
            + ("" if sig else "There is no evidence of a difference at this sample size. ")
            + "This is an association in observational data, not proof of cause and effect."
        )
        return result

    values = list(groups.values())
    all_v = np.concatenate(values)
    if normal:
        if equal_var:
            res = stats.f_oneway(*values)
            name, stat, p = "One-way ANOVA", float(res.statistic), float(res.pvalue)
        else:
            res = anova_oneway(values, use_var="unequal")
            name, stat, p = "Welch ANOVA", float(res.statistic), float(res.pvalue)
        ss_between = sum(len(v) * (v.mean() - all_v.mean()) ** 2 for v in values)
        ss_total = ((all_v - all_v.mean()) ** 2).sum()
        eff, eff_name = float(ss_between / ss_total) if ss_total else 0.0, "eta squared"
    else:
        res = stats.kruskal(*values)
        name, stat, p = "Kruskal-Wallis H", float(res.statistic), float(res.pvalue)
        eff, eff_name = float(stat * (n + 1) / (n**2 - 1)), "epsilon squared"

    pairs = list(itertools.combinations(groups, 2))
    raw = []
    for ka, kb in pairs:
        a, b = groups[ka], groups[kb]
        pp = stats.ttest_ind(a, b, equal_var=False).pvalue if normal else stats.mannwhitneyu(a, b).pvalue
        raw.append(float(pp))
    adjusted = multipletests(raw, method="holm")[1] if raw else []
    posthoc = [
        PairwiseResult(a=ka, b=kb, p_value=pr, p_adjusted=float(pa), mean_diff=float(groups[ka].mean() - groups[kb].mean()))
        for (ka, kb), pr, pa in zip(pairs, raw, adjusted, strict=True)
    ]
    sig = p < alpha
    sig_pairs = [f"{x.a} vs {x.b}" for x in posthoc if x.p_adjusted < alpha]
    result = StatTestResult(
        test=name, statistic=stat, p_value=p, effect_size=eff, effect_size_name=eff_name,
        effect_magnitude=_magnitude(eff_name, eff), n=n, groups=summaries, posthoc=posthoc,
        assumptions=checks, warnings=warns,
    )
    result.interpretation = (
        f"{value} {'differs' if sig else 'does not differ'} significantly across {group} "
        f"({name}, {_fmt_p(p)}; {eff_name} = {eff:.3f}, {result.effect_magnitude} effect). "
        + (f"Holm-corrected pairwise differences: {', '.join(sig_pairs) or 'none'}. " if sig else "")
        + "This is an association in observational data, not proof of cause and effect."
    )
    return result


def _is_categorical(s: pd.Series) -> bool:
    return (
        pd.api.types.is_bool_dtype(s)
        or not pd.api.types.is_numeric_dtype(s)
        or s.nunique(dropna=True) <= 2
    )


def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return (float(r), float(r))
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def association_test(df: pd.DataFrame, x: str, y: str, alpha: float = 0.05) -> StatTestResult:
    """Test the association between two columns, choosing the test from their types."""
    data = df[[x, y]].dropna()
    cx, cy = _is_categorical(data[x]), _is_categorical(data[y])
    if cx and cy:
        table = pd.crosstab(data[x], data[y])
        chi2, p, dof, expected = stats.chi2_contingency(table)
        warns = []
        if table.shape == (2, 2) and (expected < 5).any():
            name = "Fisher's exact test"
            stat, p = stats.fisher_exact(table.to_numpy())
            stat = float(stat)
        else:
            name, stat = "Chi-square test of independence", float(chi2)
            if (expected < 5).mean() > 0.2:
                warns.append("More than 20% of expected cell counts are below 5; the chi-square approximation may be poor.")
        n = int(table.to_numpy().sum())
        k = min(table.shape) - 1
        v = float(np.sqrt(chi2 / (n * k))) if n and k else 0.0
        sig = p < alpha
        r = StatTestResult(
            test=name, statistic=stat, p_value=float(p), effect_size=v, effect_size_name="Cramér's V",
            effect_magnitude=_magnitude("Cramér's V", v), n=n, warnings=warns,
            assumptions=[AssumptionCheck(name="expected counts >= 5", passed=bool((expected >= 5).all()),
                                         detail=f"min expected count {expected.min():.1f}")],
        )
        r.interpretation = (
            f"The association between {x} and {y} is "
            f"{'statistically significant' if sig else 'not statistically significant'} "
            f"({name}, {_fmt_p(p)}; Cramér's V = {v:.3f}, {r.effect_magnitude}). "
            + ("" if sig else f"There is no evidence that {y} varies with {x}. ")
            + "This is an association, not proof of cause and effect."
        )
        return r
    if cx != cy:
        num, cat = (y, x) if cx else (x, y)
        return compare_groups(data, num, cat, alpha)

    a, b = data[x].astype(float).to_numpy(), data[y].astype(float).to_numpy()
    checks = [_normality(a, x), _normality(b, y)]
    if all(c.passed for c in checks):
        res = stats.pearsonr(a, b)
        name = "Pearson correlation"
    else:
        res = stats.spearmanr(a, b)
        name = "Spearman correlation"
    rho, p = float(res.statistic), float(res.pvalue)
    sig = p < alpha
    r = StatTestResult(
        test=name, statistic=rho, p_value=p, effect_size=rho, effect_size_name="correlation",
        effect_magnitude=_magnitude("correlation", rho), ci=_fisher_ci(rho, len(a)), ci_label="95% CI of correlation",
        n=len(a), assumptions=checks,
    )
    r.interpretation = (
        f"{x} and {y} have a {r.effect_magnitude} {'positive' if rho > 0 else 'negative'} correlation "
        f"({name} r = {rho:.3f}, {_fmt_p(p)}), which is "
        f"{'statistically significant' if sig else 'not statistically significant'}. "
        "Correlation does not establish cause and effect."
    )
    return r
