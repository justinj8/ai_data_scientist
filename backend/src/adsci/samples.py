"""Seeded synthetic retail datasets with *planted* ground truth.

The generator plants known patterns (seasonality, a promotion uplift, a regional
price difference, anomalous days, churn drivers and a null effect) so that tests
and the evaluation harness can check the agent's answers against real answers.
Every ground-truth value is computed from the generated data, except the planted
anomaly dates, which are the truth by construction.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["Electronics", "Home", "Apparel", "Beauty", "Sports"]
CATEGORY_P = [0.20, 0.25, 0.25, 0.15, 0.15]
CATEGORY_PRICE = {"Electronics": 120.0, "Home": 60.0, "Apparel": 40.0, "Beauty": 25.0, "Sports": 55.0}
CHANNELS = ["web", "app", "store"]
CHANNEL_P = [0.5, 0.3, 0.2]
PLANS = ["basic", "pro", "enterprise"]
PLAN_P = [0.5, 0.35, 0.15]

START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")
SPIKE_DATES = ["2024-03-14", "2024-07-04", "2024-09-21", "2025-02-10", "2025-06-18"]
OUTAGE_DATES = ["2024-11-05", "2025-08-12"]
N_CUSTOMERS = 5000
BASE_DAILY_ORDERS = 50.0
YOY_GROWTH = 0.08
WEST_PRICE_FACTOR = 1.12

CHURN_FEATURES = ["support_tickets", "last_login_days", "tenure_months", "nps", "plan_basic"]


def _customers(rng: np.random.Generator) -> pd.DataFrame:
    n = N_CUSTOMERS
    signup_offsets = rng.integers(0, (pd.Timestamp("2025-06-30") - pd.Timestamp("2021-01-01")).days, n)
    signup = pd.Timestamp("2021-01-01") + pd.to_timedelta(signup_offsets, unit="D")
    tenure = np.asarray((END - signup).days // 30, dtype=int)
    plan = rng.choice(PLANS, n, p=PLAN_P)
    spend_base = np.select([plan == "basic", plan == "pro"], [20.0, 60.0], 250.0)
    monthly_spend = np.round(spend_base * rng.lognormal(0, 0.3, n), 2)
    support_tickets = rng.poisson(1.5, n)
    last_login_days = np.round(rng.exponential(20, n)).astype(int)
    nps = np.clip(np.round(rng.normal(7, 2.5, n)), 0, 10).astype(int)
    region = rng.choice(REGIONS, n)  # region deliberately has NO effect on churn (null control)

    linear = (
        0.45 * support_tickets
        + 0.06 * last_login_days
        - 0.03 * tenure
        - 0.15 * nps
        + 0.6 * (plan == "basic")
    )
    # Calibrate the intercept so the expected churn rate is 20%.
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + linear)))).mean() > 0.20:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-((lo + hi) / 2 + linear)))
    churned = rng.random(n) < p

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup.strftime("%Y-%m-%d"),
            "region": region,
            "plan": plan,
            "tenure_months": tenure,
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            "last_login_days": last_login_days,
            "nps": nps,
            "churned": churned,
        }
    )


def _orders(rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    years = (days - START).days / 365.25
    expected = BASE_DAILY_ORDERS * (1 + YOY_GROWTH) ** years
    expected = expected * np.where(days.dayofweek >= 5, 1.2, 1.0)
    expected = expected * np.where(days.quarter == 4, 1.35, 1.0)
    counts = rng.poisson(expected)
    day_str = days.strftime("%Y-%m-%d")
    counts = np.where(np.isin(day_str, SPIKE_DATES), counts * 4, counts)
    counts = np.where(np.isin(day_str, OUTAGE_DATES), 1, counts)

    order_dates = np.repeat(days.values, counts)
    n = len(order_dates)
    region = rng.choice(REGIONS, n)
    category = rng.choice(CATEGORIES, n, p=CATEGORY_P)
    channel = rng.choice(CHANNELS, n, p=CHANNEL_P)
    promo = rng.random(n) < 0.2
    units = 1 + rng.poisson(np.where(promo, 2.3, 1.6))
    base_price = np.array([CATEGORY_PRICE[c] for c in category])
    unit_price = base_price * rng.lognormal(0, 0.4, n) * np.where(region == "West", WEST_PRICE_FACTOR, 1.0)
    unit_price = np.round(unit_price, 2)
    discount = np.where(promo, np.round(rng.uniform(0.10, 0.20, n), 2), 0.0)
    revenue = np.round(units * unit_price * (1 - discount), 2)

    return pd.DataFrame(
        {
            "order_id": np.arange(1, n + 1),
            "order_date": pd.DatetimeIndex(order_dates).strftime("%Y-%m-%d"),
            "customer_id": rng.integers(1, N_CUSTOMERS + 1, n),
            "region": region,
            "category": category,
            "channel": channel,
            "units": units,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "promo": promo,
        }
    )


def _ground_truth(orders: pd.DataFrame, customers: pd.DataFrame) -> dict:
    o = orders.assign(order_date=pd.to_datetime(orders["order_date"]))
    year = o["order_date"].dt.year
    rev_by_year = o.groupby(year)["revenue"].sum()
    o24 = o[year == 2024]
    q4_2024 = o24.loc[o24["order_date"].dt.quarter == 4, "revenue"].sum()
    west = o["region"] == "West"
    top_cat = o[year == 2025].groupby("category")["revenue"].sum().idxmax()

    c = customers.assign(plan_basic=(customers["plan"] == "basic").astype(float))
    x = c[CHURN_FEATURES].astype(float)
    xz = (x - x.mean()) / x.std()
    import statsmodels.api as sm

    logit = sm.Logit(c["churned"].astype(float), sm.add_constant(xz)).fit(disp=0)
    coefs = logit.params.drop("const").abs().sort_values(ascending=False)
    _, p_region, _, _ = stats.chi2_contingency(pd.crosstab(c["region"], c["churned"]))

    return {
        "total_revenue_2025": round(float(rev_by_year[2025]), 2),
        "total_revenue_2024": round(float(rev_by_year[2024]), 2),
        "yoy_revenue_growth_2025": round(float(rev_by_year[2025] / rev_by_year[2024] - 1), 6),
        "promo_aov_uplift": round(float(o.loc[o.promo, "revenue"].mean() / o.loc[~o.promo, "revenue"].mean() - 1), 6),
        "west_aov_ratio": round(float(o.loc[west, "revenue"].mean() / o.loc[~west, "revenue"].mean()), 6),
        "top_category_2025": str(top_cat),
        "q4_share_2024": round(float(q4_2024 / rev_by_year[2024]), 6),
        "anomaly_spike_dates": list(SPIKE_DATES),
        "outage_dates": list(OUTAGE_DATES),
        "n_orders": int(len(orders)),
        "churn_rate": round(float(customers["churned"].mean()), 6),
        "churn_top_drivers": [str(k) for k in coefs.index],
        "region_churn_p_value": float(p_region),
    }


def _write_workbook(path: Path, orders: pd.DataFrame, rng: np.random.Generator) -> None:
    o25 = orders[orders["order_date"] >= "2025-01-01"]
    monthly = (
        o25.assign(month=o25["order_date"].str[:7]).groupby(["month", "region"])["revenue"].sum().reset_index()
    )
    monthly["revenue_target"] = np.round(monthly["revenue"] * rng.uniform(0.9, 1.1, len(monthly)), -2)
    targets = monthly[["month", "region", "revenue_target"]]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        o25.to_excel(xw, sheet_name="Orders_2025", index=False)
        targets.to_excel(xw, sheet_name="Targets", index=False)
        notes = pd.DataFrame(
            [
                ["Q1 promo", "Spring campaign", "Marketing", "2025-03-01"],
                ["Q4 stock", "Holiday inventory", "Ops", "2025-10-01"],
                ["West pricing", "Review premium pricing", "Finance", "2025-06-15"],
            ]
        )
        notes.to_excel(xw, sheet_name="Notes", index=False, header=False, startrow=3)
        pd.DataFrame().to_excel(xw, sheet_name="Scratch", index=False)
        ws = xw.sheets["Notes"]
        ws["A1"] = "Retail planning notes"
        ws["A2"] = "Prepared by the analytics team"
        for col, name in zip("ABCD", ["Item", "Item", "Owner", "Due"], strict=True):
            ws[f"{col}3"] = name


def generate(out_dir: Path, seed: int = 42) -> dict:
    """Generate all sample files into ``out_dir`` and return the ground-truth dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    customers = _customers(rng)
    orders = _orders(rng)
    truth = _ground_truth(orders, customers)

    orders.to_csv(out_dir / "orders.csv", index=False)
    customers.to_csv(out_dir / "customers.csv", index=False)
    _write_workbook(out_dir / "retail_workbook.xlsx", orders, rng)

    db = out_dir / "retail.db"
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        orders.to_sql("orders", con, index=False)
        customers.to_sql("customers", con, index=False)

    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def ensure_samples(settings) -> Path:
    """Return a directory with the sample files, generating them under ``data_dir`` if needed."""
    candidates = [settings.sample_dir] if settings.sample_dir else []
    candidates.append(Path(__file__).resolve().parents[3] / "sample_data" / "out")
    for d in candidates:
        if d and (Path(d) / "retail.db").exists() and (Path(d) / "orders.csv").exists():
            return Path(d)
    out = Path(settings.data_dir) / "samples"
    if not (out / "retail.db").exists():
        generate(out)
    return out
