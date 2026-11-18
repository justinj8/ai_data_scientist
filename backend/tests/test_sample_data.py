from adsci.samples import generate

SPIKES = {"2024-03-14", "2024-07-04", "2024-09-21", "2025-02-10", "2025-06-18"}


def test_planted_truths(tmp_path):
    gt = generate(tmp_path, seed=42)
    assert 0.06 <= gt["yoy_revenue_growth_2025"] <= 0.12
    assert 1.08 <= gt["west_aov_ratio"] <= 1.16
    assert gt["promo_aov_uplift"] > 0.05
    assert set(gt["anomaly_spike_dates"]) == SPIKES
    assert set(gt["outage_dates"]) == {"2024-11-05", "2025-08-12"}
    assert 0.15 <= gt["churn_rate"] <= 0.25
    assert gt["churn_top_drivers"][0] in {"last_login_days", "support_tickets"}
    assert gt["region_churn_p_value"] > 0.05
    assert gt["top_category_2025"] in {"Electronics", "Home", "Apparel", "Beauty", "Sports"}
    assert 0.25 < gt["q4_share_2024"] < 0.40


def test_outputs_exist_and_deterministic(tmp_path):
    a = generate(tmp_path / "a", seed=42)
    b = generate(tmp_path / "b", seed=42)
    assert a == b
    for name in ["orders.csv", "customers.csv", "retail_workbook.xlsx", "retail.db", "ground_truth.json"]:
        assert (tmp_path / "a" / name).exists()


def test_workbook_sheets(tmp_path):
    import openpyxl

    generate(tmp_path, seed=42)
    wb = openpyxl.load_workbook(tmp_path / "retail_workbook.xlsx", read_only=True)
    assert wb.sheetnames == ["Orders_2025", "Targets", "Notes", "Scratch"]
