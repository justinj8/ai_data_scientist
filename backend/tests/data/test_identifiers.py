import pytest

from adsci.data.identifiers import dedupe, normalize_identifier


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Order Date", "order_date"),
        ("order", "order_"),
        ("2024 Sales", "_2024_sales"),
        ("Café №", "cafe_no"),
        ("", "col"),
        ("  Revenue ($) ", "revenue"),
        ("camelCaseName", "camel_case_name"),
    ],
)
def test_normalize_identifier(raw, expected):
    assert normalize_identifier(raw) == expected


def test_dedupe():
    assert dedupe(["a", "a", "b", "a"]) == ["a", "a_2", "b", "a_3"]
