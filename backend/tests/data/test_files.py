from adsci.data.files import FileDataSource, load_csv, load_xlsx

from .conftest import SAMPLE


def test_semicolon_latin1_thousands_mixed_dates(messy_dir):
    df, warnings = load_csv(messy_dir / "messy_semicolon_latin1.csv")
    assert list(df.columns) == ["Region", "Revenue", "Order Date"]
    assert df["Region"].iloc[0] == "Zürich"
    assert df["Revenue"].dtype.kind == "f"
    assert df["Revenue"].iloc[0] == 1234.5
    assert df["Revenue"].iloc[2] == 12000.25
    assert str(df["Order Date"].dtype).startswith("datetime64")


def test_utf8_bom(messy_dir):
    df, _ = load_csv(messy_dir / "bom.csv")
    assert list(df.columns) == ["id", "value"]


def test_xlsx_title_rows_empty_sheet_dupes(messy_dir):
    sheets = load_xlsx(messy_dir / "messy.xlsx")
    assert set(sheets) == {"Data", "Notes"}  # empty "Scratch" skipped
    df, _ = sheets["Notes"]
    assert list(df.columns)[:2] == ["Item", "Item_2"]  # header found at row 3, dupes suffixed
    assert len(df) == 2


def test_file_source_xlsx_tables_and_warnings(tmp_path, messy_dir):
    src = FileDataSource("files", tmp_path / "s.duckdb")
    reports = src.add_file(messy_dir / "messy.xlsx")
    assert {r.table for r in reports} == {"messy__data", "messy__notes"}
    assert any("Scratch" in w for r in reports for w in r.warnings)
    notes = next(t for t in src.schema() if t.name == "messy__notes")
    assert [c.name for c in notes.columns][:2] == ["item", "item_2"]
    assert notes.columns[0].original_name == "Item"


def test_file_source_roundtrip_sample(tmp_path):
    src = FileDataSource("files", tmp_path / "s.duckdb")
    reports = src.add_file(SAMPLE / "retail_workbook.xlsx")
    assert {r.table for r in reports} >= {
        "retail_workbook__orders_2025",
        "retail_workbook__targets",
        "retail_workbook__notes",
    }
    df = src.query("select count(*) as n from retail_workbook__targets")
    assert df["n"][0] > 0
    t = next(t for t in src.schema() if t.name == "retail_workbook__targets")
    assert all(c.original_name for c in t.columns)
    assert t.row_count == df["n"][0]


def test_csv_table_name_and_query_limit(tmp_path):
    src = FileDataSource("files", tmp_path / "s.duckdb")
    [report] = src.add_file(SAMPLE / "orders.csv")
    assert report.table == "orders" and report.rows > 70000
    assert len(src.query("select * from orders", limit=25)) == 25
    assert src.dialect == "duckdb" and src.kind == "file"


def test_reloading_same_file_replaces_table(tmp_path, messy_dir):
    src = FileDataSource("files", tmp_path / "s.duckdb")
    src.add_file(messy_dir / "bom.csv")
    src.add_file(messy_dir / "bom.csv")
    assert [t.name for t in src.schema()] == ["bom"]


def test_file_source_query_is_guarded(tmp_path, messy_dir):
    import pytest

    from adsci.data.sql_guard import UnsafeSQLError

    src = FileDataSource("files", tmp_path / "s.duckdb", row_limit=1)
    src.add_file(messy_dir / "bom.csv")
    with pytest.raises(UnsafeSQLError):
        src.query("drop table bom")
    assert len(src.query("select * from bom")) == 1  # default row limit applied


def test_query_limit_can_exceed_default_up_to_max(tmp_path):
    src = FileDataSource("files", tmp_path / "s.duckdb", row_limit=10, max_limit=100)
    src.add_file(SAMPLE / "customers.csv")
    assert len(src.query("select * from customers")) == 10
    assert len(src.query("select * from customers", limit=50)) == 50
    assert len(src.query("select * from customers", limit=10_000)) == 100


def test_duckdb_cannot_read_host_files_even_unguarded(tmp_path):
    import duckdb
    import pytest

    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n")
    src = FileDataSource("files", tmp_path / "s.duckdb")
    with pytest.raises(duckdb.Error):
        src._con.execute(f"select * from '{secret}'").fetchall()


def _csv(tmp_path, text):
    p = tmp_path / "d.csv"
    p.write_text(text)
    return p


def test_day_first_dates_parsed_consistently(tmp_path):
    df, _ = load_csv(_csv(tmp_path, "d,v\n03/04/2024,1\n13/04/2024,2\n25/12/2024,3\n"))
    assert [str(x.date()) for x in df["d"]] == ["2024-04-03", "2024-04-13", "2024-12-25"]


def test_ambiguous_slash_dates_default_to_month_first_with_warning(tmp_path):
    df, warns = load_csv(_csv(tmp_path, "d,v\n03/04/2024,1\n05/06/2024,2\n"))
    assert str(df["d"].iloc[0].date()) == "2024-03-04"
    assert any("month/day" in w for w in warns)


def test_fractions_and_codes_are_not_dates(tmp_path):
    df, _ = load_csv(_csv(tmp_path, "ratio,code\n3/4,1-2\n1/2,3-4\n5/8,2-9\n"))
    assert df["ratio"].tolist() == ["3/4", "1/2", "5/8"] and df["code"].tolist() == ["1-2", "3-4", "2-9"]


def test_unparseable_dates_are_reported(tmp_path):
    rows = "\n".join(f"2024-01-{d:02d},{d}" for d in range(1, 21)) + "\nnot a date,21\n"
    df, warns = load_csv(_csv(tmp_path, "d,v\n" + rows))
    assert str(df["d"].dtype).startswith("datetime64") and df["d"].isna().sum() == 1
    assert any("1 value" in w and "d" in w for w in warns)


def test_xlsx_title_above_numeric_year_headers(tmp_path):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Rev"
    ws["A1"] = "Revenue by year"
    ws.append(["region", 2022, 2023, 2024])
    ws.append(["West", 10, 11, 12])
    ws.append(["East", 9, 10, 11])
    wb.save(tmp_path / "rev.xlsx")
    df, warns = load_xlsx(tmp_path / "rev.xlsx")["Rev"]
    assert list(df.columns) == ["region", "2022", "2023", "2024"] and len(df) == 2
    assert any("title row" in w for w in warns)
