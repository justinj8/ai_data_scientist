from pathlib import Path

import openpyxl
import pytest

REPO = Path(__file__).resolve().parents[3]
SAMPLE = REPO / "sample_data" / "out"


@pytest.fixture
def messy_dir(tmp_path):
    d = tmp_path / "messy"
    d.mkdir()
    # Semicolon-delimited, latin-1 encoded, thousands separators, mixed date formats.
    (d / "messy_semicolon_latin1.csv").write_bytes(
        "Region;Revenue;Order Date\n"
        "Zürich;1,234.50;2024-01-05\n"
        "Genève;987.00;2024-02-07\n"
        "Köln;12,000.25;2024-03-09\n"
        "Malmö;5.5;2024-04-11\n".encode("latin-1")
    )
    # UTF-8 with BOM, comma-delimited.
    (d / "bom.csv").write_bytes("﻿id,value\n1,10\n2,20\n".encode())

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["a", "b"])
    ws.append([1, 2])
    notes = wb.create_sheet("Notes")
    notes["A1"] = "Planning notes"
    notes["A2"] = "Prepared by analytics"
    for col, name in zip("ABCD", ["Item", "Item", "Owner", "Due"], strict=True):
        notes[f"{col}3"] = name
    notes.append(["x", "y", "Ops", "2025-01-01"])
    notes.append(["z", "w", "Fin", "2025-02-01"])
    wb.create_sheet("Scratch")
    wb.save(d / "messy.xlsx")
    return d
