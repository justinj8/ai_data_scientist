"""Regenerate the sample datasets: `uv run --project backend python sample_data/generate.py`."""

from pathlib import Path

from adsci.samples import generate

if __name__ == "__main__":
    out = Path(__file__).parent / "out"
    truth = generate(out)
    print(f"Wrote sample data to {out} ({truth['n_orders']} orders)")
