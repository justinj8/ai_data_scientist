"""Plotly >= 6 serializes numpy arrays as base64 typed arrays ({"dtype", "bdata"}); decode them to lists."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]))
            shape = obj.get("shape")
            if shape:
                dims = [int(s) for s in str(shape).split(",")] if isinstance(shape, str) else list(shape)
                arr = arr.reshape(dims)
            return arr.tolist()
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def figure_to_plain(fig: Any) -> dict:
    """Return a JSON-safe figure dict whose arrays are plain lists."""
    if hasattr(fig, "to_json"):
        fig = json.loads(fig.to_json())
    return _decode(fig)
