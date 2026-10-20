import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, indent=4, allow_nan=False)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_json_float(value):
    import math
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return number


def read_json(path):
    """Read UTF-8 report JSON, rejecting nonfinite constants and float overflow."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
