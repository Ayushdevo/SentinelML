import pandas as pd
import pytest

from src.artifacts import write_csv_atomic


def test_atomic_csv_writes_rows_and_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "analysis.csv"
    frame = pd.DataFrame({"value": [1.5, 2.0], "label": ["one", "two"]})
    write_csv_atomic(path, frame)
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)
    assert list(path.parent.iterdir()) == [path]


def test_atomic_csv_preserves_previous_report_on_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "analysis.csv"
    path.write_text("previous report\n", encoding="utf-8")
    frame = pd.DataFrame({"value": [1]})

    def fail(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(frame, "to_csv", fail)
    with pytest.raises(OSError, match="disk error"):
        write_csv_atomic(path, frame)
    assert path.read_text(encoding="utf-8") == "previous report\n"
    assert list(tmp_path.iterdir()) == [path]
