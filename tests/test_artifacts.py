import json

import pytest

from src.artifacts import write_json_atomic


def test_atomic_json_write_creates_parent_and_serializes_data(tmp_path):
    output = tmp_path / "nested" / "report.json"

    write_json_atomic(output, {"status": "ready"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "ready"}
    assert list(output.parent.iterdir()) == [output]


def test_failed_json_serialization_preserves_existing_report_and_cleans_temp(
    tmp_path,
):
    output = tmp_path / "report.json"
    output.write_text('{"status": "previous"}', encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_atomic(output, {"invalid": object()})

    assert output.read_text(encoding="utf-8") == '{"status": "previous"}'
    assert list(tmp_path.iterdir()) == [output]
