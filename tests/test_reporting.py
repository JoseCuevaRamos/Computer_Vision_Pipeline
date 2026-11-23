from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.reporting import write_csv, write_json, write_text


def test_csv_and_json_are_persisted(tmp_path: Path) -> None:
    csv_path = tmp_path / "results" / "values.csv"
    json_path = tmp_path / "results" / "values.json"
    text_path = tmp_path / "results" / "summary.md"
    write_csv(csv_path, pd.DataFrame({"value": [1, 2]}))
    write_json(json_path, {"value": 3})
    write_text(text_path, "verified")
    assert pd.read_csv(csv_path)["value"].tolist() == [1, 2]
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 3}
    assert text_path.read_text(encoding="utf-8") == "verified"
