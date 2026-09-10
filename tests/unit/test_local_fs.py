"""Tests for LocalFaceStore file artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from face_faker.domain.entities import GenerationStats, ImageRecord
from face_faker.domain.enums import GenderLabel
from face_faker.infrastructure.storage.local_fs import CSV_FIELDNAMES, LocalFaceStore


def test_save_image_writes_png(tmp_path: Path) -> None:
    store = LocalFaceStore(tmp_path / "out")
    path = store.save_image(Image.new("RGB", (8, 8), "red"), "face_0001.png")
    assert path.exists()
    assert path.suffix == ".png"


def test_save_metadata_json_stats_csv(tmp_path: Path) -> None:
    store = LocalFaceStore(tmp_path)
    records = [
        ImageRecord("face_0001.png", 1, GenderLabel.FEMALE, False, False),
        ImageRecord(
            "face_0002.png",
            2,
            GenderLabel.MALE,
            True,
            False,
            source_ref="/data/inbox/a.png",
        ),
    ]
    stats = GenerationStats(2, 2, 2, 0, 0, 1, 1, 0, 0.5).to_metadata()
    paths = store.save_metadata(records, stats)

    metadata = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
    assert metadata[0]["schema_version"] == "1"
    assert metadata[0]["gender"] == "female"
    assert metadata[1]["source_ref"] == "/data/inbox/a.png"

    stats_doc = json.loads(Path(paths["stats"]).read_text(encoding="utf-8"))
    assert stats_doc["produced"] == 2
    assert stats_doc["complete"] is True

    with Path(paths["csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["gender"] == "female"
    assert list(rows[0].keys())[0] == "filename"
    assert "gender" in CSV_FIELDNAMES
    assert "source_ref" in CSV_FIELDNAMES
    assert rows[1]["source_ref"] == "/data/inbox/a.png"
