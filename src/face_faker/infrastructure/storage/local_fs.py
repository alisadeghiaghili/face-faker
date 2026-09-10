"""Write images and metadata to a local directory."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from face_faker.domain.entities import ImageRecord, records_to_csv_rows
from face_faker.logging_config import get_logger

logger = get_logger("storage.local_fs")

CSV_FIELDNAMES = (
    "filename",
    "gender",
    "background_removed",
    "frontal_filtered",
    "yaw",
    "pitch",
    "roll",
)


class LocalFaceStore:
    """Persist faces and metadata under a single output directory.

    Args:
        output_dir: Root directory for artifacts.

    Example:
        >>> store = LocalFaceStore("out")
        >>> store.output_dir.name
        'out'
    """

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, image: Any, filename: str) -> Path:
        """Save an image under the output directory.

        Args:
            image: PIL image.
            filename: File name (PNG recommended).

        Returns:
            Absolute path of the written file.
        """
        path = self.output_dir / filename
        image.save(path)
        logger.debug("Wrote image %s", path)
        return path.resolve()

    def save_metadata(
        self,
        records: Sequence[ImageRecord],
        stats: Mapping[str, Any],
    ) -> dict[str, Path]:
        """Write metadata.json, stats.json, and gender CSV.

        CSV is always written alongside the JSON artifacts when this method
        is called; the application layer only calls it when
        ``save_metadata`` is enabled.

        Args:
            records: Per-image records.
            stats: Aggregate statistics mapping.

        Returns:
            Named paths: ``metadata``, ``stats``, ``csv``.
        """
        metadata_path = self.output_dir / "metadata.json"
        stats_path = self.output_dir / "stats.json"
        csv_path = self.output_dir / "faces.csv"

        payload = [record.to_metadata() for record in records]
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stats_path.write_text(
            json.dumps(dict(stats), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in records_to_csv_rows(records):
                writer.writerow(row)

        logger.info("Wrote metadata to %s", self.output_dir)
        return {
            "metadata": metadata_path.resolve(),
            "stats": stats_path.resolve(),
            "csv": csv_path.resolve(),
        }
