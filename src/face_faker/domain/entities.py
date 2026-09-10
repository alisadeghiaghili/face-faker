"""Immutable domain entities and value objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from face_faker.domain.enums import GenderLabel, MetadataSchemaVersion


@dataclass(frozen=True)
class FrontalMetrics:
    """Head-pose metrics produced by a :class:`FrontalFilter`.

    Angle semantics (degrees):

    - ``yaw``: left/right head turn
    - ``pitch``: up/down gaze (nod)
    - ``roll``: in-plane tilt (ear-to-shoulder)

    Attributes:
        method: Algorithm identifier (``"solvepnp"`` for the v3 filter).
        yaw: Yaw angle in degrees (positive = subject's left / image right).
        pitch: Pitch angle in degrees (positive = typically looking up).
        roll: Roll/tilt angle in degrees (positive = clockwise in image).
        yaw_threshold: Absolute yaw acceptance threshold in degrees.
        pitch_threshold: Absolute pitch acceptance threshold in degrees.
        roll_threshold: Absolute roll/tilt acceptance threshold in degrees.
    """

    method: str
    yaw: float
    pitch: float
    roll: float
    yaw_threshold: float
    pitch_threshold: float
    roll_threshold: float = 15.0

    def is_frontal(self) -> bool:
        """Return ``True`` when yaw, pitch, and roll are within thresholds.

        Returns:
            Whether the pose is considered frontal under stored thresholds.

        Example:
            >>> m = FrontalMetrics("solvepnp", 5.0, -3.0, 1.0, 15.0, 15.0, 15.0)
            >>> m.is_frontal()
            True
            >>> FrontalMetrics("solvepnp", 0.0, 0.0, 30.0, 15.0, 15.0, 15.0).is_frontal()
            False
        """
        return (
            abs(self.yaw) <= self.yaw_threshold
            and abs(self.pitch) <= self.pitch_threshold
            and abs(self.roll) <= self.roll_threshold
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize metrics for the public metadata contract.

        Returns:
            A JSON-ready mapping with rounded angles and thresholds.

        Example:
            >>> m = FrontalMetrics("solvepnp", 5.123, -2.0, 0.5, 15.0, 12.0, 10.0)
            >>> m.to_metadata()["yaw"]
            5.12
            >>> m.to_metadata()["roll_threshold"]
            10.0
        """
        return {
            "method": self.method,
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "yaw_threshold": self.yaw_threshold,
            "pitch_threshold": self.pitch_threshold,
            "roll_threshold": self.roll_threshold,
        }


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable settings for a generation run.

    Attributes:
        output_dir: Directory that will receive images and metadata.
        count: Number of successfully saved faces to produce.
        save_metadata: Write aggregate/per-face metadata and stats files.
        remove_bg: Apply background removal (transparent output).
        frontal_only: Reject images that fail the frontal filter.
        yaw_threshold: Absolute yaw (turn) limit in degrees when filtering.
        pitch_threshold: Absolute pitch (up/down gaze) limit in degrees when filtering.
        roll_threshold: Absolute roll (in-plane tilt) limit in degrees when filtering.
        classify_gender: Run gender classification when available.
        require_gender: Fail the run if gender classification is unavailable
            while ``classify_gender`` is enabled.
        grayscale: Convert final images to grayscale (LA when alpha present).
        models_dir: Explicit models directory; overrides env and defaults.
        max_attempts_factor: ``max_attempts = max(count * factor, count)``.
        request_sleep_s: Inclusive min/max sleep between source fetches.
        strict_completion: Raise :class:`GenerationIncompleteError` when the
            requested count is not reached.
    """

    output_dir: Path | str = "id_faces"
    count: int = 100
    save_metadata: bool = True
    remove_bg: bool = False
    frontal_only: bool = False
    yaw_threshold: float = 15.0
    pitch_threshold: float = 15.0
    roll_threshold: float = 15.0
    classify_gender: bool = True
    require_gender: bool = False
    grayscale: bool = True
    models_dir: Path | str | None = None
    max_attempts_factor: int = 3
    request_sleep_s: tuple[float, float] = (0.5, 1.5)
    strict_completion: bool = False

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.max_attempts_factor < 1:
            raise ValueError("max_attempts_factor must be >= 1")
        if self.yaw_threshold < 0 or self.pitch_threshold < 0 or self.roll_threshold < 0:
            raise ValueError("pose thresholds must be >= 0")
        low, high = self.request_sleep_s
        if low < 0 or high < low:
            raise ValueError("request_sleep_s must satisfy 0 <= low <= high")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.models_dir is not None:
            object.__setattr__(self, "models_dir", Path(self.models_dir))

    @property
    def max_attempts(self) -> int:
        """Upper bound on source fetch attempts for this run."""
        return max(self.count * self.max_attempts_factor, self.count)


@dataclass(frozen=True)
class ImageRecord:
    """Metadata contract for a single generated face.

    Attributes:
        filename: File name relative to the output directory.
        index: 1-based index among successfully saved faces.
        gender: Normalized gender label.
        background_removed: Whether background removal ran.
        frontal_filtered: Whether the run required a frontal pass.
        frontal: Pose metrics when frontal filtering ran and succeeded.
        schema_version: Metadata schema version string.
    """

    filename: str
    index: int
    gender: GenderLabel
    background_removed: bool
    frontal_filtered: bool
    frontal: FrontalMetrics | None = None
    schema_version: MetadataSchemaVersion = MetadataSchemaVersion.V1

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-ready dict matching the public schema.

        Returns:
            Mapping with stable keys for downstream consumers.

        Example:
            >>> rec = ImageRecord("face_0001.png", 1, GenderLabel.FEMALE, False, False)
            >>> rec.to_metadata()["gender"]
            'female'
        """
        payload: dict[str, Any] = {
            "schema_version": self.schema_version.value,
            "filename": self.filename,
            "index": self.index,
            "gender": self.gender.value,
            "background_removed": self.background_removed,
            "frontal_filtered": self.frontal_filtered,
        }
        if self.frontal is not None:
            payload["frontal"] = self.frontal.to_metadata()
        return payload


@dataclass(frozen=True)
class GenerationStats:
    """Aggregate counters for a generation run.

    Attributes:
        requested: Target image count.
        produced: Images successfully written.
        attempts: Source fetch attempts made.
        failed_fetches: Fetches that returned no usable image.
        filtered_out: Images rejected by the frontal filter.
        gender_male: Count of male-labeled outputs.
        gender_female: Count of female-labeled outputs.
        gender_unknown: Count of unknown-labeled outputs.
        elapsed_seconds: Wall-clock duration of the run.
    """

    requested: int
    produced: int
    attempts: int
    failed_fetches: int
    filtered_out: int
    gender_male: int
    gender_female: int
    gender_unknown: int
    elapsed_seconds: float

    def to_metadata(self) -> dict[str, Any]:
        """Serialize stats for ``stats.json``.

        Returns:
            JSON-ready counters including derived incomplete flag.
        """
        return {
            "requested": self.requested,
            "produced": self.produced,
            "attempts": self.attempts,
            "failed_fetches": self.failed_fetches,
            "filtered_out": self.filtered_out,
            "gender": {
                "male": self.gender_male,
                "female": self.gender_female,
                "unknown": self.gender_unknown,
            },
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "complete": self.produced >= self.requested,
        }


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of :func:`face_faker.generate_faces`.

    Attributes:
        records: Per-image metadata records in save order.
        stats: Aggregate run statistics.
        output_dir: Directory where artifacts were written.
        paths: Named paths of files written when metadata is enabled.
    """

    records: tuple[ImageRecord, ...]
    stats: GenerationStats
    output_dir: Path
    paths: Mapping[str, Path] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def to_metadata_list(self) -> list[dict[str, Any]]:
        """Return all records as metadata dicts.

        Returns:
            List of schema-stable mappings.

        Example:
            >>> from face_faker.domain.entities import GenerationStats
            >>> stats = GenerationStats(1, 0, 0, 0, 0, 0, 0, 0, 0.0)
            >>> GenerationResult((), stats, Path("out")).to_metadata_list()
            []
        """
        return [record.to_metadata() for record in self.records]


def records_to_csv_rows(records: Sequence[ImageRecord]) -> list[dict[str, Any]]:
    """Convert records into CSV-friendly row dicts.

    Args:
        records: Image records from a generation run.

    Returns:
        Row mappings suitable for :class:`csv.DictWriter`.

    Example:
        >>> rec = ImageRecord("a.png", 1, GenderLabel.MALE, True, False)
        >>> records_to_csv_rows([rec])[0]["gender"]
        'male'
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "filename": record.filename,
            "gender": record.gender.value,
            "background_removed": record.background_removed,
            "frontal_filtered": record.frontal_filtered,
        }
        if record.frontal is not None:
            row["yaw"] = record.frontal.yaw
            row["pitch"] = record.frontal.pitch
            row["roll"] = record.frontal.roll
        rows.append(row)
    return rows


def entity_asdict(obj: Any) -> dict[str, Any]:
    """Shallow ``asdict`` helper kept for debugging adapters.

    Args:
        obj: A dataclass instance.

    Returns:
        A plain dict representation.
    """
    return asdict(obj)
