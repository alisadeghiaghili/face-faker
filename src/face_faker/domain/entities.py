"""Immutable domain entities and value objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from face_faker.domain.enums import GenderLabel, MetadataSchemaVersion


def _require_non_negative(name: str, value: float) -> None:
    """Raise ``ValueError`` when ``value`` is negative.

    Args:
        name: Field name used in the error message.
        value: Numeric value to validate.

    Raises:
        ValueError: If ``value`` is less than zero.
    """
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class PoseLimits:
    """Per-direction head-pose limits in degrees.

    Sign convention matches solvePnP output:

    - ``yaw > 0``: subject turns toward their left (image right)
    - ``yaw < 0``: subject turns toward their right
    - ``pitch > 0``: looking up
    - ``pitch < 0``: looking down
    - ``roll``: in-plane tilt (absolute value compared)

    Attributes:
        yaw_left: Max positive yaw (turn toward subject's left).
        yaw_right: Max magnitude of negative yaw (turn toward subject's right).
        pitch_up: Max positive pitch (looking up).
        pitch_down: Max magnitude of negative pitch (looking down).
        roll: Max absolute roll (head tilt).
    """

    yaw_left: float = 15.0
    yaw_right: float = 15.0
    pitch_up: float = 15.0
    pitch_down: float = 15.0
    roll: float = 15.0

    def __post_init__(self) -> None:
        for name in ("yaw_left", "yaw_right", "pitch_up", "pitch_down", "roll"):
            _require_non_negative(name, getattr(self, name))

    @classmethod
    def symmetric(cls, yaw: float, pitch: float, roll: float) -> PoseLimits:
        """Build limits where left/right and up/down share one value.

        Args:
            yaw: Shared yaw limit for both directions.
            pitch: Shared pitch limit for both directions.
            roll: Roll/tilt limit.

        Returns:
            PoseLimits with mirrored thresholds.

        Example:
            >>> PoseLimits.symmetric(12.0, 10.0, 8.0).yaw_right
            12.0
        """
        return cls(
            yaw_left=yaw,
            yaw_right=yaw,
            pitch_up=pitch,
            pitch_down=pitch,
            roll=roll,
        )

    def yaw_ok(self, yaw: float) -> bool:
        """Return whether ``yaw`` is within left/right limits.

        Args:
            yaw: Yaw angle in degrees.

        Returns:
            True when the yaw direction's limit is respected.
        """
        return yaw <= self.yaw_left if yaw >= 0 else (-yaw) <= self.yaw_right

    def pitch_ok(self, pitch: float) -> bool:
        """Return whether ``pitch`` is within up/down limits.

        Args:
            pitch: Pitch angle in degrees.

        Returns:
            True when the pitch direction's limit is respected.
        """
        return pitch <= self.pitch_up if pitch >= 0 else (-pitch) <= self.pitch_down

    def roll_ok(self, roll: float) -> bool:
        """Return whether absolute roll is within the tilt limit.

        Args:
            roll: Roll angle in degrees.

        Returns:
            True when tilt is acceptable.
        """
        return abs(roll) <= self.roll

    def allows(self, yaw: float, pitch: float, roll: float) -> bool:
        """Return whether all three rotation axes are acceptable.

        Args:
            yaw: Yaw degrees.
            pitch: Pitch degrees.
            roll: Roll degrees.

        Returns:
            True when yaw, pitch, and roll each pass their direction limit.

        Example:
            >>> PoseLimits(10, 5, 8, 4, 12).allows(7.0, -3.0, 10.0)
            True
            >>> PoseLimits(10, 5, 8, 4, 12).allows(7.0, -3.0, 20.0)
            False
        """
        return self.yaw_ok(yaw) and self.pitch_ok(pitch) and self.roll_ok(roll)

    def to_metadata(self) -> dict[str, float]:
        """Serialize limits for metadata consumers.

        Returns:
            JSON-ready mapping of every direction limit.
        """
        return {
            "yaw_left": self.yaw_left,
            "yaw_right": self.yaw_right,
            "pitch_up": self.pitch_up,
            "pitch_down": self.pitch_down,
            "roll": self.roll,
        }


@dataclass(frozen=True)
class FaceBox:
    """Normalized face bounding box summary.

    All coordinates are fractions of image width/height in ``[0, 1]``.

    Attributes:
        center_x: Face-box center X (0 = left edge, 1 = right edge).
        center_y: Face-box center Y (0 = top edge, 1 = bottom edge).
        width_ratio: Box width / image width.
        height_ratio: Box height / image height.
    """

    center_x: float
    center_y: float
    width_ratio: float
    height_ratio: float

    def to_metadata(self) -> dict[str, float]:
        """Serialize the box for metadata.

        Returns:
            JSON-ready rounded geometry mapping.
        """
        return {
            "center_x": round(self.center_x, 4),
            "center_y": round(self.center_y, 4),
            "width_ratio": round(self.width_ratio, 4),
            "height_ratio": round(self.height_ratio, 4),
        }


@dataclass(frozen=True)
class FaceRegion:
    """Acceptable face-center region in normalized image coordinates.

    Defaults accept the full frame. Tighten ranges to keep faces
    toward the left/right/top/bottom of the crop.

    Attributes:
        center_x_min: Minimum allowed face-center X (inclusive).
        center_x_max: Maximum allowed face-center X (inclusive).
        center_y_min: Minimum allowed face-center Y (inclusive).
        center_y_max: Maximum allowed face-center Y (inclusive).
    """

    center_x_min: float = 0.0
    center_x_max: float = 1.0
    center_y_min: float = 0.0
    center_y_max: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.center_x_min <= self.center_x_max <= 1.0:
            raise ValueError("face center X range must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.center_y_min <= self.center_y_max <= 1.0:
            raise ValueError("face center Y range must satisfy 0 <= min <= max <= 1")

    @property
    def is_full_frame(self) -> bool:
        """True when the region accepts the entire image."""
        return (
            self.center_x_min <= 0.0
            and self.center_x_max >= 1.0
            and self.center_y_min <= 0.0
            and self.center_y_max >= 1.0
        )

    def contains(self, box: FaceBox) -> bool:
        """Return whether a face box center lies inside this region.

        Args:
            box: Normalized face geometry.

        Returns:
            True when the center is within min/max bounds.

        Example:
            >>> FaceRegion(0.3, 0.7, 0.2, 0.8).contains(FaceBox(0.5, 0.4, 0.2, 0.3))
            True
            >>> FaceRegion(0.3, 0.7, 0.2, 0.8).contains(FaceBox(0.9, 0.4, 0.2, 0.3))
            False
        """
        return (
            self.center_x_min <= box.center_x <= self.center_x_max
            and self.center_y_min <= box.center_y <= self.center_y_max
        )

    def to_metadata(self) -> dict[str, float]:
        """Serialize the region for metadata.

        Returns:
            JSON-ready min/max mapping.
        """
        return {
            "center_x_min": self.center_x_min,
            "center_x_max": self.center_x_max,
            "center_y_min": self.center_y_min,
            "center_y_max": self.center_y_max,
        }


@dataclass(frozen=True)
class FrontalMetrics:
    """Head-pose metrics and face geometry from a :class:`FrontalFilter`.

    Attributes:
        method: Algorithm identifier (``"solvepnp"``).
        yaw: Yaw angle in degrees.
        pitch: Pitch angle in degrees.
        roll: Roll/tilt angle in degrees.
        limits: Per-direction pose limits used for acceptance.
        box: Normalized face box (center + size ratios).
    """

    method: str
    yaw: float
    pitch: float
    roll: float
    limits: PoseLimits
    box: FaceBox

    def is_pose_ok(self) -> bool:
        """Return whether all rotation directions are within limits.

        Returns:
            True when yaw (L/R), pitch (U/D), and roll pass.
        """
        return self.limits.allows(self.yaw, self.pitch, self.roll)

    def is_in_region(self, region: FaceRegion | None = None) -> bool:
        """Return whether the face center lies in ``region``.

        Args:
            region: Acceptable face-center region. ``None`` accepts all.

        Returns:
            True when position is acceptable.
        """
        if region is None or region.is_full_frame:
            return True
        return region.contains(self.box)

    def is_frontal(self, region: FaceRegion | None = None) -> bool:
        """Return ``True`` when pose and optional region both pass.

        Args:
            region: Optional face-center region constraint.

        Returns:
            Combined acceptance decision.

        Example:
            >>> limits = PoseLimits(10, 10, 10, 10, 10)
            >>> box = FaceBox(0.5, 0.5, 0.3, 0.4)
            >>> m = FrontalMetrics("solvepnp", 2.0, -1.0, 0.5, limits, box)
            >>> m.is_frontal()
            True
        """
        return self.is_pose_ok() and self.is_in_region(region)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize metrics, limits, and geometry for the metadata contract.

        Returns:
            JSON-ready mapping used under ``metadata.frontal``.

        Example:
            >>> limits = PoseLimits(15, 15, 15, 15, 15)
            >>> box = FaceBox(0.5, 0.5, 0.2, 0.3)
            >>> m = FrontalMetrics("solvepnp", 1.0, 2.0, 3.0, limits, box)
            >>> sorted(m.to_metadata())
            ['box', 'limits', 'method', 'pitch', 'roll', 'yaw']
        """
        return {
            "method": self.method,
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "limits": self.limits.to_metadata(),
            "box": self.box.to_metadata(),
        }


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable settings for a generation run.

    Attributes:
        output_dir: Directory that will receive images and metadata.
        count: Number of successfully saved faces to produce.
        save_metadata: Write aggregate/per-face metadata and stats files.
        remove_bg: Apply background removal (transparent output).
        frontal_only: Enforce pose (and configured region) filtering.
        pose_limits: Per-direction rotation limits in degrees.
        face_region: Acceptable face-center region (normalized).
        classify_gender: Run gender classification when available.
        require_gender: Fail the run if gender classification is unavailable
            while ``classify_gender`` is enabled.
        grayscale: Convert final images to grayscale (LA when alpha present).
        models_dir: Explicit models directory; overrides env and defaults.
        max_attempts_factor: ``max_attempts = max(count * factor, count)``.
        request_sleep_s: Inclusive min/max sleep between source fetches.
        strict_completion: Raise :class:`GenerationIncompleteError` when the
            requested count is not reached.
        source_dir: Local image directory. When set, uses
            ``LocalDirectorySource`` instead of TPNDE.
        source_retries: Extra TPNDE attempts after the first failure.
        source_backoff_s: Base backoff seconds between TPNDE retries.
        source_shuffle: Shuffle local directory listing once at start.
        gender_max_share: Optional upper share (0..1] of produced images
            allowed for either ``male`` or ``female``. ``None`` disables
            balancing. Unknown labels are never capped.
    """

    output_dir: Path | str = "id_faces"
    count: int = 100
    save_metadata: bool = True
    remove_bg: bool = False
    frontal_only: bool = False
    pose_limits: PoseLimits = field(default_factory=PoseLimits)
    face_region: FaceRegion = field(default_factory=FaceRegion)
    classify_gender: bool = True
    require_gender: bool = False
    grayscale: bool = True
    models_dir: Path | str | None = None
    max_attempts_factor: int = 3
    request_sleep_s: tuple[float, float] = (0.5, 1.5)
    strict_completion: bool = False
    source_dir: Path | str | None = None
    source_retries: int = 2
    source_backoff_s: float = 0.25
    source_shuffle: bool = False
    gender_max_share: float | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.max_attempts_factor < 1:
            raise ValueError("max_attempts_factor must be >= 1")
        if self.source_retries < 0:
            raise ValueError("source_retries must be >= 0")
        if self.source_backoff_s < 0:
            raise ValueError("source_backoff_s must be >= 0")
        if self.gender_max_share is not None and not 0.0 < self.gender_max_share <= 1.0:
            raise ValueError("gender_max_share must be in (0, 1]")
        low, high = self.request_sleep_s
        if low < 0 or high < low:
            raise ValueError("request_sleep_s must satisfy 0 <= low <= high")
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.models_dir is not None:
            object.__setattr__(self, "models_dir", Path(self.models_dir))
        if self.source_dir is not None:
            object.__setattr__(self, "source_dir", Path(self.source_dir))

    @property
    def max_attempts(self) -> int:
        """Upper bound on source fetch attempts for this run."""
        return max(self.count * self.max_attempts_factor, self.count)

    @property
    def requires_landmark_filter(self) -> bool:
        """True when the run needs dlib pose/geometry evaluation.

        Position-only restrictions also require detection, so they imply
        the landmark filter when combined with ``frontal_only``, or when a
        non-default region is set with ``frontal_only``.
        """
        return self.frontal_only


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
        source_ref: Optional provenance string (local path or source URI).
        schema_version: Metadata schema version string.
    """

    filename: str
    index: int
    gender: GenderLabel
    background_removed: bool
    frontal_filtered: bool
    frontal: FrontalMetrics | None = None
    source_ref: str | None = None
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
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        return payload


@dataclass(frozen=True)
class GenerationStats:
    """Aggregate counters for a generation run.

    Attributes:
        requested: Target image count.
        produced: Images successfully written.
        attempts: Source fetch attempts made.
        failed_fetches: Fetches that returned no usable image.
        filtered_out: Images rejected by the pose/region filter.
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
            >>> from pathlib import Path
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
        if record.source_ref is not None:
            row["source_ref"] = record.source_ref
        if record.frontal is not None:
            row["yaw"] = record.frontal.yaw
            row["pitch"] = record.frontal.pitch
            row["roll"] = record.frontal.roll
            row["center_x"] = record.frontal.box.center_x
            row["center_y"] = record.frontal.box.center_y
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
