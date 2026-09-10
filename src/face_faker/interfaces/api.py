"""Stable public API for library consumers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from face_faker.application.generate_faces import generate_faces as _generate_faces
from face_faker.domain.entities import (
    FaceRegion,
    GenerationConfig,
    GenerationResult,
    PoseLimits,
)
from face_faker.domain.errors import (
    DependencyError,
    FaceFakerError,
    GenerationIncompleteError,
    MissingModelError,
    SourceUnavailableError,
)


def generate_faces(
    output_dir: Path | str = "id_faces",
    count: int = 100,
    *,
    save_metadata: bool = True,
    remove_bg: bool = False,
    frontal_only: bool = False,
    yaw_left_threshold: float = 15.0,
    yaw_right_threshold: float = 15.0,
    pitch_up_threshold: float = 15.0,
    pitch_down_threshold: float = 15.0,
    roll_threshold: float = 15.0,
    face_center_x_min: float = 0.0,
    face_center_x_max: float = 1.0,
    face_center_y_min: float = 0.0,
    face_center_y_max: float = 1.0,
    pose_limits: PoseLimits | None = None,
    face_region: FaceRegion | None = None,
    classify_gender: bool = True,
    require_gender: bool = False,
    grayscale: bool = True,
    models_dir: Path | str | None = None,
    max_attempts_factor: int = 3,
    request_sleep_s: tuple[float, float] = (0.5, 1.5),
    strict_completion: bool = False,
    source_dir: Path | str | None = None,
    source_retries: int = 2,
    source_backoff_s: float = 0.25,
    source_shuffle: bool = False,
    gender_max_share: float | None = None,
    prefetch_workers: int = 1,
    prefetch_buffer: int = 4,
    config: GenerationConfig | None = None,
    progress: Any | None = None,
) -> GenerationResult:
    """Generate a synthetic face dataset.

    This is the primary public entry point. Prefer passing keyword arguments;
    a full :class:`GenerationConfig` may be supplied via ``config`` instead.

    Args:
        output_dir: Directory for images and metadata.
        count: Number of faces to produce.
        save_metadata: Write ``metadata.json``, ``stats.json``, ``faces.csv``.
        remove_bg: Remove background (requires rembg).
        frontal_only: Keep only faces that pass pose + region filters.
        yaw_left_threshold: Max yaw toward subject's left (degrees).
        yaw_right_threshold: Max |yaw| toward subject's right (degrees).
        pitch_up_threshold: Max looking-up pitch (degrees).
        pitch_down_threshold: Max |looking-down| pitch (degrees).
        roll_threshold: Max absolute head tilt (degrees).
        face_center_x_min: Min face-center X in [0, 1] (left bound).
        face_center_x_max: Max face-center X in [0, 1] (right bound).
        face_center_y_min: Min face-center Y in [0, 1] (top bound).
        face_center_y_max: Max face-center Y in [0, 1] (bottom bound).
        pose_limits: Full :class:`PoseLimits` (overrides scalar pose args).
        face_region: Full :class:`FaceRegion` (overrides scalar region args).
        classify_gender: Run gender classification (requires deepface).
        require_gender: Fail if gender classification is unavailable.
        grayscale: Convert outputs to grayscale (preserves alpha).
        models_dir: Optional models directory override.
        max_attempts_factor: Attempts budget multiplier.
        request_sleep_s: Min/max sleep between source fetches.
        strict_completion: Raise if fewer images than requested are produced.
        source_dir: Local image folder (offline source; replaces TPNDE).
        source_retries: Extra TPNDE attempts after the first failure.
        source_backoff_s: Base seconds between TPNDE retries.
        source_shuffle: Shuffle local directory listing once.
        gender_max_share: Cap on male/female share of produced images.
        prefetch_workers: Concurrent source fetch threads (1 = sequential).
        prefetch_buffer: In-flight prefetch queue depth.
        config: Full config object; when set, other fields are ignored.
        progress: Optional ``(produced, requested)`` callback for UI progress.

    Returns:
        :class:`GenerationResult` with records, stats, and written paths.

    Raises:
        MissingModelError: Frontal filter requested without a landmark model.
        DependencyError: Required optional dependency missing.
        SourceUnavailableError: Zero images produced.
        GenerationIncompleteError: Incomplete run with ``strict_completion``.
        ValueError: Invalid configuration values.

    Example:
        >>> from face_faker import generate_faces
        >>> # doctest: +SKIP
        >>> result = generate_faces(
        ...     "out/faces",
        ...     count=5,
        ...     frontal_only=True,
        ...     yaw_left_threshold=12,
        ...     yaw_right_threshold=10,
        ...     pitch_up_threshold=8,
        ...     pitch_down_threshold=12,
        ...     face_center_x_min=0.3,
        ...     face_center_x_max=0.7,
        ... )
        >>> result.stats.produced
        5
    """
    if config is None:
        limits = pose_limits or PoseLimits(
            yaw_left=yaw_left_threshold,
            yaw_right=yaw_right_threshold,
            pitch_up=pitch_up_threshold,
            pitch_down=pitch_down_threshold,
            roll=roll_threshold,
        )
        region = face_region or FaceRegion(
            center_x_min=face_center_x_min,
            center_x_max=face_center_x_max,
            center_y_min=face_center_y_min,
            center_y_max=face_center_y_max,
        )
        config = GenerationConfig(
            output_dir=output_dir,
            count=count,
            save_metadata=save_metadata,
            remove_bg=remove_bg,
            frontal_only=frontal_only,
            pose_limits=limits,
            face_region=region,
            classify_gender=classify_gender,
            require_gender=require_gender,
            grayscale=grayscale,
            models_dir=models_dir,
            max_attempts_factor=max_attempts_factor,
            request_sleep_s=request_sleep_s,
            strict_completion=strict_completion,
            source_dir=source_dir,
            source_retries=source_retries,
            source_backoff_s=source_backoff_s,
            source_shuffle=source_shuffle,
            gender_max_share=gender_max_share,
            prefetch_workers=prefetch_workers,
            prefetch_buffer=prefetch_buffer,
        )
    return _generate_faces(config, progress=progress)


def generate_id_faces(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Deprecated alias for :func:`generate_faces`.

    Kept for v2 compatibility. Returns a list of metadata dicts instead of
    a :class:`GenerationResult`.

    Args:
        *args: Positional args forwarded to :func:`generate_faces`.
        **kwargs: Keyword args forwarded to :func:`generate_faces`.

    Returns:
        List of metadata dictionaries.

    Raises:
        FaceFakerError: On generation failure.

    Example:
        >>> from face_faker import generate_id_faces
        >>> # doctest: +SKIP
        >>> rows = generate_id_faces(output_dir="out", num_images=1)
    """
    warnings.warn(
        "generate_id_faces is deprecated; use generate_faces and read "
        "GenerationResult.records/stats.",
        DeprecationWarning,
        stacklevel=2,
    )
    if "num_images" in kwargs and "count" not in kwargs:
        kwargs["count"] = kwargs.pop("num_images")
    elif len(args) >= 2:
        # v2 signature: (output_dir, num_images, ...)
        args = (args[0], args[1], *args[2:])
    result = generate_faces(*args, **kwargs)
    return result.to_metadata_list()


__all__ = [
    "DependencyError",
    "FaceFakerError",
    "GenerationConfig",
    "GenerationIncompleteError",
    "GenerationResult",
    "MissingModelError",
    "SourceUnavailableError",
    "generate_faces",
    "generate_id_faces",
]
