"""Stable public API for library consumers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from face_faker.application.generate_faces import generate_faces as _generate_faces
from face_faker.domain.entities import GenerationConfig, GenerationResult
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
    yaw_threshold: float = 15.0,
    pitch_threshold: float = 15.0,
    classify_gender: bool = True,
    require_gender: bool = False,
    grayscale: bool = True,
    models_dir: Path | str | None = None,
    max_attempts_factor: int = 3,
    request_sleep_s: tuple[float, float] = (0.5, 1.5),
    strict_completion: bool = False,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    """Generate a synthetic face dataset.

    This is the primary public entry point. Prefer passing keyword arguments;
    a full :class:`GenerationConfig` may be supplied via ``config`` instead.

    Args:
        output_dir: Directory for images and metadata.
        count: Number of faces to produce.
        save_metadata: Write ``metadata.json``, ``stats.json``, ``faces.csv``.
        remove_bg: Remove background (requires rembg).
        frontal_only: Keep only frontal poses (requires dlib + OpenCV).
        yaw_threshold: Absolute yaw limit in degrees for frontal filtering.
        pitch_threshold: Absolute pitch limit in degrees for frontal filtering.
        classify_gender: Run gender classification (requires deepface).
        require_gender: Fail if gender classification is unavailable.
        grayscale: Convert outputs to grayscale (preserves alpha).
        models_dir: Optional models directory override.
        max_attempts_factor: Attempts budget multiplier.
        request_sleep_s: Min/max sleep between source fetches.
        strict_completion: Raise if fewer images than requested are produced.
        config: Full config object; when set, other fields are ignored.

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
        >>> result = generate_faces("out/faces", count=5, remove_bg=False)
        >>> result.stats.produced
        5
    """
    if config is None:
        config = GenerationConfig(
            output_dir=output_dir,
            count=count,
            save_metadata=save_metadata,
            remove_bg=remove_bg,
            frontal_only=frontal_only,
            yaw_threshold=yaw_threshold,
            pitch_threshold=pitch_threshold,
            classify_gender=classify_gender,
            require_gender=require_gender,
            grayscale=grayscale,
            models_dir=models_dir,
            max_attempts_factor=max_attempts_factor,
            request_sleep_s=request_sleep_s,
            strict_completion=strict_completion,
        )
    return _generate_faces(config)


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
