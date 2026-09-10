"""Orchestrate synthetic face generation."""

from __future__ import annotations

import random
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageOps

from face_faker.domain.entities import (
    FrontalMetrics,
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    ImageRecord,
)
from face_faker.domain.enums import GenderLabel
from face_faker.domain.errors import (
    DependencyError,
    GenerationIncompleteError,
    MissingModelError,
    SourceUnavailableError,
)
from face_faker.domain.ports import (
    BackgroundRemover,
    FaceSource,
    FaceStore,
    FrontalFilter,
    GenderClassifier,
)
from face_faker.logging_config import get_logger

logger = get_logger("application.generate")

ProgressCallback = Callable[[int, int], None]


def _to_rgb(image: Any) -> Image.Image:
    """Coerce a source payload into an RGB PIL image.

    Args:
        image: PIL image or array-like.

    Returns:
        RGB image.
    """
    if isinstance(image, Image.Image):
        if image.mode in ("RGBA", "LA", "P"):
            return image.convert("RGB")
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
    return Image.fromarray(image).convert("RGB")


def _finalize_pixels(image: Image.Image, *, grayscale: bool, remove_bg: bool) -> Image.Image:
    """Apply final color/alpha presentation rules.

    Args:
        image: Working image (RGB or RGBA).
        grayscale: Convert to grayscale while preserving alpha when present.
        remove_bg: Whether background removal already introduced alpha.

    Returns:
        Image ready for PNG serialization.
    """
    if not grayscale:
        return image
    if image.mode in ("RGBA", "LA"):
        alpha = image.getchannel("A")
        gray = image.convert("RGB").convert("L")
        out = Image.new("LA", gray.size)
        out.putdata(list(zip(gray.getdata(), alpha.getdata())))
        return out
    return ImageOps.grayscale(image)


def generate_faces(
    config: GenerationConfig,
    *,
    source: Optional[FaceSource] = None,
    frontal_filter: Optional[FrontalFilter] = None,
    gender_classifier: Optional[GenderClassifier] = None,
    background_remover: Optional[BackgroundRemover] = None,
    store: Optional[FaceStore] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    progress: Optional[ProgressCallback] = None,
) -> GenerationResult:
    """Generate a synthetic face dataset according to ``config``.

    Dependencies are injectable for tests; production callers typically use
    :func:`face_faker.generate_faces` which wires real adapters.

    Args:
        config: Immutable generation settings.
        source: Face image source. Defaults to TPNDE when omitted.
        frontal_filter: Head-pose filter used when ``config.frontal_only``.
        gender_classifier: Gender adapter when ``config.classify_gender``.
        background_remover: Background remover when ``config.remove_bg``.
        store: Persistence adapter. Defaults to local filesystem store.
        sleep_fn: Injectable sleep (tests pass a no-op).
        progress: Optional ``(produced, requested)`` progress callback.

    Returns:
        :class:`GenerationResult` with records, stats, and artifact paths.

    Raises:
        MissingModelError: Frontal filtering enabled but model file missing.
        DependencyError: Required optional dependency not installed and
            ``config.require_gender`` is set for gender, or frontal extras missing.
        SourceUnavailableError: No image could be fetched within ``max_attempts``
            and ``produced == 0``.
        GenerationIncompleteError: Fewer images than requested when
            ``config.strict_completion`` is true.

    Example:
        >>> from pathlib import Path
        >>> from face_faker.domain import GenerationConfig
        >>> from face_faker.application import generate_faces
        >>> # doctest: +SKIP
        >>> result = generate_faces(GenerationConfig(output_dir="out", count=2))
        >>> result.stats.requested
        2
    """
    if source is None:
        from face_faker.infrastructure.sources.tpnd import ThisPersonDoesNotExistSource

        source = ThisPersonDoesNotExistSource()

    if frontal_filter is None and config.frontal_only:
        from face_faker.infrastructure.vision.dlib_frontal import DlibSolvePnPFrontalFilter

        frontal_filter = DlibSolvePnPFrontalFilter(
            models_dir=config.models_dir,
            yaw_threshold=config.yaw_threshold,
            pitch_threshold=config.pitch_threshold,
        )

    if gender_classifier is None and config.classify_gender:
        from face_faker.infrastructure.vision.deepface_gender import DeepFaceGenderClassifier

        gender_classifier = DeepFaceGenderClassifier()

    if background_remover is None and config.remove_bg:
        from face_faker.infrastructure.vision.rembg_background import RembgBackgroundRemover

        background_remover = RembgBackgroundRemover()

    if store is None:
        from face_faker.infrastructure.storage.local_fs import LocalFaceStore

        store = LocalFaceStore(config.output_dir)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []
    attempts = 0
    failed_fetches = 0
    filtered_out = 0
    male = female = unknown = 0
    started = time.perf_counter()

    # Fail fast when the landmark model is required but absent.
    if config.frontal_only and frontal_filter is not None:
        predictor = getattr(frontal_filter, "predictor_path", None)
        if predictor is not None and not Path(predictor).exists():
            raise MissingModelError(
                f"Landmark model not found: {predictor}. "
                "Set FACE_FAKER_MODELS_DIR or download the dlib predictor."
            )

    while len(records) < config.count and attempts < config.max_attempts:
        attempts += 1
        try:
            raw = source.fetch()
        except Exception as exc:  # noqa: BLE001 - isolate source failures
            logger.warning("Source error on attempt %s: %s", attempts, exc)
            failed_fetches += 1
            continue

        if raw is None:
            failed_fetches += 1
            continue

        try:
            image = _to_rgb(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not decode source image: %s", exc)
            failed_fetches += 1
            continue

        metrics: FrontalMetrics | None = None
        if config.frontal_only and frontal_filter is not None:
            metrics = frontal_filter.evaluate(image)
            if metrics is None or not metrics.is_frontal():
                filtered_out += 1
                logger.debug(
                    "Filtered non-frontal image (yaw=%s pitch=%s)",
                    None if metrics is None else round(metrics.yaw, 2),
                    None if metrics is None else round(metrics.pitch, 2),
                )
                continue

        low, high = config.request_sleep_s
        if high > 0:
            sleep_fn(random.uniform(low, high))

        gender = GenderLabel.UNKNOWN
        if config.classify_gender and gender_classifier is not None:
            try:
                gender = gender_classifier.classify(image)
            except DependencyError:
                if config.require_gender:
                    raise
                logger.warning("Gender classifier unavailable; labeling UNKNOWN")
                gender = GenderLabel.UNKNOWN
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gender classification failed: %s", exc)
                gender = GenderLabel.UNKNOWN

        working = image
        if config.remove_bg and background_remover is not None:
            working = background_remover.remove(image)

        final_image = _finalize_pixels(
            working,
            grayscale=config.grayscale,
            remove_bg=config.remove_bg,
        )

        index = len(records) + 1
        filename = f"face_{index:04d}.png"
        store.save_image(final_image, filename)

        record = ImageRecord(
            filename=filename,
            index=index,
            gender=gender,
            background_removed=config.remove_bg,
            frontal_filtered=config.frontal_only,
            frontal=metrics if config.frontal_only else None,
        )
        records.append(record)

        if gender is GenderLabel.MALE:
            male += 1
        elif gender is GenderLabel.FEMALE:
            female += 1
        else:
            unknown += 1

        if progress is not None:
            progress(len(records), config.count)

    elapsed = time.perf_counter() - started
    stats = GenerationStats(
        requested=config.count,
        produced=len(records),
        attempts=attempts,
        failed_fetches=failed_fetches,
        filtered_out=filtered_out,
        gender_male=male,
        gender_female=female,
        gender_unknown=unknown,
        elapsed_seconds=elapsed,
    )

    if len(records) == 0:
        raise SourceUnavailableError(
            f"Could not produce any face after {attempts} attempts "
            f"(failed_fetches={failed_fetches}, filtered_out={filtered_out})."
        )

    if len(records) < config.count:
        message = (
            f"Produced {len(records)} of {config.count} requested faces "
            f"after {attempts} attempts."
        )
        warnings.warn(message, stacklevel=2)
        logger.warning(message)
        if config.strict_completion:
            raise GenerationIncompleteError(config.count, len(records), message)

    paths: dict[str, Path] = {}
    if config.save_metadata:
        paths = store.save_metadata(records, stats.to_metadata())

    return GenerationResult(
        records=tuple(records),
        stats=stats,
        output_dir=output_dir.resolve(),
        paths=paths,
    )
