"""DeepFace-based gender classification adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from face_faker.domain.enums import GenderLabel
from face_faker.domain.errors import DependencyError
from face_faker.logging_config import get_logger

logger = get_logger("vision.deepface_gender")


class DeepFaceGenderClassifier:
    """Classify gender using DeepFace when the optional extra is installed.

    Args:
        model_name: DeepFace model name.
        enforce_detection: Passed through to DeepFace.analyze.

    Example:
        >>> classifier = DeepFaceGenderClassifier()
        >>> classifier.model_name
        'VGG-Face'
    """

    def __init__(self, model_name: str = "VGG-Face", enforce_detection: bool = False) -> None:
        self.model_name = model_name
        self.enforce_detection = enforce_detection

    def classify(self, image: Any) -> GenderLabel:
        """Classify the dominant gender for a face image.

        Args:
            image: PIL image or ndarray.

        Returns:
            Normalized :class:`GenderLabel` (``UNKNOWN`` on soft failure).

        Raises:
            DependencyError: If DeepFace is not installed.
        """
        try:
            from deepface import DeepFace
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DependencyError(
                "Gender classification requires deepface. "
                "Install with: pip install 'face-faker[gender]'"
            ) from exc

        img_array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        try:
            analysis = DeepFace.analyze(
                img_path=img_array,
                actions=["gender"],
                enforce_detection=self.enforce_detection,
                detector_backend="opencv",
                silent=True,
            )
        except Exception as exc:  # noqa: BLE001 - classifier must not kill the batch
            logger.warning("DeepFace analyze failed: %s", exc)
            return GenderLabel.UNKNOWN

        if not analysis:
            return GenderLabel.UNKNOWN
        # DeepFace returns a list of dicts; dominant_gender is Man/Woman.
        first = analysis[0] if isinstance(analysis, list) else analysis
        raw = first.get("dominant_gender") if isinstance(first, dict) else None
        label = GenderLabel.normalize(raw)
        logger.debug("Gender raw=%r normalized=%s", raw, label.value)
        return label
