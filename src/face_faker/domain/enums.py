"""Domain enumerations."""

from __future__ import annotations

from enum import Enum


class GenderLabel(str, Enum):
    """Normalized gender labels used across the public API and metadata.

    Attributes:
        MALE: Normalized male label (``"male"``).
        FEMALE: Normalized female label (``"female"``).
        UNKNOWN: Classification unavailable or not confident.

    Example:
        >>> GenderLabel.normalize("Man") is GenderLabel.MALE
        True
        >>> GenderLabel.normalize("Woman").value
        'female'
    """

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, raw: str | None) -> GenderLabel:
        """Map free-form classifier output to a :class:`GenderLabel`.

        Accepts common DeepFace / OpenCV spellings such as ``Man``, ``Woman``,
        ``Male``, ``Female`` (any case).

        Args:
            raw: Raw gender string from a classifier, or ``None``.

        Returns:
            The matching enum member, or :attr:`UNKNOWN` when unrecognized.

        Example:
            >>> GenderLabel.normalize("FEMALE") is GenderLabel.FEMALE
            True
            >>> GenderLabel.normalize(None) is GenderLabel.UNKNOWN
            True
        """
        if raw is None:
            return cls.UNKNOWN
        token = str(raw).strip().lower()
        if token in {"male", "man", "m", "masculine"}:
            return cls.MALE
        if token in {"female", "woman", "f", "feminine"}:
            return cls.FEMALE
        return cls.UNKNOWN


class MetadataSchemaVersion(str, Enum):
    """Version identifier written into every metadata document."""

    V1 = "1"
