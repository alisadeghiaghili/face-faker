"""Ports (interfaces) for infrastructure adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from face_faker.domain.entities import FrontalMetrics, ImageRecord
from face_faker.domain.enums import GenderLabel


@runtime_checkable
class FaceSource(Protocol):
    """Source of synthetic face images."""

    def fetch(self) -> Any | None:
        """Return an image-like object or ``None`` when unavailable.

        Returns:
            An image (typically ``PIL.Image.Image``) or ``None`` on failure.
        """
        ...


@runtime_checkable
class FrontalFilter(Protocol):
    """Head-pose filter used when ``frontal_only`` is enabled."""

    def evaluate(self, image: Any) -> FrontalMetrics | None:
        """Compute pose metrics for ``image``.

        Args:
            image: Input image in RGB mode.

        Returns:
            :class:`FrontalMetrics` on success, or ``None`` when no face /
            pose solution is available.
        """
        ...


@runtime_checkable
class GenderClassifier(Protocol):
    """Gender classification adapter."""

    def classify(self, image: Any) -> GenderLabel:
        """Classify gender for a face image.

        Args:
            image: Input image.

        Returns:
            A normalized :class:`GenderLabel`.
        """
        ...


@runtime_checkable
class BackgroundRemover(Protocol):
    """Background removal adapter."""

    def remove(self, image: Any) -> Any:
        """Return an image with a transparent background.

        Args:
            image: Input RGB image.

        Returns:
            Image with an alpha channel (RGBA/LA).
        """
        ...


@runtime_checkable
class FaceStore(Protocol):
    """Persistence adapter for images and metadata."""

    def save_image(self, image: Any, filename: str) -> Path:
        """Persist an image and return its path.

        Args:
            image: Image to write.
            filename: File name relative to the store root.

        Returns:
            Absolute path of the written file.
        """
        ...

    def save_metadata(
        self,
        records: list[ImageRecord],
        stats: dict[str, Any],
    ) -> dict[str, Path]:
        """Persist aggregate metadata artifacts.

        Args:
            records: Per-image records.
            stats: Aggregate statistics mapping.

        Returns:
            Named paths of files written (``metadata``, ``stats``, ``csv``).
        """
        ...
