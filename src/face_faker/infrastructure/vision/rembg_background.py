"""rembg-based background removal adapter."""

from __future__ import annotations

from typing import Any

from face_faker.domain.errors import DependencyError
from face_faker.logging_config import get_logger

logger = get_logger("vision.rembg_background")


class RembgBackgroundRemover:
    """Remove image background using the optional ``rembg`` package.

    Args:
        alpha_matting: Enable alpha matting (slower, softer edges).

    Example:
        >>> remover = RembgBackgroundRemover()
        >>> remover.alpha_matting
        False
    """

    def __init__(self, alpha_matting: bool = False) -> None:
        self.alpha_matting = alpha_matting

    def remove(self, image: Any) -> Any:
        """Return ``image`` with a transparent background.

        Args:
            image: Input RGB image.

        Returns:
            RGBA PIL image.

        Raises:
            DependencyError: If rembg is not installed.
        """
        try:
            from rembg import remove
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DependencyError(
                "Background removal requires rembg. "
                "Install with: pip install 'face-faker[bg]'"
            ) from exc

        logger.debug("Running rembg (alpha_matting=%s)", self.alpha_matting)
        return remove(image, alpha_matting=self.alpha_matting)
