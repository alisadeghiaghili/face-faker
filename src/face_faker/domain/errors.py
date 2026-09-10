"""Typed exceptions for Face Faker.

All public failures inherit :class:`FaceFakerError` so callers can catch a
single base type at the boundary.
"""

from __future__ import annotations


class FaceFakerError(Exception):
    """Base class for all Face Faker errors."""


class MissingModelError(FaceFakerError):
    """Raised when a required model file is not present on disk.

    Example:
        >>> raise MissingModelError("landmark model missing")
        Traceback (most recent call last):
        face_faker.domain.errors.MissingModelError: landmark model missing
    """


class DependencyError(FaceFakerError):
    """Raised when an optional third-party dependency is required but absent."""


class SourceUnavailableError(FaceFakerError):
    """Raised when the face source cannot deliver any usable image."""


class GenerationIncompleteError(FaceFakerError):
    """Raised when fewer images were produced than requested.

    Attributes:
        requested: Number of images the caller asked for.
        produced: Number of images actually written.
    """

    def __init__(self, requested: int, produced: int, message: str | None = None) -> None:
        self.requested = requested
        self.produced = produced
        text = message or (
            f"Produced {produced} images but {requested} were requested."
        )
        super().__init__(text)
