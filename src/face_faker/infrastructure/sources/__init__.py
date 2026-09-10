"""Face source adapters."""

from face_faker.infrastructure.sources.local_dir import LocalDirectorySource
from face_faker.infrastructure.sources.prefetch import PrefetchingFaceSource
from face_faker.infrastructure.sources.tpnd import ThisPersonDoesNotExistSource

__all__ = [
    "LocalDirectorySource",
    "PrefetchingFaceSource",
    "ThisPersonDoesNotExistSource",
]
