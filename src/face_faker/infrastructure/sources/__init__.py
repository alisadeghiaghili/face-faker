"""Face source adapters."""

from face_faker.infrastructure.sources.local_dir import LocalDirectorySource
from face_faker.infrastructure.sources.tpnd import ThisPersonDoesNotExistSource

__all__ = ["LocalDirectorySource", "ThisPersonDoesNotExistSource"]
