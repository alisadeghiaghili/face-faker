"""Domain layer: entities, value objects, errors, and ports.

This package is pure Python (no I/O, no heavy CV imports). Infrastructure
adapters implement the ports defined here.
"""

from face_faker.domain.entities import (
    FrontalMetrics,
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    ImageRecord,
)
from face_faker.domain.enums import GenderLabel, MetadataSchemaVersion
from face_faker.domain.errors import (
    DependencyError,
    FaceFakerError,
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

__all__ = [
    "BackgroundRemover",
    "DependencyError",
    "FaceFakerError",
    "FaceSource",
    "FaceStore",
    "FrontalFilter",
    "FrontalMetrics",
    "GenderClassifier",
    "GenderLabel",
    "GenerationConfig",
    "GenerationIncompleteError",
    "GenerationResult",
    "GenerationStats",
    "ImageRecord",
    "MetadataSchemaVersion",
    "MissingModelError",
    "SourceUnavailableError",
]
