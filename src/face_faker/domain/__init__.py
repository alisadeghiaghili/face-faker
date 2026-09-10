"""Domain layer: entities, value objects, errors, and ports.

This package is pure Python (no I/O, no heavy CV imports). Infrastructure
adapters implement the ports defined here.
"""

from face_faker.domain.entities import (
    FaceBox,
    FaceRegion,
    FrontalMetrics,
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    ImageRecord,
    PoseLimits,
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
    "FaceBox",
    "FaceFakerError",
    "FaceRegion",
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
    "PoseLimits",
    "SourceUnavailableError",
]
