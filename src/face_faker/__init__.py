"""Face Faker — synthetic face dataset toolkit.

Public surface:

- :func:`generate_faces` — primary generation API
- :class:`GenerationConfig` / :class:`GenerationResult` / :class:`GenerationStats`
- :class:`GenderLabel`
- Typed errors under :mod:`face_faker.domain.errors`
"""

from face_faker._version import __version__
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
from face_faker.interfaces.api import generate_faces, generate_id_faces

__all__ = [
    "DependencyError",
    "FaceBox",
    "FaceFakerError",
    "FaceRegion",
    "FrontalMetrics",
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
    "__version__",
    "generate_faces",
    "generate_id_faces",
]
