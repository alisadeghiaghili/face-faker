"""Infrastructure adapters (I/O and heavy CV dependencies)."""

from face_faker.infrastructure.sources.tpnd import ThisPersonDoesNotExistSource
from face_faker.infrastructure.storage.local_fs import LocalFaceStore
from face_faker.infrastructure.vision.deepface_gender import DeepFaceGenderClassifier
from face_faker.infrastructure.vision.dlib_frontal import DlibSolvePnPFrontalFilter
from face_faker.infrastructure.vision.rembg_background import RembgBackgroundRemover

__all__ = [
    "DeepFaceGenderClassifier",
    "DlibSolvePnPFrontalFilter",
    "LocalFaceStore",
    "RembgBackgroundRemover",
    "ThisPersonDoesNotExistSource",
]
