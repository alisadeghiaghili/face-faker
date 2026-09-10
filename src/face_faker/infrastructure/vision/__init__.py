"""Vision adapters."""

from face_faker.infrastructure.vision.deepface_gender import DeepFaceGenderClassifier
from face_faker.infrastructure.vision.dlib_frontal import DlibSolvePnPFrontalFilter
from face_faker.infrastructure.vision.rembg_background import RembgBackgroundRemover

__all__ = [
    "DeepFaceGenderClassifier",
    "DlibSolvePnPFrontalFilter",
    "RembgBackgroundRemover",
]
