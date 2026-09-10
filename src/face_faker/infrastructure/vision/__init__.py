"""Vision adapters.

Import concrete classes from their modules; optional CV stacks are loaded
lazily inside adapters.
"""

__all__ = [
    "DeepFaceGenderClassifier",
    "DlibSolvePnPFrontalFilter",
    "RembgBackgroundRemover",
]


def __getattr__(name: str) -> object:
    """Lazily expose vision adapters."""
    if name == "DeepFaceGenderClassifier":
        from face_faker.infrastructure.vision.deepface_gender import (
            DeepFaceGenderClassifier,
        )

        return DeepFaceGenderClassifier
    if name == "DlibSolvePnPFrontalFilter":
        from face_faker.infrastructure.vision.dlib_frontal import (
            DlibSolvePnPFrontalFilter,
        )

        return DlibSolvePnPFrontalFilter
    if name == "RembgBackgroundRemover":
        from face_faker.infrastructure.vision.rembg_background import (
            RembgBackgroundRemover,
        )

        return RembgBackgroundRemover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
