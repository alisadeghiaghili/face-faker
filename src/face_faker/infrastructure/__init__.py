"""Infrastructure adapters (I/O and heavy CV dependencies).

Import concrete adapters from their modules to avoid pulling optional
scientific dependencies (numpy/cv2/dlib/deepface/rembg) at package import time.
"""

__all__ = [
    "DeepFaceGenderClassifier",
    "DlibSolvePnPFrontalFilter",
    "LocalDirectorySource",
    "LocalFaceStore",
    "RembgBackgroundRemover",
    "ThisPersonDoesNotExistSource",
]


def __getattr__(name: str):
    """Lazily expose adapter classes on attribute access."""
    if name == "ThisPersonDoesNotExistSource":
        from face_faker.infrastructure.sources.tpnd import ThisPersonDoesNotExistSource

        return ThisPersonDoesNotExistSource
    if name == "LocalDirectorySource":
        from face_faker.infrastructure.sources.local_dir import LocalDirectorySource

        return LocalDirectorySource
    if name == "LocalFaceStore":
        from face_faker.infrastructure.storage.local_fs import LocalFaceStore

        return LocalFaceStore
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
