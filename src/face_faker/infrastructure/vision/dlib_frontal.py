"""dlib landmark + OpenCV solvePnP head-pose frontal filter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from face_faker.config import landmark_model_path
from face_faker.domain.entities import FrontalMetrics
from face_faker.domain.errors import DependencyError, MissingModelError
from face_faker.logging_config import get_logger

logger = get_logger("vision.dlib_frontal")

# Canonical 3D face points (millimetres) for the six solvePnP correspondences.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip  -> landmark 30
        (0.0, -330.0, -65.0),  # Chin      -> landmark 8
        (-225.0, 170.0, -135.0),  # Left eye  -> landmark 36
        (225.0, 170.0, -135.0),  # Right eye -> landmark 45
        (-150.0, -150.0, -125.0),  # Left mouth-> landmark 48
        (150.0, -150.0, -125.0),  # Right mouth-> landmark 54
    ],
    dtype=np.float64,
)

_LANDMARK_INDICES = (30, 8, 36, 45, 48, 54)


def _require_cv_deps() -> tuple[Any, Any, Any]:
    """Import cv2, dlib, and numpy with a clear dependency error.

    Returns:
        Tuple of ``(cv2, dlib, np)`` modules.

    Raises:
        DependencyError: If opencv-python or dlib is not installed.
    """
    try:
        import cv2
        import dlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DependencyError(
            "Frontal filtering requires opencv-python and dlib. "
            "Install with: pip install 'face-faker[frontal]'"
        ) from exc
    return cv2, dlib, np


def rotation_vector_to_euler_degrees(
    rotation_vector: Any,
    cv2_module: Any,
    np_module: Any,
) -> tuple[float, float, float]:
    """Convert a Rodrigues rotation vector to (yaw, pitch, roll) degrees.

    Uses ``cv2.decomposeProjectionMatrix`` on the concatenated pose matrix.
    Angles are returned in the OpenCV camera convention commonly used for
    head-pose estimation.

    Args:
        rotation_vector: Rodrigues rotation vector from ``solvePnP``.
        cv2_module: Imported OpenCV module (injectable for tests).
        np_module: Imported NumPy module (injectable for tests).

    Returns:
        ``(yaw, pitch, roll)`` in degrees.

    Example:
        >>> import numpy as np, cv2
        >>> yaw, pitch, roll = rotation_vector_to_euler_degrees(
        ...     np.zeros(3, dtype=np.float64), cv2, np
        ... )
        >>> abs(yaw) < 1e-6 and abs(pitch) < 1e-6 and abs(roll) < 1e-6
        True
    """
    rotation_mat, _ = cv2_module.Rodrigues(rotation_vector)
    # 3x4 projection-style matrix: [R | t] with t=0 is enough for angles.
    pose_mat = np_module.hstack((rotation_mat, np_module.zeros((3, 1), dtype=np.float64)))
    _, _, _, _, _, _, euler_angles = cv2_module.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = (float(v) for v in np_module.asarray(euler_angles).reshape(-1))
    return yaw, pitch, roll


class DlibSolvePnPFrontalFilter:
    """Estimate head pose via dlib landmarks and OpenCV solvePnP.

    The filter caches the dlib shape predictor across calls.

    Args:
        models_dir: Optional explicit models directory.
        yaw_threshold: Absolute yaw acceptance threshold in degrees.
        pitch_threshold: Absolute pitch acceptance threshold in degrees.
        predictor_path: Explicit landmark model file path.

    Raises:
        MissingModelError: Only raised when :meth:`evaluate` runs and the
            model file is absent (lazy validation by design).
        DependencyError: Raised when OpenCV/dlib cannot be imported.

    Example:
        >>> filt = DlibSolvePnPFrontalFilter(yaw_threshold=10.0, pitch_threshold=10.0)
        >>> filt.yaw_threshold
        10.0
    """

    def __init__(
        self,
        models_dir: Path | str | None = None,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 15.0,
        predictor_path: Path | str | None = None,
    ) -> None:
        self.yaw_threshold = float(yaw_threshold)
        self.pitch_threshold = float(pitch_threshold)
        self._predictor_path = (
            Path(predictor_path) if predictor_path is not None else landmark_model_path(models_dir)
        )
        self._predictor: Any | None = None
        self._detector: Any | None = None

    @property
    def predictor_path(self) -> Path:
        """Resolved path of the landmark model file."""
        return self._predictor_path

    def _ensure_loaded(self) -> None:
        """Load dlib detector/predictor once and cache them."""
        if self._predictor is not None:
            return
        cv2, dlib, _ = _require_cv_deps()
        path = self._predictor_path
        if not path.exists():
            raise MissingModelError(
                f"Landmark model not found: {path}. "
                "Set FACE_FAKER_MODELS_DIR or pass models_dir, and download "
                "shape_predictor_68_face_landmarks.dat."
            )
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(path))
        logger.debug("Loaded dlib predictor from %s", path)

    def evaluate(self, image: Any) -> FrontalMetrics | None:
        """Estimate head pose for ``image``.

        Args:
            image: RGB PIL image (or array-like convertible to RGB).

        Returns:
            :class:`FrontalMetrics` when a face is detected and solvePnP
            succeeds; otherwise ``None``.

        Raises:
            MissingModelError: If the landmark model file is missing.
            DependencyError: If OpenCV/dlib is not installed.

        Example:
            >>> filt = DlibSolvePnPFrontalFilter()
            >>> # doctest: +SKIP
            >>> metrics = filt.evaluate(some_pil_image)
        """
        self._ensure_loaded()
        cv2, _, np = _require_cv_deps()

        rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        faces = self._detector(gray, 1)
        if len(faces) == 0:
            logger.debug("No face detected for pose estimation")
            return None

        landmarks = self._predictor(gray, faces[0])
        image_points = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in _LANDMARK_INDICES],
            dtype=np.float64,
        )

        focal_length = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rotation_vector, _translation = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            logger.warning("solvePnP failed")
            return None

        yaw, pitch, roll = rotation_vector_to_euler_degrees(rotation_vector, cv2, np)
        return FrontalMetrics(
            method="solvepnp",
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            yaw_threshold=self.yaw_threshold,
            pitch_threshold=self.pitch_threshold,
        )
