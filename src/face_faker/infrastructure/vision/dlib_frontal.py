"""dlib landmark + OpenCV solvePnP head-pose and face-box filter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from face_faker.config import landmark_model_path
from face_faker.domain.entities import FaceBox, FrontalMetrics, PoseLimits
from face_faker.domain.errors import DependencyError, MissingModelError
from face_faker.logging_config import get_logger

logger = get_logger("vision.dlib_frontal")

# Canonical 3D face points (millimetres) for the six solvePnP correspondences.
# Kept as plain tuples so importing this module does not require numpy.
_MODEL_POINTS_3D_SPEC = (
    (0.0, 0.0, 0.0),  # Nose tip  -> landmark 30
    (0.0, -330.0, -65.0),  # Chin      -> landmark 8
    (-225.0, 170.0, -135.0),  # Left eye  -> landmark 36
    (225.0, 170.0, -135.0),  # Right eye -> landmark 45
    (-150.0, -150.0, -125.0),  # Left mouth-> landmark 48
    (150.0, -150.0, -125.0),  # Right mouth-> landmark 54
)

_LANDMARK_INDICES = (30, 8, 36, 45, 48, 54)


def _require_cv_deps() -> tuple[Any, Any, Any]:
    """Import cv2, dlib, and numpy with a clear dependency error.

    Returns:
        Tuple of ``(cv2, dlib, np)`` modules.

    Raises:
        DependencyError: If opencv-python, dlib, or numpy is not installed.
    """
    try:
        import cv2
        import dlib
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DependencyError(
            "Frontal filtering requires opencv-python, dlib, and numpy. "
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
    pose_mat = np_module.hstack(
        (rotation_mat, np_module.zeros((3, 1), dtype=np_module.float64))
    )
    _, _, _, _, _, _, euler_angles = cv2_module.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = (float(v) for v in np_module.asarray(euler_angles).reshape(-1))
    return yaw, pitch, roll


def face_rect_to_box(rect: Any, width: int, height: int) -> FaceBox:
    """Convert a dlib rectangle to a normalized :class:`FaceBox`.

    Args:
        rect: dlib rectangle (or object with left/right/top/bottom).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Normalized face geometry.

    Example:
        >>> class R:
        ...     left, right, top, bottom = 10, 30, 5, 25
        >>> box = face_rect_to_box(R(), 100, 100)
        >>> box.center_x
        0.2
    """
    left = float(rect.left())
    right = float(rect.right())
    top = float(rect.top())
    bottom = float(rect.bottom())
    w = max(width, 1)
    h = max(height, 1)
    return FaceBox(
        center_x=((left + right) / 2.0) / w,
        center_y=((top + bottom) / 2.0) / h,
        width_ratio=(right - left) / w,
        height_ratio=(bottom - top) / h,
    )


class DlibSolvePnPFrontalFilter:
    """Estimate head pose and face box via dlib + OpenCV solvePnP.

    Caches the dlib shape predictor across calls. Evaluate returns
    :class:`FrontalMetrics` including per-direction :class:`PoseLimits` and
    normalized :class:`FaceBox` geometry for region filtering.

    Args:
        models_dir: Optional explicit models directory.
        pose_limits: Per-direction rotation limits in degrees.
        predictor_path: Explicit landmark model file path.

    Raises:
        MissingModelError: Raised when :meth:`evaluate` runs without a model.
        DependencyError: Raised when OpenCV/dlib/numpy cannot be imported.

    Example:
        >>> from face_faker.domain import PoseLimits
        >>> filt = DlibSolvePnPFrontalFilter(
        ...     pose_limits=PoseLimits(12, 10, 15, 8, 10)
        ... )
        >>> filt.pose_limits.yaw_right
        10.0
    """

    def __init__(
        self,
        models_dir: Path | str | None = None,
        pose_limits: PoseLimits | None = None,
        predictor_path: Path | str | None = None,
    ) -> None:
        self.pose_limits = pose_limits or PoseLimits()
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
        _, dlib, _ = _require_cv_deps()
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
        """Estimate head pose and face box for ``image``.

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
        detector = self._detector
        predictor = self._predictor
        if detector is None or predictor is None:  # pragma: no cover - guarded above
            raise MissingModelError("Landmark models failed to load")

        rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        faces = detector(gray, 1)
        if len(faces) == 0:
            logger.debug("No face detected for pose estimation")
            return None

        face_rect = faces[0]
        box = face_rect_to_box(face_rect, width=width, height=height)
        landmarks = predictor(gray, face_rect)
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
            np.array(_MODEL_POINTS_3D_SPEC, dtype=np.float64),
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
            limits=self.pose_limits,
            box=box,
        )
