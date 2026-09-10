"""Integration tests for dlib pose estimation (requires optional extras).

These tests skip unless ``cv2``, ``dlib``, ``numpy``, and the landmark
model file are available. Run after:

    pip install 'face-faker[frontal]'
    python scripts/download_landmark_model.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from face_faker.config import landmark_model_path
from face_faker.domain.entities import FaceRegion, PoseLimits
from face_faker.domain.errors import MissingModelError
from face_faker.infrastructure.vision.dlib_frontal import (
    DlibSolvePnPFrontalFilter,
    face_rect_to_box,
    rotation_vector_to_euler_degrees,
)


def _has_cv_stack() -> bool:
    try:
        import cv2  # noqa: F401
        import dlib  # noqa: F401
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cv_stack(),
    reason="opencv/dlib/numpy not installed",
)


def test_identity_rotation_is_zero() -> None:
    import cv2
    import numpy as np

    yaw, pitch, roll = rotation_vector_to_euler_degrees(
        np.zeros(3, dtype=np.float64), cv2, np
    )
    assert abs(yaw) < 1e-3
    assert abs(pitch) < 1e-3
    assert abs(roll) < 1e-3


def test_face_rect_geometry() -> None:
    class _Rect:
        def left(self):
            return 20

        def right(self):
            return 60

        def top(self):
            return 10

        def bottom(self):
            return 50

    box = face_rect_to_box(_Rect(), width=100, height=100)
    assert box.center_x == 0.4
    assert box.center_y == 0.3
    assert box.width_ratio == 0.4


def test_missing_model_raises(tmp_path: Path) -> None:
    from PIL import Image

    filt = DlibSolvePnPFrontalFilter(models_dir=tmp_path)
    with pytest.raises(MissingModelError):
        filt.evaluate(Image.new("RGB", (64, 64), "gray"))


def test_evaluate_on_noise_without_faces(tmp_path: Path) -> None:
    model = landmark_model_path(tmp_path)
    if not model.exists():
        pytest.skip("landmark model not downloaded")

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    image = Image.fromarray(noise, mode="RGB")

    filt = DlibSolvePnPFrontalFilter(models_dir=tmp_path, pose_limits=PoseLimits())
    metrics = filt.evaluate(image)
    # Detector may return no face on pure noise.
    assert metrics is None or metrics.method == "solvepnp"


def test_pose_limits_region_accepts_centered_box() -> None:
    from face_faker.domain.entities import FaceBox, FrontalMetrics

    metrics = FrontalMetrics(
        method="solvepnp",
        yaw=1.0,
        pitch=-2.0,
        roll=0.5,
        limits=PoseLimits(10, 10, 10, 10, 10),
        box=FaceBox(0.5, 0.5, 0.25, 0.3),
    )
    assert metrics.is_frontal(FaceRegion(0.3, 0.7, 0.2, 0.8)) is True
