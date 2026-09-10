"""Tests for euler conversion and filter construction (no dlib required)."""

from __future__ import annotations

import pytest

from face_faker.domain.entities import PoseLimits
from face_faker.domain.errors import MissingModelError
from face_faker.infrastructure.vision.dlib_frontal import (
    DlibSolvePnPFrontalFilter,
    face_rect_to_box,
    rotation_vector_to_euler_degrees,
)


def test_identity_rotation_is_zero_angles() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    yaw, pitch, roll = rotation_vector_to_euler_degrees(
        np.zeros(3, dtype=np.float64), cv2, np
    )
    assert abs(yaw) < 1e-4
    assert abs(pitch) < 1e-4
    assert abs(roll) < 1e-4


def test_filter_holds_pose_limits_and_path(tmp_path) -> None:
    limits = PoseLimits(10.0, 8.0, 12.0, 6.0, 9.0)
    filt = DlibSolvePnPFrontalFilter(models_dir=tmp_path, pose_limits=limits)
    assert filt.pose_limits.yaw_left == 10.0
    assert filt.pose_limits.yaw_right == 8.0
    assert filt.pose_limits.pitch_down == 6.0
    assert filt.pose_limits.roll == 9.0
    assert filt.predictor_path.name == "shape_predictor_68_face_landmarks.dat"


def test_face_rect_to_box_normalizes() -> None:
    class _Rect:
        def left(self):
            return 10

        def right(self):
            return 30

        def top(self):
            return 5

        def bottom(self):
            return 25

    box = face_rect_to_box(_Rect(), width=100, height=100)
    assert box.center_x == 0.2
    assert box.center_y == 0.15
    assert box.width_ratio == 0.2
    assert box.height_ratio == 0.2


def test_evaluate_raises_missing_model(tmp_path) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("dlib")
    pytest.importorskip("numpy")

    from PIL import Image

    filt = DlibSolvePnPFrontalFilter(models_dir=tmp_path / "empty")
    image = Image.new("RGB", (32, 32), "gray")
    with pytest.raises(MissingModelError):
        filt.evaluate(image)
