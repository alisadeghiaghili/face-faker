"""Tests for euler conversion and filter construction (no dlib required)."""

from __future__ import annotations

import pytest

from face_faker.domain.errors import MissingModelError
from face_faker.infrastructure.vision.dlib_frontal import (
    DlibSolvePnPFrontalFilter,
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


def test_filter_holds_thresholds_and_path(tmp_path) -> None:
    filt = DlibSolvePnPFrontalFilter(
        models_dir=tmp_path,
        yaw_threshold=10.0,
        pitch_threshold=12.0,
    )
    assert filt.yaw_threshold == 10.0
    assert filt.pitch_threshold == 12.0
    assert filt.predictor_path.name == "shape_predictor_68_face_landmarks.dat"


def test_evaluate_raises_missing_model(tmp_path) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("dlib")
    pytest.importorskip("numpy")

    from PIL import Image

    filt = DlibSolvePnPFrontalFilter(models_dir=tmp_path / "empty")
    image = Image.new("RGB", (32, 32), "gray")
    with pytest.raises(MissingModelError):
        filt.evaluate(image)
