"""Tests for domain enums, entities, and errors."""

from __future__ import annotations

import pytest

from face_faker.domain.entities import (
    FaceBox,
    FaceRegion,
    FrontalMetrics,
    GenerationConfig,
    ImageRecord,
    PoseLimits,
    records_to_csv_rows,
)
from face_faker.domain.enums import GenderLabel, MetadataSchemaVersion
from face_faker.domain.errors import GenerationIncompleteError, MissingModelError


class TestGenderLabelNormalize:
    """GenderLabel.normalize mapping contract."""

    def test_man_becomes_male(self) -> None:
        assert GenderLabel.normalize("Man") is GenderLabel.MALE

    def test_woman_becomes_female(self) -> None:
        assert GenderLabel.normalize("Woman") is GenderLabel.FEMALE

    def test_male_female_variants(self) -> None:
        assert GenderLabel.normalize("MALE") is GenderLabel.MALE
        assert GenderLabel.normalize("female") is GenderLabel.FEMALE

    def test_unknown_for_garbage(self) -> None:
        assert GenderLabel.normalize("robot") is GenderLabel.UNKNOWN
        assert GenderLabel.normalize(None) is GenderLabel.UNKNOWN

    def test_metadata_values_are_stable(self) -> None:
        assert GenderLabel.MALE.value == "male"
        assert GenderLabel.FEMALE.value == "female"
        assert GenderLabel.UNKNOWN.value == "unknown"


class TestPoseLimits:
    """Per-direction rotation limits."""

    def test_asymmetric_yaw(self) -> None:
        limits = PoseLimits(yaw_left=10.0, yaw_right=5.0, pitch_up=15.0, pitch_down=15.0, roll=15.0)
        assert limits.yaw_ok(8.0) is True
        assert limits.yaw_ok(12.0) is False
        assert limits.yaw_ok(-4.0) is True
        assert limits.yaw_ok(-6.0) is False

    def test_asymmetric_pitch(self) -> None:
        limits = PoseLimits(15.0, 15.0, 8.0, 12.0, 15.0)
        assert limits.pitch_ok(7.0) is True
        assert limits.pitch_ok(9.0) is False
        assert limits.pitch_ok(-11.0) is True
        assert limits.pitch_ok(-13.0) is False

    def test_roll_uses_absolute_value(self) -> None:
        limits = PoseLimits(15.0, 15.0, 15.0, 15.0, 10.0)
        assert limits.roll_ok(-9.0) is True
        assert limits.roll_ok(11.0) is False

    def test_negative_limits_rejected(self) -> None:
        with pytest.raises(ValueError):
            PoseLimits(yaw_left=-1.0)

    def test_symmetric_factory(self) -> None:
        limits = PoseLimits.symmetric(12.0, 10.0, 8.0)
        assert limits.yaw_left == limits.yaw_right == 12.0
        assert limits.pitch_up == limits.pitch_down == 10.0
        assert limits.roll == 8.0


class TestFaceRegion:
    """Normalized face-center region."""

    def test_full_frame_default(self) -> None:
        assert FaceRegion().is_full_frame is True

    def test_contains_center(self) -> None:
        region = FaceRegion(0.3, 0.7, 0.2, 0.8)
        assert region.contains(FaceBox(0.5, 0.4, 0.2, 0.3)) is True
        assert region.contains(FaceBox(0.9, 0.4, 0.2, 0.3)) is False
        assert region.contains(FaceBox(0.5, 0.1, 0.2, 0.3)) is False

    def test_invalid_range(self) -> None:
        with pytest.raises(ValueError):
            FaceRegion(0.8, 0.2, 0.0, 1.0)


class TestFrontalMetrics:
    """Pose + geometry acceptance."""

    def _metrics(
        self,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        limits: PoseLimits | None = None,
        center_x: float = 0.5,
        center_y: float = 0.5,
    ) -> FrontalMetrics:
        return FrontalMetrics(
            "solvepnp",
            yaw,
            pitch,
            roll,
            limits or PoseLimits(),
            FaceBox(center_x, center_y, 0.2, 0.3),
        )

    def test_accepts_centered_frontal(self) -> None:
        assert self._metrics().is_frontal() is True

    def test_rejects_excessive_yaw(self) -> None:
        assert self._metrics(yaw=20.0).is_pose_ok() is False

    def test_rejects_excessive_pitch(self) -> None:
        assert self._metrics(pitch=-18.0).is_pose_ok() is False

    def test_rejects_excessive_roll(self) -> None:
        assert self._metrics(roll=30.0).is_pose_ok() is False

    def test_asymmetric_direction_limits(self) -> None:
        limits = PoseLimits(20.0, 5.0, 15.0, 15.0, 15.0)
        assert self._metrics(yaw=15.0, limits=limits).is_pose_ok() is True
        assert self._metrics(yaw=-6.0, limits=limits).is_pose_ok() is False

    def test_region_rejects_offset_face(self) -> None:
        metrics = self._metrics(center_x=0.9, center_y=0.5)
        region = FaceRegion(0.3, 0.7, 0.2, 0.8)
        assert metrics.is_in_region(region) is False
        assert metrics.is_frontal(region) is False

    def test_region_allows_centered_face(self) -> None:
        metrics = self._metrics(center_x=0.5, center_y=0.4)
        region = FaceRegion(0.3, 0.7, 0.2, 0.8)
        assert metrics.is_frontal(region) is True

    def test_metadata_includes_limits_and_box(self) -> None:
        limits = PoseLimits(15.0, 12.0, 10.0, 8.0, 9.0)
        payload = self._metrics(yaw=1.234, limits=limits).to_metadata()
        assert payload["yaw"] == 1.23
        assert payload["limits"]["yaw_right"] == 12.0
        assert payload["limits"]["pitch_down"] == 8.0
        assert payload["box"]["center_x"] == 0.5


class TestGenerationConfig:
    """Config validation and derived fields."""

    def test_count_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(count=0)

    def test_max_attempts_factor(self) -> None:
        cfg = GenerationConfig(count=10, max_attempts_factor=3)
        assert cfg.max_attempts == 30

    def test_remove_bg_defaults_false(self) -> None:
        assert GenerationConfig().remove_bg is False

    def test_invalid_sleep_range(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(request_sleep_s=(2.0, 1.0))

    def test_output_dir_coerced_to_path(self) -> None:
        cfg = GenerationConfig(output_dir="somewhere")
        assert cfg.output_dir.name == "somewhere"

    def test_default_pose_limits_and_region(self) -> None:
        cfg = GenerationConfig()
        assert cfg.pose_limits.yaw_left == 15.0
        assert cfg.face_region.is_full_frame is True


class TestImageRecord:
    """Public metadata schema."""

    def test_schema_version_and_gender(self) -> None:
        rec = ImageRecord("face_0001.png", 1, GenderLabel.FEMALE, False, False)
        payload = rec.to_metadata()
        assert payload["schema_version"] == MetadataSchemaVersion.V1.value
        assert payload["gender"] == "female"
        assert "frontal" not in payload

    def test_frontal_block_present_when_metrics_exist(self) -> None:
        metrics = FrontalMetrics(
            "solvepnp",
            1.0,
            2.0,
            3.0,
            PoseLimits(),
            FaceBox(0.4, 0.5, 0.2, 0.3),
        )
        rec = ImageRecord("a.png", 1, GenderLabel.MALE, True, True, frontal=metrics)
        meta = rec.to_metadata()["frontal"]
        assert meta["yaw"] == 1.0
        assert meta["box"]["center_x"] == 0.4

    def test_csv_rows_include_center(self) -> None:
        metrics = FrontalMetrics(
            "solvepnp", 1.0, 2.0, 3.0, PoseLimits(), FaceBox(0.42, 0.55, 0.2, 0.3)
        )
        rec = ImageRecord("a.png", 1, GenderLabel.MALE, True, True, frontal=metrics)
        rows = records_to_csv_rows([rec])
        assert rows[0]["center_x"] == 0.42
        assert rows[0]["center_y"] == 0.55


class TestErrors:
    """Typed error payloads."""

    def test_incomplete_error_fields(self) -> None:
        err = GenerationIncompleteError(10, 3)
        assert err.requested == 10
        assert err.produced == 3
        assert "3" in str(err)

    def test_missing_model_is_face_faker_error(self) -> None:
        from face_faker.domain.errors import FaceFakerError

        assert issubclass(MissingModelError, FaceFakerError)
