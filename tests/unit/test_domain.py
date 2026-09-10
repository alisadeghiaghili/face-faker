"""Tests for domain enums, entities, and errors."""

from __future__ import annotations

import pytest

from face_faker.domain.entities import (
    FrontalMetrics,
    GenerationConfig,
    ImageRecord,
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


class TestFrontalMetrics:
    """solvePnP metrics semantics."""

    def test_is_frontal_within_thresholds(self) -> None:
        metrics = FrontalMetrics("solvepnp", 5.0, -3.0, 1.0, 15.0, 15.0)
        assert metrics.is_frontal() is True

    def test_rejects_excessive_yaw(self) -> None:
        metrics = FrontalMetrics("solvepnp", 20.0, 0.0, 0.0, 15.0, 15.0)
        assert metrics.is_frontal() is False

    def test_rejects_excessive_pitch(self) -> None:
        metrics = FrontalMetrics("solvepnp", 0.0, 18.0, 0.0, 15.0, 15.0)
        assert metrics.is_frontal() is False

    def test_metadata_uses_pose_names_not_ears(self) -> None:
        metrics = FrontalMetrics("solvepnp", 5.123, -2.0, 0.5, 15.0, 12.0)
        payload = metrics.to_metadata()
        assert set(payload) == {
            "method",
            "yaw",
            "pitch",
            "roll",
            "yaw_threshold",
            "pitch_threshold",
        }
        assert payload["yaw"] == 5.12
        assert payload["method"] == "solvepnp"


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


class TestImageRecord:
    """Public metadata schema."""

    def test_schema_version_and_gender(self) -> None:
        rec = ImageRecord("face_0001.png", 1, GenderLabel.FEMALE, False, False)
        payload = rec.to_metadata()
        assert payload["schema_version"] == MetadataSchemaVersion.V1.value
        assert payload["gender"] == "female"
        assert "frontal" not in payload

    def test_frontal_block_present_when_metrics_exist(self) -> None:
        metrics = FrontalMetrics("solvepnp", 1.0, 2.0, 3.0, 15.0, 15.0)
        rec = ImageRecord("a.png", 1, GenderLabel.MALE, True, True, frontal=metrics)
        assert rec.to_metadata()["frontal"]["yaw"] == 1.0

    def test_csv_rows(self) -> None:
        rec = ImageRecord("a.png", 1, GenderLabel.MALE, True, False)
        rows = records_to_csv_rows([rec])
        assert rows[0]["gender"] == "male"


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
