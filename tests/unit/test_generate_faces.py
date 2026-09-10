"""Tests for the generate_faces application use-case."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from face_faker.application.generate_faces import generate_faces
from face_faker.domain.entities import GenerationConfig
from face_faker.domain.enums import GenderLabel
from face_faker.domain.errors import (
    GenerationIncompleteError,
    MissingModelError,
    SourceUnavailableError,
)

from tests.conftest import (
    FakeBackgroundRemover,
    FakeFrontalFilter,
    FakeGenderClassifier,
    FakeSource,
    FakeStore,
    frontal,
    make_image,
)


def _config(tmp_path: Path, **overrides) -> GenerationConfig:
    base = dict(
        output_dir=tmp_path / "out",
        count=2,
        save_metadata=True,
        remove_bg=False,
        frontal_only=False,
        classify_gender=True,
        request_sleep_s=(0.0, 0.0),
        max_attempts_factor=5,
    )
    base.update(overrides)
    return GenerationConfig(**base)


def test_produces_requested_count_and_metadata(tmp_path, sleep_noop) -> None:
    source = FakeSource([make_image("red"), make_image("blue")])
    store = FakeStore()
    gender = FakeGenderClassifier(GenderLabel.FEMALE)

    result = generate_faces(
        _config(tmp_path, count=2),
        source=source,
        gender_classifier=gender,
        store=store,
        sleep_fn=sleep_noop,
    )

    assert result.stats.produced == 2
    assert result.stats.requested == 2
    assert len(result.records) == 2
    assert result.records[0].filename == "face_0001.png"
    assert result.records[0].gender is GenderLabel.FEMALE
    assert store.metadata_calls == 1
    assert set(store.images) == {"face_0001.png", "face_0002.png"}


def test_metadata_schema_version_present(tmp_path, sleep_noop) -> None:
    result = generate_faces(
        _config(tmp_path, count=1),
        source=FakeSource([make_image()]),
        gender_classifier=FakeGenderClassifier(GenderLabel.MALE),
        store=FakeStore(),
        sleep_fn=sleep_noop,
    )
    payload = result.records[0].to_metadata()
    assert payload["schema_version"] == "1"
    assert payload["gender"] == "male"


def test_frontal_filter_rejects_and_accepts(tmp_path, sleep_noop) -> None:
    images = [make_image("red"), make_image("green"), make_image("blue")]
    source = FakeSource(images)
    store = FakeStore()
    # First image non-frontal, second frontal — third unused once count met.
    filt = FakeFrontalFilter(
        [
            frontal(yaw=30.0),
            frontal(yaw=2.0),
            frontal(yaw=1.0),
        ]
    )

    result = generate_faces(
        _config(tmp_path, count=1, frontal_only=True),
        source=source,
        frontal_filter=filt,
        gender_classifier=FakeGenderClassifier(),
        store=store,
        sleep_fn=sleep_noop,
    )

    assert result.stats.produced == 1
    assert result.stats.filtered_out == 1
    assert result.records[0].frontal is not None
    assert result.records[0].frontal.method == "solvepnp"
    assert result.records[0].frontal.yaw == 2.0


def test_frontal_metadata_not_labeled_ears(tmp_path, sleep_noop) -> None:
    result = generate_faces(
        _config(tmp_path, count=1, frontal_only=True),
        source=FakeSource([make_image()]),
        frontal_filter=FakeFrontalFilter([frontal(yaw=1.5, pitch=-2.5)]),
        gender_classifier=FakeGenderClassifier(),
        store=FakeStore(),
        sleep_fn=sleep_noop,
    )
    frontal_meta = result.records[0].to_metadata()["frontal"]
    assert set(frontal_meta) >= {
        "yaw",
        "pitch",
        "roll",
        "method",
        "yaw_threshold",
        "pitch_threshold",
        "roll_threshold",
    }
    assert "left_ear" not in frontal_meta
    assert "ear_asymmetry" not in frontal_meta


def test_frontal_filter_rejects_excessive_roll_tilt(tmp_path, sleep_noop) -> None:
    images = [make_image("red"), make_image("green")]
    filt = FakeFrontalFilter(
        [
            frontal(yaw=0.0, pitch=0.0, roll=40.0),
            frontal(yaw=0.0, pitch=0.0, roll=2.0),
        ]
    )
    result = generate_faces(
        _config(tmp_path, count=1, frontal_only=True),
        source=FakeSource(images),
        frontal_filter=filt,
        gender_classifier=FakeGenderClassifier(),
        store=FakeStore(),
        sleep_fn=sleep_noop,
    )
    assert result.stats.filtered_out == 1
    assert result.records[0].frontal is not None
    assert result.records[0].frontal.roll == 2.0


def test_remove_bg_default_false_keeps_rgb(tmp_path, sleep_noop) -> None:
    store = FakeStore()
    generate_faces(
        _config(tmp_path, count=1, remove_bg=False, grayscale=False),
        source=FakeSource([make_image()]),
        gender_classifier=FakeGenderClassifier(),
        store=store,
        sleep_fn=sleep_noop,
    )
    assert store.images["face_0001.png"].mode == "RGB"


def test_remove_bg_true_uses_remover(tmp_path, sleep_noop) -> None:
    store = FakeStore()
    result = generate_faces(
        _config(tmp_path, count=1, remove_bg=True, grayscale=False),
        source=FakeSource([make_image()]),
        gender_classifier=FakeGenderClassifier(),
        background_remover=FakeBackgroundRemover(),
        store=store,
        sleep_fn=sleep_noop,
    )
    assert result.records[0].background_removed is True
    assert store.images["face_0001.png"].mode == "RGBA"


def test_grayscale_preserves_alpha_when_removed(tmp_path, sleep_noop) -> None:
    store = FakeStore()
    generate_faces(
        _config(tmp_path, count=1, remove_bg=True, grayscale=True),
        source=FakeSource([make_image()]),
        gender_classifier=FakeGenderClassifier(),
        background_remover=FakeBackgroundRemover(),
        store=store,
        sleep_fn=sleep_noop,
    )
    assert store.images["face_0001.png"].mode == "LA"


def test_missing_model_raises_when_frontal_only(tmp_path, sleep_noop) -> None:
    class BrokenFilter:
        predictor_path = tmp_path / "does_not_exist.dat"

        def evaluate(self, image):
            return frontal()

    with pytest.raises(MissingModelError):
        generate_faces(
            _config(tmp_path, count=1, frontal_only=True),
            source=FakeSource([make_image()]),
            frontal_filter=BrokenFilter(),
            store=FakeStore(),
            sleep_fn=sleep_noop,
        )


def test_zero_success_raises_source_unavailable(tmp_path, sleep_noop) -> None:
    with pytest.raises(SourceUnavailableError):
        generate_faces(
            _config(tmp_path, count=2, max_attempts_factor=1),
            source=FakeSource([], failures=10),
            gender_classifier=FakeGenderClassifier(),
            store=FakeStore(),
            sleep_fn=sleep_noop,
        )


def test_incomplete_warns_by_default(tmp_path, sleep_noop) -> None:
    with pytest.warns(UserWarning, match="Produced"):
        result = generate_faces(
            _config(tmp_path, count=5, max_attempts_factor=1),
            source=FakeSource([make_image()]),
            gender_classifier=FakeGenderClassifier(),
            store=FakeStore(),
            sleep_fn=sleep_noop,
        )
    assert result.stats.produced == 1
    assert result.stats.to_metadata()["complete"] is False


def test_strict_completion_raises(tmp_path, sleep_noop) -> None:
    with pytest.warns(UserWarning, match="Produced"):
        with pytest.raises(GenerationIncompleteError):
            generate_faces(
                _config(tmp_path, count=5, max_attempts_factor=1, strict_completion=True),
                source=FakeSource([make_image()]),
                gender_classifier=FakeGenderClassifier(),
                store=FakeStore(),
                sleep_fn=sleep_noop,
            )


def test_csv_only_when_save_metadata(tmp_path, sleep_noop) -> None:
    """save_metadata=False must not call store.save_metadata (no CSV side channel)."""
    store = FakeStore()
    generate_faces(
        _config(tmp_path, count=1, save_metadata=False),
        source=FakeSource([make_image()]),
        gender_classifier=FakeGenderClassifier(),
        store=store,
        sleep_fn=sleep_noop,
    )
    assert store.metadata_calls == 0
    assert len(store.images) == 1


def test_gender_counts_use_normalized_labels(tmp_path, sleep_noop) -> None:
    class SwitchingGender:
        def __init__(self) -> None:
            self.n = 0

        def classify(self, image):
            self.n += 1
            return GenderLabel.MALE if self.n == 1 else GenderLabel.FEMALE

    result = generate_faces(
        _config(tmp_path, count=2),
        source=FakeSource([make_image("red"), make_image("blue")]),
        gender_classifier=SwitchingGender(),
        store=FakeStore(),
        sleep_fn=sleep_noop,
    )
    assert result.stats.gender_male == 1
    assert result.stats.gender_female == 1
    assert result.stats.gender_unknown == 0


def test_progress_callback(tmp_path, sleep_noop) -> None:
    seen: list[tuple[int, int]] = []
    generate_faces(
        _config(tmp_path, count=2),
        source=FakeSource([make_image(), make_image()]),
        gender_classifier=FakeGenderClassifier(),
        store=FakeStore(),
        sleep_fn=sleep_noop,
        progress=lambda produced, requested: seen.append((produced, requested)),
    )
    assert seen == [(1, 2), (2, 2)]
