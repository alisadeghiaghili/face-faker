"""Tests for public package exports and deprecation alias."""

from __future__ import annotations

import warnings

import face_faker
from face_faker.domain.enums import GenderLabel


def test_version() -> None:
    assert face_faker.__version__ == "3.5.0"


def test_public_exports() -> None:
    for name in (
        "generate_faces",
        "GenerationConfig",
        "GenerationResult",
        "GenerationStats",
        "GenderLabel",
        "MissingModelError",
    ):
        assert hasattr(face_faker, name), name


def test_generate_id_faces_is_deprecated_alias(monkeypatch, tmp_path) -> None:
    from face_faker import interfaces
    from face_faker.domain.entities import GenerationResult, GenerationStats

    stats = GenerationStats(1, 1, 1, 0, 0, 0, 0, 1, 0.0)

    def fake_generate_faces(*args, **kwargs):
        return GenerationResult(records=(), stats=stats, output_dir=tmp_path)

    monkeypatch.setattr(interfaces.api, "generate_faces", fake_generate_faces)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = face_faker.generate_id_faces(output_dir=str(tmp_path), num_images=1)
    assert rows == []
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_gender_label_is_exported_enum() -> None:
    assert face_faker.GenderLabel is GenderLabel
