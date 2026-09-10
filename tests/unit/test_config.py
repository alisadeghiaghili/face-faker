"""Tests for model path resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from face_faker import config as config_mod


def test_explicit_dir_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config_mod.MODELS_DIR_ENV, "/tmp/env-models")
    resolved = config_mod.resolve_models_dir(tmp_path / "explicit")
    assert resolved == (tmp_path / "explicit").resolve()


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "from-env"
    monkeypatch.setenv(config_mod.MODELS_DIR_ENV, str(target))
    assert config_mod.resolve_models_dir() == target.resolve()


def test_landmark_model_path_joins_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config_mod.MODELS_DIR_ENV, str(tmp_path))
    path = config_mod.landmark_model_path()
    assert path.name == config_mod.DEFAULT_LANDMARK_MODEL_NAME
    assert path.parent == tmp_path.resolve()


def test_ensure_models_dir_creates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "nested" / "models"
    monkeypatch.setenv(config_mod.MODELS_DIR_ENV, str(target))
    created = config_mod.ensure_models_dir()
    assert created.exists()
    assert created.is_dir()
