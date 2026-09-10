"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from face_faker.domain.entities import (
    FaceBox,
    FrontalMetrics,
    GenerationConfig,
    PoseLimits,
)
from face_faker.domain.enums import GenderLabel


class FakeSource:
    """Deterministic in-memory face source for tests."""

    def __init__(self, images: list[Image.Image] | None = None, failures: int = 0) -> None:
        self.images = list(images or [])
        self.failures = failures
        self.calls = 0

    def fetch(self) -> Image.Image | None:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            return None
        if not self.images:
            return None
        return self.images.pop(0)


class FakeFrontalFilter:
    """Pose filter returning a scripted sequence of metrics/None."""

    def __init__(self, results: list[FrontalMetrics | None]) -> None:
        self.results = list(results)
        self.predictor_path = None  # skip model existence pre-check

    def evaluate(self, image: Any) -> FrontalMetrics | None:
        if not self.results:
            return None
        return self.results.pop(0)


class FakeGenderClassifier:
    """Gender classifier returning a fixed label."""

    def __init__(self, label: GenderLabel = GenderLabel.FEMALE) -> None:
        self.label = label
        self.calls = 0

    def classify(self, image: Any) -> GenderLabel:
        self.calls += 1
        return self.label


class FakeBackgroundRemover:
    """Background remover that adds a dummy alpha channel."""

    def remove(self, image: Any) -> Image.Image:
        rgba = image.convert("RGBA")
        return rgba


class FakeStore:
    """In-memory store capturing images and metadata."""

    def __init__(self) -> None:
        self.images: dict[str, Image.Image] = {}
        self.metadata_calls = 0

    def save_image(self, image: Any, filename: str):
        self.images[filename] = image
        return filename

    def save_metadata(self, records, stats):
        self.metadata_calls += 1
        return {}


def make_image(color: str = "red", size: tuple[int, int] = (64, 64)) -> Image.Image:
    """Create a solid-color RGB test image."""
    return Image.new("RGB", size, color=color)


def make_image_bytes(image: Image.Image | None = None) -> bytes:
    """Encode an image to JPEG bytes."""
    img = image or make_image()
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def frontal(
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    threshold: float = 15.0,
    limits: PoseLimits | None = None,
    center_x: float = 0.5,
    center_y: float = 0.5,
    box: FaceBox | None = None,
) -> FrontalMetrics:
    """Build frontal metrics with angles, limits, and normalized face box."""
    pose_limits = limits or PoseLimits.symmetric(threshold, threshold, threshold)
    face_box = box or FaceBox(center_x, center_y, 0.25, 0.35)
    return FrontalMetrics(
        method="solvepnp",
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        limits=pose_limits,
        box=face_box,
    )


@pytest.fixture
def sleep_noop() -> Any:
    """Sleep function that records delays instead of sleeping."""

    def _sleep(seconds: float) -> None:
        return None

    return _sleep


@pytest.fixture
def basic_config(tmp_path) -> GenerationConfig:
    """Minimal generation config writing into a temp directory."""
    return GenerationConfig(
        output_dir=tmp_path / "out",
        count=2,
        remove_bg=False,
        frontal_only=False,
        classify_gender=True,
        request_sleep_s=(0.0, 0.0),
        max_attempts_factor=5,
    )
