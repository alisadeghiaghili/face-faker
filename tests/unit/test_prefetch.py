"""Tests for PrefetchingFaceSource."""

from __future__ import annotations

import threading

from face_faker.infrastructure.sources.prefetch import PrefetchingFaceSource


class _CountingSource:
    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.calls = 0
        self.last_source_ref = None
        self._lock = threading.Lock()

    def fetch(self):
        with self._lock:
            self.calls += 1
            idx = self.calls
        self.last_source_ref = f"item-{idx}"
        return f"image-{idx}"


def test_inline_when_workers_is_one() -> None:
    src = PrefetchingFaceSource(_CountingSource(3), max_workers=1)
    img, ref = src.fetch_item()
    assert img == "image-1"
    assert ref == "item-1"
    src.shutdown()


def test_parallel_preserves_refs() -> None:
    inner = _CountingSource(8)
    src = PrefetchingFaceSource(inner, max_workers=4, max_buffer=4)
    seen_refs = []
    for _ in range(5):
        img, ref = src.fetch_item()
        assert img is not None
        assert ref is not None
        assert ref.startswith("item-")
        assert img.endswith(ref.split("-")[1])
        seen_refs.append(ref)
    assert len(set(seen_refs)) == 5
    src.shutdown()


def test_invalid_workers() -> None:
    try:
        PrefetchingFaceSource(_CountingSource(), max_workers=0)
    except ValueError as exc:
        assert "max_workers" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_generate_with_prefetch(tmp_path) -> None:

    from PIL import Image

    from face_faker.application.generate_faces import generate_faces
    from face_faker.domain.entities import GenerationConfig
    from face_faker.infrastructure.storage.local_fs import LocalFaceStore

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for i in range(4):
        Image.new("RGB", (16, 16), "blue").save(inbox / f"f{i}.png")
    out = tmp_path / "out"
    config = GenerationConfig(
        output_dir=out,
        count=4,
        classify_gender=False,
        request_sleep_s=(0.0, 0.0),
        source_dir=inbox,
        max_attempts_factor=6,
        prefetch_workers=3,
        prefetch_buffer=3,
    )
    result = generate_faces(config, store=LocalFaceStore(out), sleep_fn=lambda s: None)
    assert result.stats.produced == 4
    refs = [r.source_ref for r in result.records]
    assert all(ref and ref.endswith(".png") for ref in refs)
