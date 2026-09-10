"""Tests for face source adapters."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
from PIL import Image

from face_faker.infrastructure.sources.local_dir import LocalDirectorySource
from face_faker.infrastructure.sources.tpnd import ThisPersonDoesNotExistSource


class TestLocalDirectorySource:
    """Offline directory source."""

    def _write_images(self, directory: Path, names: list[str]) -> None:
        for name in names:
            Image.new("RGB", (16, 16), color="blue").save(directory / name)

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LocalDirectorySource(tmp_path / "nope")

    def test_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No images"):
            LocalDirectorySource(tmp_path)

    def test_cycles_images(self, tmp_path: Path) -> None:
        self._write_images(tmp_path, ["a.png", "b.jpg"])
        source = LocalDirectorySource(tmp_path)
        assert source.file_count == 2
        first = source.fetch()
        second = source.fetch()
        third = source.fetch()
        assert first is not None and second is not None and third is not None
        assert first.size == (16, 16)

    def test_ignores_non_images(self, tmp_path: Path) -> None:
        self._write_images(tmp_path, ["ok.png"])
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        source = LocalDirectorySource(tmp_path)
        assert source.file_count == 1

    def test_recursive_listing(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub"
        nested.mkdir()
        Image.new("RGB", (8, 8), "red").save(nested / "n.png")
        source = LocalDirectorySource(tmp_path, recursive=True)
        assert source.file_count == 1

    def test_generate_from_local_dir(self, tmp_path: Path) -> None:
        from face_faker.application.generate_faces import generate_faces
        from face_faker.domain.entities import GenerationConfig
        from face_faker.infrastructure.storage.local_fs import LocalFaceStore

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        Image.new("RGB", (32, 32), "green").save(inbox / "one.png")
        Image.new("RGB", (32, 32), "yellow").save(inbox / "two.png")
        out = tmp_path / "out"
        config = GenerationConfig(
            output_dir=out,
            count=3,
            classify_gender=False,
            request_sleep_s=(0.0, 0.0),
            source_dir=inbox,
            max_attempts_factor=5,
        )
        result = generate_faces(config, store=LocalFaceStore(out), sleep_fn=lambda s: None)
        assert result.stats.produced == 3
        assert len(list(out.glob("face_*.png"))) == 3


class TestThisPersonDoesNotExistSource:
    """TPNDE HTTP client including retry behavior."""

    def test_invalid_retries(self) -> None:
        with pytest.raises(ValueError):
            ThisPersonDoesNotExistSource(retries=-1)

    def test_retries_transient_then_succeeds(self) -> None:
        class _Resp:
            def __init__(self, status: int, content: bytes = b"") -> None:
                self.status_code = status
                self.content = content

        calls = {"n": 0}

        class _Session:
            headers: dict[str, str] = {}

            def get(self, url, timeout=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise requests.Timeout("boom")
                buf_ok = _encode_png()
                return _Resp(200, buf_ok)

        source = ThisPersonDoesNotExistSource(
            session=_Session(),
            retries=2,
            backoff_s=0.0,
            sleep_fn=lambda s: None,
        )
        image = source.fetch()
        assert image is not None
        assert calls["n"] == 2

    def test_exhausts_retries(self) -> None:
        class _Resp:
            status_code = 503
            content = b""

        class _Session:
            headers: dict[str, str] = {}

            def get(self, url, timeout=None):
                return _Resp()

        source = ThisPersonDoesNotExistSource(
            session=_Session(),
            retries=1,
            backoff_s=0.0,
            sleep_fn=lambda s: None,
        )
        assert source.fetch() is None


def _encode_png() -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    return buffer.getvalue()
