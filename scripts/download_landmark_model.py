"""Download the dlib 68-point landmark model into the models directory.

Usage:
    python scripts/download_landmark_model.py [--dest DIR] [--force]

The official archive is served from dlib.net. This script downloads,
decompresses, and writes ``shape_predictor_68_face_landmarks.dat``.
"""

from __future__ import annotations

import argparse
import bz2
import shutil
import sys
import urllib.request
from pathlib import Path

ARCHIVE_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_NAME = "shape_predictor_68_face_landmarks.dat"
DEFAULT_CHUNK = 1024 * 256


def download_archive(url: str, dest: Path, chunk: int = DEFAULT_CHUNK) -> None:
    """Stream-download ``url`` to ``dest``.

    Args:
        url: HTTP(S) URL of the archive.
        dest: Target file path for the compressed archive.
        chunk: Read chunk size in bytes.

    Example:
        >>> # doctest: +SKIP
        >>> download_archive(ARCHIVE_URL, Path("/tmp/model.bz2"))
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "face-faker/3.6 (+https://github.com/alisadeghiaghili/face-faker)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        while True:
            block = response.read(chunk)
            if not block:
                break
            handle.write(block)


def decompress_bz2(archive: Path, target: Path) -> None:
    """Decompress a ``.bz2`` archive to ``target``.

    Args:
        archive: Path to the compressed file.
        target: Output path for the decompressed model.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(archive, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def ensure_model(dest_dir: Path, *, force: bool = False, url: str = ARCHIVE_URL) -> Path:
    """Ensure the landmark model exists under ``dest_dir``.

    Args:
        dest_dir: Models directory.
        force: Redownload even if the model file already exists.
        url: Archive URL override.

    Returns:
        Path to the model file.

    Raises:
        RuntimeError: If download or decompression fails.
    """
    model_path = dest_dir / MODEL_NAME
    if model_path.exists() and not force:
        print(f"Model already present: {model_path}")
        return model_path

    archive_path = dest_dir / f"{MODEL_NAME}.bz2"
    print(f"Downloading {url} ...")
    try:
        download_archive(url, archive_path)
        print("Decompressing ...")
        decompress_bz2(archive_path, model_path)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        raise RuntimeError(f"Failed to prepare landmark model: {exc}") from exc
    finally:
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)

    print(f"Wrote {model_path} ({model_path.stat().st_size} bytes)")
    return model_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the download helper.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Download dlib 68-point landmark model")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Models directory (default: FACE_FAKER_MODELS_DIR or ./models)",
    )
    parser.add_argument("--force", action="store_true", help="Redownload if present")
    args = parser.parse_args(argv)

    if args.dest is not None:
        dest = args.dest
    else:
        import os

        env = os.environ.get("FACE_FAKER_MODELS_DIR")
        dest = Path(env) if env else Path("models")

    try:
        ensure_model(dest, force=args.force)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
