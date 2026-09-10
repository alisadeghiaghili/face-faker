"""Local directory face source adapter."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from face_faker.logging_config import get_logger

logger = get_logger("sources.local_dir")

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class LocalDirectorySource:
    """Read face images from a local directory (cycling).

    Useful for offline generation, regression fixtures, and private
    datasets that must not leave the machine.

    Args:
        directory: Folder containing image files.
        recursive: Recurse into subdirectories when listing files.
        shuffle: Randomize the listing once at construction.
        seed: RNG seed used when ``shuffle`` is true.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        ValueError: If no supported images are found.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from PIL import Image
        >>> d = Path(tempfile.mkdtemp())
        >>> Image.new("RGB", (8, 8), "red").save(d / "a.png")
        >>> src = LocalDirectorySource(d)
        >>> src.file_count
        1
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        recursive: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Source directory not found: {self.directory}")
        files = self._list_images(self.directory, recursive=recursive)
        if not files:
            raise ValueError(f"No images found in {self.directory}")
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(files)
        self._files: tuple[Path, ...] = tuple(files)
        self._index = 0

    @staticmethod
    def _list_images(directory: Path, *, recursive: bool) -> list[Path]:
        """Return sorted image paths under ``directory``.

        Args:
            directory: Root folder.
            recursive: Include nested folders.

        Returns:
            Sorted list of image file paths.
        """
        pattern = "**/*" if recursive else "*"
        found = [
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        return sorted(found)

    @property
    def file_count(self) -> int:
        """Number of unique source images available."""
        return len(self._files)

    @property
    def files(self) -> Sequence[Path]:
        """Immutable listing of source image paths."""
        return self._files

    def fetch(self) -> Any | None:
        """Load the next image in the cycle.

        Returns:
            RGB :class:`PIL.Image.Image`, or ``None`` on read failure.
        """
        from PIL import Image

        path = self._files[self._index % len(self._files)]
        self._index += 1
        try:
            image = Image.open(path)
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    def __iter__(self) -> Iterable[Path]:
        """Iterate the unique file listing (not the infinite cycle)."""
        return iter(self._files)
