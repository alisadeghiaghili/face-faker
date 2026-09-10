"""Prefetching wrapper that overlaps source I/O with pipeline work."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from face_faker.logging_config import get_logger

logger = get_logger("sources.prefetch")


class PrefetchingFaceSource:
    """Wrap a :class:`FaceSource` and fetch images on a worker pool.

    Fetching (especially TPNDE HTTP) can overlap with pose/gender/rembg
    processing. This adapter keeps a bounded in-flight queue.

    Thread-safety: the inner source is only touched under a lock so that
    ``last_source_ref`` stays consistent with the returned image.

    Args:
        inner: Source to wrap.
        max_workers: Number of concurrent fetch threads (``1`` disables
            pooling and fetches inline).
        max_buffer: Maximum in-flight futures kept filled ahead of the
            consumer.

    Example:
        >>> class S:
        ...     last_source_ref = None
        ...     def fetch(self):
        ...         return "img"
        >>> src = PrefetchingFaceSource(S(), max_workers=2, max_buffer=2)
        >>> img, ref = src.fetch_item()
        >>> img
        'img'
    """

    def __init__(
        self,
        inner: Any,
        *,
        max_workers: int = 2,
        max_buffer: int = 4,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_buffer < 1:
            raise ValueError("max_buffer must be >= 1")
        self._inner = inner
        self._max_workers = max_workers
        self._max_buffer = max_buffer
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._pending: deque[Future[tuple[Any, str | None]]] = deque()
        if max_workers > 1:
            self._pool = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="face-faker-fetch",
            )
            self._fill()

    def _fetch_one_locked(self) -> tuple[Any, str | None]:
        """Fetch a single item while holding the inner-source lock."""
        with self._lock:
            image = self._inner.fetch()
            ref = getattr(self._inner, "last_source_ref", None)
            return image, (str(ref) if ref is not None else None)

    def _fill(self) -> None:
        """Submit fetch tasks until the buffer is full."""
        if self._pool is None:
            return
        while len(self._pending) < self._max_buffer:
            self._pending.append(self._pool.submit(self._fetch_one_locked))

    def fetch_item(self) -> tuple[Any, str | None]:
        """Fetch the next image and its provenance ref.

        Returns:
            ``(image_or_None, source_ref_or_None)``.
        """
        if self._pool is None:
            return self._fetch_one_locked()

        if not self._pending:
            self._fill()
        future = self._pending.popleft()
        try:
            item = future.result()
        except Exception as exc:  # noqa: BLE001 - batch must not crash
            logger.warning("Prefetch worker failed: %s", exc)
            item = (None, None)
        self._fill()
        return item

    def fetch(self) -> Any | None:
        """Protocol-compatible fetch (drops ``source_ref``)."""
        image, _ = self.fetch_item()
        return image

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool.

        Args:
            wait: Block until pending workers finish.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None
            self._pending.clear()

    def __enter__(self) -> PrefetchingFaceSource:
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown(wait=False)
