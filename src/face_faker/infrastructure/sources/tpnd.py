"""HTTP client for thispersondoesnotexist.com with retry/backoff."""

from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Callable, Optional

import requests

from face_faker.logging_config import get_logger

logger = get_logger("sources.tpnd")

DEFAULT_URL = "https://thispersondoesnotexist.com/"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_USER_AGENT = "face-faker/3.3 (+https://github.com/alisadeghiaghili/face-faker)"
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF_S = 0.25
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class ThisPersonDoesNotExistSource:
    """Fetch synthetic faces from thispersondoesnotexist.com.

    Failed requests are retried with linear backoff for transient errors
    (timeouts, connection errors, and selected HTTP status codes).

    Args:
        url: Endpoint URL.
        timeout_s: Per-request timeout in seconds.
        session: Optional shared :class:`requests.Session`.
        user_agent: User-Agent header value.
        retries: Extra attempts after the first failure (``retries=2`` means
            up to 3 total tries).
        backoff_s: Base sleep between retries; multiplied by attempt index.
        sleep_fn: Injectable sleep (tests pass a no-op).

    Example:
        >>> source = ThisPersonDoesNotExistSource(retries=1, sleep_fn=lambda s: None)
        >>> source.retries
        1
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        session: Optional[requests.Session] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        retries: int = DEFAULT_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be >= 0")
        if backoff_s < 0:
            raise ValueError("backoff_s must be >= 0")
        self._url = url
        self._timeout_s = timeout_s
        self._session = session or requests.Session()
        self._user_agent = user_agent
        self._session.headers.setdefault("User-Agent", user_agent)
        self.retries = retries
        self.backoff_s = backoff_s
        self._sleep = sleep_fn

    def fetch(self) -> Any | None:
        """Download one face image, retrying transient failures.

        Returns:
            A :class:`PIL.Image.Image` in RGB mode, or ``None`` when every
            attempt failed.
        """
        from PIL import Image

        last_error: Exception | None = None
        attempts = self.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.get(self._url, timeout=self._timeout_s)
                if response.status_code in _RETRYABLE_STATUS:
                    last_error = RuntimeError(f"HTTP {response.status_code}")
                    logger.warning(
                        "TPNDE HTTP %s (attempt %s/%s)",
                        response.status_code,
                        attempt,
                        attempts,
                    )
                elif response.status_code != 200:
                    logger.warning("TPNDE returned HTTP %s", response.status_code)
                    return None
                else:
                    image = Image.open(BytesIO(response.content))
                    image.load()
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image
            except Exception as exc:  # noqa: BLE001 - batch must not crash
                last_error = exc
                logger.warning(
                    "TPNDE fetch failed (attempt %s/%s): %s", attempt, attempts, exc
                )
            if attempt < attempts:
                self._sleep(self.backoff_s * attempt)
        logger.warning("TPNDE exhausted retries: %s", last_error)
        return None
