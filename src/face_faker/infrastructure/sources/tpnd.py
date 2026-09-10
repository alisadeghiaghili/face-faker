"""HTTP client for thispersondoesnotexist.com."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

import requests

from face_faker.logging_config import get_logger

logger = get_logger("sources.tpnd")

DEFAULT_URL = "https://thispersondoesnotexist.com/"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_USER_AGENT = "face-faker/3.0 (+https://github.com/alisadeghiaghili/face-faker)"


class ThisPersonDoesNotExistSource:
    """Fetch synthetic faces from thispersondoesnotexist.com.

    Args:
        url: Endpoint URL.
        timeout_s: Per-request timeout in seconds.
        session: Optional shared :class:`requests.Session`.
        user_agent: User-Agent header value.

    Example:
        >>> source = ThisPersonDoesNotExistSource(timeout_s=1.0)
        >>> callable(source.fetch)
        True
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        session: Optional[requests.Session] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._session = session or requests.Session()
        self._user_agent = user_agent
        self._session.headers.setdefault("User-Agent", user_agent)

    def fetch(self) -> Any | None:
        """Download one face image.

        Returns:
            A :class:`PIL.Image.Image` in RGB mode, or ``None`` on any failure.
        """
        from PIL import Image

        try:
            response = self._session.get(self._url, timeout=self._timeout_s)
            if response.status_code != 200:
                logger.warning("TPNDE returned HTTP %s", response.status_code)
                return None
            image = Image.open(BytesIO(response.content))
            image.load()
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as exc:  # noqa: BLE001 - boundary must not crash the batch
            logger.warning("TPNDE fetch failed: %s", exc)
            return None
