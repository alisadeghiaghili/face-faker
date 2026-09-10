"""Backward-compatible CLI module path.

Prefer :mod:`face_faker.interfaces.cli`.
"""

from face_faker.interfaces.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
