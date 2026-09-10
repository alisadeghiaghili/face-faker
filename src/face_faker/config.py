"""Runtime configuration and model-path resolution.

Model directory resolution order:

1. Explicit ``GenerationConfig.models_dir``
2. Environment variable ``FACE_FAKER_MODELS_DIR``
3. Platform user data directory (``%APPDATA%`` / ``~/.local/share``)
4. Repository ``models/`` folder when running from a source checkout

Examples:
    >>> import os
    >>> os.environ["FACE_FAKER_MODELS_DIR"] = "/tmp/models"
    >>> resolve_models_dir().name
    'models'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_MODELS_DIR = PACKAGE_DIR.parent.parent / "models"

DEFAULT_LANDMARK_MODEL_NAME = "shape_predictor_68_face_landmarks.dat"
DEFAULT_YAW_THRESHOLD_DEG = 15.0
DEFAULT_PITCH_THRESHOLD_DEG = 15.0
DEFAULT_COUNT = 100

MODELS_DIR_ENV = "FACE_FAKER_MODELS_DIR"


def _user_data_dir() -> Path:
    """Return the platform-appropriate user data directory for Face Faker.

    Returns:
        Path under the OS user data root (``face-faker`` subdirectory).
    """
    if sys.platform == "win32":
        root = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(root) / "face-faker"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "face-faker"
    base = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(base) / "face-faker"


def resolve_models_dir(explicit: Path | str | None = None) -> Path:
    """Resolve the directory that should contain model files.

    Args:
        explicit: Optional explicit directory from config/CLI. When provided
            it wins over environment and defaults.

    Returns:
        Absolute path of the models directory (may not exist yet).

    Example:
        >>> resolve_models_dir("/tmp/x").name
        'x'
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_value = os.environ.get(MODELS_DIR_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    user_dir = _user_data_dir() / "models"
    if user_dir.exists():
        return user_dir
    if REPO_MODELS_DIR.exists():
        return REPO_MODELS_DIR.resolve()
    return user_dir


def landmark_model_path(models_dir: Path | str | None = None) -> Path:
    """Return the expected path of the dlib 68-point landmark model.

    Args:
        models_dir: Optional explicit models directory.

    Returns:
        Path to ``shape_predictor_68_face_landmarks.dat``.
    """
    return resolve_models_dir(models_dir) / DEFAULT_LANDMARK_MODEL_NAME


def ensure_models_dir(models_dir: Path | str | None = None) -> Path:
    """Create the models directory if needed and return it.

    Args:
        models_dir: Optional explicit models directory.

    Returns:
        Path of an existing models directory.
    """
    path = resolve_models_dir(models_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
