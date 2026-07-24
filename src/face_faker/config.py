"""
Configuration for Face Faker
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Default paths
DEFAULT_DLIB_PREDICTOR = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"

# Face generation defaults
DEFAULT_FACE_OUTPUT_DIR = "id_faces"
DEFAULT_NUM_IMAGES = 100
DEFAULT_FRONTAL_THRESHOLD = 15


def get_model_path(model_name: str = "shape_predictor_68_face_landmarks.dat") -> Path:
    """Get the path to a model file."""
    return MODELS_DIR / model_name


def ensure_models_dir() -> Path:
    """Ensure the models directory exists and return its path."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR
