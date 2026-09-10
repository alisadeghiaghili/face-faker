# Face Faker

**Synthetic face dataset toolkit for computer-vision testing and research.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

Generate labeled synthetic face images for smoke-testing face pipelines, QA,
and research datasets. Not intended for identity fraud, impersonation, or
circumventing identity verification systems.

## What changed in v3

v3 is a **breaking correctness release**:

- Head-pose filtering uses **solvePnP** with real `yaw` / `pitch` / `roll` (degrees).
- Gender labels are normalized to `male` / `female` / `unknown`.
- `remove_bg` default is **`False`** in both API and CLI.
- Metadata schema is versioned (`schema_version: "1"`).
- Model paths resolve via `FACE_FAKER_MODELS_DIR` (install-safe).
- Typed errors, structured logging, and injectable adapters for tests.

See [docs/MIGRATION_v3.md](docs/MIGRATION_v3.md).

## Installation

```bash
pip install face-faker
```

Optional extras:

```bash
pip install 'face-faker[frontal]'   # dlib + OpenCV head pose
pip install 'face-faker[gender]'    # DeepFace gender labels
pip install 'face-faker[bg]'        # rembg background removal
pip install 'face-faker[full]'      # everything
```

### Landmark model (frontal filter only)

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
export FACE_FAKER_MODELS_DIR="$PWD/models"
mkdir -p "$FACE_FAKER_MODELS_DIR"
mv shape_predictor_68_face_landmarks.dat "$FACE_FAKER_MODELS_DIR/"
```

## Quick start (Python)

```python
from face_faker import GenerationConfig, generate_faces

result = generate_faces(
    output_dir="out/faces",
    count=20,
    remove_bg=False,
    frontal_only=False,
    classify_gender=True,
)

print(result.stats.produced, result.stats.gender_female)
print(result.records[0].to_metadata())
```

Frontal-only with explicit pose thresholds:

```python
from face_faker import generate_faces

result = generate_faces(
    "out/frontal",
    count=50,
    frontal_only=True,
    yaw_threshold=12.0,    # left/right turn
    pitch_threshold=15.0,  # up/down gaze
    roll_threshold=10.0,   # in-plane head tilt
    strict_completion=True,
)
```

## CLI

```bash
face-faker generate --count 20 --output-dir ./faces
face-faker generate --count 50 --frontal-only \
  --yaw-threshold 12 --pitch-threshold 15 --roll-threshold 10
face-faker generate --count 10 --remove-bg --color
face-faker info
python -m face_faker --version
```

Exit codes:

| Code | Meaning |
|-----:|---------|
| 0 | Success |
| 1 | Usage error |
| 2 | Missing model file |
| 3 | Missing optional dependency |
| 4 | Source unavailable (zero images) |
| 5 | Incomplete batch (`--strict`) |
| 10 | Unexpected error |

## Outputs

```
out/faces/
  face_0001.png
  face_0002.png
  metadata.json      # per-face schema v1
  stats.json         # aggregate counters
  faces.csv          # tabular summary
```

### Metadata record (schema v1)

```json
{
  "schema_version": "1",
  "filename": "face_0001.png",
  "index": 1,
  "gender": "female",
  "background_removed": false,
  "frontal_filtered": true,
  "frontal": {
    "method": "solvepnp",
    "yaw": 3.21,
    "pitch": -1.05,
    "roll": 0.4,
    "yaw_threshold": 15.0,
    "pitch_threshold": 15.0,
    "roll_threshold": 15.0
  }
}
```

## Defaults (single source of truth)

| Setting | Default |
|---------|---------|
| `count` | 100 |
| `remove_bg` | `false` |
| `frontal_only` | `false` |
| `yaw_threshold` | `15.0` degrees (turn) |
| `pitch_threshold` | `15.0` degrees (up/down gaze) |
| `roll_threshold` | `15.0` degrees (head tilt) |
| `classify_gender` | `true` |
| `grayscale` | `true` |
| `save_metadata` | `true` |
| `max_attempts_factor` | `3` |

## Intended use

- Unit/integration fixtures for face detection and attribute models
- Synthetic augmentation experiments
- QA smoke datasets

**Out of scope:** forging identity documents, bypassing KYC/biometric checks,
or creating content intended to impersonate a real person.

## Development

```bash
pip install -e '.[dev]'
pytest
```

Architecture notes: [docs/DESIGN.md](docs/DESIGN.md).

## License

Proprietary — see [LICENSE](LICENSE).

## Author

Ali Sadeghi Aghili
