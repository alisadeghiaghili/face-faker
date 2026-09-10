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

Frontal-only with per-direction rotation limits and face-center region:

```python
from face_faker import generate_faces

result = generate_faces(
    "out/frontal",
    count=50,
    frontal_only=True,
    # rotation (degrees) — four directions + tilt
    yaw_left_threshold=12.0,    # turn toward subject's left
    yaw_right_threshold=8.0,    # turn toward subject's right
    pitch_up_threshold=6.0,     # looking up
    pitch_down_threshold=10.0,  # looking down
    roll_threshold=8.0,         # in-plane head tilt
    # face position in frame (normalized 0..1)
    face_center_x_min=0.30,
    face_center_x_max=0.70,
    face_center_y_min=0.20,
    face_center_y_max=0.80,
    strict_completion=True,
)
```

## Sources

| Source | How to enable |
|--------|----------------|
| TPNDE (default) | leave `source_dir` unset; retries via `source_retries` / `source_backoff_s` |
| Local folder | `source_dir=...` or `--source-dir` (cycles images; optional `--source-shuffle`) |

```python
result = generate_faces(
    "out/from_local",
    count=30,
    source_dir="datasets/faces",
    source_shuffle=True,
    classify_gender=False,
)
```

```bash
face-faker generate --count 30 --source-dir ./datasets/faces --source-shuffle
```

## Balancing & provenance

```python
# Cap male/female share of the produced batch (e.g. max 55% each)
result = generate_faces(
    "out/balanced",
    count=100,
    gender_max_share=0.55,
    source_dir="datasets/faces",
)
# Each record may include source_ref (local path or tpnd:// URI)
print(result.records[0].source_ref)
```

```bash
face-faker generate --count 100 --gender-max-share 0.55 --progress
```

## CLI

```bash
face-faker generate --count 20 --output-dir ./faces
face-faker generate --count 50 --frontal-only \
  --yaw-left 12 --yaw-right 8 \
  --pitch-up 6 --pitch-down 10 \
  --roll-threshold 8 \
  --face-x-min 0.3 --face-x-max 0.7 \
  --face-y-min 0.2 --face-y-max 0.8
face-faker generate --count 10 --remove-bg --color
face-faker generate --count 30 --source-dir ./inbox --source-shuffle --progress
face-faker generate --count 100 --gender-max-share 0.55 --progress
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
    "limits": {
      "yaw_left": 15.0,
      "yaw_right": 15.0,
      "pitch_up": 15.0,
      "pitch_down": 15.0,
      "roll": 15.0
    },
    "box": {
      "center_x": 0.51,
      "center_y": 0.42,
      "width_ratio": 0.28,
      "height_ratio": 0.36
    }
  }
}
```

## Defaults (single source of truth)

| Setting | Default |
|---------|---------|
| `count` | 100 |
| `remove_bg` | `false` |
| `frontal_only` | `false` |
| `pose_limits.yaw_left` | `15.0` degrees |
| `pose_limits.yaw_right` | `15.0` degrees |
| `pose_limits.pitch_up` | `15.0` degrees |
| `pose_limits.pitch_down` | `15.0` degrees |
| `pose_limits.roll` | `15.0` degrees |
| `face_region` | full frame `[0,1] × [0,1]` |
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
