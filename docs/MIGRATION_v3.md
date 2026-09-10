# Migration guide: v2 → v3

v3.0.0 is a **breaking** release focused on correctness.

## API

| v2 | v3 |
|----|----|
| `generate_id_faces(...)` | `generate_faces(...)` returns `GenerationResult` |
| `num_images` | `count` |
| `remove_bg=True` default | `remove_bg=False` default |
| `frontal_threshold=15` (degrees, but used as EAR ratio) | `PoseLimits` (yaw L/R, pitch U/D, roll) in degrees + optional `FaceRegion` |
| metadata `pose.yaw/pitch/roll` from EAR values | metadata `frontal.yaw/pitch/roll` from solvePnP |
| gender `Male`/`Female`/other | gender `male`/`female`/`unknown` |
| silent failures | typed errors + logging |
| CSV always written | CSV only with `save_metadata=True` |

`generate_id_faces` remains as a **deprecated** wrapper that returns a list of
metadata dicts and emits `DeprecationWarning`.

## CLI

```bash
# v2
face-faker generate --count 100 --remove-bg --frontal-only --threshold 15

# v3.2
face-faker generate --count 100 --remove-bg --frontal-only \
  --yaw-left 15 --yaw-right 15 \
  --pitch-up 15 --pitch-down 15 \
  --roll-threshold 15 \
  --face-x-min 0.2 --face-x-max 0.8 \
  --face-y-min 0.15 --face-y-max 0.85
```

### v3.1 note

`roll_threshold` (head tilt) participates in frontal acceptance.

### v3.2 note

Symmetric `--yaw-threshold` / `--pitch-threshold` are replaced by
per-direction limits (`--yaw-left/--yaw-right`, `--pitch-up/--pitch-down`).
Face-center region flags control where the face may sit in the frame.

## Models

Set `FACE_FAKER_MODELS_DIR` (or `--models-dir`) to the directory containing
`shape_predictor_68_face_landmarks.dat`. Package-relative `models/` is only a
development fallback.

## Metadata consumers

Read `schema_version`. For v1:

- `gender` is lowercase enum value
- pose lives under `frontal` with `method: "solvepnp"`
- aggregate file is `stats.json`; tabular file is `faces.csv`
