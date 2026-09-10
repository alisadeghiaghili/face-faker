# face-faker design policy (v3)

## Product

Synthetic face dataset toolkit for CV testing and research. Not identity fraud tooling.

## Principles

1. **Claim = Behavior** — every documented feature has a test.
2. **Fail loud** — missing model/dependency/source is typed, logged, and non-zero exit.
3. **Units are sacred** — pose thresholds are degrees (yaw/pitch/roll).
4. **Metadata is a contract** — versioned schema; never label EAR as pose.
5. **Install-safe resources** — models resolve via env/user-data, not `__file__` parents.
6. **Pluggable ports** — source/filter/classifier/remover/store are injectable.
7. **Cache hot resources** — dlib predictor and HTTP session loaded once.
8. **TDD** — red → green → refactor for behavior changes.
9. **Docs = contract** — docstrings with Args/Returns/Raises/Examples + type hints.
10. **No AI footprint in git history** — author identity only.

## Layering

```
interfaces (CLI / public API)
    → application (generate_faces use-case)
        → domain (entities, ports, errors)
            ← infrastructure (tpnd, dlib solvePnP, deepface, rembg, local fs)
```

## Frontal filter (v3.0+)

dlib 68 landmarks → six 2D points → OpenCV `solvePnP` → euler `(yaw, pitch, roll)` degrees.

Accept when all of the following hold:

- `|yaw| <= yaw_threshold` (left/right turn)
- `|pitch| <= pitch_threshold` (up/down gaze)
- `|roll| <= roll_threshold` (in-plane head tilt)

## Versioning

- v3.0.0 — breaking correctness release (pose units, gender labels, remove_bg default, schema).
- v3.1.0 — roll/tilt threshold participates in frontal acceptance.

## Gender

Classifier raw strings (`Man`, `Woman`, …) normalize to `GenderLabel`.

## Out of scope

Identity document forgery, KYC bypass, impersonation of real individuals.
