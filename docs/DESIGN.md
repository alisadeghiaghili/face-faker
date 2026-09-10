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

## Face geometry filter (v3.2+)

dlib 68 landmarks → six 2D points → OpenCV `solvePnP` → euler `(yaw, pitch, roll)` degrees,
plus dlib face rectangle → normalized `FaceBox`.

Accept when **all** of the following hold:

**Rotation (four directions + tilt)**

- `yaw <= pose_limits.yaw_left` when `yaw >= 0` (turn toward subject's left)
- `(-yaw) <= pose_limits.yaw_right` when `yaw < 0` (turn toward subject's right)
- `pitch <= pose_limits.pitch_up` when `pitch >= 0` (looking up)
- `(-pitch) <= pose_limits.pitch_down` when `pitch < 0` (looking down)
- `|roll| <= pose_limits.roll` (in-plane head tilt)

**Position (four frame directions)**

- `face_region.center_x_min <= box.center_x <= face_region.center_x_max` (left/right)
- `face_region.center_y_min <= box.center_y <= face_region.center_y_max` (top/bottom)

Default region is the full frame (`0..1`), so position filtering is opt-in.

## Versioning

- v3.0.0 — breaking correctness release (pose units, gender labels, remove_bg default, schema).
- v3.1.0 — roll/tilt threshold participates in frontal acceptance.
- v3.2.0 — per-direction pose limits (L/R/U/D) + face-center region filter.
- v3.3.0 — local directory source, TPNDE retry/backoff, ruff+mypy CI.
- v3.4.0 — CLI progress, gender_max_share balancing, source_ref provenance, vision CI job.
- v3.5.0 — packaging CI (build+twine), CLI e2e for local source, source_ref in CSV.
- v3.6.0 — landmark download script, dlib integration tests, CI model cache.

## Gender

Classifier raw strings (`Man`, `Woman`, …) normalize to `GenderLabel`.

## Out of scope

Identity document forgery, KYC bypass, impersonation of real individuals.
