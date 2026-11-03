# Baselines

Baseline planners for the nuReasoning planning benchmark live under `nureasoning.planning.baselines`. Every baseline implements the benchmark's `trajectory_provider` interface (see `base.py`):

```python
provider(clip_path: str, key_frame_idx: int, ego_state) -> np.ndarray  # (N, 3) global (x, y, yaw)
```

The returned trajectory is sampled at 0.1 s from t=0 (inclusive) to the 5 s planning horizon (51 points), in the global map frame.

| Baseline | Module | Status |
| --- | --- | --- |
| nuVLA (this devkit) | `nureasoning.nuvla.trajectory_provider` | Implemented (train with `python -m nureasoning.nuvla.train`) |
| Constant velocity | `nureasoning.planning.baselines.constant_velocity` | Implemented (sanity-check lower bound) |
| UniAD | `nureasoning.planning.baselines.uniad` | Placeholder |
| DiffusionDrive | `nureasoning.planning.baselines.diffusion_drive` | Implemented |

## Using a baseline

Evaluate on the validation split (ground truth available) by passing the registered name as `--mode`:

```bash
python -m nureasoning.planning.benchmark \
  --data_root ./dataset/data/validation \
  --mode constant_velocity
```

Generate the official challenge submission with a specific planning provider:

```bash
python -m nureasoning.submission.challenge \
  --planning-provider constant_velocity \
  --reasoning-provider stub \
  --data-root ./dataset/data/test \
  --output challenge_submission.json
```

## Adding a new baseline

1. Create a module under `nureasoning/planning/baselines/<name>/` with a class deriving from `BaseTrajectoryProvider` and register it with `@register_baseline("<name>")`.
2. Load your model in `__init__` and implement `__call__` to run inference at the clip's key frame.
3. Convert ego-frame model outputs to the global frame using the key-frame ego pose (see `nureasoning.nuvla.trajectory_provider.VLATrajectoryProvider._ego_to_global`). The local planning benchmark consumes global waypoints; challenge / planning submission JSON is converted to the ego frame at write time.
4. Add the name to `BASELINE_MODES` in `nureasoning/planning/benchmark.py`, then evaluate with `python -m nureasoning.planning.benchmark --mode <name>` and submit with `python -m nureasoning.submission.challenge --planning-provider <name>`.

The placeholder modules (`uniad`, `diffusion_drive`) contain model-specific integration notes.
