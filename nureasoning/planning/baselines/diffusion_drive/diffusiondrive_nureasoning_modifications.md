# DiffusionDrive on nuReasoning: Modifications and Hyperparameters

Summary of what was changed in this checkout relative to upstream DiffusionDrive
(NAVSIM branch), and the hyperparameters of the checkpoint currently used for
benchmark and challenge inference. Sources: `git diff` of `navsim/`,
`nureasoning_training_validation.md`,
`train_eval.md`, `../notes/onboarding.md`,
`../notes/diffusiondrive_setup.md`, and the
Hydra `overrides.yaml` of each run.

## 1. Training lineage

| Stage | Run | Horizon | Init | LR | Epochs | Warmup |
| --- | --- | --- | --- | --- | --- | --- |
| 1. 4 s from scratch | `exp/nureasoning_non_vqa_full_stable/2026.09.30.00.37.41` -> `.02.22.56` -> `.05.05.03` (resumed twice via `resume_checkpoint_path`) | 4 s, 8 poses | random | 6e-4 | 100 | 3 |
| 2. 5 s fine-tune | `exp/nureasoning_non_vqa_5s_finetune/2026.10.01.06.21.20` | 5 s, 10 poses | stage-1 `last.ckpt` (shape-compatible tensors only) | 1e-4 | 20 | 1 |

**Fine-tuned from:** `exp/nureasoning_non_vqa_full_stable/2026.09.30.05.05.03/checkpoints/last.ckpt`
(a link to `epoch=099-step=108000.ckpt`, the end of the 100-epoch 4 s run).
That run was never initialized from ImageNet weights (`image_pretrained=false`)
and was not itself a fine-tune of the public NAVSIM DiffusionDrive checkpoint.
Within stage 1, `.00.37.41` left no checkpoints, `.02.22.56` trained from
scratch to epoch 12, and `.05.05.03` resumed it to epoch 99.

Warm start details (from the stage-2 console log): 753 tensors were copied; 10
shape-mismatched tensors kept their fresh initialization, all in the trajectory
head and tied to the pose count (8 -> 10):
`plan_anchor`, `plan_anchor_encoder.0.weight` (512 -> 640 inputs), and in each
of the two diffusion decoder layers `cross_bev_attention.attention_weights.{weight,bias}`
and `task_decoder.plan_reg_branch.4.{weight,bias}`. The backbone, BEV
fusion, query decoder and other head layers carry over unchanged.

The earlier `nureasoning_non_vqa_full` lineage hit non-finite losses and was
abandoned. `*_stable` restarted from random init with the NaN guard below.
Stage 2 is the checkpoint used for inference. Its `last.ckpt` is a link to
`epoch=019-step=21500.ckpt`, so the benchmark and challenge defaults are the same weights.

## 2. Hyperparameters of the 5 s checkpoint

Launcher: `run_training_5s.sh`. Values are from the run's `overrides.yaml`.

| Group | Setting |
| --- | --- |
| Data | cache `data/nureasoning_non_vqa_cache`; 69,230 train / 8,642 val samples; `use_cache_without_dataset=True`; frame stride 4 |
| Targets | `transfuser_target_5s`; 5 s at 0.5 s steps (10 poses) |
| Anchors | `nureasoning_train_20x10_anchors.npy` (20 modes x 10 poses x xy), generated from training targets only |
| Model | `trajectory_only=true`; `image_pretrained=false`; ResNet-34 image and LiDAR encoders |
| Optimization | lr 1e-4; batch 32 x accumulate 2 (effective 64); grad clip 1.0 (norm); `32-true`; 1 GPU; 4 workers |
| Schedule | 20 epochs, `scheduler_epochs=20`, `warmup_epochs=1`, cosine decay to `min_lr=1e-6` |
| Checkpointing | every 500 steps, `save_top_k=3`, monitor `step` (max), plus `last.ckpt` |
| Seed | 0 |

Stage-1 differences: lr 6e-4, 100 epochs, warmup 3, 8 poses, 20x8 anchors,
default target cache name `transfuser_target`.

## 3. Code modifications (all under `navsim/`)

**Trajectory-only mode** (camera + ego status in, trajectory out)
- `transfuser_config.py`: new `trajectory_only` flag. Agent-detection and
  BEV-semantic heads and losses are skipped when set.
- `transfuser_model_v2.py`: BEV-semantic and agent heads only run when
  `trajectory_only` is false.
- `transfuser_loss.py`: with `trajectory_only`, loss is
  `trajectory_weight * trajectory_loss`, plus the trajectory head's own loss terms.
- `transfuser_features.py`: target builder returns only `{"trajectory"}` and
  casts to float32.

**Missing LiDAR**
- `transfuser_features.py`: nuReasoning clips have no LiDAR for many frames, so a
  `None` point cloud becomes an empty cloud and the BEV histogram is all zeros.
  The backbone still receives a zero LiDAR tensor as a shape placeholder.

**Horizon and anchors (4 s -> 5 s)**
- `transfuser_model_v2.py` (`TrajectoryHead`): validates the anchor shape
  `(modes, num_poses, 2)` and sizes `plan_anchor_encoder` as `num_poses * 64`
  (was hard-coded 512 for 8 poses).
- `transfuser_config.py`: `target_cache_name` keeps 4 s and 5 s caches separate
  (`transfuser_features.py` uses it as the builder's unique name).

**Warm start**
- `transfuser_agent.py` (`init_from_pretrained`): strips `agent.`, loads only
  tensors that exist and match in shape, and prints skipped mismatches. This is
  what lets the 8-pose checkpoint initialize the 10-pose model.
- `transfuser_config.py` and `transfuser_backbone.py`: `image_pretrained` flag
  (set false so weights come from the checkpoint, not timm).

**Scheduler**
- `transfuser_config.py` and `transfuser_agent.py`: `scheduler_epochs` and
  `warmup_epochs` replace the hard-coded 100 / 3. Keep `scheduler_epochs` equal
  to `trainer.params.max_epochs`.
- `modules/scheduler.py` (`WarmupCosLR`): scales each param group from its own
  base LR, so per-group multipliers (such as the 0.5x image-backbone group)
  survive warmup and decay. The old code only applied `lr_scale` when present.

**Training harness**
- `run_training.py` and `default_training.yaml`: `resume_checkpoint_path`
  restores full Lightning state; `checkpoint.*` adds step-based checkpoints.
- `agent_lightning_module.py`: raises `FloatingPointError` with per-term losses
  on a non-finite loss instead of continuing silently.

## 4. nuReasoning data pipeline (outside `navsim/`)

Raw clips are not read at train time. `scripts/nureasoning/` converts them once
into the NAVSIM cache:
- `nureasoning_clip_converter.py`, `nureasoning_map_adapter.py`,
  `nureasoning_pkl_loader.py`: clip -> NAVSIM `Scene`.
- `generate_training_cache.py`: writes `transfuser_feature.gz` and
  `transfuser_target[_5s].gz`; restart-safe.
- `generate_plan_anchors.py`: k-means anchors from training targets.
- `validate_cached_sample.py`: strict checkpoint load, finite-loss check, devkit
  visualization.

Command ordering: `[LEFT, STRAIGHT, RIGHT, UNKNOWN]`. Only `GO_STRAIGHT` has been
seen in the data so far.

## 5. Inference: devkit baseline

`nureasoning-devkit/nureasoning/planning/baselines/diffusion_drive/model.py`
registers `diffusion_drive` and reproduces the training-time settings above.
It was previously a placeholder that raised `NotImplementedError`.

- Inputs: 4 history frames at 2 Hz ending at the key frame; front, front-left
  and front-right cameras only; LiDAR disabled when the clip has none.
- Output: 10 ego-frame poses at 0.5 s, resampled to 51 points at 0.1 s
  (linear in x, y; unwrapped linear in yaw), then rotated into the global frame.
- Checkpoint is loaded with `strict=True`. Wrong anchors fail loudly.
- Config via env vars (the devkit builds the provider with no arguments):
  `DIFFUSIONDRIVE_ROOT`, `DIFFUSIONDRIVE_CHECKPOINT`,
  `DIFFUSIONDRIVE_PLAN_ANCHORS`, `DIFFUSIONDRIVE_DEVICE`.
- Run it with the `navsim` conda env, e.g.
  `python -m nureasoning.submission.challenge --planning-provider diffusion_drive ...`.
- The same logic lives in `scripts/nureasoning/diffusiondrive_trajectory_provider.py`.
  The two were checked to give identical output on one validation clip.

## 6. Caveats

- **Source is not snapshotted per run.** Runs store only Hydra configs. The
  `navsim/` changes above are uncommitted, so a run's code is whatever the tree
  held at launch time.
- **Unverified data assumptions** from `notes/onboarding.md` (ego-state and
  annotation schema, category map) were written before real data was seen. The
  onboarding doc's "Current state of the data" still describes the old 3-clip
  cache and is stale.
- **Stage-1 anchors** (`20x8`) are incompatible with the 5 s model. Never mix
  anchor files and checkpoints across horizons.
- `nureasoning_training_validation.md` documents the stage-1 defaults (lr 6e-4,
  100 epochs, 20x8 anchors). The 5 s settings are only in `run_training_5s.sh`.
