# DiffusionDrive on nuReasoning: Training and Inference

This folder holds the nuReasoning DiffusionDrive baseline.

| Path | Purpose |
| --- | --- |
| `model.py` | Inference: registers the `diffusion_drive` planning baseline |
| `diffusiondrive_nureasoning_modifications.md` | What was changed in DiffusionDrive and the hyperparameters used |
| `training/navsim/` | Vendored DiffusionDrive `navsim/` package at upstream commit `9b52ed0` with the nuReasoning changes already applied; the launchers run from `training/` and import it |
| `training/reference/navsim_nureasoning.patch` | The same nuReasoning changes as a patch against upstream DiffusionDrive `9b52ed0`, for review or for rebuilding the checkout from scratch |
| `training/scripts/nureasoning/` | Clip converter, cache and anchor generation, validation, benchmark and challenge runners |
| `training/run_*.sh` | Launchers: `run_training.sh` (4 s, stage 1), `run_training_5s.sh` (5 s fine-tune, stage 2), `run_nureasoning_pilot.sh` (50-clip pilot), `run_official_benchmark_5s.sh` |
| `training/docs/nureasoning_training_validation.md` | Detailed stage-1 workflow, resume behavior and troubleshooting |

Training needs a DiffusionDrive checkout with the nuReasoning changes applied.
Inference needs the same checkout, because `model.py` imports its `navsim`
package. The `training/` folder here is already such a checkout, so you can use
it directly (set `DIFFUSIONDRIVE_ROOT=$DF/training`) and skip section 1.
Paths written `/path/to/...` are placeholders to replace.

## 1. Set up the DiffusionDrive checkout (optional)

Only needed if you want a checkout separate from `training/`, for example to
confirm the patch against a clean upstream tree. Otherwise use `training/`
as is.

```bash
git clone https://github.com/hustvl/DiffusionDrive.git /path/to/DiffusionDrive
cd /path/to/DiffusionDrive
git checkout 9b52ed0                 # base the patch was made against

DF=/path/to/nureasoning-devkit/nureasoning/planning/baselines/diffusion_drive
git apply "$DF/training/reference/navsim_nureasoning.patch"
cp -r "$DF"/training/scripts "$DF"/training/run_*.sh "$DF"/training/docs .
```

Create the `navsim` conda environment by following upstream `docs/install.md`
(Python 3.9, torch 2.0.1+cu118, pytorch-lightning 2.2.1, nuplan-devkit 1.2.0,
`diffusers`, `einops`), then install the devkit into it
(`pip install -e /path/to/nureasoning-devkit`). The launchers call
`conda run -n navsim`.

Check `git apply` succeeded before continuing. Without the patch, training
fails on the `trajectory_only` and `image_pretrained` config fields, and
inference fails to load the checkpoint.

## 2. Data needed

- Non-VQA training clips (camera, ego state, trajectory) and validation clips.
- The devkit checkout (the launchers add it to `PYTHONPATH`).

Point the launchers at them with environment variables:

```bash
export NUREASONING_DEVKIT_ROOT=/path/to/nureasoning-devkit
export NUREASONING_TRAIN_ROOT=/path/to/nureasoning_training_data
export NUREASONING_VAL_ROOT=/path/to/nureasoning_testing_data
```

The clips contain no LiDAR. The model is trained camera + ego status only, with a
zero LiDAR tensor kept as a shape placeholder. The cache needs roughly
60-65 GiB under `data/nureasoning_non_vqa_cache/`.

## 3. Stage 1: 4 s model from scratch

```bash
cd /path/to/DiffusionDrive
./run_nureasoning_pilot.sh 2>&1 | tee exp/nureasoning_non_vqa_pilot_console.log   # optional 50-clip check
NUREASONING_EXPERIMENT_NAME=nureasoning_non_vqa_full_stable \
NUREASONING_RESUME_CHECKPOINT=none \
./run_training.sh 2>&1 | tee -a exp/nureasoning_non_vqa_full_stable_console.log
```

`run_training.sh` caches the training and validation splits, generates the
20x8 anchors from training targets, then trains. Re-running the same command
resumes from the newest checkpoint of that experiment.

Stage-1 settings: 100 epochs, warmup 3, lr 6e-4, batch 32 x accumulation 2,
gradient clip 1.0, 32-bit precision, 8 poses over 4 s. Output goes to
`exp/<experiment>/<timestamp>/` with `checkpoints/last.ckpt`.

## 4. Stage 2: 5 s fine-tune

`run_training_5s.sh` caches 10-pose targets under the name
`transfuser_target_5s` in the same cache directory, generates the 20x10 anchors,
and warm-starts from the stage-1 `last.ckpt`:

```bash
export NUREASONING_INIT_CHECKPOINT=/path/to/DiffusionDrive/exp/nureasoning_non_vqa_full_stable/<timestamp>/checkpoints/last.ckpt
./run_training_5s.sh 2>&1 | tee -a exp/nureasoning_non_vqa_5s_finetune_console.log
```

Stage-2 settings: 20 epochs, warmup 1, lr 1e-4, batch 32 x accumulation 2,
gradient clip 1.0, 10 poses over 5 s. The launcher's experiment name defaults
to `nureasoning_non_vqa_5s_finetune`. Ten tensors in the trajectory
head are re-initialized because their shapes change (the console log lists
them). Everything else is copied from stage 1.

Useful overrides (both launchers): `NUREASONING_MAX_EPOCHS`,
`NUREASONING_BATCH_SIZE`, `NUREASONING_ACCUMULATE_GRAD_BATCHES`,
`NUREASONING_LEARNING_RATE`, `NUREASONING_RESUME_CHECKPOINT`
(`auto`, `none`, or a path), `NUREASONING_EXPERIMENT_NAME`. If 32 runs out of
GPU memory, use batch 16 with accumulation 4 to keep the effective batch of 64.

Watch `train/loss_epoch` and `val/loss_epoch`. Training stops with a
`FloatingPointError` on a non-finite loss.

## 5. Check a checkpoint

Run one cached validation sample through strict checkpoint loading and the
devkit visualization:

```bash
cd /path/to/DiffusionDrive
CACHE=$PWD/data/nureasoning_non_vqa_cache
export PYTHONPATH=/path/to/nureasoning-devkit:$PWD/scripts/nureasoning:$PWD
conda run --no-capture-output -n navsim python scripts/nureasoning/validate_cached_sample.py \
  --cache-path "$CACHE" --log-name nureasoning_val \
  --plan-anchor-path "$CACHE/nureasoning_train_20x10_anchors.npy" \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --device cuda --trajectory-only --sample-index 0 \
  --clips-root /path/to/nureasoning_testing_data \
  --visualization-output exp/nureasoning_production_validation
```

The 5 s model needs the 20x10 anchors; the 4 s model needs 20x8. Mixing them
fails the strict load.

## 6. Inference with the devkit baseline

`model.py` registers `diffusion_drive`. It reads its configuration from the
environment because the devkit builds baselines with no arguments:

```bash
export DIFFUSIONDRIVE_ROOT=/path/to/DiffusionDrive          # required
export DIFFUSIONDRIVE_CHECKPOINT=/path/to/checkpoint.ckpt   # default: root/exp/nureasoning_non_vqa_5s_finetune/<run>/checkpoints/last.ckpt
export DIFFUSIONDRIVE_PLAN_ANCHORS=/path/to/nureasoning_train_20x10_anchors.npy
export DIFFUSIONDRIVE_DEVICE=cuda                           # or cpu / auto
```

Use the `navsim` conda environment. `model.py` supports only the 5 s model
(10 poses at 0.5 s, resampled to 51 points at 0.1 s). It does not load the
stage-1 checkpoint.

**Challenge submission**

```bash
python -m nureasoning.submission.challenge \
  --data-root /path/to/challenge/data/test \
  --planning-provider diffusion_drive
```

**Planning benchmark** (`nureasoning.planning.benchmark --mode` only accepts
`gt` and `vla`, so use the runner script):

```bash
cd /path/to/DiffusionDrive
export PYTHONPATH=/path/to/nureasoning-devkit:$PWD/scripts/nureasoning:$PWD
python scripts/nureasoning/materialize_planning_eval_data.py \
  --source-root /path/to/validation_archives \
  --output-root data/nureasoning_validation_planning_eval --key-frame-index 100
python scripts/nureasoning/run_diffusiondrive_official_benchmark.py \
  --data-root data/nureasoning_validation_planning_eval \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --plan-anchor-path /path/to/nureasoning_train_20x10_anchors.npy \
  --output-dir exp/nureasoning_diffusiondrive_official_benchmark_5s \
  --key-frame-index 100 --device cuda
```

`run_official_benchmark_5s.sh` wraps both steps; edit its path variables at the
top first.

## 7. Notes

- Checkpoints (about 700 MB each), caches and anchors are not part of this
  folder. Share them separately.
- Runs store only their Hydra configs, not source. Reproducing a checkpoint
  means using the patch and launcher settings documented here.
- Provider output was checked to match
  `training/scripts/nureasoning/diffusiondrive_trajectory_provider.py` exactly
  on one validation clip (CPU). The full benchmark and challenge flows have not
  been re-run through `model.py`.
