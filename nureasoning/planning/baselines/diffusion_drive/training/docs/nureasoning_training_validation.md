# nuReasoning DiffusionDrive Training and Validation

This guide covers the local camera/trajectory workflow for the non-VQA
nuReasoning data. It uses:

- training clips: `/path/to/nureasoning_training_data`
- validation clips: `/path/to/nureasoning_testing_data`
- nuReasoning devkit: `/path/to/nureasoning-devkit`
- writable repository: `/path/to/DiffusionDrive`

The source datasets and devkit are read-only. Caches, checkpoints, logs, and
visualizations are written only under `DiffusionDrive`.

## What the workflow trains

The current mode uses camera and vehicle-status sensor features with trajectory
targets. Agent-detection and BEV-semantic targets and losses are disabled. The
raw clips do not contain LiDAR files, so a zero LiDAR tensor is retained only as
the shape placeholder expected by the TransFuser backbone.

Raw nuReasoning clips are converted to DiffusionDrive tensors before training.
This one-time cache avoids decoding images and rebuilding scene windows during
every epoch. With the default `frame_stride=4`, a typical 41-frame, 2 Hz clip
produces seven samples. The full cache is expected to require roughly 60-65 GiB.

## Requirements

- The `navsim` Conda environment must exist.
- The three paths listed above must be mounted in the workspace.
- Full training requires an NVIDIA GPU. Cache generation and anchor generation
  do not require a GPU.
- Do not run two cache generators against the same cache directory at once.

The repository scripts invoke `conda run -n navsim`, so activating the
environment first is optional.

## Recommended pilot

Before a full run, execute the tested 50-clip pilot:

```bash
cd /path/to/DiffusionDrive
./run_nureasoning_pilot.sh 2>&1 | tee exp/nureasoning_non_vqa_pilot_console.log
```

The pilot performs all of the following:

1. Caches one sample from each of 40 training clips.
2. Caches one sample from each of 10 held-out validation clips.
3. Generates 20 trajectory anchors from training targets only.
4. Trains epoch 1 from scratch and saves a checkpoint.
5. Resumes the full Lightning state and continues through epoch 3.
6. Checks that validation loss is finite and improves overall.
7. Produces a devkit validation PNG and JSON report.

Pilot outputs are stored under `exp/nureasoning_non_vqa_pilot/<run-id>/`.

## Full training

Run the production launcher from a GPU workspace:

```bash
cd /path/to/DiffusionDrive
./run_training.sh 2>&1 | tee -a exp/nureasoning_non_vqa_full_console.log
```

The launcher runs four stages in order:

1. Cache the complete non-VQA training split.
2. Cache the complete non-VQA validation split.
3. Generate 20 trajectory anchors from the training cache only.
4. Train DiffusionDrive and validate after every epoch.

Cache generation is restart-safe: compatible completed samples are reused.
The anchor file is not created until both caching stages have completed:

```text
data/nureasoning_non_vqa_cache/nureasoning_train_20x8_anchors.npy
```

Training defaults are:

- 100 epochs
- physical batch size 32
- gradient accumulation over 2 batches
- effective optimizer batch size 64
- learning rate `6e-4`
- gradient clipping with max norm `1.0`
- `32-true` precision
- 4 DataLoader workers
- one GPU
- checkpoint every 500 training steps
- retain the three most recent step checkpoints plus `last.ckpt`
- automatic resume from the newest production checkpoint

Training runs are stored under:

```text
exp/nureasoning_non_vqa_full/<timestamp>/
```

Step checkpoints are stored in that run's `checkpoints/` directory. Each
checkpoint contains model, optimizer, scheduler, epoch, and global-step state.

### Resume behavior

By default, `run_training.sh` finds the newest checkpoint anywhere under
`exp/nureasoning_non_vqa_full/` and resumes it. Tiny smoke and pilot checkpoints
live under different experiment directories and are not selected.

Start a clean training lineage after a non-finite run as follows:

1. Stop the old training job with `Ctrl-C` in its terminal.
2. Verify that no training process remains:

```bash
pgrep -af 'run_training.sh|run_training.py'
```

The command must print no matching training processes before continuing. The
launcher also refuses to start while another NAVSIM training process is active.

3. Launch from random model and optimizer initialization under a new experiment
   name, isolated from the old NaN-contaminated checkpoints:

```bash
cd /path/to/DiffusionDrive
NUREASONING_EXPERIMENT_NAME=nureasoning_non_vqa_full_stable \
NUREASONING_RESUME_CHECKPOINT=none \
./run_training.sh 2>&1 | tee -a exp/nureasoning_non_vqa_full_stable_console.log
```

This reuses the verified cache and regenerates anchors, but initializes the
model, optimizer, and scheduler from scratch. Do not delete or rebuild the
60-65 GiB cache. The separate experiment name prevents automatic resume from
the old run.

After the new run has produced a finite checkpoint, resume that lineage with the
same experiment name and the default automatic checkpoint selection:

```bash
NUREASONING_EXPERIMENT_NAME=nureasoning_non_vqa_full_stable \
./run_training.sh 2>&1 | tee -a exp/nureasoning_non_vqa_full_stable_console.log
```

Resume a specific production checkpoint:

```bash
NUREASONING_RESUME_CHECKPOINT=/absolute/path/to/checkpoint.ckpt ./run_training.sh
```

List production checkpoints by modification time before choosing one:

```bash
find -L exp/nureasoning_non_vqa_full -type f -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort -r
```

### Cache on a non-GPU workspace

It is safe to start `run_training.sh` in a CPU-only workspace. It will finish
the training cache, validation cache, and anchors before attempting the GPU
stage. After this message appears, all CPU preparation is complete:

```text
[non-vqa-training] Starting camera/trajectory training ...
```

At that point the GPU stage can be stopped. Verify the anchor exists:

```bash
ls -lh data/nureasoning_non_vqa_cache/nureasoning_train_20x8_anchors.npy
```

Then rerun `run_training.sh` from the GPU workspace. It will scan and reuse the
completed cache before starting or resuming training. Both workspaces must see
the same filesystem.

## Monitoring

The launchers use unbuffered output, so caching and Lightning progress appear
directly in the terminal. If the command was started with `tee`, follow its log
from another terminal:

```bash
tail -f exp/nureasoning_non_vqa_full_console.log
```

Monitor the GPU:

```bash
watch -n 2 nvidia-smi
```

Check whether a cache or training process is active in the current workspace:

```bash
pgrep -af 'run_training.sh|generate_training_cache.py|run_training.py'
```

Count currently cached trajectory targets:

```bash
find data/nureasoning_non_vqa_cache/nureasoning_train \
  -type f -name 'transfuser_target.gz' | wc -l

find data/nureasoning_non_vqa_cache/nureasoning_val \
  -type f -name 'transfuser_target.gz' | wc -l
```

During training, watch `train/loss_epoch` and `val/loss_epoch`. Both must remain
finite. A tiny validation subset can be noisy, so use the multi-epoch trend
rather than requiring every epoch to improve.

## Validate a production checkpoint

Training already evaluates the complete cached validation split at the end of
every epoch. Use the following evaluation-only command for one selected sample,
strict checkpoint loading, finite trajectory loss, and a devkit visualization.

First locate and assign the checkpoint:

```bash
cd /path/to/DiffusionDrive
find -L exp/nureasoning_non_vqa_full -type f -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort -r | head

CHECKPOINT=/absolute/path/to/selected/checkpoint.ckpt
```

Then run validation:

```bash
cd /path/to/DiffusionDrive

CACHE="$PWD/data/nureasoning_non_vqa_cache"
ANCHORS="$CACHE/nureasoning_train_20x8_anchors.npy"
VAL_ROOT=/path/to/nureasoning_testing_data
DEVKIT_ROOT=/path/to/nureasoning-devkit
OUTPUT="$PWD/exp/nureasoning_production_validation"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DEVKIT_ROOT:$PWD/scripts/nureasoning:$PWD${PYTHONPATH:+:$PYTHONPATH}"

conda run --no-capture-output -n navsim python \
  scripts/nureasoning/validate_cached_sample.py \
  --cache-path "$CACHE" \
  --log-name nureasoning_val \
  --plan-anchor-path "$ANCHORS" \
  --checkpoint-path "$CHECKPOINT" \
  --device cuda \
  --trajectory-only \
  --sample-index 0 \
  --clips-root "$VAL_ROOT" \
  --visualization-output "$OUTPUT"
```

Successful validation prints JSON containing:

- `status: passed`
- the loaded checkpoint tensor count
- finite trajectory loss
- predicted trajectory shape `[1, 8, 3]`
- the selected validation token
- devkit component and planning scores

The visualization artifacts are:

```text
exp/nureasoning_production_validation/planning_result.png
exp/nureasoning_production_validation/planning_result.json
```

Use `--sample-index N` to inspect another cached validation sample. Use
`--device cpu` only when GPU inference is unavailable; it will be slower.

`run_nureasoning_validation_smoke.sh` remains useful for a one-sample
architecture smoke test, but its defaults point to the separate COCO devkit
validation sample. The direct command above is the production non-VQA
validation path.

## Configuration overrides

Set environment variables before `run_training.sh` to change its defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NUREASONING_TRAIN_ROOT` | `.../nureasoning_training_data` | Raw non-VQA training clips |
| `NUREASONING_VAL_ROOT` | `.../nureasoning_testing_data` | Raw non-VQA validation clips |
| `NUREASONING_CACHE_PATH` | `data/nureasoning_non_vqa_cache` | Prepared tensor cache |
| `NUREASONING_EXP_ROOT` | `exp` | Training outputs |
| `NUREASONING_EXPERIMENT_NAME` | `nureasoning_non_vqa_full` | Checkpoint lineage/output subdirectory |
| `NUREASONING_MAX_EPOCHS` | `100` | Total epoch target, including resumed epochs |
| `NUREASONING_WARMUP_EPOCHS` | `3` | Scheduler warmup |
| `NUREASONING_BATCH_SIZE` | `32` | Physical per-device batch size |
| `NUREASONING_ACCUMULATE_GRAD_BATCHES` | `2` | Batches per optimizer update; effective batch is physical batch times this value |
| `NUREASONING_LEARNING_RATE` | `6e-4` | Base optimizer learning rate |
| `NUREASONING_GRADIENT_CLIP_VAL` | `1.0` | Gradient max norm |
| `NUREASONING_NUM_WORKERS` | `4` | DataLoader workers |
| `NUREASONING_FRAME_STRIDE` | `4` | Spacing between eligible cached windows |
| `NUREASONING_CHECKPOINT_EVERY_N_STEPS` | `500` | Step-checkpoint interval |
| `NUREASONING_CHECKPOINT_SAVE_TOP_K` | `3` | Number of rolling step checkpoints to retain |
| `NUREASONING_RESUME_CHECKPOINT` | `auto` | `auto`, `none`, or an explicit checkpoint |
| `NAVSIM_PRECISION` | `32-true` | Lightning precision |
| `NAVSIM_CONDA_ENV` | `navsim` | Conda environment name |

If physical batch 32 runs out of GPU memory, keep the effective batch at 64 by
using physical batch 16 with four-step accumulation:

```bash
NUREASONING_BATCH_SIZE=16 \
NUREASONING_ACCUMULATE_GRAD_BATCHES=4 \
./run_training.sh
```

Example shorter run from scratch:

```bash
NUREASONING_MAX_EPOCHS=10 \
NUREASONING_WARMUP_EPOCHS=1 \
NUREASONING_RESUME_CHECKPOINT=none \
./run_training.sh
```

## Common problems

### Anchor file does not exist

Anchor generation has not run yet. Wait for both training and validation cache
stages to print `Done`, then for the launcher to print `Saved 20 anchors`.

### A cache job was interrupted

Rerun the same command. Completed compatible samples are skipped. Do not delete
the partial cache and do not start a second writer concurrently.

### Training starts from an unexpected checkpoint

The default is automatic resume. Inspect the path printed after `Resuming
camera/trajectory training`. Use `NUREASONING_RESUME_CHECKPOINT=none` to force
scratch training or provide the exact desired checkpoint path.

### CUDA is unavailable

Cache preparation can continue on CPU. Move to a GPU workspace before the
training stage, or use `--device cpu` only for the single-sample validator.

### Checkpoint validation fails to load completely

Use the training cache's anchor file and include `--trajectory-only`. Do not use
anchors from a smoke or pilot cache with a production checkpoint.
