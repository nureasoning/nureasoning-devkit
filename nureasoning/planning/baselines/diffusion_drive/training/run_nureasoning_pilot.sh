#!/usr/bin/env bash
set -euo pipefail

# Minimum 50-clip pilot: 40 train clips + 10 held-out validation clips, one
# sample from each. Stage 2 resumes the full Lightning state saved by stage 1.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVKIT_ROOT="${NUREASONING_DEVKIT_ROOT:-/path/to/nureasoning-devkit}"
TRAIN_ROOT="${NUREASONING_TRAIN_ROOT:-/path/to/nureasoning_training_data}"
VAL_ROOT="${NUREASONING_VAL_ROOT:-/path/to/nureasoning_testing_data}"
BASE_EXP_ROOT="${NUREASONING_PILOT_EXP_ROOT:-$SCRIPT_DIR/exp/nureasoning_non_vqa_pilot}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="${NUREASONING_PILOT_RUN_ROOT:-$BASE_EXP_ROOT/$RUN_ID}"
CACHE_PATH="${NUREASONING_PILOT_CACHE_PATH:-$RUN_ROOT/cache}"
TRAIN_LOG_NAME="nureasoning_train"
VAL_LOG_NAME="nureasoning_val"
ANCHOR_PATH="$CACHE_PATH/nureasoning_train_20x8_anchors.npy"
STAGE1_OUTPUT_DIR="$RUN_ROOT/stage1"
STAGE2_OUTPUT_DIR="$RUN_ROOT/stage2_resumed"
VISUALIZATION_OUTPUT="$RUN_ROOT/validation_visualization"
CONDA_ENV="${NAVSIM_CONDA_ENV:-navsim}"
ACCELERATOR="${NAVSIM_ACCELERATOR:-gpu}"
PRECISION="${NAVSIM_PRECISION:-32-true}"
BATCH_SIZE="${NUREASONING_PILOT_BATCH_SIZE:-2}"
NUM_WORKERS="${NUREASONING_PILOT_NUM_WORKERS:-2}"

stage() {
    printf '\n[non-vqa-pilot] %s\n' "$1"
}

for required_dir in "$DEVKIT_ROOT/nureasoning" "$TRAIN_ROOT" "$VAL_ROOT"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "Required directory does not exist: $required_dir" >&2
        exit 1
    fi
done
if [[ "$PRECISION" != "32" && "$PRECISION" != "32-true" ]]; then
    echo "The pilot uses 32-true precision to make the checkpoint/resume result deterministic." >&2
    exit 1
fi

cd "$SCRIPT_DIR"
export NAVSIM_DEVKIT_ROOT="$SCRIPT_DIR"
export NAVSIM_EXP_ROOT="$RUN_ROOT"
export OPENSCENE_DATA_ROOT="$TRAIN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$DEVKIT_ROOT:$SCRIPT_DIR/scripts/nureasoning:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

stage "Run directory: $RUN_ROOT"
stage "Caching 40 training clips (one sample per clip)"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_training_cache.py" \
    --clips_root "$TRAIN_ROOT" \
    --cache_path "$CACHE_PATH" \
    --num_history_frames 4 \
    --num_future_frames 10 \
    --target-frame-rate-hz 2 \
    --frame_stride 4 \
    --max-clips 40 \
    --max-samples 40 \
    --max-samples-per-clip 1 \
    --cache-log-name "$TRAIN_LOG_NAME" \
    --trajectory-only \
    --fail-fast

stage "Caching 10 held-out validation clips (one sample per clip)"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_training_cache.py" \
    --clips_root "$VAL_ROOT" \
    --cache_path "$CACHE_PATH" \
    --num_history_frames 4 \
    --num_future_frames 10 \
    --target-frame-rate-hz 2 \
    --frame_stride 4 \
    --max-clips 10 \
    --max-samples 10 \
    --max-samples-per-clip 1 \
    --cache-log-name "$VAL_LOG_NAME" \
    --trajectory-only \
    --fail-fast

stage "Generating trajectory anchors from the 40 training samples only"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_plan_anchors.py" \
    --cache-path "$CACHE_PATH" \
    --log-name "$TRAIN_LOG_NAME" \
    --output "$ANCHOR_PATH" \
    --num-modes 20

COMMON_TRAIN_ARGS=(
    agent=diffusiondrive_agent
    train_test_split=navtrain
    cache_path="$CACHE_PATH"
    use_cache_without_dataset=True
    force_cache_computation=False
    "train_logs=[$TRAIN_LOG_NAME]"
    "val_logs=[$VAL_LOG_NAME]"
    agent.checkpoint_path=null
    +agent.config.plan_anchor_path="$ANCHOR_PATH"
    +agent.config.image_pretrained=false
    +agent.config.trajectory_only=true
    +agent.config.scheduler_epochs=3
    +agent.config.warmup_epochs=1
    trainer.params.accelerator="$ACCELERATOR"
    +trainer.params.devices=1
    trainer.params.strategy=auto
    trainer.params.precision="$PRECISION"
    trainer.params.fast_dev_run=false
    trainer.params.num_sanity_val_steps=0
    dataloader.params.batch_size="$BATCH_SIZE"
    dataloader.params.num_workers="$NUM_WORKERS"
    dataloader.params.pin_memory=true
    dataloader.params.prefetch_factor=2
    +trainer.params.enable_checkpointing=true
    +trainer.params.enable_progress_bar=true
    +trainer.params.log_every_n_steps=1
)

stage "Stage 1/2: training epoch 1 from scratch"
conda run --no-capture-output -n "$CONDA_ENV" python \
    navsim/planning/script/run_training.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    experiment_name=nureasoning_non_vqa_pilot_stage1 \
    output_dir="$STAGE1_OUTPUT_DIR" \
    resume_checkpoint_path=null \
    trainer.params.max_epochs=1

STAGE1_CHECKPOINT="$(find "$STAGE1_OUTPUT_DIR" -type f -name '*.ckpt' -print | sort | tail -n 1)"
if [[ -z "$STAGE1_CHECKPOINT" || ! -f "$STAGE1_CHECKPOINT" ]]; then
    echo "Stage 1 did not create a checkpoint under $STAGE1_OUTPUT_DIR" >&2
    exit 1
fi
stage "Stage 1 checkpoint: $STAGE1_CHECKPOINT"

stage "Stage 2/2: resuming epoch 1 and continuing through epoch 3"
conda run --no-capture-output -n "$CONDA_ENV" python \
    navsim/planning/script/run_training.py \
    "${COMMON_TRAIN_ARGS[@]}" \
    experiment_name=nureasoning_non_vqa_pilot_stage2 \
    output_dir="$STAGE2_OUTPUT_DIR" \
    resume_checkpoint_path="'$STAGE1_CHECKPOINT'" \
    trainer.params.max_epochs=3

FINAL_CHECKPOINT="$(find "$STAGE2_OUTPUT_DIR" -type f -name '*.ckpt' -print | sort | tail -n 1)"
if [[ -z "$FINAL_CHECKPOINT" || ! -f "$FINAL_CHECKPOINT" ]]; then
    echo "Resumed training did not create a checkpoint under $STAGE2_OUTPUT_DIR" >&2
    exit 1
fi
stage "Resumed checkpoint: $FINAL_CHECKPOINT"

stage "Verifying finite validation loss with an overall decrease"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/verify_pilot_metrics.py" \
    --stage1-log-dir "$STAGE1_OUTPUT_DIR/lightning_logs/version_0" \
    --resumed-log-dir "$STAGE2_OUTPUT_DIR/lightning_logs/version_0"

stage "Running held-out trajectory loss and devkit visualization"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/validate_cached_sample.py" \
    --cache-path "$CACHE_PATH" \
    --log-name "$VAL_LOG_NAME" \
    --plan-anchor-path "$ANCHOR_PATH" \
    --checkpoint-path "$FINAL_CHECKPOINT" \
    --device cuda \
    --trajectory-only \
    --sample-index 0 \
    --clips-root "$VAL_ROOT" \
    --visualization-output "$VISUALIZATION_OUTPUT"

stage "Pilot passed. Outputs are under $RUN_ROOT"
find "$RUN_ROOT" -type f \( -name '*.ckpt' -o -name 'planning_result.png' -o -name 'planning_result.json' \) -print | sort
