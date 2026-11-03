#!/usr/bin/env bash
set -euo pipefail

# Full non-VQA camera/trajectory training. Cache generation and model training
# are restart-safe. Smoke checkpoints produced from tiny test datasets are not
# reused by automatic resume.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVKIT_ROOT="${NUREASONING_DEVKIT_ROOT:-/path/to/nureasoning-devkit}"
# Patched DiffusionDrive checkout (9b52ed0 + reference/navsim_nureasoning.patch).
# Defaults to this folder, which is the case after copying the launchers into the checkout.
DD_ROOT="${DIFFUSIONDRIVE_ROOT:-$SCRIPT_DIR}"
TRAIN_ROOT="${NUREASONING_TRAIN_ROOT:-/path/to/nureasoning_training_data}"
VAL_ROOT="${NUREASONING_VAL_ROOT:-/path/to/nureasoning_testing_data}"
CACHE_PATH="${NUREASONING_CACHE_PATH:-$SCRIPT_DIR/data/nureasoning_non_vqa_cache}"
EXP_ROOT="${NUREASONING_EXP_ROOT:-$SCRIPT_DIR/exp}"
TRAIN_LOG_NAME="nureasoning_train"
VAL_LOG_NAME="nureasoning_val"
ANCHOR_PATH="$CACHE_PATH/nureasoning_train_20x8_anchors.npy"
CONDA_ENV="${NAVSIM_CONDA_ENV:-navsim}"
ACCELERATOR="${NAVSIM_ACCELERATOR:-gpu}"
PRECISION="${NAVSIM_PRECISION:-32-true}"
MAX_EPOCHS="${NUREASONING_MAX_EPOCHS:-100}"
WARMUP_EPOCHS="${NUREASONING_WARMUP_EPOCHS:-3}"
BATCH_SIZE="${NUREASONING_BATCH_SIZE:-32}"
ACCUMULATE_GRAD_BATCHES="${NUREASONING_ACCUMULATE_GRAD_BATCHES:-2}"
LEARNING_RATE="${NUREASONING_LEARNING_RATE:-6e-4}"
GRADIENT_CLIP_VAL="${NUREASONING_GRADIENT_CLIP_VAL:-1.0}"
NUM_WORKERS="${NUREASONING_NUM_WORKERS:-4}"
FRAME_STRIDE="${NUREASONING_FRAME_STRIDE:-4}"
CHECKPOINT_EVERY_N_STEPS="${NUREASONING_CHECKPOINT_EVERY_N_STEPS:-500}"
CHECKPOINT_SAVE_TOP_K="${NUREASONING_CHECKPOINT_SAVE_TOP_K:-3}"
EXPERIMENT_NAME="${NUREASONING_EXPERIMENT_NAME:-nureasoning_non_vqa_full}"

stage() {
    printf '\n[non-vqa-training] %s\n' "$1"
}

for required_dir in "$DD_ROOT/navsim" "$DEVKIT_ROOT/nureasoning" "$TRAIN_ROOT" "$VAL_ROOT"; do
    if [[ ! -d "$required_dir" ]]; then
        echo "Required directory does not exist: $required_dir" >&2
        exit 1
    fi
done
if (( MAX_EPOCHS <= WARMUP_EPOCHS )); then
    echo "NUREASONING_MAX_EPOCHS must be greater than NUREASONING_WARMUP_EPOCHS." >&2
    exit 1
fi
if [[ ! "$CHECKPOINT_EVERY_N_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUREASONING_CHECKPOINT_EVERY_N_STEPS must be a positive integer." >&2
    exit 1
fi
for integer_setting in BATCH_SIZE ACCUMULATE_GRAD_BATCHES NUM_WORKERS CHECKPOINT_SAVE_TOP_K; do
    value="${!integer_setting}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$integer_setting must be a positive integer (received: $value)." >&2
        exit 1
    fi
done

if pgrep -f '[p]ython .*navsim/planning/script/run_training.py' >/dev/null; then
    echo "Another NAVSIM training process is already active. Stop it before starting this run:" >&2
    pgrep -af '[p]ython .*navsim/planning/script/run_training.py' >&2
    exit 1
fi

cd "$DD_ROOT"
export NAVSIM_DEVKIT_ROOT="$DD_ROOT"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="$TRAIN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$DEVKIT_ROOT:$SCRIPT_DIR/scripts/nureasoning:$DD_ROOT${PYTHONPATH:+:$PYTHONPATH}"

stage "Caching the full non-VQA training split from $TRAIN_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_training_cache.py" \
    --clips_root "$TRAIN_ROOT" \
    --cache_path "$CACHE_PATH" \
    --num_history_frames 4 \
    --num_future_frames 10 \
    --target-frame-rate-hz 2 \
    --frame_stride "$FRAME_STRIDE" \
    --cache-log-name "$TRAIN_LOG_NAME" \
    --trajectory-only \
    --fail-fast

stage "Caching the full non-VQA validation split from $VAL_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_training_cache.py" \
    --clips_root "$VAL_ROOT" \
    --cache_path "$CACHE_PATH" \
    --num_history_frames 4 \
    --num_future_frames 10 \
    --target-frame-rate-hz 2 \
    --frame_stride "$FRAME_STRIDE" \
    --cache-log-name "$VAL_LOG_NAME" \
    --trajectory-only \
    --fail-fast

stage "Generating 20 trajectory anchors from the training split only"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_plan_anchors.py" \
    --cache-path "$CACHE_PATH" \
    --log-name "$TRAIN_LOG_NAME" \
    --output "$ANCHOR_PATH" \
    --num-modes 20

RESUME_CHECKPOINT="${NUREASONING_RESUME_CHECKPOINT:-auto}"
case "$RESUME_CHECKPOINT" in
    auto)
        latest_checkpoint=""
        while IFS= read -r -d '' candidate; do
            if [[ -z "$latest_checkpoint" || "$candidate" -nt "$latest_checkpoint" ]]; then
                latest_checkpoint="$candidate"
            fi
        done < <(find -L "$EXP_ROOT/$EXPERIMENT_NAME" -type f -name '*.ckpt' -print0 2>/dev/null)
        RESUME_CHECKPOINT="${latest_checkpoint:-null}"
        ;;
    ""|none|null)
        RESUME_CHECKPOINT=null
        ;;
    *)
        if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
            echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
            exit 1
        fi
        ;;
esac

if [[ "$RESUME_CHECKPOINT" == null ]]; then
    stage "Starting camera/trajectory training from scratch for $MAX_EPOCHS epochs"
else
    stage "Resuming camera/trajectory training from $RESUME_CHECKPOINT"
fi
stage "Training config: batch=$BATCH_SIZE, accumulation=$ACCUMULATE_GRAD_BATCHES, effective_batch=$((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES)), lr=$LEARNING_RATE, gradient_clip=$GRADIENT_CLIP_VAL"
conda run --no-capture-output -n "$CONDA_ENV" python \
    navsim/planning/script/run_training.py \
    agent=diffusiondrive_agent \
    experiment_name="$EXPERIMENT_NAME" \
    train_test_split=navtrain \
    cache_path="$CACHE_PATH" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    "train_logs=[$TRAIN_LOG_NAME]" \
    "val_logs=[$VAL_LOG_NAME]" \
    agent.checkpoint_path=null \
    agent.lr="$LEARNING_RATE" \
    resume_checkpoint_path="$RESUME_CHECKPOINT" \
    checkpoint.every_n_train_steps="$CHECKPOINT_EVERY_N_STEPS" \
    checkpoint.save_top_k="$CHECKPOINT_SAVE_TOP_K" \
    checkpoint.monitor=step \
    checkpoint.mode=max \
    +agent.config.plan_anchor_path="$ANCHOR_PATH" \
    +agent.config.image_pretrained=false \
    +agent.config.trajectory_only=true \
    +agent.config.scheduler_epochs="$MAX_EPOCHS" \
    +agent.config.warmup_epochs="$WARMUP_EPOCHS" \
    trainer.params.accelerator="$ACCELERATOR" \
    +trainer.params.devices=1 \
    trainer.params.strategy=auto \
    trainer.params.precision="$PRECISION" \
    trainer.params.max_epochs="$MAX_EPOCHS" \
    trainer.params.fast_dev_run=false \
    trainer.params.num_sanity_val_steps=0 \
    trainer.params.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    trainer.params.gradient_clip_val="$GRADIENT_CLIP_VAL" \
    dataloader.params.batch_size="$BATCH_SIZE" \
    dataloader.params.num_workers="$NUM_WORKERS" \
    +trainer.params.enable_checkpointing=true \
    +trainer.params.enable_progress_bar=true \
    +trainer.params.log_every_n_steps=10
