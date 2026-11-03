#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the completed 4-second nuReasoning model for the devkit's native
# 5-second trajectory horizon. The existing 4-second cache and checkpoints are
# preserved; 10-pose targets are stored under a separate cache filename.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEVKIT_ROOT="${NUREASONING_DEVKIT_ROOT:-/path/to/nureasoning-devkit}"
TRAIN_ROOT="${NUREASONING_TRAIN_ROOT:-/path/to/nureasoning_training_data}"
VAL_ROOT="${NUREASONING_VAL_ROOT:-/path/to/nureasoning_testing_data}"
CACHE_PATH="${NUREASONING_CACHE_PATH:-$SCRIPT_DIR/data/nureasoning_non_vqa_cache}"
EXP_ROOT="${NUREASONING_EXP_ROOT:-$SCRIPT_DIR/exp}"
TRAIN_LOG_NAME="nureasoning_train"
VAL_LOG_NAME="nureasoning_val"
TARGET_CACHE_NAME="transfuser_target_5s"
ANCHOR_PATH="$CACHE_PATH/nureasoning_train_20x10_anchors.npy"
INIT_CHECKPOINT="${NUREASONING_INIT_CHECKPOINT:?set NUREASONING_INIT_CHECKPOINT to the stage-1 (4 s) last.ckpt}"
CONDA_ENV="${NAVSIM_CONDA_ENV:-navsim}"
ACCELERATOR="${NAVSIM_ACCELERATOR:-gpu}"
PRECISION="${NAVSIM_PRECISION:-32-true}"
MAX_EPOCHS="${NUREASONING_MAX_EPOCHS:-20}"
WARMUP_EPOCHS="${NUREASONING_WARMUP_EPOCHS:-1}"
BATCH_SIZE="${NUREASONING_BATCH_SIZE:-32}"
ACCUMULATE_GRAD_BATCHES="${NUREASONING_ACCUMULATE_GRAD_BATCHES:-2}"
LEARNING_RATE="${NUREASONING_LEARNING_RATE:-1e-4}"
GRADIENT_CLIP_VAL="${NUREASONING_GRADIENT_CLIP_VAL:-1.0}"
NUM_WORKERS="${NUREASONING_NUM_WORKERS:-4}"
FRAME_STRIDE="${NUREASONING_FRAME_STRIDE:-4}"
CHECKPOINT_EVERY_N_STEPS="${NUREASONING_CHECKPOINT_EVERY_N_STEPS:-500}"
CHECKPOINT_SAVE_TOP_K="${NUREASONING_CHECKPOINT_SAVE_TOP_K:-3}"
EXPERIMENT_NAME="${NUREASONING_EXPERIMENT_NAME:-nureasoning_non_vqa_5s_finetune}"

stage() {
    printf '\n[non-vqa-5s-finetune] %s\n' "$1"
}

for required_dir in "$DEVKIT_ROOT/nureasoning" "$TRAIN_ROOT" "$VAL_ROOT" "$CACHE_PATH"; do
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

cd "$SCRIPT_DIR"
export NAVSIM_DEVKIT_ROOT="$SCRIPT_DIR"
export NAVSIM_EXP_ROOT="$EXP_ROOT"
export OPENSCENE_DATA_ROOT="$TRAIN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$DEVKIT_ROOT:$SCRIPT_DIR/scripts/nureasoning:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

stage "Caching isolated 10-pose training targets"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_training_cache.py" \
    --clips_root "$TRAIN_ROOT" \
    --cache_path "$CACHE_PATH" \
    --num_history_frames 4 \
    --num_future_frames 10 \
    --target-frame-rate-hz 2 \
    --trajectory-time-horizon 5 \
    --trajectory-interval-length 0.5 \
    --target-cache-name "$TARGET_CACHE_NAME" \
    --reuse-existing-features \
    --frame_stride "$FRAME_STRIDE" \
    --cache-log-name "$TRAIN_LOG_NAME" \
    --trajectory-only \
    --fail-fast

stage "Caching isolated 10-pose validation targets"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_training_cache.py" \
    --clips_root "$VAL_ROOT" \
    --cache_path "$CACHE_PATH" \
    --num_history_frames 4 \
    --num_future_frames 10 \
    --target-frame-rate-hz 2 \
    --trajectory-time-horizon 5 \
    --trajectory-interval-length 0.5 \
    --target-cache-name "$TARGET_CACHE_NAME" \
    --reuse-existing-features \
    --frame_stride "$FRAME_STRIDE" \
    --cache-log-name "$VAL_LOG_NAME" \
    --trajectory-only \
    --fail-fast

stage "Generating 20 ten-pose trajectory anchors from the training split"
conda run --no-capture-output -n "$CONDA_ENV" python \
    "$SCRIPT_DIR/scripts/nureasoning/generate_plan_anchors.py" \
    --cache-path "$CACHE_PATH" \
    --log-name "$TRAIN_LOG_NAME" \
    --target-cache-name "$TARGET_CACHE_NAME" \
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
    if [[ ! -f "$INIT_CHECKPOINT" ]]; then
        echo "Warm-start checkpoint does not exist: $INIT_CHECKPOINT" >&2
        exit 1
    fi
    AGENT_CHECKPOINT="$INIT_CHECKPOINT"
    stage "Warm-starting the 5-second model from $INIT_CHECKPOINT"
else
    AGENT_CHECKPOINT=null
    stage "Resuming 5-second training state from $RESUME_CHECKPOINT"
fi

stage "Training config: horizon=5s, poses=10, batch=$BATCH_SIZE, accumulation=$ACCUMULATE_GRAD_BATCHES, lr=$LEARNING_RATE"
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
    agent.checkpoint_path="$AGENT_CHECKPOINT" \
    agent.lr="$LEARNING_RATE" \
    resume_checkpoint_path="$RESUME_CHECKPOINT" \
    checkpoint.every_n_train_steps="$CHECKPOINT_EVERY_N_STEPS" \
    checkpoint.save_top_k="$CHECKPOINT_SAVE_TOP_K" \
    checkpoint.monitor=step \
    checkpoint.mode=max \
    agent.config.trajectory_sampling.time_horizon=5 \
    agent.config.trajectory_sampling.interval_length=0.5 \
    +agent.config.target_cache_name="$TARGET_CACHE_NAME" \
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
