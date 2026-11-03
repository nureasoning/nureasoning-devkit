#!/usr/bin/env bash
set -euo pipefail

repo_root="/path/to/DiffusionDrive"
devkit_root="/path/to/nureasoning-devkit"
source_root="${NUREASONING_VALIDATION_ARCHIVES:-/path/to/validation_archives}"
data_root="${NUREASONING_PLANNING_EVAL_DATA:-${repo_root}/data/nureasoning_validation_planning_eval}"
checkpoint_path="${DIFFUSIONDRIVE_CHECKPOINT:?set DIFFUSIONDRIVE_CHECKPOINT to the 5 s checkpoint}"
plan_anchor_path="${DIFFUSIONDRIVE_PLAN_ANCHORS:-${repo_root}/data/nureasoning_non_vqa_cache/nureasoning_train_20x10_anchors.npy}"
output_dir="${NUREASONING_BENCHMARK_OUTPUT:-${repo_root}/exp/nureasoning_diffusiondrive_official_benchmark_5s}"
max_clips="${NUREASONING_MAX_CLIPS:-0}"
vis_clips="${NUREASONING_VIS_CLIPS:-5}"
materialize_workers="${NUREASONING_MATERIALIZE_WORKERS:-8}"

cd "${repo_root}"

PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n navsim \
  python scripts/nureasoning/materialize_planning_eval_data.py \
    --source-root "${source_root}" \
    --output-root "${data_root}" \
    --key-frame-index 100 \
    --max-clips "${max_clips}" \
    --workers "${materialize_workers}"

PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
PYTHONPATH="${devkit_root}:${repo_root}/scripts/nureasoning:${repo_root}" \
  conda run --no-capture-output -n navsim \
  python scripts/nureasoning/run_diffusiondrive_official_benchmark.py \
    --data-root "${data_root}" \
    --checkpoint-path "${checkpoint_path}" \
    --plan-anchor-path "${plan_anchor_path}" \
    --output-dir "${output_dir}" \
    --key-frame-index 100 \
    --max-clips "${max_clips}" \
    --vis-clips "${vis_clips}" \
    --device cuda
