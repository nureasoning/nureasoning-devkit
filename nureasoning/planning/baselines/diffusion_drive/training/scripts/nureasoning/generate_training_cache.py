"""
Generates a DiffusionDrive-compatible feature/target cache from a
directory of nuReasoning clips, so training can run with
`use_cache_without_dataset=True` (no live SceneLoader/Dataset needed).

Verified cache format (navsim_ref/navsim/planning/training/dataset.py,
`CacheOnlyDataset` / `Dataset._cache_scene_with_token`):

    {cache_path}/{log_name}/{initial_token}/{builder.get_unique_name()}.gz

...where each .gz file is a gzip-compressed pickle of a
Dict[str, torch.Tensor], written with compresslevel=1.

For DiffusionDrive specifically (navsim/agents/diffusiondrive/
transfuser_features.py), that's exactly two files per token:
    transfuser_feature.gz   (camera_feature, lidar_feature, status_feature)
    transfuser_target.gz    (trajectory, agent_states, agent_labels, bev_semantic_map)

With ``--trajectory-only``, the target file contains only ``trajectory``.

The last fully completed clip is recorded atomically under
``{cache_path}/.clip_progress``. Restarts continue at the following clip;
a mismatched clip identity makes the script fall back to cache validation.

Run this on your actual training machine, where navsim + DiffusionDrive
are installed -- it imports their real classes rather than reimplementing
the feature/target math, so it stays correct if either repo changes.

Usage:
    python generate_training_cache.py \
        --clips_root /path/to/nureasoning/train \
        --cache_path $NAVSIM_EXP_ROOT/nureasoning_cache \
        --num_history_frames 4 --num_future_frames 10
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from pyquaternion import Quaternion
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from nureasoning.common.clips import discover_clips

from nureasoning_clip_converter import (
    _ego_state_to_raw,
    build_scene_dict_list,
    build_scene,
    get_scene_windows,
)
from nureasoning_map_adapter import NuReasoningMapAdapter

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent  # noqa: F401 (for reference)
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import (
    TransfuserFeatureBuilder,
    TransfuserTargetBuilder,
)


CACHE_PROGRESS_SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON without exposing a partially written final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary_path, "w") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class ClipProgressCursor:
    """Atomic cursor pointing to the last fully completed clip."""

    def __init__(self, cache_path: Path, log_name: str, cache_config: Dict[str, Any]) -> None:
        progress_config = {
            "progress_schema_version": CACHE_PROGRESS_SCHEMA_VERSION,
            **cache_config,
        }
        encoded_config = json.dumps(
            progress_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._config_signature = hashlib.sha256(encoded_config).hexdigest()

        namespace = hashlib.sha256(log_name.encode("utf-8")).hexdigest()[:16]
        progress_dir = cache_path / ".clip_progress"
        progress_stem = f"{namespace}.{self._config_signature}"
        self._cursor_path = progress_dir / f"{progress_stem}.cursor.json"
        self._config_path = progress_dir / f"{progress_stem}.config.json"

        progress_dir.mkdir(parents=True, exist_ok=True)
        self._record = self._load_cursor()
        if not self._config_path.is_file():
            _atomic_write_json(
                self._config_path,
                {
                    "config_signature": self._config_signature,
                    "log_name": log_name,
                    "cache_config": progress_config,
                },
            )

    @property
    def cursor_path(self) -> Path:
        return self._cursor_path

    @staticmethod
    def _clip_state(clip_dir: Path) -> Dict[str, Any]:
        metadata_path = clip_dir / "metadata.json"
        metadata_stat = metadata_path.stat()
        clip_path = str(clip_dir.resolve())
        return {
            "clip_key": hashlib.sha256(clip_path.encode("utf-8")).hexdigest(),
            "clip_path": clip_path,
            "metadata_size": metadata_stat.st_size,
            "metadata_mtime_ns": metadata_stat.st_mtime_ns,
        }

    def _load_cursor(self) -> Optional[Dict[str, Any]]:
        if not self._cursor_path.is_file():
            return None
        try:
            with open(self._cursor_path) as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"[clip-progress] ignoring unreadable cursor: {self._cursor_path}")
            return None
        if record.get("config_signature") != self._config_signature:
            print(f"[clip-progress] ignoring mismatched cursor: {self._cursor_path}")
            return None
        return record

    def resume_position(self, clip_dirs) -> tuple[int, int]:
        """Return ``(next_clip_index, completed_sample_count)``."""
        if self._record is None:
            return 0, 0
        last_index = self._record.get("last_completed_index")
        total_cached = self._record.get("completed_sample_count")
        if (
            type(last_index) is not int
            or type(total_cached) is not int
            or last_index < 0
            or last_index >= len(clip_dirs)
            or total_cached < 0
        ):
            print(f"[clip-progress] ignoring invalid cursor: {self._cursor_path}")
            return 0, 0

        clip_state = self._clip_state(clip_dirs[last_index])
        if any(self._record.get(key) != value for key, value in clip_state.items()):
            print(
                "[clip-progress] clip ordering or metadata changed; "
                "validating the cache from clip 1."
            )
            return 0, 0
        return last_index + 1, total_cached

    def mark_complete(
        self,
        clip_dir: Path,
        clip_index: int,
        completed_sample_count: int,
    ) -> None:
        """Advance the cursor after an entire clip completes."""
        if clip_index < 0 or completed_sample_count < 0:
            raise ValueError("clip progress values cannot be negative")
        record = {
            "config_signature": self._config_signature,
            "last_completed_index": clip_index,
            "completed_sample_count": completed_sample_count,
            **self._clip_state(clip_dir),
        }
        _atomic_write_json(self._cursor_path, record)
        self._record = record


def dump_gz(path: Path, data_dict) -> None:
    """Atomically write one cache entry.

    A workspace shutdown can interrupt a write. Writing beside the final
    path and replacing it only after the gzip stream is closed prevents a
    truncated file from looking like a completed cache entry on restart.
    """
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary_path, "wb", compresslevel=1) as f:
            pickle.dump(data_dict, f)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def target_cache_is_compatible(path: Path, expected_num_poses: int) -> bool:
    """Check target invariants that changed after older caches were written."""
    if not path.is_file():
        return False
    try:
        with gzip.open(path, "rb") as f:
            target = pickle.load(f)
    except Exception:
        # In particular, treat files truncated by an interrupted older run
        # as cache misses so they are regenerated rather than failing later
        # in training.
        return False
    trajectory = target.get("trajectory") if isinstance(target, dict) else None
    return (
        isinstance(trajectory, torch.Tensor)
        and trajectory.dtype == torch.float32
        and trajectory.ndim == 2
        and tuple(trajectory.shape) == (expected_num_poses, 3)
    )


def load_global_ego_poses(
    clip_dir: Path,
    metadata: Dict[str, Any],
    source_indices,
) -> np.ndarray:
    """Load global ``(x, y, yaw)`` poses once for a clip span."""
    global_ego_poses = []
    for source_index in source_indices:
        frame = metadata["frames"][source_index]
        ego_raw = _ego_state_to_raw(clip_dir / frame["ego_state"])
        ego_yaw = Quaternion(*ego_raw["ego2global_rotation"]).yaw_pitch_roll[0]
        global_ego_poses.append(
            [ego_raw["ego_x"], ego_raw["ego_y"], ego_yaw]
        )
    return np.asarray(global_ego_poses, dtype=np.float64)


def build_trajectory_target_from_global_poses(
    global_ego_poses: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Build the same local trajectory as ``Scene.get_future_trajectory``.

    This path reads only the small ego-state pickles. It deliberately avoids
    maps, annotations, lidar, and camera images, none of which contribute to a
    trajectory-only target.
    """
    local_ego_poses = convert_absolute_to_relative_se2_array(
        StateSE2(*global_ego_poses[0]),
        global_ego_poses[1:],
    )
    return {
        "trajectory": torch.tensor(local_ego_poses, dtype=torch.float32),
    }


def cache_one_clip(
    clip_dir: Path,
    cache_path: Path,
    feature_builder: TransfuserFeatureBuilder,
    target_builder: TransfuserTargetBuilder,
    num_history_frames: int,
    num_future_frames: int,
    frame_stride: int = 1,
    frame_interval: int = 1,
    max_samples: int = 0,
    cache_log_name: Optional[str] = None,
    reuse_existing_features: bool = False,
    fail_fast: bool = False,
) -> int:
    """
    Cache selected valid windows in one clip.

    Returns the number of complete samples, including existing cache entries.

    :param frame_stride: only cache every Nth window's center frame, to
        avoid near-duplicate overlapping windows dominating the dataset
        (e.g. stride=4 at 2Hz gives one cached sample every 2 seconds).
    """
    with open(clip_dir / "metadata.json") as f:
        metadata = json.load(f)

    sampled_source_indices = list(range(0, len(metadata["frames"]), frame_interval))
    window_starts = get_scene_windows(len(sampled_source_indices), num_history_frames, num_future_frames)
    selected_starts = window_starts[::frame_stride]
    if max_samples > 0:
        selected_starts = selected_starts[:max_samples]
    if not selected_starts:
        return 0

    log_name = cache_log_name or metadata["log_name"]
    cache_entries = []
    for start in selected_starts:
        initial_source_index = sampled_source_indices[start + num_history_frames - 1]
        initial_token = metadata["frames"][initial_source_index]["token"]
        token_path = cache_path / log_name / initial_token
        feature_cache_file = token_path / (feature_builder.get_unique_name() + ".gz")
        target_cache_file = token_path / (target_builder.get_unique_name() + ".gz")
        expected_num_poses = target_builder._config.trajectory_sampling.num_poses
        target_is_cached = target_cache_is_compatible(
            target_cache_file,
            expected_num_poses=expected_num_poses,
        )

        # Targets are written after features. If an older run was killed
        # while writing a feature, its target is absent/invalid; recomputing
        # both files repairs the potentially truncated feature as well.
        feature_is_cached = feature_cache_file.is_file() and (
            target_is_cached or reuse_existing_features
        )
        cache_entries.append(
            (
                start,
                token_path,
                feature_cache_file,
                target_cache_file,
                feature_is_cached,
                target_is_cached,
            )
        )

    num_cached = sum(
        feature_is_cached and target_is_cached
        for _, _, _, _, feature_is_cached, target_is_cached in cache_entries
    )
    if num_cached == len(cache_entries):
        # This is the important restart fast path: do not load map, ego,
        # annotation, or camera data for a clip that is already complete.
        return num_cached

    window_size = num_history_frames + num_future_frames
    first_start = selected_starts[0]
    last_end = selected_starts[-1] + window_size
    source_indices = sampled_source_indices[first_start:last_end]
    trajectory_only = target_builder._config.trajectory_only
    global_ego_poses = None
    if trajectory_only and any(
        not target_is_cached
        for _, _, _, _, _, target_is_cached in cache_entries
    ):
        global_ego_poses = load_global_ego_poses(
            clip_dir,
            metadata,
            source_indices,
        )

    # Construct heavyweight scene resources lazily. A 5-second target refresh
    # can reuse all existing features and should never decode camera images.
    map_adapter = None
    scene_dict_list = None

    def get_scene_for_window(local_start: int):
        nonlocal map_adapter, scene_dict_list
        if map_adapter is None:
            map_adapter = NuReasoningMapAdapter(
                str(clip_dir / metadata["map_annotation"])
            )
        if scene_dict_list is None:
            scene_dict_list = build_scene_dict_list(
                clip_dir,
                metadata,
                frame_indices=source_indices,
            )
        end = local_start + window_size
        return build_scene(
            clip_dir=clip_dir,
            metadata=metadata,
            map_adapter=map_adapter,
            scene_dict_list=scene_dict_list[local_start:end],
            sensor_blobs_path=clip_dir,
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )

    for (
        start,
        token_path,
        feature_cache_file,
        target_cache_file,
        feature_is_cached,
        target_is_cached,
    ) in cache_entries:
        local_start = start - first_start
        if feature_is_cached and target_is_cached:
            continue

        try:
            scene = None
            if not feature_is_cached or (not trajectory_only and not target_is_cached):
                scene = get_scene_for_window(local_start)
            os.makedirs(token_path, exist_ok=True)
            if not feature_is_cached:
                dump_gz(feature_cache_file, feature_builder.compute_features(scene.get_agent_input()))
            if not target_is_cached:
                if trajectory_only:
                    current_offset = local_start + num_history_frames - 1
                    target_global_poses = global_ego_poses[
                        current_offset:current_offset + expected_num_poses + 1
                    ]
                    target = build_trajectory_target_from_global_poses(
                        target_global_poses,
                    )
                else:
                    target = target_builder.compute_targets(scene)
                dump_gz(target_cache_file, target)
            num_cached += 1
        except Exception:
            if fail_fast:
                raise
            print(f"  [skip] {clip_dir.name} window@{start}: {traceback.format_exc(limit=2)}")

    return num_cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_root", type=Path, required=True, help="Directory containing one subdirectory per clip")
    parser.add_argument("--cache_path", type=Path, required=True)
    parser.add_argument("--num_history_frames", type=int, default=4)
    parser.add_argument("--num_future_frames", type=int, default=10)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument(
        "--target-frame-rate-hz", type=float, default=2.0,
        help="Cadence expected by NAVSIM/DiffusionDrive (default: 2 Hz)",
    )
    parser.add_argument("--max-clips", type=int, default=0, help="0 means all discovered clips")
    parser.add_argument("--max-samples", type=int, default=0, help="Global cap; 0 means all samples")
    parser.add_argument(
        "--max-samples-per-clip",
        type=int,
        default=0,
        help="Per-clip cap; 0 means all samples allowed by the global cap",
    )
    parser.add_argument("--cache-log-name", help="Put samples under one named cache split")
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="Cache only trajectory targets; omit unused agent and BEV targets",
    )
    parser.add_argument(
        "--trajectory-time-horizon",
        type=float,
        default=4.0,
        help="Trajectory target horizon in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--trajectory-interval-length",
        type=float,
        default=0.5,
        help="Spacing between trajectory target poses in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--target-cache-name",
        default="transfuser_target",
        help="Target builder/cache filename stem",
    )
    parser.add_argument(
        "--reuse-existing-features",
        action="store_true",
        help="Trust and reuse feature files when generating a separate target cache",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.target_frame_rate_hz <= 0:
        parser.error("--target-frame-rate-hz must be positive")
    if args.frame_stride <= 0:
        parser.error("--frame_stride must be positive")
    if args.max_samples < 0:
        parser.error("--max-samples cannot be negative")
    if args.max_samples_per_clip < 0:
        parser.error("--max-samples-per-clip cannot be negative")
    if args.trajectory_time_horizon <= 0:
        parser.error("--trajectory-time-horizon must be positive")
    if args.trajectory_interval_length <= 0:
        parser.error("--trajectory-interval-length must be positive")

    trajectory_sampling = TrajectorySampling(
        time_horizon=args.trajectory_time_horizon,
        interval_length=args.trajectory_interval_length,
    )
    required_future_frames = trajectory_sampling.num_poses
    if required_future_frames > args.num_future_frames:
        parser.error(
            "--num_future_frames must be at least the number of target poses "
            f"({required_future_frames})"
        )

    config = TransfuserConfig(
        trajectory_sampling=trajectory_sampling,
        trajectory_only=args.trajectory_only,
        target_cache_name=args.target_cache_name,
    )
    feature_builder = TransfuserFeatureBuilder(config=config)
    target_builder = TransfuserTargetBuilder(config=config)

    args.cache_path.mkdir(parents=True, exist_ok=True)
    progress_cursor = ClipProgressCursor(
        cache_path=args.cache_path,
        log_name=args.cache_log_name or "<metadata-log-name>",
        cache_config={
            "cache_content_version": 2,
            "feature_builder": feature_builder.get_unique_name(),
            "target_builder": target_builder.get_unique_name(),
            "num_history_frames": args.num_history_frames,
            "num_future_frames": args.num_future_frames,
            "frame_stride": args.frame_stride,
            "target_frame_rate_hz": args.target_frame_rate_hz,
            "trajectory_time_horizon": args.trajectory_time_horizon,
            "trajectory_interval_length": args.trajectory_interval_length,
            "reuse_existing_features": args.reuse_existing_features,
            "max_samples": args.max_samples,
            "max_samples_per_clip": args.max_samples_per_clip,
            "trajectory_only": args.trajectory_only,
        },
    )
    clip_dirs = [
        Path(path)
        for path in discover_clips(str(args.clips_root), max_clips=args.max_clips)
    ]
    print(f"Found {len(clip_dirs)} clips under {args.clips_root}")
    if not clip_dirs:
        raise SystemExit("No nuReasoning clips found (a clip directory contains metadata.json)")

    start_index, total_cached = progress_cursor.resume_position(clip_dirs)
    if start_index > 0:
        if args.max_samples > 0 and total_cached >= args.max_samples:
            resume_status = "the configured sample limit is already satisfied."
        elif start_index == len(clip_dirs):
            resume_status = "all clip work is already complete."
        else:
            resume_status = f"resuming at clip {start_index + 1}."
        print(
            f"[clip-progress] Skipping {start_index}/{len(clip_dirs)} completed clips "
            f"({total_cached} samples); {resume_status}"
        )
    print(f"Progress cursor: {progress_cursor.cursor_path}")

    cursor_can_advance = True
    for i in range(start_index, len(clip_dirs)):
        if args.max_samples > 0 and total_cached >= args.max_samples:
            break

        clip_dir = clip_dirs[i]
        with open(clip_dir / "metadata.json") as f:
            metadata = json.load(f)
        source_rate_hz = float(metadata["frame_rate_hz"])
        frame_interval = max(1, round(source_rate_hz / args.target_frame_rate_hz))
        actual_rate_hz = source_rate_hz / frame_interval
        if abs(actual_rate_hz - args.target_frame_rate_hz) > 1e-6:
            raise ValueError(
                f"{clip_dir}: {source_rate_hz} Hz cannot be evenly resampled to "
                f"{args.target_frame_rate_hz} Hz"
            )
        remaining = args.max_samples - total_cached if args.max_samples > 0 else 0
        clip_sample_limit = args.max_samples_per_clip
        if remaining > 0:
            clip_sample_limit = min(clip_sample_limit, remaining) if clip_sample_limit > 0 else remaining

        sampled_frame_count = len(range(0, len(metadata["frames"]), frame_interval))
        expected_sample_count = len(
            get_scene_windows(
                sampled_frame_count,
                args.num_history_frames,
                args.num_future_frames,
            )[::args.frame_stride]
        )
        if clip_sample_limit > 0:
            expected_sample_count = min(expected_sample_count, clip_sample_limit)

        n = cache_one_clip(
            clip_dir=clip_dir,
            cache_path=args.cache_path,
            feature_builder=feature_builder,
            target_builder=target_builder,
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_stride=args.frame_stride,
            frame_interval=frame_interval,
            max_samples=clip_sample_limit,
            cache_log_name=args.cache_log_name,
            reuse_existing_features=args.reuse_existing_features,
            fail_fast=args.fail_fast,
        )
        total_cached += n
        if n == expected_sample_count and cursor_can_advance:
            progress_cursor.mark_complete(
                clip_dir=clip_dir,
                clip_index=i,
                completed_sample_count=total_cached,
            )
        elif n != expected_sample_count:
            cursor_can_advance = False
            print(
                f"[clip-progress] {clip_dir.name} remains incomplete "
                f"({n}/{expected_sample_count} samples); the cursor was not advanced."
            )
        print(
            f"[{i + 1}/{len(clip_dirs)}] {clip_dir.name}: {source_rate_hz:g} Hz -> "
            f"{actual_rate_hz:g} Hz, {n} available samples (running total: {total_cached})"
        )

    if total_cached == 0:
        raise SystemExit("No valid DiffusionDrive training samples were cached")
    print(f"Done. {total_cached} samples available under {args.cache_path}")
    print("Train with: use_cache_without_dataset=True cache_path=" + str(args.cache_path))


if __name__ == "__main__":
    main()
