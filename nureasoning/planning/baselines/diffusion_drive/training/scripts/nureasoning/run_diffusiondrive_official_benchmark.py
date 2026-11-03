"""Run the unmodified nuReasoning planning benchmark with DiffusionDrive."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

from nureasoning.planning.benchmark import (
    BenchmarkConfig,
    BenchmarkSummary,
    discover_clips,
    run_benchmark,
    set_random_seed,
)
from nureasoning.visualization.plotting import (
    visualize_clip_result,
    visualize_summary,
)

from diffusiondrive_trajectory_provider import DiffusionDriveTrajectoryProvider


LOGGER = logging.getLogger(__name__)
REQUIRED_CAMERA_KEYS = ("front", "front_left", "front_right")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/nureasoning_validation_planning_eval"
DEFAULT_ANCHORS = (
    REPO_ROOT / "data/nureasoning_non_vqa_cache/nureasoning_train_20x10_anchors.npy"
)
DEFAULT_OUTPUT = REPO_ROOT / "exp/nureasoning_diffusiondrive_official_benchmark_5s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--plan-anchor-path", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--vis-clips", type=int, default=5)
    parser.add_argument("--key-frame-index", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-progress", type=float, default=0.3)
    parser.add_argument("--w-comfort", type=float, default=0.2)
    parser.add_argument("--w-human", type=float, default=0.5)
    return parser.parse_args()


def validate_benchmark_data(
    data_root: Path,
    config: BenchmarkConfig,
    max_clips: int,
) -> Dict[str, Any]:
    """Reject inputs that cannot represent the devkit's temporal protocol."""
    clips = discover_clips(str(data_root))
    if max_clips > 0:
        clips = clips[:max_clips]
    if not clips:
        raise SystemExit(f"No clips with metadata.json found under {data_root}")

    frame_rates: List[float] = []
    frame_intervals: List[float] = []
    available_annotation_steps: List[int] = []
    for clip_path in clips:
        metadata_path = Path(clip_path) / "metadata.json"
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        frames = metadata.get("frames") or []
        if config.key_frame_index >= len(frames):
            raise SystemExit(
                f"{metadata_path}: key frame {config.key_frame_index} would be "
                f"clamped in a {len(frames)}-frame clip"
            )

        key_frame = frames[config.key_frame_index]
        sensors = key_frame.get("sensors") or {}
        cameras = sensors.get("cameras") or {}
        required_sensor_paths = [
            cameras.get(camera_key) for camera_key in REQUIRED_CAMERA_KEYS
        ]
        lidar_path = (sensors.get("lidar") or {}).get("point_cloud_path") or ""
        if lidar_path:
            required_sensor_paths.append(lidar_path)
        missing_sensor_paths = [
            relative_path
            for relative_path in required_sensor_paths
            if not relative_path or not (Path(clip_path) / relative_path).is_file()
        ]
        if missing_sensor_paths:
            raise SystemExit(
                f"{metadata_path}: missing required keyframe sensor files: "
                f"{missing_sensor_paths}"
            )

        frame_dt_s = None
        for previous, current in zip(frames, frames[1:]):
            t0 = previous.get("relative_time_s")
            t1 = current.get("relative_time_s")
            if (
                isinstance(t0, (int, float))
                and isinstance(t1, (int, float))
                and t1 > t0
            ):
                frame_dt_s = float(t1) - float(t0)
                break
        if frame_dt_s is None:
            raise SystemExit(f"{metadata_path}: no increasing frame timestamps")

        stride = max(1, int(round(config.trajectory_dt_s / frame_dt_s)))
        actual_dt_s = stride * frame_dt_s
        if not math.isclose(
            actual_dt_s,
            config.trajectory_dt_s,
            rel_tol=0.02,
            abs_tol=0.005,
        ):
            raise SystemExit(
                f"{metadata_path}: {frame_dt_s:g}s source frames cannot represent "
                f"the official {config.trajectory_dt_s:g}s metric interval"
            )

        frame_rates.append(
            round(float(metadata.get("frame_rate_hz", 1.0 / frame_dt_s)), 6)
        )
        frame_intervals.append(round(frame_dt_s, 6))
        available_annotation_steps.append(
            min(
                config.trajectory_steps,
                ((len(frames) - 1 - config.key_frame_index) // stride) + 1,
            )
        )

    return {
        "input_clip_count": len(clips),
        "input_frame_rates_hz": sorted(set(frame_rates)),
        "observed_frame_intervals_s": sorted(set(frame_intervals)),
        "minimum_available_annotation_steps": min(available_annotation_steps),
        "requested_annotation_steps": config.trajectory_steps,
    }


def _result_record(result: Any) -> Dict[str, Any]:
    return {
        "clip": result.clip_name,
        "clip_path": result.clip_path,
        "key_frame": result.key_frame_idx,
        "collision": result.s_collision,
        "driveable": result.s_driveable,
        "progress": result.s_progress,
        "comfort": result.s_comfort,
        "human": result.s_human,
        "ADE_m": result.details_human.get("ade") if result.details_human else None,
        "FDE_m": result.details_human.get("fde") if result.details_human else None,
        "planning_score": result.planning_score,
        "error": result.error,
        "details": {
            "collision": result.details_collision,
            "driveable": result.details_driveable,
            "progress": result.details_progress,
            "comfort": result.details_comfort,
            "human": result.details_human,
        },
    }


def save_reports(
    summary: BenchmarkSummary,
    config: BenchmarkConfig,
    provider: DiffusionDriveTrajectoryProvider,
    args: argparse.Namespace,
    data_profile: Dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    valid = [result for result in summary.results if result.error is None]

    for index, result in enumerate(valid[: args.vis_clips]):
        output = args.output_dir / f"clip_{index:03d}_{result.clip_name[:40]}.png"
        visualize_clip_result(result, config, LOGGER, save_path=str(output), show=False)
    visualize_summary(
        summary,
        LOGGER,
        save_path=str(args.output_dir / "benchmark_summary.png"),
        show=False,
    )

    report = {
        "total_clips": summary.total_clips,
        "evaluated_clips": summary.evaluated_clips,
        "failed_clips": summary.failed_clips,
        "mean_collision": summary.mean_collision,
        "mean_driveable": summary.mean_driveable,
        "mean_progress": summary.mean_progress,
        "mean_comfort": summary.mean_comfort,
        "mean_human": summary.mean_human,
        "mean_ADE_m": summary.mean_ade_m,
        "mean_FDE_m": summary.mean_fde_m,
        "mean_planning_score": summary.mean_planning_score,
        "config": {
            "seed": args.seed,
            "w_progress": config.w_progress,
            "w_comfort": config.w_comfort,
            "w_human": config.w_human,
            "trajectory_dt_s": config.trajectory_dt_s,
            "trajectory_steps": config.trajectory_steps,
            "comfort_dt_s": config.comfort_dt_s,
            "human_fde_full_score_m": config.human_fde_full_score_m,
            "human_fde_threshold_m": config.human_fde_threshold_m,
            "key_frame_index": config.key_frame_index,
            "checkpoint_path": str(provider.checkpoint_path),
            "plan_anchor_path": str(provider.plan_anchor_path),
            "checkpoint_tensors_loaded": provider.checkpoint_tensor_count,
            **data_profile,
        },
        "per_clip": [_result_record(result) for result in summary.results],
    }
    with open(args.output_dir / "benchmark_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    csv_fields = [
        "clip",
        "key_frame",
        "collision",
        "driveable",
        "progress",
        "comfort",
        "human",
        "ADE_m",
        "FDE_m",
        "planning_score",
        "error",
    ]
    with open(args.output_dir / "benchmark_report.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in report["per_clip"]:
            writer.writerow({field: record[field] for field in csv_fields})

    summary_fields = [
        "total_clips",
        "evaluated_clips",
        "failed_clips",
        "mean_collision",
        "mean_driveable",
        "mean_progress",
        "mean_comfort",
        "mean_human",
        "mean_ADE_m",
        "mean_FDE_m",
        "mean_planning_score",
    ]
    with open(args.output_dir / "benchmark_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({field: report[field] for field in summary_fields})


def print_summary(summary: BenchmarkSummary) -> None:
    print("\n" + "=" * 70)
    print("OFFICIAL NUREASONING PLANNING BENCHMARK — DIFFUSIONDRIVE")
    print("=" * 70)
    print(f"  Clips evaluated : {summary.evaluated_clips} / {summary.total_clips}")
    print(f"  Failed          : {summary.failed_clips}")
    print(f"  Collision       : {summary.mean_collision:.4f}")
    print(f"  Driveable Area  : {summary.mean_driveable:.4f}")
    print(f"  Progress        : {summary.mean_progress:.4f}")
    print(f"  Comfort         : {summary.mean_comfort:.4f}")
    print(f"  Human Likeness  : {summary.mean_human:.4f}")
    print(f"  ADE / FDE       : {summary.mean_ade_m:.3f} m / {summary.mean_fde_m:.3f} m")
    print(f"  Planning Score  : {summary.mean_planning_score:.4f}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {args.data_root}")
    if args.max_clips < 0:
        raise SystemExit("--max-clips cannot be negative")
    if args.vis_clips < 0:
        raise SystemExit("--vis-clips cannot be negative")

    set_random_seed(args.seed)
    config = BenchmarkConfig(
        w_progress=args.w_progress,
        w_comfort=args.w_comfort,
        w_human=args.w_human,
        key_frame_index=args.key_frame_index,
    )
    data_profile = validate_benchmark_data(args.data_root, config, args.max_clips)
    LOGGER.info(
        "Validated %d clips at frame rate(s) %s Hz",
        data_profile["input_clip_count"],
        data_profile["input_frame_rates_hz"],
    )
    provider = DiffusionDriveTrajectoryProvider(
        checkpoint_path=str(args.checkpoint_path),
        plan_anchor_path=str(args.plan_anchor_path),
        device=args.device,
    )
    summary = run_benchmark(
        data_root=str(args.data_root),
        config=config,
        max_clips=args.max_clips,
        trajectory_provider=provider,
    )
    print_summary(summary)
    save_reports(summary, config, provider, args, data_profile)
    print(f"Outputs saved to {args.output_dir}")
    if summary.failed_clips:
        raise SystemExit(
            f"Benchmark is incomplete: {summary.failed_clips} of "
            f"{summary.total_clips} clips failed"
        )


if __name__ == "__main__":
    main()
