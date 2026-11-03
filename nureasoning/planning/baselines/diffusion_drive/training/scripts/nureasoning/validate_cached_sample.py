"""Run one evaluation-only DiffusionDrive pass and optionally visualize it."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from nureasoning.common.clips import discover_clips
from nureasoning.common.pickle_io import load_pickle
from nureasoning.planning.benchmark import BenchmarkConfig, ClipResult, evaluate_clip
from nureasoning.visualization.plotting import visualize_clip_result

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.planning.training.dataset import CacheOnlyDataset


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--log-name", required=True)
    parser.add_argument("--plan-anchor-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="Evaluate only the trajectory objective used by camera/trajectory training",
    )
    parser.add_argument(
        "--clips-root",
        type=Path,
        help="Source nuReasoning clips; required with --visualization-output",
    )
    parser.add_argument(
        "--visualization-output",
        type=Path,
        help="Directory for the devkit planning PNG and JSON report",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return torch.device(requested)


def move_to_device(values: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    return {name: value.to(device) for name, value in values.items()}


def scalar_losses(losses: Dict[str, Tensor]) -> Dict[str, float]:
    result = {}
    for name, value in losses.items():
        scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(scalar):
            raise RuntimeError(f"Non-finite validation loss {name}: {scalar}")
        result[name] = scalar
    return result


def find_source_frame(clips_root: Path, token: str) -> Tuple[Path, int, Dict[str, Any]]:
    """Resolve a cached NAVSIM token back to its nuReasoning clip and frame."""
    for clip_path_str in discover_clips(str(clips_root)):
        clip_path = Path(clip_path_str)
        with open(clip_path / "metadata.json") as f:
            metadata = json.load(f)
        for frame_idx, frame in enumerate(metadata.get("frames", [])):
            if frame.get("token") == token:
                return clip_path, frame_idx, metadata
    raise RuntimeError(f"Token {token} was not found under {clips_root}")


def ego_to_global(trajectory: np.ndarray, ego_pose: Dict[str, Any]) -> np.ndarray:
    """Convert NAVSIM [x-forward, y-left, heading] poses to global poses."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) trajectory, got {trajectory.shape}")

    ego_x = float(ego_pose.get("x", 0.0))
    ego_y = float(ego_pose.get("y", 0.0))
    ego_yaw = float(ego_pose.get("yaw", 0.0))
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)

    global_trajectory = np.empty_like(trajectory, dtype=np.float64)
    global_trajectory[:, 0] = ego_x + cos_yaw * trajectory[:, 0] - sin_yaw * trajectory[:, 1]
    global_trajectory[:, 1] = ego_y + sin_yaw * trajectory[:, 0] + cos_yaw * trajectory[:, 1]
    global_trajectory[:, 2] = (trajectory[:, 2] + ego_yaw + math.pi) % (2 * math.pi) - math.pi
    return global_trajectory


def json_compatible(value: Any) -> Any:
    """Convert numpy values in a devkit result into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def render_devkit_visualization(
    predicted_local_trajectory: np.ndarray,
    token: str,
    clips_root: Path,
    output_dir: Path,
    model_config: TransfuserConfig,
) -> Dict[str, Any]:
    """Score and render one DiffusionDrive result with the nuReasoning devkit."""
    clip_path, frame_idx, metadata = find_source_frame(clips_root, token)
    frame = metadata["frames"][frame_idx]
    ego_state = load_pickle(str(clip_path / frame["ego_state"]))
    ego_pose = ego_state.pose

    predicted_global = ego_to_global(predicted_local_trajectory, ego_pose)
    current_global = np.array(
        [[float(ego_pose["x"]), float(ego_pose["y"]), float(ego_pose["yaw"])]],
        dtype=np.float64,
    )
    predicted_global_with_current = np.vstack([current_global, predicted_global])

    sampling = model_config.trajectory_sampling
    benchmark_config = BenchmarkConfig(
        trajectory_dt_s=float(sampling.interval_length),
        trajectory_steps=int(sampling.num_poses) + 1,
        comfort_dt_s=float(sampling.interval_length),
        key_frame_index=frame_idx,
    )
    result: ClipResult = evaluate_clip(
        str(clip_path),
        benchmark_config,
        planned_trajectory=predicted_global_with_current,
    )
    if result.error is not None:
        raise RuntimeError(f"Devkit evaluation failed: {result.error}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "planning_result.png"
    result_path = output_dir / "planning_result.json"
    visualize_clip_result(result, benchmark_config, logger, save_path=str(image_path), show=False)

    result_payload = {
        "clip_path": str(clip_path),
        "clip_name": result.clip_name,
        "key_frame_idx": result.key_frame_idx,
        "token": token,
        "trajectory_interval_seconds": benchmark_config.trajectory_dt_s,
        "predicted_local_trajectory": predicted_local_trajectory,
        "planned_global_trajectory": result.planned_traj,
        "ground_truth_global_trajectory": result.gt_traj,
        "scores": {
            "collision": result.s_collision,
            "driveable": result.s_driveable,
            "progress": result.s_progress,
            "comfort": result.s_comfort,
            "human": result.s_human,
            "planning": result.planning_score,
        },
        "details": {
            "collision": result.details_collision,
            "driveable": result.details_driveable,
            "progress": result.details_progress,
            "comfort": result.details_comfort,
            "human": result.details_human,
        },
    }
    with open(result_path, "w") as f:
        json.dump(json_compatible(result_payload), f, indent=2, sort_keys=True)

    return {
        "image": str(image_path),
        "report": str(result_path),
        "clip_path": str(clip_path),
        "key_frame_idx": frame_idx,
        "scores": json_compatible(result_payload["scores"]),
    }


def load_checkpoint(agent: TransfuserAgent, checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    normalized_state_dict = {}
    for name, value in state_dict.items():
        if name.startswith("agent."):
            name = name.removeprefix("agent.")
        elif name.startswith("model."):
            # The completed nuReasoning training run saved the bare model
            # beneath ``model`` instead of the NAVSIM agent wrapper.
            name = "_transfuser_model." + name.removeprefix("model.")
        normalized_state_dict[name] = value

    missing, unexpected = agent.load_state_dict(normalized_state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not load completely: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total)"
        )
    return len(normalized_state_dict)


def main() -> None:
    args = parse_args()
    if not args.cache_path.is_dir():
        raise SystemExit(f"Cache path does not exist: {args.cache_path}")
    if not args.plan_anchor_path.is_file():
        raise SystemExit(f"Plan anchors do not exist: {args.plan_anchor_path}")
    if args.checkpoint_path is not None and not args.checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {args.checkpoint_path}")
    if args.visualization_output is not None and args.clips_root is None:
        raise SystemExit("--clips-root is required with --visualization-output")
    if args.clips_root is not None and not args.clips_root.is_dir():
        raise SystemExit(f"Clips root does not exist: {args.clips_root}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    config = TransfuserConfig(
        image_pretrained=False,
        plan_anchor_path=str(args.plan_anchor_path),
        trajectory_only=args.trajectory_only,
    )
    agent = TransfuserAgent(config=config, lr=6e-4, checkpoint_path=None)
    checkpoint_tensors = load_checkpoint(agent, args.checkpoint_path) if args.checkpoint_path else 0
    dataset = CacheOnlyDataset(
        cache_path=str(args.cache_path),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=[args.log_name],
    )
    if not 0 <= args.sample_index < len(dataset):
        raise RuntimeError(
            f"Sample index {args.sample_index} is outside the validation dataset "
            f"with {len(dataset)} samples"
        )

    selected_dataset = Subset(dataset, [args.sample_index])
    features, targets = next(
        iter(DataLoader(selected_dataset, batch_size=1, shuffle=False, num_workers=0))
    )
    features = move_to_device(features, device)
    targets = move_to_device(targets, device)
    agent = agent.to(device).eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    with torch.inference_mode():
        predictions = agent(features, targets)
        losses = scalar_losses(agent.compute_loss(features, targets, predictions))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_elapsed_seconds = round(time.perf_counter() - start_time, 3)

    visualization = None
    selected_token = dataset.tokens[args.sample_index]
    if args.visualization_output is not None:
        visualization = render_devkit_visualization(
            predicted_local_trajectory=predictions["trajectory"][0].detach().cpu().numpy(),
            token=selected_token,
            clips_root=args.clips_root,
            output_dir=args.visualization_output,
            model_config=config,
        )

    report = {
        "status": "passed",
        "samples": len(dataset),
        "selected_sample_index": args.sample_index,
        "token": selected_token,
        "device": str(device),
        "checkpoint": str(args.checkpoint_path) if args.checkpoint_path else None,
        "checkpoint_tensors_loaded": checkpoint_tensors,
        "elapsed_seconds": inference_elapsed_seconds,
        "losses": losses,
        "prediction_shapes": {
            name: list(value.shape) for name, value in predictions.items() if torch.is_tensor(value)
        },
        "visualization": visualization,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
