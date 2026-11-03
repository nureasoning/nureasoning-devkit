"""Build DiffusionDrive trajectory anchors from cached nuReasoning targets."""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np


def load_trajectories(
    cache_path: Path,
    log_name: Optional[str],
    target_cache_name: str,
) -> np.ndarray:
    """Load all ``(T, 2)`` ego trajectories from a DiffusionDrive cache."""
    search_root = cache_path / log_name if log_name else cache_path
    target_paths = sorted(search_root.rglob(f"{target_cache_name}.gz"))
    trajectories: List[np.ndarray] = []
    for target_path in target_paths:
        with gzip.open(target_path, "rb") as f:
            target = pickle.load(f)
        trajectory = target["trajectory"]
        if hasattr(trajectory, "detach"):
            trajectory = trajectory.detach().cpu().numpy()
        trajectory = np.asarray(trajectory, dtype=np.float32)
        if trajectory.ndim != 2 or trajectory.shape[1] < 2:
            raise ValueError(f"Invalid trajectory shape {trajectory.shape} in {target_path}")
        trajectories.append(trajectory[:, :2])

    if not trajectories:
        raise ValueError(f"No {target_cache_name}.gz files found under {search_root}")
    shapes = {trajectory.shape for trajectory in trajectories}
    if len(shapes) != 1:
        raise ValueError(f"Cached trajectories have inconsistent shapes: {sorted(shapes)}")
    return np.stack(trajectories)


def build_anchors(trajectories: np.ndarray, num_modes: int, iterations: int) -> np.ndarray:
    """Cluster trajectories; repeat samples when a smoke cache has fewer modes."""
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    num_samples, num_poses, _ = trajectories.shape
    flattened = trajectories.reshape(num_samples, -1).astype(np.float64)
    if num_samples < num_modes:
        indices = np.arange(num_modes) % num_samples
        return trajectories[indices].astype(np.float32)

    # Deterministic farthest-point initialization avoids an sklearn dependency.
    center_indices = [0]
    min_distances = np.full(num_samples, np.inf)
    for _ in range(1, num_modes):
        latest = flattened[center_indices[-1]]
        distances = np.sum((flattened - latest) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
        center_indices.append(int(np.argmax(min_distances)))
    centers = flattened[center_indices].copy()

    for _ in range(iterations):
        distances = np.sum((flattened[:, None] - centers[None]) ** 2, axis=2)
        assignments = np.argmin(distances, axis=1)
        updated = centers.copy()
        for mode in range(num_modes):
            members = flattened[assignments == mode]
            if len(members):
                updated[mode] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated

    return centers.reshape(num_modes, num_poses, 2).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--log-name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-modes", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--target-cache-name", default="transfuser_target")
    args = parser.parse_args()

    trajectories = load_trajectories(
        args.cache_path,
        args.log_name,
        args.target_cache_name,
    )
    anchors = build_anchors(trajectories, args.num_modes, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, anchors)
    print(
        f"Saved {anchors.shape[0]} anchors with {anchors.shape[1]} poses from "
        f"{len(trajectories)} cached trajectories to {args.output}"
    )


if __name__ == "__main__":
    main()
