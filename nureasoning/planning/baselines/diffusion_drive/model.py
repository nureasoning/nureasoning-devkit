"""
DiffusionDrive baseline.

Runs a nuReasoning 5 s DiffusionDrive checkpoint (the stage-2 fine-tune
described in ``diffusiondrive_nureasoning_modifications.md``) on the key frame
of a clip and returns a global-frame ``(N, 3)`` array of ``(x, y, yaw)``
waypoints at 0.1 s resolution over the 5 s horizon (see
``nureasoning.planning.baselines.base.BaseTrajectoryProvider``).

The model definition is the adapted NAVSIM DiffusionDrive code in the
DiffusionDrive repository, which must be importable (``navsim`` package) and
needs the ``navsim`` conda environment (torch, nuplan-devkit, diffusers).
Point ``DIFFUSIONDRIVE_ROOT`` at a working copy of ``training/`` (it also
provides ``scripts/nureasoning/nureasoning_clip_converter.py``, reused here so
inference encodes inputs exactly as training did) that contains the nuReasoning
changes (``trajectory_only``, ``image_pretrained``, ``target_cache_name``,
pose-count-aware ``TrajectoryHead``, empty-LiDAR fallback); an unmodified
NAVSIM/DiffusionDrive tree will fail to load the checkpoint. See
``diffusiondrive_nureasoning_modifications.md`` in this folder.

Training-time settings that inference must match (taken from the run's
``code/hydra/overrides.yaml``):

  * trajectory sampling: 5 s horizon, 0.5 s interval -> 10 poses
  * ``trajectory_only=True`` (no BEV-semantic / agent heads)
  * ``image_pretrained=False`` (weights come from the checkpoint)
  * ``target_cache_name="transfuser_target_5s"``
  * plan anchors: ``nureasoning_train_20x10_anchors.npy`` (20 modes x 10 poses)
  * 4 history frames at 2 Hz; front / front-left / front-right cameras only

Configuration (used when the provider is built without arguments, as
``--planning-provider diffusion_drive`` does):

  ``DIFFUSIONDRIVE_ROOT``         DiffusionDrive repo root (required)
  ``DIFFUSIONDRIVE_CHECKPOINT``   5 s Lightning checkpoint (required)
  ``DIFFUSIONDRIVE_PLAN_ANCHORS`` 20x10 anchor file the checkpoint was trained
                                  with (required)
  ``DIFFUSIONDRIVE_DEVICE``       ``cuda`` (default), ``cpu`` or ``auto``

Produce a challenge submission with
``python -m nureasoning.submission.challenge --planning-provider diffusion_drive``.
``python -m nureasoning.planning.benchmark`` only offers ``--mode gt|vla``; to
run the planning benchmark with this provider use
``training/scripts/nureasoning/run_diffusiondrive_official_benchmark.py``.
Training code and full instructions: ``training_and_inference_instructions.md``.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import BaseTrajectoryProvider, register_baseline


@register_baseline("diffusion_drive")
class DiffusionDriveTrajectoryProvider(BaseTrajectoryProvider):
    """Five-second DiffusionDrive planner for the nuReasoning benchmark."""

    horizon_s = 5.0
    dt_s = 0.1
    model_dt_s = 0.5
    model_num_poses = 10
    model_history_frames = 4
    model_frame_rate_hz = 2.0

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        plan_anchor_path: Optional[str] = None,
        device: Optional[str] = None,
        diffusiondrive_root: Optional[str] = None,
    ) -> None:
        root_str = diffusiondrive_root or os.environ.get("DIFFUSIONDRIVE_ROOT")
        if not root_str:
            raise ValueError(
                "Set DIFFUSIONDRIVE_ROOT (or pass diffusiondrive_root) to the "
                "DiffusionDrive checkout that contains the nuReasoning changes"
            )
        root = Path(root_str).expanduser()
        checkpoint_path = checkpoint_path or os.environ.get("DIFFUSIONDRIVE_CHECKPOINT")
        plan_anchor_path = plan_anchor_path or os.environ.get("DIFFUSIONDRIVE_PLAN_ANCHORS")
        if not checkpoint_path or not plan_anchor_path:
            raise ValueError(
                "Set DIFFUSIONDRIVE_CHECKPOINT and DIFFUSIONDRIVE_PLAN_ANCHORS "
                "(or pass checkpoint_path and plan_anchor_path)"
            )
        device = device or os.environ.get("DIFFUSIONDRIVE_DEVICE", "cuda")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.plan_anchor_path = Path(plan_anchor_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")
        if not self.plan_anchor_path.is_file():
            raise FileNotFoundError(f"Plan anchors do not exist: {self.plan_anchor_path}")

        # Heavy / optional dependencies are imported here so that registering
        # the baseline never breaks environments without the NAVSIM stack.
        scripts_dir = root / "scripts" / "nureasoning"
        for entry in (scripts_dir, root):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        import torch
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )
        from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
        from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
        from navsim.common.dataclasses import SensorConfig

        # The training-side converter is the single source of truth for the
        # camera/command encodings. Importing it also installs the patch that
        # lets nuplan read "binary_compressed" LiDAR PCD files; without it,
        # clips with such LiDAR fail to load.
        import nureasoning_clip_converter as converter

        self._converter = converter
        self._torch = torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)

        config = TransfuserConfig(
            trajectory_sampling=TrajectorySampling(
                time_horizon=self.horizon_s,
                interval_length=self.model_dt_s,
            ),
            image_pretrained=False,
            plan_anchor_path=str(self.plan_anchor_path),
            target_cache_name="transfuser_target_5s",
            trajectory_only=True,
        )
        if config.trajectory_sampling.num_poses != self.model_num_poses:
            raise ValueError(
                f"Five-second configuration must produce {self.model_num_poses} "
                f"poses, got {config.trajectory_sampling.num_poses}"
            )

        self.config = config
        # TransfuserAgent prints "Initializing from scratch" without a
        # checkpoint; weights are loaded below with a strict state-dict check.
        with redirect_stdout(io.StringIO()):
            self.agent = TransfuserAgent(config=config, lr=0.0, checkpoint_path=None)
        self.checkpoint_tensor_count = self._load_checkpoint(self.checkpoint_path)
        self.agent = self.agent.to(self.device).eval()

        # The feature builder consumes only the front / front-left / front-right
        # cameras at the key frame; other cameras would only add I/O.
        key = [self.model_history_frames - 1]
        self.sensor_config = SensorConfig(
            cam_f0=key, cam_l0=key, cam_l1=False, cam_l2=False,
            cam_r0=key, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=key,
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        checkpoint = self._torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        normalized: Dict[str, Any] = {}
        for name, value in state_dict.items():
            if name.startswith("agent."):
                name = name.removeprefix("agent.")
            elif name.startswith("model."):
                name = "_transfuser_model." + name.removeprefix("model.")
            normalized[name] = value
        self.agent.load_state_dict(normalized, strict=True)
        return len(normalized)

    @staticmethod
    def _load_metadata(clip_path: Path) -> Dict[str, Any]:
        with open(clip_path / "metadata.json", encoding="utf-8") as handle:
            return json.load(handle)

    def _history_indices(self, metadata: Dict[str, Any], key_frame_idx: int) -> List[int]:
        source_rate_hz = float(metadata.get("frame_rate_hz", 0.0))
        if source_rate_hz <= 0:
            raise ValueError("metadata.json has no positive frame_rate_hz")
        frame_interval = max(1, round(source_rate_hz / self.model_frame_rate_hz))
        actual_rate_hz = source_rate_hz / frame_interval
        if not math.isclose(actual_rate_hz, self.model_frame_rate_hz, abs_tol=1e-6):
            raise ValueError(
                f"Clip rate {source_rate_hz:g} Hz cannot be sampled at "
                f"{self.model_frame_rate_hz:g} Hz"
            )
        return [
            max(0, key_frame_idx - offset * frame_interval)
            for offset in range(self.model_history_frames - 1, -1, -1)
        ]

    def _frame_to_agent_input_dict(
        self, clip_path: Path, metadata: Dict[str, Any], frame_idx: int
    ) -> Dict[str, Any]:
        conv = self._converter
        frame = metadata["frames"][frame_idx]
        ego_raw = conv._ego_state_to_raw(clip_path / frame["ego_state"])
        sensors = frame.get("sensors") or {}
        cameras: Dict[str, Dict[str, Any]] = {}
        for camera_key, relative_path in (sensors.get("cameras") or {}).items():
            calibration = metadata["camera_calibrations"][
                conv._CAMERA_KEY_TO_CALIB_NAME[camera_key]
            ]
            cameras[conv._CAMERA_KEY_TO_NAVSIM_NAME[camera_key]] = {
                "data_path": relative_path,
                "sensor2lidar_rotation": calibration["sensor2lidar_rotation"],
                "sensor2lidar_translation": calibration["sensor2lidar_translation"],
                "cam_intrinsic": calibration["intrinsic"],
                "distortion": calibration.get("distortion", [0.0] * 5),
            }
        # Challenge clips carry cameras but no LiDAR; the LiDAR channel is
        # disabled in ``_build_agent_input`` when this path is empty.
        lidar_path = (sensors.get("lidar") or {}).get("point_cloud_path") or ""
        mission_goal = frame.get("mission_goal") or {}
        command = mission_goal.get("command", "") if isinstance(mission_goal, dict) else ""
        return {
            "ego2global_translation": ego_raw["ego2global_translation"],
            "ego2global_rotation": ego_raw["ego2global_rotation"],
            "ego_dynamic_state": ego_raw["ego_dynamic_state"],
            "driving_command": conv.driving_command_onehot(command),
            "cams": cameras,
            "lidar_path": lidar_path,
        }

    def _build_agent_input(
        self, clip_path: Path, metadata: Dict[str, Any], key_frame_idx: int
    ):
        from navsim.common.dataclasses import AgentInput

        scene_dicts = [
            self._frame_to_agent_input_dict(clip_path, metadata, idx)
            for idx in self._history_indices(metadata, key_frame_idx)
        ]
        sensor_config = self.sensor_config
        if not scene_dicts[-1]["lidar_path"]:
            sensor_config = replace(sensor_config, lidar_pc=False)
        return AgentInput.from_scene_dict_list(
            scene_dict_list=scene_dicts,
            sensor_blobs_path=clip_path,
            num_history_frames=self.model_history_frames,
            sensor_config=sensor_config,
        )

    def _predict_local_trajectory(self, agent_input) -> np.ndarray:
        features: Dict[str, Any] = {}
        for builder in self.agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0).to(self.device) for k, v in features.items()}
        with self._torch.inference_mode():
            prediction = self.agent(features)["trajectory"]
        trajectory = prediction[0].detach().float().cpu().numpy()
        expected_shape = (self.model_num_poses, 3)
        if trajectory.shape != expected_shape:
            raise ValueError(
                f"DiffusionDrive produced {trajectory.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("DiffusionDrive produced non-finite trajectory values")
        return trajectory.astype(np.float64, copy=False)

    def _interpolate_to_benchmark_grid(self, local_trajectory: np.ndarray) -> np.ndarray:
        """Resample the 10 x 0.5 s poses (plus the origin) onto the 0.1 s grid."""
        source = np.vstack([np.zeros((1, 3)), local_trajectory])
        source_times = np.arange(len(source), dtype=np.float64) * self.model_dt_s
        target_times = np.arange(self.num_steps, dtype=np.float64) * self.dt_s

        x = np.interp(target_times, source_times, source[:, 0])
        y = np.interp(target_times, source_times, source[:, 1])
        yaw = np.interp(target_times, source_times, np.unwrap(source[:, 2]))
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        out = np.column_stack([x, y, yaw])
        out[0] = 0.0
        return out

    @staticmethod
    def _ego_to_global(local_trajectory: np.ndarray, ego_state: Any) -> np.ndarray:
        pose_raw = (
            ego_state.get("pose", {})
            if isinstance(ego_state, dict)
            else getattr(ego_state, "pose", {})
        )
        pose = pose_raw if isinstance(pose_raw, dict) else {}
        ego_x = float(pose.get("x", 0.0))
        ego_y = float(pose.get("y", 0.0))
        ego_yaw = float(pose.get("yaw", 0.0))
        cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)

        out = np.empty_like(local_trajectory, dtype=np.float64)
        out[:, 0] = ego_x + cos_yaw * local_trajectory[:, 0] - sin_yaw * local_trajectory[:, 1]
        out[:, 1] = ego_y + sin_yaw * local_trajectory[:, 0] + cos_yaw * local_trajectory[:, 1]
        out[:, 2] = (ego_yaw + local_trajectory[:, 2] + math.pi) % (2.0 * math.pi) - math.pi
        return out

    def __call__(
        self,
        clip_path: str,
        key_frame_idx: int,
        ego_state: Any,
    ) -> Optional[np.ndarray]:
        clip = Path(clip_path)
        metadata = self._load_metadata(clip)
        frames = metadata.get("frames") or []
        if not 0 <= key_frame_idx < len(frames):
            raise IndexError(
                f"Key frame {key_frame_idx} is outside clip with {len(frames)} frames"
            )
        agent_input = self._build_agent_input(clip, metadata, key_frame_idx)
        local_native = self._predict_local_trajectory(agent_input)
        local_benchmark = self._interpolate_to_benchmark_grid(local_native)
        global_trajectory = self._ego_to_global(local_benchmark, ego_state)
        if global_trajectory.shape != (self.num_steps, 3):
            raise ValueError(
                f"Provider produced {global_trajectory.shape}; "
                f"expected ({self.num_steps}, 3)"
            )
        if not np.all(np.isfinite(global_trajectory)):
            raise ValueError("Provider produced non-finite global trajectory values")
        return global_trajectory
