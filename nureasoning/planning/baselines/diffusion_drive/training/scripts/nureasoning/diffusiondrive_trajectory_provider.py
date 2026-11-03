"""nuReasoning planning-benchmark provider for the adapted DiffusionDrive model."""

from __future__ import annotations

import json
import math
import io
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from nureasoning.planning.baselines.base import BaseTrajectoryProvider

from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput, SensorConfig

from nureasoning_clip_converter import (
    _CAMERA_KEY_TO_CALIB_NAME,
    _CAMERA_KEY_TO_NAVSIM_NAME,
    _ego_state_to_raw,
    driving_command_onehot,
)


class DiffusionDriveTrajectoryProvider(BaseTrajectoryProvider):
    """Run the five-second DiffusionDrive checkpoint on one devkit key frame."""

    horizon_s = 5.0
    dt_s = 0.1
    model_dt_s = 0.5
    model_num_poses = 10
    model_history_frames = 4
    model_frame_rate_hz = 2.0

    def __init__(
        self,
        checkpoint_path: str,
        plan_anchor_path: str,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.plan_anchor_path = Path(plan_anchor_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")
        if not self.plan_anchor_path.is_file():
            raise FileNotFoundError(f"Plan anchors do not exist: {self.plan_anchor_path}")

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
                "Five-second DiffusionDrive configuration must produce "
                f"{self.model_num_poses} poses, got "
                f"{config.trajectory_sampling.num_poses}"
            )

        self.config = config
        # The NAVSIM agent prints "Initializing from scratch" whenever its
        # constructor is used without a checkpoint.  Loading is handled below
        # with a strict state-dict check, so suppress that misleading message.
        with redirect_stdout(io.StringIO()):
            self.agent = TransfuserAgent(config=config, lr=0.0, checkpoint_path=None)
        self.checkpoint_tensor_count = self._load_checkpoint(self.checkpoint_path)
        self.agent = self.agent.to(self.device).eval()

        # DiffusionDrive's feature builder consumes only these three cameras.
        # Loading the other five cameras would add I/O without changing input.
        lidar_include = [self.model_history_frames - 1]
        self.sensor_config = SensorConfig(
            cam_f0=[self.model_history_frames - 1],
            cam_l0=[self.model_history_frames - 1],
            cam_l1=False,
            cam_l2=False,
            cam_r0=[self.model_history_frames - 1],
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=lidar_include,
        )

    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        normalized_state: Dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            if name.startswith("agent."):
                name = name.removeprefix("agent.")
            elif name.startswith("model."):
                name = "_transfuser_model." + name.removeprefix("model.")
            normalized_state[name] = value
        self.agent.load_state_dict(normalized_state, strict=True)
        return len(normalized_state)

    @staticmethod
    def _load_metadata(clip_path: Path) -> Dict[str, Any]:
        with open(clip_path / "metadata.json", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _frame_rate_hz(metadata: Dict[str, Any]) -> float:
        frame_rate_hz = float(metadata.get("frame_rate_hz", 0.0))
        if frame_rate_hz <= 0:
            raise ValueError("metadata.json has no positive frame_rate_hz")
        return frame_rate_hz

    def _history_indices(
        self,
        metadata: Dict[str, Any],
        key_frame_idx: int,
    ) -> List[int]:
        source_rate_hz = self._frame_rate_hz(metadata)
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

    @staticmethod
    def _frame_to_agent_input_dict(
        clip_path: Path,
        metadata: Dict[str, Any],
        frame_idx: int,
    ) -> Dict[str, Any]:
        frame = metadata["frames"][frame_idx]
        ego_raw = _ego_state_to_raw(clip_path / frame["ego_state"])
        sensors = frame.get("sensors") or {}
        cameras: Dict[str, Dict[str, Any]] = {}
        for camera_key, relative_path in (sensors.get("cameras") or {}).items():
            calibration = metadata["camera_calibrations"][
                _CAMERA_KEY_TO_CALIB_NAME[camera_key]
            ]
            cameras[_CAMERA_KEY_TO_NAVSIM_NAME[camera_key]] = {
                "data_path": relative_path,
                "sensor2lidar_rotation": calibration["sensor2lidar_rotation"],
                "sensor2lidar_translation": calibration["sensor2lidar_translation"],
                "cam_intrinsic": calibration["intrinsic"],
                "distortion": calibration.get(
                    "distortion", [0.0, 0.0, 0.0, 0.0, 0.0]
                ),
            }
        # Challenge test clips contain cameras but intentionally omit LiDAR.
        # AgentInput disables the LiDAR channel below when this path is empty.
        lidar_path = (sensors.get("lidar") or {}).get("point_cloud_path") or ""
        mission_goal = frame.get("mission_goal") or {}
        command = mission_goal.get("command", "") if isinstance(mission_goal, dict) else ""
        return {
            "ego2global_translation": ego_raw["ego2global_translation"],
            "ego2global_rotation": ego_raw["ego2global_rotation"],
            "ego_dynamic_state": ego_raw["ego_dynamic_state"],
            "driving_command": driving_command_onehot(command),
            "cams": cameras,
            "lidar_path": lidar_path,
        }

    def _build_agent_input(
        self,
        clip_path: Path,
        metadata: Dict[str, Any],
        key_frame_idx: int,
    ) -> AgentInput:
        frame_indices = self._history_indices(metadata, key_frame_idx)
        scene_dicts = [
            self._frame_to_agent_input_dict(clip_path, metadata, frame_idx)
            for frame_idx in frame_indices
        ]

        current_lidar = scene_dicts[-1]["lidar_path"]
        sensor_config = self.sensor_config
        if not current_lidar:
            sensor_config = replace(sensor_config, lidar_pc=False)

        return AgentInput.from_scene_dict_list(
            scene_dict_list=scene_dicts,
            sensor_blobs_path=clip_path,
            num_history_frames=self.model_history_frames,
            sensor_config=sensor_config,
        )

    def _predict_local_trajectory(self, agent_input: AgentInput) -> np.ndarray:
        features: Dict[str, torch.Tensor] = {}
        for builder in self.agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in features.items()
        }
        with torch.inference_mode():
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
        current = np.zeros((1, 3), dtype=np.float64)
        source = np.vstack([current, local_trajectory])
        source_times = np.arange(len(source), dtype=np.float64) * self.model_dt_s
        target_times = np.arange(self.num_steps, dtype=np.float64) * self.dt_s

        x = np.interp(target_times, source_times, source[:, 0])
        y = np.interp(target_times, source_times, source[:, 1])
        yaw = np.interp(target_times, source_times, np.unwrap(source[:, 2]))
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        interpolated = np.column_stack([x, y, yaw])
        interpolated[0] = 0.0
        return interpolated

    @staticmethod
    def _ego_to_global(local_trajectory: np.ndarray, ego_state: Any) -> np.ndarray:
        pose_raw = ego_state.get("pose", {}) if isinstance(ego_state, dict) else getattr(ego_state, "pose", {})
        pose = pose_raw if isinstance(pose_raw, dict) else {}
        ego_x = float(pose.get("x", 0.0))
        ego_y = float(pose.get("y", 0.0))
        ego_yaw = float(pose.get("yaw", 0.0))
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)

        global_trajectory = np.empty_like(local_trajectory, dtype=np.float64)
        global_trajectory[:, 0] = (
            ego_x
            + cos_yaw * local_trajectory[:, 0]
            - sin_yaw * local_trajectory[:, 1]
        )
        global_trajectory[:, 1] = (
            ego_y
            + sin_yaw * local_trajectory[:, 0]
            + cos_yaw * local_trajectory[:, 1]
        )
        global_trajectory[:, 2] = (
            ego_yaw + local_trajectory[:, 2] + math.pi
        ) % (2.0 * math.pi) - math.pi
        return global_trajectory

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
