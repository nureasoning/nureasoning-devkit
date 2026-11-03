"""
Converts one nuReasoning clip directory into NAVSIM `Scene` objects.

Verified against the nuReasoning devkit and sample files (metadata.json,
map.pkl, ego_state/*.pkl, annotations/*.pkl):
  - metadata.json: frame_rate_hz=10, so callers must resample to NAVSIM's
    native 2 Hz cadence before building a Scene; 8 cameras
    (front, front_left, front_right, left, right, back, back_left,
    back_right -> CAM_M_F/L0/R0/L1/R1/B/L2/R2), per-frame paths for
    ego_state/annotations/cameras/reasoning, camera_calibrations with
    intrinsic + sensor2lidar_rotation/translation + width/height,
    mission_goal.command + route_path per frame.
  - map.pkl geometries are in the SAME global planar frame as
    mission_goal.route_path (large easting/northing values) -- see
    nureasoning_map_adapter.py.
  - `sensors.lidar.point_cloud_path` was EMPTY in the sample frame. If
    that holds across the dataset, DiffusionDrive's lidar-histogram
    feature branch has nothing to consume -- see the flag at the bottom
    of this file. Don't assume this is just a fluke of one frame; check
    a few more clips before writing training code around it.

The pickle files are loaded through ``nureasoning.common.pickle_io`` so
the converter uses the devkit's schema aliases and typed records rather
than maintaining a second, permissive pickle implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from nureasoning.common.pickle_io import load_pickle
from nureasoning_map_adapter import NuReasoningMapAdapter

# These come from the real navsim/DiffusionDrive install in your training
# environment -- not available in this inspection sandbox, imported lazily
# so this module can still be unit-tested against metadata.json/map.pkl
# without the full navsim/nuplan stack present.
try:
    from navsim.common.dataclasses import (
        Scene,
        SceneMetadata,
        Frame,
        Cameras,
        Lidar,
    )
except ImportError:
    Scene = SceneMetadata = Frame = Cameras = Lidar = None
# --- Patch: support PCD "binary_compressed" lidar files -------------------
# nuplan-devkit's PointCloud.parse_points only supports the plain 'binary'
# PCD variant. A subset of nuReasoning clips uses PCL's 'binary_compressed'
# encoding. Add that branch without modifying the installed nuplan package.
try:
    import struct
    from nuplan.database.utils.pointclouds.pointcloud import PointCloud
except ImportError:
    pass  # navsim/nuplan not importable in this environment -- see below
else:
    try:
        import lzf as _lzf
    except ImportError:
        _lzf = None

    _ORIGINAL_PARSE_POINTS = PointCloud.parse_points

    def _decompress_lzf(compressed_data: bytes, uncompressed_size: int) -> bytes:
        """Decode the LZF stream used by PCL binary_compressed PCD files."""
        if _lzf is not None:
            decompressed = _lzf.decompress(compressed_data, uncompressed_size)
            if decompressed is None:
                raise RuntimeError("LZF decompression failed")
            return decompressed

        output = bytearray(uncompressed_size)
        input_index = 0
        output_index = 0
        compressed_size = len(compressed_data)

        while input_index < compressed_size:
            control = compressed_data[input_index]
            input_index += 1

            if control < 32:
                literal_length = control + 1
                input_end = input_index + literal_length
                output_end = output_index + literal_length
                if input_end > compressed_size or output_end > uncompressed_size:
                    raise RuntimeError("Invalid LZF literal run")
                output[output_index:output_end] = compressed_data[input_index:input_end]
                input_index = input_end
                output_index = output_end
                continue

            match_length = control >> 5
            reference_index = output_index - ((control & 0x1F) << 8) - 1
            if match_length == 7:
                if input_index >= compressed_size:
                    raise RuntimeError("Truncated LZF match length")
                match_length += compressed_data[input_index]
                input_index += 1
            if input_index >= compressed_size:
                raise RuntimeError("Truncated LZF back-reference")
            reference_index -= compressed_data[input_index]
            input_index += 1
            match_length += 2

            if reference_index < 0 or output_index + match_length > uncompressed_size:
                raise RuntimeError("Invalid LZF back-reference")
            for _ in range(match_length):
                output[output_index] = output[reference_index]
                output_index += 1
                reference_index += 1

        if output_index != uncompressed_size:
            raise RuntimeError(
                f"LZF decompression size mismatch: expected {uncompressed_size}, "
                f"got {output_index}"
            )
        return bytes(output)

    def _parse_points_with_compressed_support(stream, header):
        if header.data != "binary_compressed":
            return _ORIGINAL_PARSE_POINTS(stream, header)

        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise RuntimeError("Incomplete binary_compressed size header")
        compressed_size, uncompressed_size = struct.unpack("<II", size_bytes)

        compressed_data = stream.read(compressed_size)
        if len(compressed_data) != compressed_size:
            raise RuntimeError(
                f"Incomplete compressed pointcloud stream: {compressed_size} bytes "
                f"expected, {len(compressed_data)} got"
            )

        expected_layout_size = sum(int(size) for size in header.size) * header.points
        if uncompressed_size != expected_layout_size:
            raise RuntimeError(
                f"PCD field layout requires {expected_layout_size} bytes, "
                f"but the compressed header declares {uncompressed_size}"
            )
        decompressed = _decompress_lzf(compressed_data, uncompressed_size)

        row_type = PointCloud.np_type(header)
        points = np.empty(header.points, dtype=row_type)
        type_mapping = {"I": "int", "U": "uint", "F": "float"}
        offset = 0
        for field_name, size, type_char in zip(header.fields, header.size, header.type):
            np_dtype = np.dtype(type_mapping[type_char] + str(int(size) * 8))
            nbytes = np_dtype.itemsize * header.points
            field_bytes = decompressed[offset:offset + nbytes]
            points[field_name] = np.frombuffer(field_bytes, dtype=np_dtype)
            offset += nbytes
        return points

    PointCloud.parse_points = staticmethod(_parse_points_with_compressed_support)
# ---------------------------------------------------------------------------

# nuScenes-style dotted categories -> NAVSIM's 8 coarse buckets
# (vehicle, pedestrian, bicycle, traffic_cone, barrier, czone_sign,
# generic_object, ego). Only "vehicle.car" has actually been observed in
# the uploaded sample -- extend this once more categories show up.
CATEGORY_MAP = {
    "vehicle.car": "vehicle",
    "vehicle.truck": "vehicle",
    "vehicle.bus": "vehicle",
    "vehicle.construction": "vehicle",
    "vehicle.trailer": "vehicle",
    "vehicle.emergency": "vehicle",
    "vehicle.bicycle": "bicycle",
    "vehicle.personal_mobility.bicycle": "bicycle",
    "human": "pedestrian",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "other.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "generic_object",
}


def map_category(raw_category: str) -> str:
    return CATEGORY_MAP.get(raw_category, "generic_object")


# Observed commands so far: "GO_STRAIGHT". Extend the left/right branches
# once more mission_goal.command values are seen in the wild.
_COMMAND_ORDER = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]


def driving_command_onehot(command_str: str) -> np.ndarray:
    onehot = np.zeros(len(_COMMAND_ORDER), dtype=np.int64)
    normalized = (command_str or "").upper()
    if "LEFT" in normalized:
        onehot[_COMMAND_ORDER.index("LEFT")] = 1
    elif "RIGHT" in normalized:
        onehot[_COMMAND_ORDER.index("RIGHT")] = 1
    elif "STRAIGHT" in normalized:
        onehot[_COMMAND_ORDER.index("STRAIGHT")] = 1
    else:
        onehot[_COMMAND_ORDER.index("UNKNOWN")] = 1
    return onehot


def _ego_state_to_raw(ego_state_path: Path) -> Dict[str, Any]:
    obj = load_pickle(str(ego_state_path))
    pose = obj.pose
    vel = obj.velocity
    acc = obj.acceleration
    return {
        "ego2global_translation": [pose["x"], pose["y"], pose["z"]],
        "ego2global_rotation": [pose["qw"], pose["qx"], pose["qy"], pose["qz"]],
        "ego_dynamic_state": [vel["vx"], vel["vy"], acc["ax"], acc["ay"]],
        "ego_x": pose["x"],
        "ego_y": pose["y"],
        "ego_yaw": pose["yaw"],
    }


def _annotations_to_raw(
    annotations_path: Optional[Path],
    reasoning_path: Optional[Path],
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
) -> Dict[str, Any]:
    import math
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)

    def to_local_xy(gx, gy):
        dx, dy = gx - ego_x, gy - ego_y
        return dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw

    def to_local_vxy(vx, vy):
        return vx * cos_yaw + vy * sin_yaw, -vx * sin_yaw + vy * cos_yaw

    if annotations_path is not None and annotations_path.exists():
        obj = load_pickle(str(annotations_path))
        boxes, names, velocity_3d, track_tokens = [], [], [], []
        for o in obj.objects:
            p, d, v = o.pose, o.dimensions, o.velocity
            lx, ly = to_local_xy(p["x"], p["y"])
            local_yaw = p["yaw"] - ego_yaw
            lvx, lvy = to_local_vxy(v["vx"], v["vy"])
            boxes.append([lx, ly, p["z"], d["l"], d["w"], d["h"], local_yaw])
            names.append(map_category(o.category))
            velocity_3d.append([lvx, lvy, v["vz"]])
            track_tokens.append(o.track_token)
        return {
            "anns": {
                "gt_boxes": np.array(boxes, dtype=np.float32).reshape(-1, 7),
                "gt_names": names,
                "gt_velocity_3d": np.array(velocity_3d, dtype=np.float32).reshape(-1, 3),
                "instance_tokens": list(track_tokens),
                "track_tokens": list(track_tokens),
            }
        }

    # No annotations.pkl for this frame -> zero boxes.
    return {
        "anns": {
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_names": [],
            "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32),
            "instance_tokens": [],
            "track_tokens": [],
        }
    }


def build_scene_dict_list(
    clip_dir: Path,
    metadata: Dict[str, Any],
    frame_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Builds the list of raw per-frame dicts in exactly the schema
    `navsim.common.dataclasses.Scene.from_scene_dict_list` expects
    (verified against navsim_ref/navsim/common/dataclasses.py):
    token, timestamp, log_name, scene_token, map_location, roadblock_ids,
    traffic_lights, ego2global_translation, ego2global_rotation,
    ego_dynamic_state, driving_command, cams, lidar_path, anns.
    """
    log_name = metadata["log_name"]
    scene_dicts: List[Dict[str, Any]] = []

    frames = metadata["frames"]
    if frame_indices is not None:
        frames = [frames[index] for index in frame_indices]
    for frame in frames:
        ego_raw = _ego_state_to_raw(clip_dir / frame["ego_state"])


        annotations_path = clip_dir / frame["annotations"] if frame.get("annotations") else None
        anns_raw = _annotations_to_raw(
            annotations_path,
            None,
            ego_x=ego_raw["ego_x"],
            ego_y=ego_raw["ego_y"],
            ego_yaw=ego_raw["ego_yaw"],
        )
        cams = {}
        for cam_key, rel_path in frame["sensors"]["cameras"].items():
            calib = metadata["camera_calibrations"][_CAMERA_KEY_TO_CALIB_NAME[cam_key]]
            cams[_CAMERA_KEY_TO_NAVSIM_NAME[cam_key]] = {
                "data_path": rel_path,
                "sensor2lidar_rotation": calib["sensor2lidar_rotation"],
                "sensor2lidar_translation": calib["sensor2lidar_translation"],
                "cam_intrinsic": calib["intrinsic"],
                "distortion": calib.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0]),
            }

        lidar_rel_path = frame["sensors"]["lidar"].get("point_cloud_path") or None

        scene_dicts.append(
            {
                "token": frame["token"],
                "timestamp": frame["timestamp_us"],
                "log_name": log_name,
                "scene_token": metadata["clip_token"],
                "map_location": metadata["clip_location"],  # not a nuplan MAP_LOCATIONS value -- see below
                "roadblock_ids": [],  # not used by DiffusionDrive's builders
                "traffic_lights": [],  # not used by DiffusionDrive's builders
                "ego2global_translation": ego_raw["ego2global_translation"],
                "ego2global_rotation": ego_raw["ego2global_rotation"],
                "ego_dynamic_state": ego_raw["ego_dynamic_state"],
                "driving_command": driving_command_onehot(frame["mission_goal"]["command"]),
                "cams": cams,
                "lidar_path": lidar_rel_path,
                "anns": anns_raw["anns"],
            }
        )
    return scene_dicts


# metadata.json camera keys -> (a) matching camera_calibrations key, (b) navsim Cameras field name
_CAMERA_KEY_TO_CALIB_NAME = {
    "front": "CAM_M_F",
    "front_left": "CAM_M_L0",
    "front_right": "CAM_M_R0",
    "left": "CAM_M_L1",
    "right": "CAM_M_R1",
    "back": "CAM_M_B",
    "back_left": "CAM_M_L2",
    "back_right": "CAM_M_R2",
}
_CAMERA_KEY_TO_NAVSIM_NAME = {
    "front": "cam_f0",
    "front_left": "cam_l0",
    "front_right": "cam_r0",
    "left": "cam_l1",
    "right": "cam_r1",
    "back": "cam_b0",
    "back_left": "cam_l2",
    "back_right": "cam_r2",
}


def get_scene_windows(
    num_total_frames: int, num_history_frames: int, num_future_frames: int
) -> List[int]:
    """
    Returns valid start indices for scene windows -- i.e. every possible
    center frame that has enough history and future padding available.
    The input count must already be sampled at NAVSIM's 2 Hz cadence.
    A 41-frame, 2 Hz sequence (20s) with the NAVSIM default (4 history +
    10 future = 14 total) gives 41 - 14 + 1 = 28 usable windows.
    """
    window_size = num_history_frames + num_future_frames
    return list(range(0, num_total_frames - window_size + 1))


def build_scene(
    clip_dir: Path,
    metadata: Dict[str, Any],
    map_adapter: NuReasoningMapAdapter,
    scene_dict_list: List[Dict[str, Any]],
    sensor_blobs_path: Path,
    num_history_frames: int,
    num_future_frames: int,
):
    """
    Builds a real navsim `Scene`, reusing navsim's own per-frame builder
    logic (`Scene._build_ego_status`, `Scene._build_annotations`) for
    correctness, but substituting our NuReasoningMapAdapter in place of
    `Scene._build_map_api` -- nuPlan's map factory only covers 4 fixed
    cities (Las Vegas, Pittsburgh, Boston, Singapore) loaded from its own
    GPKG database; nuReasoning's own per-clip map.pkl is what actually
    has this clip's geometry, and is required regardless of city for
    clips outside those 4.
    """
    if Scene is None:
        raise ImportError(
            "navsim is not importable in this environment. Run this on your "
            "actual NAVSIM/DiffusionDrive training machine, not this sandbox."
        )

    from navsim.common.dataclasses import SensorConfig  # local import, see try/except above

    sensor_config = SensorConfig.build_all_sensors(include=[num_history_frames - 1])

    scene_metadata = SceneMetadata(
        log_name=scene_dict_list[num_history_frames - 1]["log_name"],
        scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
        map_name=scene_dict_list[num_history_frames - 1]["map_location"],
        initial_token=scene_dict_list[num_history_frames - 1]["token"],
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
    )

    frames: List[Frame] = []
    for frame_idx, frame_dict in enumerate(scene_dict_list):
        ego_status = Scene._build_ego_status(frame_dict)
        annotations = Scene._build_annotations(frame_dict)

        sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
        cameras = Cameras.from_camera_dict(
            sensor_blobs_path=sensor_blobs_path,
            camera_dict=frame_dict["cams"],
            sensor_names=sensor_names,
        )
        lidar_path = Path(frame_dict["lidar_path"]) if frame_dict["lidar_path"] else None
        # NAVSIM's Lidar.from_paths only dereferences lidar_path when "lidar_pc"
        # is present in sensor_names -- strip it out for frames with no lidar
        # data so it correctly falls through to the empty-Lidar() branch.
        lidar_sensor_names = sensor_names if lidar_path is not None else [
            s for s in sensor_names if s != "lidar_pc"
        ]
        lidar = Lidar.from_paths(
            sensor_blobs_path=sensor_blobs_path,
            lidar_path=lidar_path,
            sensor_names=lidar_sensor_names,
        )

        frames.append(
            Frame(
                token=frame_dict["token"],
                timestamp=frame_dict["timestamp"],
                roadblock_ids=frame_dict["roadblock_ids"],
                traffic_lights=frame_dict["traffic_lights"],
                annotations=annotations,
                ego_status=ego_status,
                lidar=lidar,
                cameras=cameras,
            )
        )

    return Scene(scene_metadata=scene_metadata, map_api=map_adapter, frames=frames)


if __name__ == "__main__":
    # Runs the parts that don't need navsim/nuplan installed -- validates
    # metadata parsing and window computation against the real uploaded
    # metadata.json.
    import sys

    clip_dir = Path(sys.argv[1])
    with open(clip_dir / "metadata.json") as f:
        metadata = json.load(f)

    windows = get_scene_windows(len(metadata["frames"]), num_history_frames=4, num_future_frames=10)
    print(f"Clip: {metadata['clip_token']} ({metadata['clip_location']})")
    print(f"Total frames: {metadata['total_frames']} @ {metadata['frame_rate_hz']} Hz")
    print(f"Usable scene windows (4 hist + 10 future): {len(windows)}")

    frame = metadata["frames"][0]
    ego = _ego_state_to_raw(clip_dir / frame["ego_state"])
    result = _annotations_to_raw(
        clip_dir / frame["annotations"], None,
        ego_x=ego["ego_x"], ego_y=ego["ego_y"], ego_yaw=ego["ego_yaw"],
    )
    print(f"Frame 0 boxes from devkit annotations: {result['anns']['gt_boxes'].shape}")
    print(f"  names: {result['anns']['gt_names']}")