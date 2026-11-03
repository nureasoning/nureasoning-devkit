"""Materialize the minimal files needed for the official planning benchmark.

The current nuReasoning validation download stores one scenario per TAR file.
Extracting every camera and lidar sample would duplicate multiple terabytes even
though DiffusionDrive and the planning evaluator consume only a small subset.
This script creates a resumable, non-destructive evaluation view and retains all
source archives unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tarfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/path/to/validation_archives")
DEFAULT_OUTPUT = REPO_ROOT / "data/nureasoning_validation_planning_eval"
CAMERA_KEYS = ("front", "front_left", "front_right")
KEY_FRAME_INDEX = 100
TRAJECTORY_DT_S = 0.1
TRAJECTORY_STEPS = 51
MODEL_FRAME_RATE_HZ = 2.0
MODEL_HISTORY_FRAMES = 4
COMPLETE_MARKER = ".planning_eval_complete.json"
MATERIALIZATION_SCHEMA_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--key-frame-index", type=int, default=KEY_FRAME_INDEX)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of independent archives to materialize concurrently.",
    )
    parser.add_argument(
        "--clip-name",
        help="Materialize one exact clip name (useful for a smoke test).",
    )
    return parser.parse_args()


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe relative path in metadata: {value!r}")
    return path


def _frame_dt_s(frames: Sequence[Mapping[str, Any]]) -> float:
    for previous, current in zip(frames, frames[1:]):
        t0 = previous.get("relative_time_s")
        t1 = current.get("relative_time_s")
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and t1 > t0:
            return float(t1) - float(t0)
    raise ValueError("Metadata has no increasing relative frame timestamps")


def _selection(
    metadata: Mapping[str, Any], key_frame_index: int
) -> Tuple[List[PurePosixPath], Dict[str, Any]]:
    frames = metadata.get("frames") or []
    if not frames:
        raise ValueError("Metadata contains no frames")
    if not 0 <= key_frame_index < len(frames):
        raise ValueError(
            f"Key frame {key_frame_index} is outside clip with {len(frames)} frames"
        )

    frame_rate_hz = float(metadata.get("frame_rate_hz", 0.0))
    if frame_rate_hz <= 0:
        raise ValueError("Metadata has no positive frame_rate_hz")

    frame_dt_s = _frame_dt_s(frames)
    annotation_stride = max(1, int(round(TRAJECTORY_DT_S / frame_dt_s)))
    actual_annotation_dt_s = annotation_stride * frame_dt_s
    if not math.isclose(
        actual_annotation_dt_s,
        TRAJECTORY_DT_S,
        rel_tol=0.02,
        abs_tol=0.005,
    ):
        raise ValueError(
            f"Frame interval {frame_dt_s:g}s cannot represent the official "
            f"{TRAJECTORY_DT_S:g}s evaluation interval"
        )

    history_stride = max(1, int(round(frame_rate_hz / MODEL_FRAME_RATE_HZ)))
    actual_history_rate_hz = frame_rate_hz / history_stride
    if not math.isclose(actual_history_rate_hz, MODEL_FRAME_RATE_HZ, abs_tol=1e-6):
        raise ValueError(
            f"Clip rate {frame_rate_hz:g}Hz cannot be sampled at "
            f"{MODEL_FRAME_RATE_HZ:g}Hz"
        )
    first_history_index = key_frame_index - (
        MODEL_HISTORY_FRAMES - 1
    ) * history_stride
    if first_history_index < 0:
        raise ValueError("Key frame does not have enough DiffusionDrive history")

    selected = {
        _safe_relative_path("metadata.json"),
        _safe_relative_path(str(metadata.get("map_annotation", "map.pkl"))),
    }

    history_indices = [
        key_frame_index - offset * history_stride
        for offset in range(MODEL_HISTORY_FRAMES - 1, -1, -1)
    ]
    for frame_index in history_indices:
        selected.add(_safe_relative_path(str(frames[frame_index]["ego_state"])))

    current_cameras = frames[key_frame_index]["sensors"]["cameras"]
    for camera_key in CAMERA_KEYS:
        selected.add(_safe_relative_path(str(current_cameras[camera_key])))

    current_lidar = (
        frames[key_frame_index]
        .get("sensors", {})
        .get("lidar", {})
        .get("point_cloud_path")
        or ""
    )
    if current_lidar:
        selected.add(_safe_relative_path(str(current_lidar)))

    annotation_indices = [
        key_frame_index + step * annotation_stride
        for step in range(TRAJECTORY_STEPS)
        if key_frame_index + step * annotation_stride < len(frames)
    ]
    for frame_index in annotation_indices:
        selected.add(_safe_relative_path(str(frames[frame_index]["annotations"])))

    profile = {
        "frame_rate_hz": frame_rate_hz,
        "frame_dt_s": frame_dt_s,
        "key_frame_index": key_frame_index,
        "history_indices": history_indices,
        "annotation_indices": annotation_indices,
        "available_annotation_steps": len(annotation_indices),
        "requested_annotation_steps": TRAJECTORY_STEPS,
        "camera_keys": list(CAMERA_KEYS),
        "lidar_included": bool(current_lidar),
        "trajectory_dt_s": TRAJECTORY_DT_S,
        "trajectory_steps": TRAJECTORY_STEPS,
    }
    return sorted(selected, key=str), profile


def _write_stream(stream: Any, destination: Path, expected_size: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    with open(temporary, "wb") as output:
        shutil.copyfileobj(stream, output, length=1024 * 1024)
    if temporary.stat().st_size != expected_size:
        raise IOError(
            f"Short copy for {destination}: got {temporary.stat().st_size}, "
            f"expected {expected_size}"
        )
    os.replace(temporary, destination)


def _copy_file(source: Path, destination: Path) -> None:
    expected_size = source.stat().st_size
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != expected_size:
        raise IOError(f"Short copy for {destination}")
    os.replace(temporary, destination)


def _load_directory_metadata(clip_dir: Path) -> Dict[str, Any]:
    with open(clip_dir / "metadata.json", encoding="utf-8") as handle:
        return json.load(handle)


def _materialize_directory(
    clip_dir: Path,
    output_dir: Path,
    selected: Iterable[PurePosixPath],
) -> None:
    for relative_path in selected:
        source = clip_dir.joinpath(*relative_path.parts)
        if not source.is_file():
            raise FileNotFoundError(f"Required source file is missing: {source}")
        _copy_file(source, output_dir.joinpath(*relative_path.parts))


def _tar_members(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        name = str(_safe_relative_path(member.name.rstrip("/")))
        if name in members:
            raise ValueError(f"Duplicate TAR member: {name}")
        members[name] = member
    return members


def _read_tar_metadata(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    clip_name: str,
) -> Dict[str, Any]:
    metadata_name = f"{clip_name}/metadata.json"
    member = members.get(metadata_name)
    if member is None or not member.isfile():
        raise FileNotFoundError(f"{metadata_name} is missing from {archive.name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise IOError(f"Could not read {metadata_name} from {archive.name}")
    with stream:
        return json.load(stream)


def _materialize_tar(
    archive_path: Path,
    output_dir: Path,
    key_frame_index: int,
) -> Tuple[List[PurePosixPath], Dict[str, Any]]:
    clip_name = archive_path.stem
    with tarfile.open(archive_path, mode="r:") as archive:
        members = _tar_members(archive)
        metadata = _read_tar_metadata(archive, members, clip_name)
        selected, profile = _selection(metadata, key_frame_index)
        for relative_path in selected:
            member_name = f"{clip_name}/{relative_path}"
            member = members.get(member_name)
            if member is None or not member.isfile():
                raise FileNotFoundError(
                    f"Required TAR member is missing: {archive_path}:{member_name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise IOError(f"Could not read {archive_path}:{member_name}")
            with stream:
                _write_stream(
                    stream,
                    output_dir.joinpath(*relative_path.parts),
                    member.size,
                )
    return selected, profile


def _is_complete(
    output_dir: Path,
    source_path: Path,
    key_frame_index: int,
) -> bool:
    marker_path = output_dir / COMPLETE_MARKER
    if not marker_path.is_file():
        return False
    try:
        with open(marker_path, encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, ValueError):
        return False
    if marker.get("source_path") != str(source_path.resolve()):
        return False
    if marker.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        return False
    if marker.get("key_frame_index") != key_frame_index:
        return False
    required_files = marker.get("required_files") or []
    return bool(required_files) and all(
        output_dir.joinpath(*_safe_relative_path(path).parts).is_file()
        for path in required_files
    )


def _write_marker(
    output_dir: Path,
    source_path: Path,
    selected: Iterable[PurePosixPath],
    profile: Mapping[str, Any],
) -> None:
    marker = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "source_path": str(source_path.resolve()),
        "key_frame_index": profile["key_frame_index"],
        "required_files": [str(path) for path in selected],
        "profile": dict(profile),
    }
    marker_path = output_dir / COMPLETE_MARKER
    temporary = marker_path.with_name(marker_path.name + ".part")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2)
    os.replace(temporary, marker_path)


def _discover_sources(source_root: Path) -> List[Tuple[str, Path]]:
    archives = {path.stem: path for path in source_root.rglob("*.tar")}
    directories = {
        path.parent.name: path.parent for path in source_root.rglob("metadata.json")
    }
    clip_names = sorted(set(archives) | set(directories))
    sources: List[Tuple[str, Path]] = []
    for clip_name in clip_names:
        # A complete extracted directory avoids reopening its retained TAR.
        if clip_name in directories:
            sources.append((clip_name, directories[clip_name]))
        else:
            sources.append((clip_name, archives[clip_name]))
    return sources


def _materialize_one(
    clip_name: str,
    source_path: Path,
    output_root: Path,
    key_frame_index: int,
) -> str:
    output_dir = output_root / clip_name
    if _is_complete(output_dir, source_path, key_frame_index):
        return "skip"

    if source_path.is_dir():
        metadata = _load_directory_metadata(source_path)
        selected, profile = _selection(metadata, key_frame_index)
        _materialize_directory(source_path, output_dir, selected)
    else:
        selected, profile = _materialize_tar(
            source_path,
            output_dir,
            key_frame_index,
        )
    _write_marker(output_dir, source_path, selected, profile)
    return "materialized"


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {args.source_root}")
    if args.key_frame_index < 0:
        raise SystemExit("--key-frame-index cannot be negative")
    if args.max_clips < 0:
        raise SystemExit("--max-clips cannot be negative")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    sources = _discover_sources(args.source_root)
    if args.clip_name:
        sources = [item for item in sources if item[0] == args.clip_name]
        if not sources:
            raise SystemExit(f"Clip was not found: {args.clip_name}")
    if args.max_clips:
        sources = sources[: args.max_clips]
    if not sources:
        raise SystemExit(f"No clip directories or TAR files found under {args.source_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Found {len(sources)} unique clips to materialize with "
        f"{args.workers} worker(s)",
        flush=True,
    )
    completed = 0
    skipped = 0
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures: Dict[Future[str], str] = {
        executor.submit(
            _materialize_one,
            clip_name,
            source_path,
            args.output_root,
            args.key_frame_index,
        ): clip_name
        for clip_name, source_path in sources
    }
    try:
        for index, future in enumerate(as_completed(futures), 1):
            clip_name = futures[future]
            status = future.result()
            if status == "skip":
                skipped += 1
            else:
                completed += 1
            print(
                f"[{index}/{len(sources)}] {status} {clip_name}",
                flush=True,
            )
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    print(
        f"Materialization complete: {completed} written, {skipped} already complete, "
        f"{len(sources)} total",
        flush=True,
    )


if __name__ == "__main__":
    main()
