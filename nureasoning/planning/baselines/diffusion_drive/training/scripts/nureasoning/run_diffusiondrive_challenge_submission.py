"""Generate a nuReasoning challenge submission with the trained DiffusionDrive planner."""

from __future__ import annotations

import argparse
from pathlib import Path

from nureasoning.submission.challenge import run_challenge_submission
from nureasoning.submission.reasoning import StubChallengeAnswerer

from diffusiondrive_trajectory_provider import DiffusionDriveTrajectoryProvider


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/nureasoning_official/data/test"
DEFAULT_ANCHORS = (
    REPO_ROOT
    / "data/nureasoning_non_vqa_cache/nureasoning_train_20x10_anchors.npy"
)
DEFAULT_OUTPUT = REPO_ROOT / "exp/nureasoning_diffusiondrive_challenge_submission.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--plan-anchor-path", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument(
        "--reasoning-seed",
        type=int,
        default=0,
        help="Seed for the official random-answer stub (this checkpoint is non-VQA)",
    )
    parser.add_argument(
        "--skip-reasoning",
        action="store_true",
        help="Write empty answer lists; intended only for partial/debug output",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write debugging output despite failed clips or missing answers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"Challenge data root not found: {args.data_root}")
    if not args.checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint_path}")
    if not args.plan_anchor_path.is_file():
        raise SystemExit(f"Plan anchors not found: {args.plan_anchor_path}")

    provider = DiffusionDriveTrajectoryProvider(
        checkpoint_path=str(args.checkpoint_path),
        plan_anchor_path=str(args.plan_anchor_path),
        device=args.device,
    )
    reasoning_answerer = (
        None if args.skip_reasoning else StubChallengeAnswerer(seed=args.reasoning_seed)
    )
    run_challenge_submission(
        data_root=str(args.data_root),
        output_path=str(args.output),
        planning_provider=provider,
        reasoning_answerer=reasoning_answerer,
        max_clips=args.max_clips,
        skip_reasoning=args.skip_reasoning,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
