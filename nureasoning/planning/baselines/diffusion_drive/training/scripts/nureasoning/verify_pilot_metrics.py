"""Verify that a resumed pilot recorded finite, improving validation loss."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_epoch_losses(log_dir: Path, tag: str) -> List[float]:
    event_paths = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_paths:
        raise RuntimeError(f"No TensorBoard event files found under {log_dir}")

    losses: List[float] = []
    for event_path in event_paths:
        events = EventAccumulator(str(event_path))
        events.Reload()
        if tag not in events.Tags().get("scalars", []):
            continue
        losses.extend(float(event.value) for event in events.Scalars(tag))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-log-dir", type=Path, required=True)
    parser.add_argument("--resumed-log-dir", type=Path, required=True)
    parser.add_argument("--tag", default="val/loss_epoch")
    args = parser.parse_args()

    losses = load_epoch_losses(args.stage1_log_dir, args.tag)
    losses.extend(load_epoch_losses(args.resumed_log_dir, args.tag))
    if len(losses) < 2:
        raise RuntimeError(f"Expected at least two epoch losses for {args.tag}, found {losses}")
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"Non-finite validation loss found: {losses}")
    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"Validation loss did not improve overall: first={losses[0]}, last={losses[-1]}"
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "metric": args.tag,
                "epoch_losses": losses,
                "overall_improvement": losses[0] - losses[-1],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
