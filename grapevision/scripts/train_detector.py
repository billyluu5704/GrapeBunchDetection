"""Train the stage-1 bunch detector.

python scripts/train_detector.py --data datasets/west_zone2/detector/data.yaml [--epochs 150]
"""

from __future__ import annotations

import argparse

from grapevision.config import load_config, resolve_path
from grapevision.detect import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--name", default="detector")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    det = cfg["detector"]
    train_detector(
        args.data,
        weights=det["weights"],
        imgsz=det["imgsz"],
        epochs=args.epochs,
        device=det["device"].split(":")[-1],
        workers=args.workers,
        project=resolve_path("runs"),
        name=args.name,
        seed=cfg["split"]["seed"],
    )


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    main()
