"""Train the stage-1 bunch detector.

python scripts/train_detector.py --dataset-dir ../GrapevineBunchDetection [--epochs 150 --device 0]
python scripts/train_detector.py --data datasets/west_zone2/detector/data.yaml --weights runs/detector/weights/best.pt
python scripts/train_detector.py --dataset-dir ../GrapevineBunchDetection --name zenodo_base --resume

Training options live in the `train:` block of the config (batch, patience, cache, scale...); the flags below
override the common ones. Fine-tuning: pass the earlier run's best.pt as --weights.
"""

from __future__ import annotations

import argparse

from grapevision.config import load_config, resolve_path
from grapevision.detect import train_detector, write_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="an existing Ultralytics data.yaml")
    src.add_argument("--dataset-dir", help="folder of {train,val,test}/{images,labels}; yaml goes to outputs/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--name", default="detector")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weights", default=None, help="starting checkpoint (default: detector.weights)")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help='e.g. "0", "0,1,2" or "cpu" (default: detector.device)')
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="continue an interrupted run from runs/<name>/weights/last.pt"
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    data = args.data
    if args.dataset_dir:
        out = resolve_path(cfg["paths"]["output_dir"]) / f"{args.name}_data.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = str(write_data_yaml(args.dataset_dir, out))
        print(f"data.yaml -> {data}")

    det = cfg["detector"]
    train_args = dict(cfg.get("train", {}))
    for key in ("batch", "patience"):
        if getattr(args, key) is not None:
            train_args[key] = getattr(args, key)
    weights = args.weights or det["weights"]
    if args.resume:
        weights = str(resolve_path("runs") / args.name / "weights" / "last.pt")
        train_args["resume"] = True
    device = args.device if args.device is not None else det["device"].split(":")[-1]
    train_detector(
        data,
        weights=weights,
        imgsz=args.imgsz or det["imgsz"],
        epochs=args.epochs,
        device=device,
        workers=args.workers,
        project=resolve_path("runs"),
        name=args.name,
        seed=cfg["split"]["seed"],
        **train_args,
    )


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    main()
