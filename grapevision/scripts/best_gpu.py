"""Pick the least-loaded GPU and, optionally, run a command pinned to it.

    .venv/bin/python scripts/best_gpu.py                       # print the chosen GPU index
    .venv/bin/python scripts/best_gpu.py -- python scripts/train_detector.py ... --device 0

With a command after `--`, CUDA_VISIBLE_DEVICES is set to the chosen GPU's UUID, so inside the command the GPU
is always device 0. The train_detector.py and eval_counts.py scripts also accept `--device auto`.
Selection logic lives in grapevision/gpu.py.
"""

from __future__ import annotations

import argparse
import os
import sys

from grapevision.gpu import select_gpu


def main() -> None:
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        k = argv.index("--")
        argv, cmd = argv[:k], argv[k + 1 :]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-free-mb", type=int, default=10000, help="skip GPUs with less free memory (default 10000)")
    parser.add_argument("--min-total-mb", type=int, default=16000, help="skip small cards such as display adapters")
    parser.add_argument("--samples", type=int, default=3, help="utilization readings to average")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
    args = parser.parse_args(argv)
    try:
        best = select_gpu(args.min_free_mb, args.min_total_mb, args.samples, args.interval)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    if not cmd:
        print(best["index"])
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=best["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
