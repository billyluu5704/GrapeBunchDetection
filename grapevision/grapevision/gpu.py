"""Choose the least-loaded GPU from nvidia-smi and pin this process to it.

`use_best_gpu()` sets CUDA_VISIBLE_DEVICES to the chosen GPU's UUID, so afterwards it is always device 0
(`"0"` / `"cuda:0"`) and the nvidia-smi versus CUDA index-order mismatch cannot pick the wrong card. Call it
before the first CUDA call in the process. Standard library only.

GPUs with less than min_total_mb memory (display adapters) or less than min_free_mb free are skipped. The rest are
ranked by average utilization (lowest first, rounded to 10%), then by free memory (most first). It is a
snapshot: another job can still start on the same GPU afterwards.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

QUERY = "index,uuid,name,memory.total,memory.used,utilization.gpu"


def read_gpus() -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        index, uuid, name, total, used, util = (f.strip() for f in line.split(","))
        gpus.append(
            {"index": int(index), "uuid": uuid, "name": name, "total": int(total), "free": int(total) - int(used),
             "util": float(util)}
        )
    return gpus


def sample(samples: int, interval: float) -> list[dict]:
    """Read several times and average utilization, because a single reading is noisy."""
    runs = []
    for i in range(samples):
        if i:
            time.sleep(interval)
        runs.append(read_gpus())
    gpus = runs[-1]
    for g in gpus:
        g["util"] = sum(r_g["util"] for r in runs for r_g in r if r_g["index"] == g["index"]) / len(runs)
    return gpus


def choose(gpus: list[dict], min_free_mb: int, min_total_mb: int) -> dict | None:
    ok = [g for g in gpus if g["total"] >= min_total_mb and g["free"] >= min_free_mb]
    return min(ok, key=lambda g: (round(g["util"] / 10), -g["free"]), default=None)


def select_gpu(
    min_free_mb: int = 10000, min_total_mb: int = 16000, samples: int = 3, interval: float = 1.0, verbose: bool = True
) -> dict:
    gpus = sample(samples, interval)
    if verbose:
        for g in gpus:
            print(f"  GPU {g['index']}: {g['name']}  free {g['free']} MiB  util {g['util']:.0f}%", file=sys.stderr)
    best = choose(gpus, min_free_mb, min_total_mb)
    if best is None:
        raise RuntimeError(f"no GPU with {min_free_mb} MiB free; lower min_free_mb or wait")
    if verbose:
        print(f"chosen: GPU {best['index']} ({best['uuid']})", file=sys.stderr)
    return best


def use_best_gpu(**kwargs) -> str:
    """Pin this process to the best GPU and return its device id inside the process ("0")."""
    best = select_gpu(**kwargs)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = best["uuid"]
    return "0"
