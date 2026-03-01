#!/usr/bin/env python3

import math
import os
import struct
import sys
from pathlib import Path


def read_f32(path: Path):
    data = path.read_bytes()
    n = len(data) // 4
    return struct.unpack("<" + "f" * n, data[: n * 4])


def compare_pair(cpu_file: Path, cuda_file: Path, name: str):
    if not cpu_file.exists() or not cuda_file.exists():
        print(f"{name}: missing file cpu={cpu_file.exists()} cuda={cuda_file.exists()}")
        return None

    a = read_f32(cpu_file)
    b = read_f32(cuda_file)
    n = min(len(a), len(b))
    if n == 0:
        print(f"{name}: empty")
        return None

    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    for i in range(n):
        d = abs(a[i] - b[i])
        if d > max_abs:
            max_abs = d
        sum_abs += d
        sum_sq += d * d

    mean_abs = sum_abs / n
    rmse = math.sqrt(sum_sq / n)
    print(f"{name}: n={n} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} rmse={rmse:.6f}")

    if name in ("classifier_mm", "classifier_out"):
        classes = 7
        frames = n // classes
        diff = 0
        for t in range(frames):
            sa = a[t * classes : (t + 1) * classes]
            sb = b[t * classes : (t + 1) * classes]
            ia = max(range(classes), key=lambda i: sa[i])
            ib = max(range(classes), key=lambda i: sb[i])
            if ia != ib:
                diff += 1
        print(f"  argmax_diff_frames={diff}/{frames}")

    return max_abs, mean_abs, rmse


def main():
    if len(sys.argv) != 3:
        print("Usage: compare-seg-dumps.py <cpu_dump_dir> <cuda_dump_dir>")
        return 1

    cpu_dir = Path(sys.argv[1])
    cuda_dir = Path(sys.argv[2])

    pairs = [
        ("lstm_out_cont", "seg_lstm_out_cont_0000.bin"),
        ("linear1_mm", "seg_linear1_mm_0000.bin"),
        ("linear2_mm", "seg_linear2_mm_0000.bin"),
        ("classifier_mm_src0", "seg_classifier_mm_src0_0000.bin"),
        ("classifier_mm_src1", "seg_classifier_mm_src1_0000.bin"),
        ("classifier_mm", "seg_classifier_mm_0000.bin"),
        ("classifier_out", "seg_classifier_out_0000.bin"),
    ]

    print(f"CPU dump:  {cpu_dir}")
    print(f"CUDA dump: {cuda_dir}")

    worst = None
    for name, filename in pairs:
        res = compare_pair(cpu_dir / filename, cuda_dir / filename, name)
        if res is not None:
            if worst is None or res[1] > worst[1]:
                worst = (name, *res)

    if worst:
        print(
            f"Worst mean_abs layer: {worst[0]} "
            f"(max_abs={worst[1]:.6f}, mean_abs={worst[2]:.6f}, rmse={worst[3]:.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
