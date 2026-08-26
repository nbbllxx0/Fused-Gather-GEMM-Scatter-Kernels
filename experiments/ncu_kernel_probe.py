"""
ncu_kernel_probe.py
-------------------
Minimal launcher to be profiled by Nsight Compute.

Run *under* ncu:

    ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,... --csv \\
        python experiments/phase3/ncu_kernel_probe.py --size 216k --path fused_fp32

It builds the operator, does a few untimed warm-up launches, then launches
the kernel of interest exactly `--launches` times so the profiler has a
clean set of identical invocations to average.  Nothing else is on the
device timeline apart from setup, which ncu reports separately by kernel
name.

This exists so that "effective bandwidth" and "% of peak DRAM bandwidth"
can be measurements.  Dividing a byte *model* by an event timing produces
numbers with those units, but they are the model restated -- they cannot
disagree with it, so they cannot test it.  Only a hardware counter can.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZE_LADDER = {
    "64k":  (80, 40, 20),
    "216k": (120, 60, 30),
    "512k": (160, 80, 40),
    "1M":   (200, 100, 50),
    "2M":   (252, 126, 63),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="216k")
    ap.add_argument("--path", default="fused_fp32")
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite, PATH_SPEC, NEEDS_EDOF

    nelx, nely, nelz = SIZE_LADDER[args.size]
    n_elem = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof,
                          build_edof=(args.path in NEEDS_EDOF))
    rng = np.random.default_rng(42)
    u = cp.asarray(rng.standard_normal(ndof), dtype=cp.float64)
    E = cp.asarray(0.1 + 0.9 * rng.random(n_elem), dtype=cp.float64)

    for _ in range(args.warmup):
        suite.matvec_full(u, E, path=args.path)
    cp.cuda.Stream.null.synchronize()

    for _ in range(args.launches):
        suite.matvec_full(u, E, path=args.path)
    cp.cuda.Stream.null.synchronize()

    print(f"probe done: {args.path} at {args.size} "
          f"({n_elem} elem, {ndof} dof), {args.launches} launches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
