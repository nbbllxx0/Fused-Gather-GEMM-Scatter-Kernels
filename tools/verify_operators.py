"""
verify_operators.py
-------------------
Parity harness for the operator suite.

Nothing downstream is allowed to run until this passes.  The bar:

    matvec agreement vs the three-stage FP64 reference,
    relative L2 <= 1e-6 for FP64 paths and <= 1e-5 for FP32 paths,
    at two mesh sizes.

Three things are checked, in order of how badly a failure would corrupt the
results:

  1. The analytic index map reproduces pub_simp_solver._edof_table_3d
     *exactly* (integer equality, every element, every DOF).  If this is
     wrong the analytic and node-owned kernels solve a different problem and
     every other check would still pass by symmetry.
  2. Every operator path agrees with a NumPy/CuPy FP64 three-stage reference.
  3. The node-owned path is bitwise deterministic across repeats, and the
     atomic paths are not.  The determinism claim for the node-owned mapping
     is therefore a measurement rather than an expectation.

Usage:
    python tools/verify_operators.py            # 8k + 64k
    python tools/verify_operators.py --sizes 64k,216k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

SIZES = {
    "2k":   (20, 10, 10),
    "8k":   (40, 20, 10),
    "64k":  (80, 40, 20),
    "216k": (120, 60, 30),
    "512k": (160, 80, 40),
}


def rel_l2(a, b):
    import cupy as cp
    num = float(cp.linalg.norm(a - b))
    den = float(cp.linalg.norm(b))
    return num / den if den > 0 else num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="8k,64k")
    ap.add_argument("--out", default="results/G1/operator_parity.json")
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import _edof_table_3d, KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite, ALL_PATHS, PATH_SPEC

    report = {"sizes": {}, "pass": True}
    print("=" * 78)
    print("Operator parity harness")
    print("=" * 78)

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZES[tag]
        n_elem = nelx * nely * nelz
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        print(f"\n[{tag}]  {nelx}x{nely}x{nelz} = {n_elem:,} elements, "
              f"{ndof:,} DOF")
        entry = {"grid": [nelx, nely, nelz], "n_elem": n_elem, "ndof": ndof,
                 "checks": {}}

        # -- 1. index map ------------------------------------------------
        edof_ref = _edof_table_3d(nelx, nely, nelz)          # (n_elem, 24) int32
        suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof)
        edof_gpu = suite._edof32.reshape(n_elem, 24)
        exact = bool(cp.all(edof_gpu == cp.asarray(edof_ref)))
        print(f"  index map == _edof_table_3d : {'EXACT' if exact else 'MISMATCH'}")
        entry["checks"]["index_map_exact"] = exact
        if not exact:
            report["pass"] = False
            bad = int(cp.sum(edof_gpu != cp.asarray(edof_ref)))
            entry["checks"]["index_mismatches"] = bad
            report["sizes"][tag] = entry
            continue

        # -- reference solution ------------------------------------------
        rng = np.random.default_rng(42)
        u_np = rng.standard_normal(ndof)
        E_np = 0.1 + 0.9 * rng.random(n_elem)      # non-uniform modulus field
        u64 = cp.asarray(u_np, dtype=cp.float64)
        E64 = cp.asarray(E_np, dtype=cp.float64)

        y_ref = suite.matvec_full(u64, E64, path="three_stage_fp64").copy()

        # -- 2. every path vs the reference ------------------------------
        for path in ALL_PATHS:
            if path == "three_stage_fp64":
                continue
            prec = PATH_SPEC[path][0]
            bar = 1e-6 if prec == "fp64" else 1e-5
            y = suite.matvec_full(u64, E64, path=path)
            err = rel_l2(y.astype(cp.float64), y_ref)
            ok = err <= bar
            report["pass"] &= ok
            entry["checks"][path] = {"rel_l2": err, "bar": bar, "pass": ok}
            print(f"  {path:<22s} rel L2 = {err:.3e}  (bar {bar:.0e})  "
                  f"{'PASS' if ok else 'FAIL'}")

        # -- 3. determinism ----------------------------------------------
        det = {}
        for path in ("fused_fp32", "fused_ai_fp32", "node_fp32", "node_fp64"):
            outs = []
            for _ in range(5):
                outs.append(suite.matvec_full(u64, E64, path=path).copy())
            bitwise = all(bool(cp.all(outs[0] == o)) for o in outs[1:])
            spread = max(float(cp.max(cp.abs(outs[0].astype(cp.float64)
                                             - o.astype(cp.float64))))
                         for o in outs[1:])
            det[path] = {"bitwise_identical": bitwise, "max_abs_spread": spread}
            print(f"  determinism {path:<16s} bitwise={bitwise}  "
                  f"max|dy|={spread:.3e}")
        entry["checks"]["determinism"] = det

        report["sizes"][tag] = entry
        del suite
        cp.get_default_memory_pool().free_all_blocks()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 78)
    print("PARITY: " + ("PASS" if report["pass"] else "FAIL"))
    print(f"written to {args.out}")
    print("=" * 78)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
