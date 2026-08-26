"""
run_ladder_v11.py
-----------------
The end-to-end ladder, measured against the true residual.

Three families of run, and the difference between them is the point:

  fp64      double precision throughout. The control, and the only direct
            route that reaches the tolerance.
  fp32      single precision throughout. Expected to fail: its attainable
            residual floors two to three orders above the tolerance, so the
            fail-closed rule invalidates the run. Recorded as invalid with the
            residual it actually reached, because that number is the result.
  ir        residual in double precision, correction in single by the fast
            mapping, update in double. Reaches the tolerance and keeps most of
            the single-precision speed.

Every run is fail-closed on the true residual. A run that cannot hold the
tolerance is written to the results file as invalid rather than dropped, so
the failure is visible in the audit trail.

Usage:
    python experiments/phase3/run_ladder_v11.py --sizes 216k,512k --tag s1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZES = {"64k": (80, 40, 20), "216k": (120, 60, 30), "512k": (160, 80, 40),
         "1M": (200, 100, 50), "2M": (252, 126, 63), "4.9M": (170, 170, 170)}

# (label, kwargs) -- label is what tables and figures key on
SPECS = [
    ("three_stage_fp64", dict(path="three_stage_fp64")),
    ("fused_fp64",       dict(path="fused_fp64")),
    ("three_stage_fp32", dict(path="three_stage_fp32")),
    ("fused_fp32",       dict(path="fused_fp32")),
    ("fused_ai_fp32",    dict(path="fused_ai_fp32")),
    ("node_fp32",        dict(path="node_fp32")),
    ("ir_fused_fp32",    dict(path="fused_fp32", ir_inner="fused_fp32")),
    ("ir_fused_ai_fp32", dict(path="fused_ai_fp32", ir_inner="fused_ai_fp32")),
    ("ir_node_fp32",     dict(path="node_fp32", ir_inner="node_fp32")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="216k")
    ap.add_argument("--specs", default=",".join(s for s, _ in SPECS))
    ap.add_argument("--tag", default="v11")
    ap.add_argument("--volfrac", type=float, default=0.30)
    ap.add_argument("--rmin", type=float, default=1.5)
    ap.add_argument("--physical-rmin", type=float, default=None,
                    help="hold the filter radius fixed in physical length "
                         "rather than in element widths; this is what makes "
                         "designs comparable across meshes, and it is the "
                         "only mode a refinement claim may be drawn from")
    ap.add_argument("--cg-tol", type=float, default=1e-5)
    ap.add_argument("--cg-maxiter", type=int, default=20000)
    ap.add_argument("--max-outer", type=int, default=400)
    ap.add_argument("--ir-tol", type=float, default=1e-1)
    ap.add_argument("--gate", default="G6",
                    help="name of the results/ subdirectory to write to")
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.simp_r2 import build_cantilever, run_simp

    out = os.path.join(_ROOT, "results", args.gate,
                       f"ladder_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    want = [s.strip() for s in args.specs.split(",") if s.strip()]
    byname = dict(SPECS)

    print("=" * 78, flush=True)
    print("end-to-end ladder, true-residual protocol", flush=True)
    print(f"tol={args.cg_tol:g} on ||b-Ax||/||b||, guard={args.max_outer} "
          f"design iterations, ir inner tol={args.ir_tol:g}", flush=True)
    print("=" * 78, flush=True)

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZES[tag]
        prob = build_cantilever(nelx, nely, nelz, load="patch")
        print(f"\n### {tag}: {prob['n_elem']:,} elem, {prob['ndof']:,} DOF",
              flush=True)
        for name in want:
            if any(r.get("size") == tag and r.get("path") == name
                   for r in rows):
                print(f"  {name:<18s} already done", flush=True)
                continue
            kw = dict(byname[name])
            cp.get_default_memory_pool().free_all_blocks()
            t0 = time.perf_counter()
            try:
                r = run_simp(prob, volfrac=args.volfrac, rmin=args.rmin,
                             physical_rmin=args.physical_rmin,
                             cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
                             max_outer=args.max_outer, ir_tol=args.ir_tol,
                             verbose=False, record_history=True, **kw)
                r["size"] = tag
                r["path"] = name
                r["scheme"] = ("ir" if kw.get("ir_inner")
                               else ("fp64" if "fp64" in kw["path"]
                                     else "fp32"))
                r["wall_s"] = time.perf_counter() - t0
                rho = r.pop("rho_phys", None)
                if rho is not None:
                    f = os.path.join(_ROOT, "results", args.gate,
                                     f"run_{args.tag}_{tag}_{name}_rho.npy")
                    np.save(f, cp.asnumpy(rho) if hasattr(rho, "device")
                            else rho)
                # run_simp returns device/host arrays alongside the
                # scalars; the results file holds numbers, and the fields
                # are written to .npy beside it.
                for key in [k for k, v in list(r.items())
                            if hasattr(v, "shape") and getattr(v, "ndim", 0)]:
                    arr = r.pop(key)
                    np.save(os.path.join(
                        _ROOT, "results", args.gate,
                        f"run_{args.tag}_{tag}_{name}_{key}.npy"),
                        cp.asnumpy(arr) if hasattr(arr, "device") else arr)
                rows.append(r)
                print(f"  {name:<18s} outer={r['outer_iters']:3d} "
                      f"conv={str(r['outer_converged']):5s} "
                      f"c={r['final_compliance']:.4f} "
                      f"maxres={r['max_rel_resid']:.2e} "
                      f"CG={r['total_cg_iters']:7d} "
                      f"{r['wall_s']:8.1f}s", flush=True)
            except Exception as ex:                        # noqa: BLE001
                msg = str(ex)
                rows.append(dict(size=tag, path=name, invalid=True,
                                 scheme=("ir" if kw.get("ir_inner")
                                         else ("fp64" if "fp64" in kw["path"]
                                               else "fp32")),
                                 reason=msg[:400],
                                 wall_s=time.perf_counter() - t0))
                print(f"  {name:<18s} INVALID  {msg[:110]}", flush=True)
                if not isinstance(ex, Exception):
                    traceback.print_exc()
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1)
            cp.get_default_memory_pool().free_all_blocks()

    print(f"\nwritten: {os.path.relpath(out, _ROOT)}  ({len(rows)} rows)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
