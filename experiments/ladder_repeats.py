"""
ladder_repeats.py
-----------------
The end-to-end ladder, repeated, because single runs are not measurements.

The operator microbenchmark in this paper has always reported a median over
independent timing blocks, on the stated grounds that a single run is not
reproducible at these timescales. The end-to-end ladder never got the same
treatment, and it needed it more: three repeats of the same 512k configuration
gave 36.1, 36.8 and 49.2 s for the double-precision control -- a 36 per cent
spread -- while the refinement path held 21.5 to 23.4 s. Reading one run
against one run turned a 1.6x speedup into a 2.1x speedup.

Every cell here is the median of `--reps` complete optimizations, with the
spread recorded. Only paths that converge are repeated; the direct
single-precision paths do not converge at any mesh and are recorded once, by
run_ladder_v11, with the residual floor they reach.

Usage:
    python experiments/phase3/ladder_repeats.py --sizes 216k,512k,1M,2M --reps 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZES = {"216k": (120, 60, 30), "512k": (160, 80, 40),
         "1M": (200, 100, 50), "2M": (252, 126, 63)}

PATHS = [
    ("three_stage_fp64", dict(path="three_stage_fp64")),
    ("fused_fp64",       dict(path="fused_fp64")),
    ("ir_fused_fp32",    dict(path="fused_fp32", ir_inner="fused_fp32")),
    ("ir_fused_ai_fp32", dict(path="fused_ai_fp32", ir_inner="fused_ai_fp32")),
    ("ir_node_fp32",     dict(path="node_fp32", ir_inner="node_fp32")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="216k,512k,1M,2M")
    ap.add_argument("--paths", default=",".join(p for p, _ in PATHS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="rep")
    ap.add_argument("--gate", default="G6")
    args = ap.parse_args()

    import cupy as cp
    from gpu_fem.simp_r2 import build_cantilever, run_simp

    out = os.path.join(_ROOT, "results", args.gate, f"ladder_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    byname = dict(PATHS)

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZES[tag]
        prob = build_cantilever(nelx, nely, nelz, load="patch")
        print(f"\n### {tag}: {prob['n_elem']:,} elements", flush=True)
        for name in [p.strip() for p in args.paths.split(",") if p.strip()]:
            done = [r for r in rows if r.get("size") == tag
                    and r.get("path") == name]
            if len(done) >= args.reps:
                print(f"  {name:<18s} already have {len(done)} reps",
                      flush=True)
                continue
            walls = [r["wall_s"] for r in done]
            for rep in range(len(done), args.reps):
                cp.get_default_memory_pool().free_all_blocks()
                t0 = time.perf_counter()
                r = run_simp(prob, volfrac=0.30, rmin=1.5, cg_tol=1e-5,
                             cg_maxiter=20000, max_outer=400, ir_tol=1e-1,
                             verbose=False, record_history=False,
                             **byname[name])
                w = time.perf_counter() - t0
                walls.append(w)
                rows.append(dict(size=tag, path=name, rep=rep, wall_s=w,
                                 outer_iters=r["outer_iters"],
                                 outer_converged=r["outer_converged"],
                                 total_cg_iters=r["total_cg_iters"],
                                 final_compliance=r["final_compliance"],
                                 final_vol_phys=r["final_vol_phys"],
                                 max_rel_resid=r["max_rel_resid"]))
                with open(out, "w", encoding="utf-8") as fh:
                    json.dump(rows, fh, indent=1)
                print(f"  {name:<18s} rep{rep} {w:8.1f}s outer={r['outer_iters']:3d} "
                      f"c={r['final_compliance']:.6f} "
                      f"res={r['max_rel_resid']:.2e}", flush=True)
            med = statistics.median(walls)
            print(f"  {name:<18s} MEDIAN {med:8.1f}s  "
                  f"spread {min(walls):.1f}-{max(walls):.1f}s", flush=True)

    print(f"\nwritten: {os.path.relpath(out, _ROOT)} ({len(rows)} rows)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
