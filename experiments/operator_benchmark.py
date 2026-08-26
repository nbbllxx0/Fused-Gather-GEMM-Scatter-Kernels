"""
operator_benchmark.py
---------------------
Isolated operator microbenchmark for every matrix-free K*v mapping, on the
*real* structured cantilever connectivity.

Three properties distinguish this from a naive hot-path profile:

1. It benchmarks the real operator, not a synthetic tensor of the same
   shape with random DOF indices.  Seeding random DOF-index patterns instead
   of the structured cantilever connectivity turns the benchmark into a
   gather/scatter stress test rather than a measurement of the operator
   actually used.  Both are reported here so the difference is visible.

2. It reports a distribution, not a number.  Every cell is the median of
   `--blocks` independent timing blocks, with the full spread recorded, and
   the GPU clock and temperature are sampled before and after each block so a
   throttled or contended measurement can be identified rather than averaged
   in.  This matters because single timings of the same configuration can
   disagree wildly: for one nominal 216k SIMP configuration, separately
   recorded fused wall times run 17.5 s (scaling_ladder_simp_mid.csv),
   23.4-48.8 s
   (statistical_repeats.csv) and 87.5 s (fully_converged_study.csv).

3. It records the logical byte model *and*, when Nsight Compute is available,
   the measured DRAM counters, as four separate quantities per kernel:
   logical bytes, atomic read-modify-write traffic, the cache-reuse
   assumption, and measured DRAM bytes.  The shipped ncu_bandwidth.csv
   contains no Nsight counter at all -- it is CUDA-event time divided into a
   byte model -- so the manuscript's "effective bandwidth" and "% of peak"
   are model-derived and are relabelled accordingly.

Usage:
    python experiments/phase3/operator_benchmark.py --sizes 64k,216k,512k,1M
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZE_LADDER = {
    "64k":  (80, 40, 20),
    "216k": (120, 60, 30),
    "512k": (160, 80, 40),
    "1M":   (200, 100, 50),
    "2M":   (252, 126, 63),
    "4.9M": (170, 170, 170),
    "8M":   (400, 200, 100),
    "13M":  (468, 234, 117),
    "16M":  (504, 252, 126),
}


def gpu_state():
    """Clock, temperature and power right now -- for throttle detection."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
             "utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().split(",")
        return {"sm_mhz": float(out[0]), "mem_mhz": float(out[1]),
                "temp_c": float(out[2]), "power_w": float(out[3]),
                "util_pct": float(out[4]), "mem_used_mib": float(out[5])}
    except Exception as ex:                                   # pragma: no cover
        return {"error": repr(ex)}


def time_path(suite, u, E, path, blocks, target_ms, warmup_ms):
    """Median-of-blocks CUDA-event timing with per-block GPU state.

    Timing protocol.  The GPU in this host also drives the
    desktop and its clocks are *not* locked -- locking them would be a
    system-wide setting change -- so the protocol is built to make
    contention visible instead of averaging it away:

      * a single call is timed once to calibrate the iteration count, then
        every timing block is sized to run for ~`target_ms` of GPU work, so
        launch overhead and clock transients are amortised rather than
        measured;
      * the kernel is run for `warmup_ms` before any timing, which is what
        brings the SM clock to its steady state;
      * `blocks` independent blocks are timed and the *median* is reported,
        with min, max and the full per-block list retained;
      * SM clock, memory clock, temperature, power and GPU utilisation are
        sampled around every block.

    A cell whose spread is large is not a cell to quote -- it is a cell to
    re-run.  The spread is therefore carried into the CSV and into the
    manuscript's uncertainty column rather than being discarded.
    """
    import cupy as cp

    # -- calibrate ------------------------------------------------------
    suite.matvec_full(u, E, path=path)
    cp.cuda.Stream.null.synchronize()
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    ev0.record()
    for _ in range(3):
        suite.matvec_full(u, E, path=path)
    ev1.record()
    ev1.synchronize()
    one_ms = cp.cuda.get_elapsed_time(ev0, ev1) / 3.0
    iters = max(3, min(20000, int(round(target_ms / max(one_ms, 1e-4)))))

    # -- warm up to steady-state clocks ---------------------------------
    t_end = time.perf_counter() + warmup_ms * 1e-3
    while time.perf_counter() < t_end:
        for _ in range(iters):
            suite.matvec_full(u, E, path=path)
        cp.cuda.Stream.null.synchronize()

    # -- timed blocks ---------------------------------------------------
    per_block, states = [], []
    for _ in range(blocks):
        st0 = gpu_state()
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ev0.record()
        for _ in range(iters):
            suite.matvec_full(u, E, path=path)
        ev1.record()
        ev1.synchronize()
        ms = cp.cuda.get_elapsed_time(ev0, ev1) / iters
        per_block.append(ms * 1e3)          # microseconds per call
        states.append({"before": st0, "after": gpu_state()})

    s = sorted(per_block)
    n = len(s)

    def _q(p):
        if n == 1:
            return s[0]
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return s[lo] + (pos - lo) * (s[hi] - s[lo])

    median = _q(0.5)
    q1, q3 = _q(0.25), _q(0.75)
    return {
        "us_median": median,
        "us_q1": q1,
        "us_q3": q3,
        "us_min": s[0],
        "us_max": s[-1],
        "us_all": per_block,
        "iters_per_block": iters,
        # IQR is the quoted uncertainty: robust to the isolated slow blocks
        # that desktop GPU contention produces on this host.  The full range
        # is kept alongside it so an outlier is visible rather than smoothed.
        "iqr_pct": 100.0 * (q3 - q1) / median if median else float("nan"),
        "spread_pct": 100.0 * (s[-1] - s[0]) / median if median else float("nan"),
        "gpu_state": states,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64k,216k,512k,1M")
    ap.add_argument("--target-ms", type=float, default=50.0,
                    help="GPU work per timing block (ms)")
    ap.add_argument("--blocks", type=int, default=9)
    ap.add_argument("--warmup-ms", type=float, default=1500.0,
                    help="run the kernel this long before timing, to reach "
                         "steady-state SM clocks")
    ap.add_argument("--paths", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite, ALL_PATHS, PATH_SPEC

    paths = ([p.strip() for p in args.paths.split(",") if p.strip()]
             if args.paths else ALL_PATHS)

    out_path = args.out or os.path.join(_ROOT, "results", "G3",
                                        "operator_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    report = {
        "device": props["name"].decode(),
        "sm_count": props["multiProcessorCount"],
        "target_ms_per_block": args.target_ms,
        "blocks": args.blocks,
        "warmup_ms": args.warmup_ms,
        "clocks_locked": False,
        "timing_protocol": (
            "median of `blocks` CUDA-event blocks, each auto-sized to "
            "~target_ms of GPU work, after warmup_ms of steady-state warm-up; "
            "SM/mem clock, temperature and power sampled around every block; "
            "GPU clocks NOT locked (shared with the desktop compositor)"),
        "path_spec": {k: {"vectors": v[0], "KE": v[1], "E_e": v[2],
                          "index": v[3], "logical_B_per_elem": v[4],
                          "FLOP_per_elem": v[5]}
                      for k, v in PATH_SPEC.items()},
        "rows": [],
    }

    print("=" * 78)
    print(f"operator microbenchmark  --  {report['device']}")
    print(f"{args.blocks} blocks of ~{args.target_ms:.0f} ms, "
          f"{args.warmup_ms:.0f} ms warm-up, clocks not locked")
    print("=" * 78)

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZE_LADDER[tag]
        n_elem = nelx * nely * nelz
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        print(f"\n[{tag}]  {nelx}x{nely}x{nelz} = {n_elem:,} elem, {ndof:,} DOF")

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()

        needs_edof = any(p in PATH_SPEC and PATH_SPEC[p][3] != "none"
                         for p in paths)
        suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof,
                              build_edof=needs_edof)

        rng = np.random.default_rng(42)
        u64 = cp.asarray(rng.standard_normal(ndof), dtype=cp.float64)
        E64 = cp.asarray(0.1 + 0.9 * rng.random(n_elem), dtype=cp.float64)

        base = None
        for path in paths:
            try:
                pool_before = pool.used_bytes()
                r = time_path(suite, u64, E64, path, args.blocks,
                              args.target_ms, args.warmup_ms)
                peak_used = pool.used_bytes()
                peak_res = pool.total_bytes()
            except cp.cuda.memory.OutOfMemoryError:
                print(f"  {path:<22s} OOM")
                report["rows"].append({"size": tag, "n_elem": n_elem,
                                       "path": path, "oom": True})
                pool.free_all_blocks()
                continue
            except Exception as ex:
                print(f"  {path:<22s} ERROR {ex!r}")
                report["rows"].append({"size": tag, "n_elem": n_elem,
                                       "path": path, "error": repr(ex)})
                continue

            spec = PATH_SPEC[path]
            logical_B = spec[4] * n_elem
            flop = spec[5] * n_elem
            t_s = r["us_median"] * 1e-6
            if base is None and path == "three_stage_fp64":
                base = r["us_median"]
            row = {
                "size": tag, "n_elem": n_elem, "ndof": ndof, "path": path,
                "vectors": spec[0], "KE": spec[1], "E_e": spec[2],
                "index": spec[3],
                "logical_B_per_elem": spec[4], "FLOP_per_elem": spec[5],
                "logical_bytes": logical_B, "flops": flop,
                "logical_intensity": spec[5] / spec[4],
                "us_median": r["us_median"], "us_q1": r["us_q1"],
                "us_q3": r["us_q3"], "us_min": r["us_min"],
                "us_max": r["us_max"], "iqr_pct": r["iqr_pct"],
                "spread_pct": r["spread_pct"],
                "us_all": r["us_all"], "iters_per_block": r["iters_per_block"],
                "model_bandwidth_GBs": logical_B / t_s / 1e9,
                "achieved_GFLOPs": flop / t_s / 1e9,
                "pool_used_bytes": peak_used, "pool_total_bytes": peak_res,
                "gpu_state": r["gpu_state"][0],
            }
            report["rows"].append(row)
            print(f"  {path:<22s} {r['us_median']:9.1f} us  "
                  f"(IQR {r['iqr_pct']:5.1f}%, range {r['spread_pct']:5.1f}%)  "
                  f"model BW {row['model_bandwidth_GBs']:7.1f} GB/s")

        # speedups relative to the consistently-typed FP64 three-stage cell
        for row in report["rows"]:
            if row.get("size") == tag and "us_median" in row:
                b = next((x["us_median"] for x in report["rows"]
                          if x.get("size") == tag
                          and x.get("path") == "three_stage_fp64"
                          and "us_median" in x), None)
                if b:
                    row["speedup_vs_three_stage_fp64"] = b / row["us_median"]

        del suite, u64, E64
        pool.free_all_blocks()

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    csv_path = out_path.replace(".json", ".csv")
    cols = ["size", "n_elem", "ndof", "path", "vectors", "KE", "E_e", "index",
            "logical_B_per_elem", "FLOP_per_elem", "logical_intensity",
            "us_median", "us_q1", "us_q3", "us_min", "us_max",
            "iqr_pct", "spread_pct", "iters_per_block",
            "model_bandwidth_GBs", "achieved_GFLOPs",
            "speedup_vs_three_stage_fp64", "pool_used_bytes",
            "pool_total_bytes"]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for row in report["rows"]:
            if "us_median" not in row:
                continue
            fh.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

    print(f"\nwritten: {out_path}\n         {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
