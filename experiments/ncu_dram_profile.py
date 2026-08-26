"""
ncu_dram_profile.py
-------------------
Measured DRAM traffic per operator path, via Nsight Compute.

STATUS: blocked on hosts without counter access, and deliberately not worked around.

    ncu reports ERR_NVGPUCTRPERM: access to NVIDIA GPU performance counters
    is restricted to administrators by default on Windows.  Enabling it is a
    system-level setting, so it needs an explicit decision by the machine's
    owner:

        NVIDIA Control Panel -> Desktop -> Developer -> Manage GPU
        Performance Counters -> "Allow access to the GPU performance
        counters to all users"       (requires admin; then reboot)

    or run this script from an elevated shell.

Until that happens the paper reports the two analytic bounds (logical and
compulsory, from gpu_fem.cuda_operators.traffic_terms) and states plainly
that measured DRAM counters were not collected.  It does *not* present a
byte model as a measurement, which is the easy mistake to make here: a file
named for bandwidth can contain no Nsight counter at all and still end up
quoted as "effective bandwidth" or "per cent of peak DRAM bandwidth".

The gap matters most for the node-owned path, whose logical and compulsory
bounds differ by a factor of 29 (812 vs 28 B/element) and whose *logical*
bandwidth at 1M elements evaluates to 1360 GB/s -- above the RTX 4090's
1008 GB/s peak, which is itself proof that the logical model is not DRAM
traffic.

Usage once permissions allow:
    python experiments/phase3/ncu_dram_profile.py --sizes 216k,1M
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
]

DEFAULT_PATHS = ["three_stage_fp64", "three_stage_fp32", "fused_fp64",
                 "fused_fp32", "fused_ai_fp32", "node_fp32"]


def find_ncu():
    exe = shutil.which("ncu") or shutil.which("ncu.bat")
    if exe:
        return exe
    base = r"C:\Program Files\NVIDIA Corporation"
    if os.path.isdir(base):
        cands = []
        for d in os.listdir(base):
            if d.lower().startswith("nsight compute"):
                for name in ("ncu.bat", "ncu.exe"):
                    p = os.path.join(base, d, name)
                    if os.path.exists(p):
                        cands.append(p)
        if cands:
            return sorted(cands)[-1]
    return None


def run_one(ncu, python, size, path, launches):
    cmd = [ncu, "--metrics", ",".join(METRICS), "--csv",
           "--target-processes", "all", "--kernel-name-base", "function",
           python, os.path.join(_HERE, "ncu_kernel_probe.py"),
           "--size", size, "--path", path, "--launches", str(launches)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = proc.stdout
    if "ERR_NVGPUCTRPERM" in out or "ERR_NVGPUCTRPERM" in proc.stderr:
        return {"blocked": True,
                "reason": "ERR_NVGPUCTRPERM: GPU performance counters are "
                          "restricted to administrators on this host"}
    rows = []
    start = out.find('"ID"')
    if start < 0:
        return {"blocked": True, "reason": "no CSV section in ncu output",
                "stdout_tail": out[-2000:], "stderr_tail": proc.stderr[-2000:]}
    reader = csv.DictReader(io.StringIO(out[start:]))
    for r in reader:
        rows.append(r)
    return {"blocked": False, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="216k,1M")
    ap.add_argument("--paths", default=",".join(DEFAULT_PATHS))
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gpu_fem.cuda_operators import traffic_terms

    ncu = find_ncu()
    python = sys.executable
    out_path = args.out or os.path.join(_ROOT, "results", "G3",
                                        "dram_measured.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    report = {"ncu": ncu, "metrics": METRICS, "blocked": False,
              "analytic_bounds": {p: traffic_terms(p)
                                  for p in args.paths.split(",")},
              "runs": []}

    if ncu is None:
        report["blocked"] = True
        report["reason"] = "Nsight Compute (ncu) not found on PATH"
        print("ncu not found -- cannot profile; analytic bounds only")
    else:
        print(f"ncu: {ncu}")
        for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
            for path in [p.strip() for p in args.paths.split(",") if p.strip()]:
                print(f"  profiling {path} @ {size} ...", flush=True)
                res = run_one(ncu, python, size, path, args.launches)
                res.update({"size": size, "path": path})
                report["runs"].append(res)
                if res.get("blocked"):
                    report["blocked"] = True
                    report["reason"] = res.get("reason")
                    print(f"    BLOCKED: {res.get('reason')}")
                    break
            if report["blocked"]:
                break

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if report["blocked"]:
        print("\n" + "=" * 74)
        print("BLOCKED -- measured DRAM counters were NOT collected.")
        print(report.get("reason", ""))
        print("To enable: NVIDIA Control Panel -> Desktop -> Developer ->")
        print("  Manage GPU Performance Counters -> allow all users")
        print("  (requires administrator, then reboot), or run ncu elevated.")
        print("The paper reports analytic logical/compulsory bounds and")
        print("states that DRAM counters are unavailable; it does not quote")
        print("a modelled number as a measured bandwidth.")
        print("=" * 74)
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
