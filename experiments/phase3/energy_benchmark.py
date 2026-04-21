"""
energy_benchmark.py
-------------------
Measure GPU energy consumption during SIMP-120 runs using nvidia-smi
power sampling.

Method
------
nvidia-smi is polled in a background thread at ~100 ms intervals during
the benchmark run.  Total GPU energy = trapezoid integral of power 鑴?time
reported in Wh and J.

Reports
-------
  - GPU power trace (CSV) for each run
  - Aggregate: total Wh, mean W, peak W, run duration
  - Comparison: fused FP32 vs FP64 energy ratio

Also reports the same number estimated from TDP 鑴?wall_time as a sanity
check ("loose upper bound" used in the paper).

Usage
-----
    python experiments/phase3/energy_benchmark.py
    python experiments/phase3/energy_benchmark.py --sizes 216k,1M --paths fp64,fused
    python experiments/phase3/energy_benchmark.py --niters 60       # faster test
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def _prefer_pytorch_env() -> None:
    root        = Path(__file__).resolve().parents[2]
    runtime_tmp = root / ".runtime_tmp"
    cupy_cache  = root / ".cupy_cache"
    runtime_tmp.mkdir(exist_ok=True)
    cupy_cache.mkdir(exist_ok=True)
    os.environ["TMP"]            = str(runtime_tmp)
    os.environ["TEMP"]           = str(runtime_tmp)
    os.environ["CUPY_CACHE_DIR"] = str(cupy_cache)
    os.environ["CUPY_TEMPDIR"]   = str(runtime_tmp)
    tempfile.tempdir             = str(runtime_tmp)

    preferred_python = os.environ.get("GPU_FEM_PYTHON")
    if not preferred_python:
        return
    env_python = Path(preferred_python).expanduser()
    if os.environ.get("GPU_FEM_ENV_BOOTSTRAPPED") == "1":
        return
    if not env_python.exists():
        return
    env = os.environ.copy()
    env["GPU_FEM_ENV_BOOTSTRAPPED"] = "1"
    os.execve(str(env_python), [str(env_python), __file__, *sys.argv[1:]], env)


_prefer_pytorch_env()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phase3"))

import numpy as np

SIZE_LADDER = {
    "64k":  (80,  40,  20),
    "216k": (120, 60,  30),
    "512k": (160, 80,  40),
    "1M":   (200, 100, 50),
}

RTX_4090_TDP_W = 450.0   # Rated TDP for paper's GPU

# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Power sampler thread
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

class _PowerSampler:
    """Polls nvidia-smi in a background thread and stores (timestamp, watts)."""

    def __init__(self, interval_s: float = 0.1, gpu_index: int = 0):
        self._interval    = interval_s
        self._gpu_index   = gpu_index
        self._samples: list[tuple[float, float]] = []
        self._stop_event  = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)
        self._available   = self._check_available()

    def _check_available(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits", f"--id={self._gpu_index}"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return True
        except Exception:
            return False

    def _run(self):
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            if self._available:
                try:
                    r = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=power.draw",
                         "--format=csv,noheader,nounits",
                         f"--id={self._gpu_index}"],
                        capture_output=True, text=True, timeout=3,
                    )
                    watts_str = r.stdout.strip().split("\n")[0].strip()
                    if watts_str and watts_str not in ("N/A", ""):
                        watts = float(watts_str)
                        self._samples.append((time.perf_counter(), watts))
                except Exception:
                    pass
            elapsed = time.perf_counter() - t0
            self._stop_event.wait(max(0.0, self._interval - elapsed))

    def start(self):
        if self._available:
            self._samples.clear()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def energy_J(self) -> float:
        """Trapezoid integration of power over time (Joules)."""
        if len(self._samples) < 2:
            return float("nan")
        ts = np.array([s[0] for s in self._samples])
        ws = np.array([s[1] for s in self._samples])
        try:
            return float(np.trapezoid(ws, ts))   # NumPy 閳?.0
        except AttributeError:
            return float(np.trapz(ws, ts))        # NumPy <2.0 fallback

    def mean_W(self) -> float:
        if not self._samples:
            return float("nan")
        return float(np.mean([s[1] for s in self._samples]))

    def peak_W(self) -> float:
        if not self._samples:
            return float("nan")
        return float(np.max([s[1] for s in self._samples]))

    def n_samples(self) -> int:
        return len(self._samples)


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Benchmark helpers
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def _free_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def build_cantilever_problem(nelx, nely, nelz):
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc

    spec = ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
        rmin=1.5,
    )
    bc = generate_bc(spec)
    return dict(
        spec=spec,
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F, ndof=bc.ndof,
    )


def run_simp_with_power(size_tag, path, n_iters, sampler_interval=0.1):
    from gpu_fem.solver_v2 import SolverV2
    from gpu_fem.simp_gpu import run_simp_surrogate_gpu, TO3DParams
    from gpu_fem.local_agents import PureFEMRouter
    from gpu_fem.pub_baseline_controller import ScheduleOnlyController

    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz)

    params = TO3DParams(
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=prob["spec"].volfrac,
        rmin=1.5,
        max_iter=n_iters,
    )
    solver_kwargs = dict(
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=True,
        enable_matrix_free=True,
        enable_rediscr_gmg=False,
    )
    if path == "fp64":
        solver_kwargs.update(enable_mixed_precision=False, enable_fused_cuda=False)
    elif path == "fp32":
        solver_kwargs.update(enable_mixed_precision=True, enable_fused_cuda=False)
    elif path == "fused":
        solver_kwargs.update(enable_mixed_precision=True, enable_fused_cuda=True)
    else:
        raise ValueError(f"Unknown path {path}")

    sampler = _PowerSampler(interval_s=sampler_interval)

    sampler.start()
    t_start = time.perf_counter()
    result = run_simp_surrogate_gpu(
        params=params,
        fixed=prob["fixed"], free=prob["free"],
        F=prob["F"], ndof=prob["ndof"],
        surrogate=None,
        router=PureFEMRouter(),
        device="auto",
        param_controller=ScheduleOnlyController(),
        verbose=False,
        solver_class=SolverV2,
        solver_kwargs=solver_kwargs,
    )
    wall_s = time.perf_counter() - t_start
    sampler.stop()

    compliance = result.get("best_compliance", result.get("final_compliance", float("nan")))
    energy_J   = sampler.energy_J()
    energy_Wh  = energy_J / 3600.0
    mean_W     = sampler.mean_W()
    peak_W     = sampler.peak_W()
    tdp_Wh     = RTX_4090_TDP_W * wall_s / 3600.0   # upper-bound estimate

    if not sampler._available:
        print(f"  WARNING: nvidia-smi not available 閳?energy from TDP鑴硉ime only")
        energy_J  = float("nan")
        energy_Wh = float("nan")

    _free_gpu_memory()

    return dict(
        size=size_tag, path=path, n_iters=n_iters,
        n_elem=nelx * nely * nelz,
        wall_s=float(wall_s),
        compliance=float(compliance),
        energy_J=energy_J,
        energy_Wh=energy_Wh,
        tdp_upper_Wh=tdp_Wh,
        mean_W=mean_W, peak_W=peak_W,
        n_power_samples=sampler.n_samples(),
        nvidia_smi_available=sampler._available,
        power_samples=[(t, w) for t, w in sampler._samples],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",   default="216k,1M")
    parser.add_argument("--paths",   default="fp64,fused")
    parser.add_argument("--niters",  type=int, default=120)
    parser.add_argument("--interval", type=float, default=0.1,
                        help="nvidia-smi poll interval in seconds")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    paths = [p.strip() for p in args.paths.split(",")]

    print("=" * 70)
    print(f"  Energy benchmark 閳?SIMP-{args.niters}  (RTX 4090 TDP={RTX_4090_TDP_W}W)")
    print(f"  sizes={sizes}  paths={paths}")
    print("=" * 70)

    all_rows = []
    for sz in sizes:
        if sz not in SIZE_LADDER:
            print(f"  WARN: unknown size {sz}, skipping.")
            continue
        for path in paths:
            print(f"\n  Running {path} @ {sz} ...")
            row = run_simp_with_power(sz, path, args.niters, args.interval)
            all_rows.append(row)
            smi_str = "measured" if row["nvidia_smi_available"] else "TDP-upper-bound"
            print(
                f"    wall={row['wall_s']:.1f}s  "
                f"energy={row['energy_Wh']:.4f}Wh ({smi_str})"
                f"  TDP_upper={row['tdp_upper_Wh']:.4f}Wh"
                f"  mean_W={row['mean_W']:.1f}  peak_W={row['peak_W']:.1f}"
            )

    # 閳光偓閳光偓 Summary 閳光偓閳光偓
    print(f"\n{'='*90}")
    print("  ENERGY SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Size':>5}  {'Path':>6}  {'wall(s)':>8}  {'Energy(Wh)':>11}  "
          f"{'TDP-ub(Wh)':>11}  {'mean_W':>8}  {'E_ratio(F/64)':>14}")
    print(f"  {'閳光偓'*82}")

    by_size: dict = {}
    for r in all_rows:
        by_size.setdefault(r["size"], {})[r["path"]] = r

    for sz in sizes:
        if sz not in by_size:
            continue
        sr = by_size[sz]
        fp64_Wh = sr.get("fp64", {}).get("energy_Wh", None)
        for path in paths:
            if path not in sr:
                continue
            r = sr[path]
            if fp64_Wh and r["energy_Wh"] and not np.isnan(fp64_Wh) and not np.isnan(r["energy_Wh"]):
                ratio = fp64_Wh / r["energy_Wh"]
                rs = f"{ratio:.2f}x"
            else:
                rs = "   --   "
            e_str   = f"{r['energy_Wh']:.4f}" if not np.isnan(r["energy_Wh"]) else "  N/A "
            tdp_str = f"{r['tdp_upper_Wh']:.4f}"
            print(f"  {r['size']:>5}  {r['path']:>6}  {r['wall_s']:>8.1f}  "
                  f"{e_str:>11}  {tdp_str:>11}  {r['mean_W']:>8.1f}  {rs:>14}")

    # 閳光偓閳光偓 Save 閳光偓閳光偓
    out_dir   = ROOT / "experiments" / "phase3"
    csv_path  = out_dir / "energy_benchmark.csv"
    json_path = out_dir / "energy_benchmark.json"

    flat = []
    for r in all_rows:
        flat.append({k: v for k, v in r.items() if k != "power_samples"})

    if flat:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)

        # Power traces in JSON (separate per run)
        json_data = []
        for r in all_rows:
            d = {k: v for k, v in r.items() if k != "power_samples"}
            d["power_trace"] = r["power_samples"]
            # Convert float pairs to list-of-lists for JSON
            d["power_trace"] = [[float(t), float(w)] for t, w in r["power_samples"]]
            json_data.append(d)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"\n  Saved: {csv_path}")
        print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
