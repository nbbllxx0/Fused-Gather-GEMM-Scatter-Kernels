"""
test_bf16_ir.py
---------------
Smoke test for the three fused_dtype paths: fp32, bf16 (raw), bf16_ir.

Runs a single cold-start 64k cantilever solve on each path and reports
CG iterations, compliance, and residual history.

Expected outcome
----------------
  fp32     : ~345 CG, compliance matches FP64 to ~1e-5
  bf16     : stalls (CG hits maxiter or plateaus above 1e-3 rel residual)
  bf16_ir  : outer residual correction still stagnates badly on the deposited
             cantilever smoke tests, despite much larger total inner-CG counts
"""
from __future__ import annotations

import os
import sys
import tempfile
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

import numpy as np  # noqa: E402
from scaling_ladder import build_cantilever_problem  # noqa: E402
from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D  # noqa: E402
from gpu_fem.solver_v2 import SolverV2  # noqa: E402


def make_solver(prob, fused_dtype: str, inner_tol: float = 1e-3, max_outer: int = 8):
    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    return SolverV2(
        edof=edof, row_idx=row_idx, col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D, free=prob["free"], F=prob["F"], ndof=prob["ndof"],
        backend="auto", grid_dims=(nelx, nely, nelz),
        enable_warm_start=False,
        enable_matrix_free=True,
        enable_mixed_precision=True,
        enable_fused_cuda=True,
        fused_dtype=fused_dtype,
        bf16_ir_inner_tol=inner_tol,
        bf16_ir_max_outer=max_outer,
    )


def run(fused_dtype: str, prob, **kwargs):
    s = make_solver(prob, fused_dtype, **kwargs)
    # Leave verbose off for the batch run 閳?keeps output readable.
    rho = np.full(prob["n_elem"], 0.5)
    t0 = time.perf_counter()
    c, _ = s.solve(rho, penal=3.0)
    dt = time.perf_counter() - t0
    tag = f"{fused_dtype}" + (f"_tol{kwargs.get('inner_tol', 1e-3):.0e}"
                              if fused_dtype == "bf16_ir" else "")
    print(f"  [{tag:18s}]  cg={s.last_cg_iters:>4d}   wall={dt*1e3:7.1f} ms   c={c:.6f}")
    return {"fused_dtype": tag, "cg": s.last_cg_iters,
            "wall_ms": dt * 1e3, "compliance": float(c)}


def main():
    import csv
    import json

    SIZES = [(80, 40, 20, "64k"), (120, 60, 30, "216k")]
    print("=" * 66)
    print("  BF16-in-CG smoke test  閳? fused_dtype 閳?{fp32, bf16, bf16_ir}")
    print("=" * 66)

    all_rows = []
    for NELX, NELY, NELZ, tag in SIZES:
        print(f"\n  cantilever {tag} ({NELX}x{NELY}x{NELZ} = {NELX*NELY*NELZ:,} elem)")
        print("  " + "-" * 60)
        prob = build_cantilever_problem(NELX, NELY, NELZ)
        configs = [
            ("fp32",    {}),
            ("bf16",    {}),
            ("bf16_ir", {"inner_tol": 1e-3, "max_outer": 8}),
            ("bf16_ir", {"inner_tol": 1e-5, "max_outer": 8}),
        ]
        ref_c = None
        for dtype, kwargs in configs:
            try:
                r = run(dtype, prob, **kwargs)
                r["size_tag"] = tag
                r["n_elem"]   = NELX * NELY * NELZ
                if ref_c is None:
                    ref_c = r["compliance"]
                r["compliance_rel_err"] = abs(r["compliance"] - ref_c) / abs(ref_c)
                all_rows.append(r)
            except Exception as exc:
                print(f"  [{dtype}] FAILED: {type(exc).__name__}: {exc}")

    # Save CSV + JSON
    out_csv  = ROOT / "experiments" / "phase3" / "bf16_ir_smoke.csv"
    out_json = ROOT / "experiments" / "phase3" / "bf16_ir_smoke.json"
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_rows)
        out_json.write_text(json.dumps(all_rows, indent=2))
        print(f"\n  Saved: {out_csv.relative_to(ROOT)}")
        print(f"  Saved: {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
