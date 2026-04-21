"""
capture_cg_residuals.py
-----------------------
Capture per-iteration CG residual ||r_k||/||r_0|| for a representative
cold-start solve, for FP64 / FP32 / Fused paths. Output feeds F6 figure.

Uses 216k cantilever (120x60x30) 閳?the canonical mid-ladder size.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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
from scaling_ladder import build_cantilever_problem, _make_solver  # noqa: E402


SIZE_TAG = "216k"
NELX, NELY, NELZ = 120, 60, 30


def capture_path(path: str, prob: dict) -> dict:
    solver = _make_solver(prob, path=path, warm_start=False)
    solver.capture_cg_history = True
    n_elem = prob["n_elem"]
    rho_phys = np.full(n_elem, 0.5)
    c, _ = solver.solve(rho_phys, penal=3.0)
    hist = list(solver.last_cg_history)
    n_cg = int(solver.last_cg_iters)
    print(f"  [{path:5s}] CG iters = {n_cg:>4d}   first 3 ratios = "
          f"{[f'{v:.3e}' for v in hist[:3]]}   last = {hist[-1]:.3e}")
    return {
        "label":     f"{path.upper()} matfree",
        "path":      path,
        "n_cg":      n_cg,
        "compliance": float(c),
        "residuals": hist,
    }


def main():
    print("=" * 66)
    print(f"  Capturing CG residual history 閳?cantilever {SIZE_TAG}")
    print(f"  {NELX}x{NELY}x{NELZ} = {NELX*NELY*NELZ:,} elements")
    print("=" * 66)
    prob = build_cantilever_problem(NELX, NELY, NELZ)

    out = []
    for path in ("fp64", "fp32", "fused"):
        try:
            out.append(capture_path(path, prob))
        except Exception as exc:
            print(f"  [{path}] FAILED: {type(exc).__name__}: {exc}")

    for entry in out:
        entry["problem_title"] = (
            f"Cold-start CG residual decay - cantilever {SIZE_TAG} "
            f"({NELX}x{NELY}x{NELZ}, uniform rho=0.5)"
        )

    out_json = ROOT / "experiments" / "phase3" / "cg_residual_history.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

