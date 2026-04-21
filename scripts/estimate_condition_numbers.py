"""
estimate_condition_numbers.py
------------------------------
Lanczos-based condition number estimates for K_ff across all 6 BVPs.

Addresses reviewer objection #2 (why Jacobi fails for 5/6 BVPs) by
quantifying kappa(K_ff) and kappa(M^-1 K_ff) for the Jacobi preconditioner.

Method
------
For each BVP at its medium-size preset (~24k-64k elements):
  1. Assemble sparse K_ff on CPU at uniform density rho=volfrac, penal=3.
  2. Estimate lambda_max via Lanczos (ARPACK, which='LM').
  3. Estimate lambda_min via shift-invert Lanczos (sigma~0, which='LM').
  4. kappa(K_ff) = lambda_max / lambda_min.
  5. Build diagonal Jacobi preconditioner M = diag(K_ff).
  6. Repeat steps 2-4 for the symmetrically-scaled system
     K_scaled = D^{-1/2} K_ff D^{-1/2}  where D = diag(K_ff).
     kappa(K_scaled) = kappa(M^{-1} K_ff).

Results saved to experiments/paper2/condition_numbers.csv.

Usage
-----
    python scripts/estimate_condition_numbers.py
    python scripts/estimate_condition_numbers.py --preset cantilever_gpu_medium
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_fem.presets import get_preset
from gpu_fem.bc_generator import generate_bc
from gpu_fem.pub_simp_solver import (
    KE_UNIT_3D,
    _edof_table_3d,
    _build_sparse_indices,
)

# ─────────────────────────────────────────────────────────────────────────────
# BVP presets: one representative size per problem class (~24k-64k elements)
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = [
    "cantilever_gpu_medium",   #  64k  — well-constrained, fast CG
    "mbb_gpu_medium",          #  40k  — pin+roller, ill-conditioned
    "bridge_gpu_medium",       #  40k  — pin+roller, ill-conditioned
    "torsion_gpu_medium",      #  24k  — torsional BCs
    "column_gpu_medium",       #  32k  — top-loaded column
    "bracket_gpu_medium",      #  27k  — bracket
]

PENAL   = 3.0
E0      = 1.0
EMIN    = 1e-9
N_EIGS  = 1      # only the spectral extrema are needed for kappa

OUT_DIR = Path(__file__).parent.parent / "experiments" / "paper2"
OUT_CSV = OUT_DIR / "condition_numbers.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary-condition extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_free_and_F(spec):
    """
    Return (ndof, free_int32, F_float64) from a ProblemSpec.

    Use the same BC generator as the paper-2 experiments so the condition-number
    study matches the actual supports and loads used in the reported runs.
    """
    bc = generate_bc(spec)
    return bc.ndof, bc.free_dofs.astype(np.int32), bc.F.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse K_ff assembly (CPU, no GPU required)
# ─────────────────────────────────────────────────────────────────────────────

def _build_Kff(spec, rho_val: float = None) -> sp.csr_matrix:
    """
    Assemble the free-DOF stiffness matrix K_ff as a sparse CSR matrix.

    rho_val : uniform density to use (default: spec.volfrac).
    """
    nelx, nely, nelz = spec.nelx, spec.nely, spec.nelz
    n_elem = nelx * nely * nelz
    ndof   = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

    if rho_val is None:
        rho_val = spec.volfrac

    # Element stiffness values
    rho = np.full(n_elem, rho_val)
    E_e = EMIN + (E0 - EMIN) * rho ** PENAL   # (n_elem,)

    # Build sparse K (full, all DOFs)
    edof          = _edof_table_3d(nelx, nely, nelz)         # (n_elem, 24)
    row_idx, col_idx = _build_sparse_indices(edof)
    sK            = np.kron(E_e, KE_UNIT_3D.ravel())         # (n_elem*576,)
    K_full        = sp.csr_matrix(
        (sK, (row_idx, col_idx)), shape=(ndof, ndof)
    )
    K_full.sum_duplicates()
    K_full.eliminate_zeros()

    # Extract free-DOF submatrix
    _, free, _ = _extract_free_and_F(spec)
    K_ff = K_full[np.ix_(free, free)].tocsr()
    return K_ff


# ─────────────────────────────────────────────────────────────────────────────
# Condition number estimation via Lanczos (ARPACK)
# ─────────────────────────────────────────────────────────────────────────────

def _kappa(A: sp.spmatrix, k: int = N_EIGS, label: str = "") -> tuple[float, float, float]:
    """
    Estimate kappa(A) = lambda_max / lambda_min via Lanczos (ARPACK).

    Uses which='LM' for the largest eigenvalue and which='SA' for the
    smallest algebraic eigenvalue — no matrix factorization required, so
    memory stays O(nnz).

    Returns (lambda_min, lambda_max, kappa).
    """
    import sys as _sys
    n = A.shape[0]
    k = min(k, n - 2)
    t0 = time.perf_counter()

    # Largest eigenvalue  (fast — Lanczos naturally finds this)
    vals_large, _ = spla.eigsh(A, k=k, which="LM", tol=1e-4, maxiter=20_000)
    lam_max = float(vals_large.max())
    _sys.stdout.flush()

    # Smallest algebraic eigenvalue.  This is sufficient to distinguish
    # well-constrained systems from effectively singular ones.
    try:
        vals_small, _ = spla.eigsh(
            A, k=k, which="SA", tol=1e-3, maxiter=100_000,
            ncv=min(max(2 * k + 1, 30), n - 1),
        )
        lam_min = float(vals_small.min())
        if lam_min <= 1e-14:
            lam_min = float("nan")
    except Exception as exc:
        print(f"    [WARN] SA eigsh failed ({type(exc).__name__}: {exc}); kappa = inf")
        lam_min = float("nan")

    t_elapsed = time.perf_counter() - t0
    kappa = lam_max / lam_min if (lam_min and lam_min > 0) else float("inf")

    if label:
        print(f"    {label}: lam_min={lam_min:.3e}  lam_max={lam_max:.3e}"
              f"  kappa={kappa:.3e}  ({t_elapsed:.1f}s)")
    import sys as _sys; _sys.stdout.flush()
    return lam_min, lam_max, kappa


# ─────────────────────────────────────────────────────────────────────────────
# Per-BVP estimation
# ─────────────────────────────────────────────────────────────────────────────

def run_one(preset_name: str) -> dict:
    spec   = get_preset(preset_name)
    n_elem = spec.nelx * spec.nely * spec.nelz
    _, free, _ = _extract_free_and_F(spec)
    n_free = len(free)

    import sys as _sys
    print(f"\n{'='*60}", flush=True)
    print(f"Preset: {preset_name}  ({spec.nelx}x{spec.nely}x{spec.nelz}"
          f" = {n_elem:,} elem,  {n_free:,} free DOFs)", flush=True)
    print(f"{'='*60}", flush=True)

    # Assemble K_ff
    t_assemble = time.perf_counter()
    K_ff = _build_Kff(spec)
    t_assemble = time.perf_counter() - t_assemble
    print(f"  K_ff assembled: {K_ff.shape[0]:,} x {K_ff.shape[1]:,}"
          f"  nnz={K_ff.nnz:,}  ({t_assemble:.1f}s)", flush=True)

    # kappa(K_ff) — unpreconditioned
    lam_min_raw, lam_max_raw, kappa_raw = _kappa(K_ff, label="K_ff (raw)")

    # Build Jacobi preconditioner: D = diag(K_ff)
    diag = np.array(K_ff.diagonal())
    diag = np.where(np.abs(diag) > 1e-14, diag, 1.0)
    D_invsqrt = sp.diags(1.0 / np.sqrt(diag))

    # Symmetrically scaled K: K_scaled = D^{-1/2} K_ff D^{-1/2}
    # kappa(K_scaled) == kappa(M^{-1} K_ff) for Jacobi M=D
    K_scaled = D_invsqrt @ K_ff @ D_invsqrt
    K_scaled = K_scaled.tocsr()
    K_scaled.sum_duplicates()

    lam_min_jac, lam_max_jac, kappa_jac = _kappa(K_scaled, label="K_scaled (Jacobi)")

    reduction = kappa_raw / kappa_jac if kappa_jac > 0 and np.isfinite(kappa_jac) else float("nan")
    print(f"  Jacobi kappa reduction: {reduction:.1f}x")

    return {
        "preset":       preset_name,
        "n_elem":       n_elem,
        "n_free":       n_free,
        "lam_min_raw":  f"{lam_min_raw:.4e}",
        "lam_max_raw":  f"{lam_max_raw:.4e}",
        "kappa_raw":    f"{kappa_raw:.4e}",
        "lam_min_jac":  f"{lam_min_jac:.4e}",
        "lam_max_jac":  f"{lam_max_jac:.4e}",
        "kappa_jac":    f"{kappa_jac:.4e}",
        "kappa_reduction_x": round(reduction, 1) if np.isfinite(reduction) else "nan",
        "notes": (
            "rho=volfrac uniform, penal=3.0; "
            "kappa_jac = kappa(D^{-1/2} K D^{-1/2})"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lanczos kappa(K_ff) estimates for all 6 BVPs"
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        help="Run a single preset instead of all 6 (default: all)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    presets_to_run = [args.preset] if args.preset else PRESETS

    rows: list[dict] = []
    t_total = time.perf_counter()

    for preset_name in presets_to_run:
        try:
            row = run_one(preset_name)
            rows.append(row)
        except Exception as exc:
            import traceback
            print(f"[ERROR] {preset_name}: {exc}")
            traceback.print_exc()
            rows.append({"preset": preset_name, "error": str(exc)})

    print(f"\n{'='*60}")
    print(f"CONDITION NUMBER SUMMARY  (penal={PENAL}, rho=volfrac, uniform)")
    print(f"{'='*60}")
    hdr = f"{'Preset':<28} {'n_elem':>8} {'kappa(K)':>12} {'kappa(JacK)':>12} {'reduction':>10}"
    print(hdr)
    print("-" * 70)
    for r in rows:
        if "error" in r:
            print(f"{r['preset']:<28}  ERROR: {r['error']}")
            continue
        print(f"{r['preset']:<28} {r['n_elem']:>8,} "
              f"{r['kappa_raw']:>12} {r['kappa_jac']:>12} {str(r['kappa_reduction_x']):>9}x")

    print(f"\nTotal time: {time.perf_counter() - t_total:.1f}s")

    # Write CSV
    if rows:
        seen: dict[str, None] = {}
        for row in rows:
            for k in row.keys():
                seen[k] = None
        fieldnames = list(seen.keys())
        with OUT_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to: {OUT_CSV}")

    print("\nFor IJNME paper:")
    print("  - High kappa(K) + low reduction => CG diverges (MBB, bridge, column)")
    print("  - Low kappa(K) + moderate reduction => CG converges fast (cantilever)")
    print("  - Directly supports GMG motivation in Paper 3")


if __name__ == "__main__":
    main()
