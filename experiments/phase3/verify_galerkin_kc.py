"""
verify_galerkin_kc.py  鈥? Galerkin K_c correctness + spectral quality

Two tests on the 1.7k cantilever (6,552 free DOFs, coarsest level 144 DOFs):

TEST 1 鈥?K_c correctness
    Build K_ff_free by standard sparse assembly (O(n_elem*576) scatter-add).
    Build the exact Galerkin K_c via sparse triple product P^T K_ff P.
    Compare with the element-wise Q_stu assembly used in MatFreeGMG.setup().
    Expected: ||K_c_exact - K_c_galerkin||_F / ||K_c_exact||_F < 1e-10.
    A larger error means there is a bug in the Q_stu assembly.

TEST 2 鈥?Spectral quality of M^{-1}K
    For a uniform-density field (early SIMP): compute all eigenvalues of
    M^{-1}K where M is the V-cycle preconditioner (built by MatFreeGMG).
    Compare 魏(M^{-1}K) = 位_max/位_min versus 魏(D^{-1}K) for Jacobi.
    If 魏(M^{-1}K) 鈮?魏(D^{-1}K) 鈫?V-cycle gives no spectral improvement
    despite z/jac 鈮?1 (norm quality metric is misleading for PCG).

Usage
-----
python experiments/phase3/verify_galerkin_kc.py
"""

from __future__ import annotations
import sys, os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("GPU_FEM_ENV_BOOTSTRAPPED", "1")

from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
from gpu_fem.solver_v2 import (
    _build_scalar_prolongation,
    _coarse_free_dofs_injection,
)
from solver_v3 import (
    MatFreeGMG,
    MatrixFreeKff,
    _build_coarse_level_galerkin,
)

import cupy as cp
import cupyx.scipy.sparse as cpsp

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Mesh setup: 1.7k cantilever
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
nelx, nely, nelz = 12, 6, 6
n_elem           = nelx * nely * nelz
n_nodes          = (nelx+1) * (nely+1) * (nelz+1)
ndof             = 3 * n_nodes

print(f"\n{'='*68}")
print(f"  verify_galerkin_kc.py  ({nelx}脳{nely}脳{nelz} = {n_elem:,d} elements)")
print(f"{'='*68}")

# Fix x=0 face
fixed = []
for iy in range(nely+1):
    for iz in range(nelz+1):
        nid = iy*(nelz+1) + iz
        fixed += [3*nid, 3*nid+1, 3*nid+2]
fixed   = np.unique(fixed).astype(np.int32)
free    = np.setdiff1d(np.arange(ndof, dtype=np.int32), fixed)
n_free  = len(free)

# Load: -y force on x=nelx, iy=0 face
F = np.zeros(ndof)
for iz in range(nelz+1):
    nid = nelx*(nely+1)*(nelz+1) + 0*(nelz+1) + iz
    F[3*nid+1] = -1.0
F_free = F[free]

print(f"  free DOFs  : {n_free:,d}")
print(f"  fixed DOFs : {len(fixed):,d}")

E0, Emin, penal = 1.0, 1e-9, 3.0
KE_UNIT = KE_UNIT_3D   # (24, 24)

# Coarse level 1 dimensions
nelx_c = max(1, nelx//2)
nely_c = max(1, nely//2)
nelz_c = max(1, nelz//2)
free_c = _coarse_free_dofs_injection(nelx, nely, nelz, nelx_c, nely_c, nelz_c, free)
n_free_c = len(free_c)
print(f"  coarse lv1 : {nelx_c}脳{nely_c}脳{nelz_c}  n_free_c={n_free_c:,d}")

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Build K_ff_free from standard scatter-add assembly (reference truth)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
edof_f = _edof_table_3d(nelx, nely, nelz)                 # (n_elem, 24)
row_idx_all, col_idx_all = _build_sparse_indices(edof_f)  # each (n_elem*576,)

free_mask = np.zeros(ndof, dtype=bool)
free_mask[free] = True
keep_f     = free_mask[row_idx_all] & free_mask[col_idx_all]
keep_idx_f = np.nonzero(keep_f)[0]

free_local_f = np.full(ndof, -1, dtype=np.int32)
free_local_f[free] = np.arange(n_free, dtype=np.int32)

r_loc = free_local_f[row_idx_all[keep_idx_f]]
c_loc = free_local_f[col_idx_all[keep_idx_f]]
elem_of_kept_f = keep_idx_f // 576
ke_of_kept_f   = keep_idx_f % 576
KE_flat        = KE_UNIT.ravel()  # (576,)

def assemble_Kff(rho: np.ndarray) -> sp.csr_matrix:
    """Assemble K_ff_free for the given density array (numpy)."""
    E_e    = Emin + (E0 - Emin) * rho**penal
    data   = E_e[elem_of_kept_f] * KE_flat[ke_of_kept_f]
    K      = sp.csr_matrix((data, (r_loc, c_loc)), shape=(n_free, n_free), dtype=np.float64)
    K.sum_duplicates()
    return K

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Build P_free (from level 0 to level 1)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
P_sc   = _build_scalar_prolongation(nelx, nely, nelz, nelx_c, nely_c, nelz_c)
P_vec  = sp.kron(P_sc, sp.eye(3, format="csr", dtype=np.float64), format="csr")
P_free = P_vec[free, :][:, free_c].tocsr()
print(f"  P_free     : {P_free.shape}  nnz={P_free.nnz:,d}")

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Build Galerkin struct once
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
galerkin_cl = _build_coarse_level_galerkin(
    nelx_c, nely_c, nelz_c,
    nelx, nely, nelz,
    free_c, E0, Emin, KE_UNIT,
)
print(f"  Galerkin CL: n_free={galerkin_cl.n_free}  nnz_per_row~={galerkin_cl.Kff_indices.shape[0]/max(galerkin_cl.n_free,1):.1f}")

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# TEST 1: K_c correctness across three density cases
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
print(f"\n{'鈹€'*68}")
print("TEST 1: K_c element-wise Galerkin vs. sparse triple product P^T K_ff P")
print(f"{'鈹€'*68}")

rng = np.random.default_rng(42)
density_cases = [
    ("uniform 0.5",   np.full(n_elem, 0.5)),
    ("uniform 1.0",   np.full(n_elem, 1.0)),
    ("random [0,1]",  rng.random(n_elem)),
    ("SIMP-like",     np.clip(rng.standard_normal(n_elem)*0.3 + 0.5, 1e-3, 1.0)),
]

all_pass = True
for case_name, rho in density_cases:
    E_e_np  = Emin + (E0 - Emin) * rho**penal
    E_e_gpu = cp.asarray(E_e_np)

    # 鈹€鈹€ Exact Galerkin K_c via sparse triple product 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    K_ff_free = assemble_Kff(rho)
    K_c_trip  = (P_free.T @ K_ff_free @ P_free)
    K_c_exact = K_c_trip.toarray()
    fro_exact = np.linalg.norm(K_c_exact, "fro")

    # 鈹€鈹€ Galerkin K_c from element-wise Q_stu assembly 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    Kff_data = cp.zeros(int(galerkin_cl.Kff_indices.shape[0]), dtype=cp.float64)
    for stu_idx in range(8):
        Kff_data += (
            E_e_gpu[galerkin_cl.fine_elem_sorted_stu[stu_idx]]
            * galerkin_cl.Q_stu_sorted[stu_idx]
        )
    K_c_gpusp = cpsp.csr_matrix(
        (Kff_data, galerkin_cl.Kff_indices, galerkin_cl.Kff_indptr),
        shape=(n_free_c, n_free_c),
    )
    K_c_gal = cp.asnumpy(K_c_gpusp.toarray())

    # 鈹€鈹€ Compare 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rel_err    = np.linalg.norm(K_c_exact - K_c_gal, "fro") / max(fro_exact, 1e-300)
    sym_err    = np.linalg.norm(K_c_gal - K_c_gal.T, "fro") / max(fro_exact, 1e-300)
    ev_gal     = np.linalg.eigvalsh(K_c_gal)
    ev_ex      = np.linalg.eigvalsh(K_c_exact)
    pass_flag  = rel_err < 1e-8
    all_pass   = all_pass and pass_flag
    status     = "PASS" if pass_flag else "FAIL"

    print(f"  [{status}] {case_name:20s}  rel_err={rel_err:.3e}"
          f"  sym_err={sym_err:.3e}"
          f"  ev_min: exact={ev_ex.min():.4g} gal={ev_gal.min():.4g}")

print()
if all_pass:
    print("  [OK] TEST 1 PASSED -- K_c assembly is numerically exact (Galerkin = triple-product)")
else:
    print("  [!!] TEST 1 FAILED -- Bug in element-wise Galerkin K_c assembly")

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# TEST 2: Spectral quality of M^{-1}K
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
print(f"\n{'鈹€'*68}")
print("TEST 2: Spectral quality 鈥?eigenvalues of M^{{-1}}K and D^{{-1}}K")
print(f"{'鈹€'*68}")
print("  (Uses uniform rho=0.5, penal=3 鈥?representative early SIMP iteration)")

rho_test  = np.full(n_elem, 0.5)
K_ff      = assemble_Kff(rho_test)            # scipy sparse (n_free, n_free)
K_ff_np   = K_ff.toarray()
diag_kff  = np.diag(K_ff_np)

# Eigenvalues of D^{-1} K (Jacobi preconditioned system)
DinvK     = np.diag(1.0/diag_kff) @ K_ff_np
ev_jacpre = np.linalg.eigvalsh(DinvK)         # eigenvalues of D^{-1/2} K D^{-1/2}
kappa_jac = ev_jacpre.max() / ev_jacpre.min()
print(f"\n  Jacobi D^{{-1}}K:  位_min={ev_jacpre.min():.4g}  位_max={ev_jacpre.max():.4g}"
      f"  魏={kappa_jac:.4g}")

# Build MatFreeGMG
edof_gpu_f = cp.asarray(edof_f)
KE_unit_gpu = cp.asarray(KE_UNIT)
free_gpu    = cp.asarray(free)
mf_op = MatrixFreeKff(
    edof_gpu    = edof_gpu_f,
    KE_unit_gpu = KE_unit_gpu,
    free_gpu    = free_gpu,
    n_free      = n_free,
    ndof        = ndof,
)

n_levels = MatFreeGMG.suggest_n_levels(n_free)
print(f"\n  Building MatFreeGMG: {nelx}脳{nely}脳{nelz}, {n_levels} levels, n_smooth=3, omega=0.5")
gmg = MatFreeGMG(
    mf_op    = mf_op,
    nelx     = nelx, nely=nely, nelz=nelz,
    free     = free,
    E0       = E0, Emin = Emin,
    KE_UNIT  = KE_UNIT,
    n_levels = n_levels,
    n_smooth = 3,
    omega    = 0.5,
)
print(f"  Free DOFs/level: {gmg._n_free}")

E_e_test = Emin + (E0 - Emin) * rho_test**penal
E_e_gpu  = cp.asarray(E_e_test)
_kff_diag = mf_op.extract_diagonal(E_e_gpu)
gmg.setup(E_e_gpu, rho_test, penal, fine_diag=_kff_diag)
print(f"  GMG setup complete.  omega_used={gmg._omega:.4f}")

# V-cycle as a matrix: V_mat[j] = vcycle(e_j) for each standard basis vector
print(f"\n  Computing V-cycle matrix columns (n_free={n_free} matvecs)...")
V_mat = np.zeros((n_free, n_free), dtype=np.float64)
for j in range(n_free):
    e_j = cp.zeros(n_free, dtype=cp.float64)
    e_j[j] = 1.0
    V_mat[:, j] = cp.asnumpy(gmg.vcycle(e_j))

# Check symmetry of V-cycle matrix
sym_err_V = np.linalg.norm(V_mat - V_mat.T, "fro") / max(np.linalg.norm(V_mat, "fro"), 1e-300)
print(f"  ||V - V^T|| / ||V|| = {sym_err_V:.3e}  (must be < 1e-10 for PCG to work)")

# Eigenvalues of V^{-1} K (V is the V-cycle approximation to K^{-1})
# Method: eigvals of V K  (V 鈮?K^{-1}, so VK 鈮?I 鈫?eigenvalues 鈮?1 if good)
VK_mat   = V_mat @ K_ff_np
ev_VK    = np.linalg.eigvalsh((VK_mat + VK_mat.T) / 2.0)  # symmetrize for eigvalsh
kappa_V  = ev_VK.max() / ev_VK.min()
print(f"\n  V-cycle (VK):  位_min={ev_VK.min():.4g}  位_max={ev_VK.max():.4g}"
      f"  魏={kappa_V:.4g}")
print(f"  Improvement factor: 魏_jac/魏_V = {kappa_jac/kappa_V:.3g}x")
print(f"  (> 1 means V-cycle is better preconditioner; < 1 means WORSE)")

# Also check if V is PD
ev_V = np.linalg.eigvalsh(V_mat)
pd_flag = ev_V.min() > 0
print(f"\n  Eigenvalues of V itself: min={ev_V.min():.4g}  max={ev_V.max():.4g}  PD={pd_flag}")

# Check coarse K_c condition numbers per level
print(f"\n{'鈹€'*68}")
print("  Coarse K_c condition numbers (post-setup):")
print(f"{'鈹€'*68}")
for lv in range(1, n_levels):
    K_lv_np = cp.asnumpy(gmg._K_gpu[lv].toarray())
    ev_lv   = np.linalg.eigvalsh(K_lv_np)
    kap_lv  = ev_lv.max() / max(ev_lv.min(), 1e-300)
    n_lv    = gmg._n_free[lv]
    mode    = "chol" if (lv==n_levels-1 and gmg._coarse_chol_mode=="dense") else "iterative"
    print(f"  lv={lv}  n={n_lv:5d}  位_min={ev_lv.min():.4g}  位_max={ev_lv.max():.4g}"
          f"  魏={kap_lv:.4g}  ({mode})")

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# TEST 3: V-cycle residual reduction ratio (convergence factor)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
print(f"\n{'鈹€'*68}")
print("TEST 3: Two-grid convergence factor estimate")
print(f"{'鈹€'*68}")
print("  ||e_{k+1}|| / ||e_k||_K for 5 V-cycle iterations starting from random e")

K_ff_cp = cpsp.csr_matrix(cp.asarray(K_ff_np))
rng2    = np.random.default_rng(7)
e       = cp.asarray(rng2.standard_normal(n_free))

norms = []
for k in range(6):
    # K-energy norm: sqrt(e^T K e)
    Ke    = K_ff_cp @ e
    enorm = float(cp.sqrt(cp.dot(e, Ke)))
    norms.append(enorm)
    if k < 5:
        # One V-cycle step: e 鈫?e - V(K e)
        Ve = gmg.vcycle(Ke)
        e  = e - Ve

rho_factors = [norms[k+1]/norms[k] for k in range(5)]
print(f"  ||e||_K per iteration: " + "  ".join(f"{v:.4g}" for v in norms))
print(f"  Convergence factors:   " + "  ".join(f"{r:.4g}" for r in rho_factors))
avg_rho = (norms[-1]/norms[0]) ** (1/5)
print(f"  Average 蟻 (5 iters):   {avg_rho:.4g}")
print(f"  (good multigrid: 蟻 < 0.5; 蟻 鈮?1 means V-cycle is useless)")

print(f"\n{'='*68}")
print("SUMMARY")
print(f"{'='*68}")
print(f"  TEST 1 (K_c correctness):        {'PASS' if all_pass else 'FAIL'}")
print(f"  TEST 2 (spectral 魏):  Jacobi={kappa_jac:.3g}  V-cycle={kappa_V:.3g}"
      f"  ratio={kappa_jac/kappa_V:.3g}x")
print(f"  TEST 3 (convergence factor):     蟻_avg={avg_rho:.4g}")
print()
if avg_rho < 0.3:
    print("  [OK] V-cycle is an EXCELLENT preconditioner")
elif avg_rho < 0.6:
    print("  [OK] V-cycle is a GOOD preconditioner")
elif avg_rho < 0.9:
    print("  [~~] V-cycle provides MODEST improvement (consider more levels/smoothing)")
else:
    print("  [!!] V-cycle is POOR (rho~1 => no improvement over Jacobi; investigate root cause)")
    if sym_err_V > 1e-8:
        print(f"    CAUSE: V-cycle is NOT symmetric (||V-V^T||/||V||={sym_err_V:.3e})")
        print("    FIX: Check smoother symmetry and boundary handling in _vcycle().")
    elif kappa_jac / kappa_V < 2.0:
        print("    CAUSE: V-cycle does NOT improve spectral condition number.")
        print("    Hypothesis: coarse correction is proportional to Jacobi (no complementarity).")
        print("    Check: coarse K_c condition, level 2 eigenvalues, smoother range.")
