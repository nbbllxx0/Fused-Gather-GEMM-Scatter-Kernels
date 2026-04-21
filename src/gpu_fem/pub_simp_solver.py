"""
pub_simp_solver.py
------------------
Publication-grade 2D/3D SIMP topology optimizer.
Three-field formulation: design -> density filter -> Heaviside projection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg
from scipy.spatial import cKDTree

try:
    import pyamg
    _PYAMG_AVAILABLE = True
except ImportError:
    _PYAMG_AVAILABLE = False


class _AMGSolver:
    def __init__(self, ndof_threshold: int = 8000, rebuild_tol: float = 0.15,
                 cg_tol: float = 1e-5, cg_maxiter: int = 800):
        self.ndof_threshold = ndof_threshold
        self.rebuild_tol    = rebuild_tol
        self.cg_tol         = cg_tol
        self.cg_maxiter     = cg_maxiter
        self._ml            = None        
        self._last_E_e      = None        
        self._use_amg       = _PYAMG_AVAILABLE

    def solve(self, K_csc: csc_matrix, F: np.ndarray, free: np.ndarray,
              E_e: np.ndarray, ndof: int) -> np.ndarray:
        U = np.zeros(ndof)
        Kff = K_csc[free][:, free]
        Ff  = F[free]

        if not self._use_amg or ndof < self.ndof_threshold:
            if not self._use_amg and ndof > 15000:
                print(f"\n[WARNING] 'pyamg' module not found! Solving {ndof} DOFs with SciPy's spsolve.")
                print("[WARNING] This will be EXTREMELY SLOW for 3D meshes and may look like a freeze/hang!\n")
            U[free] = spsolve(Kff, Ff)
            return U

        rebuild = (self._ml is None or self._last_E_e is None)
        if not rebuild:
            delta = np.max(np.abs(E_e - self._last_E_e)) / max(np.max(self._last_E_e), 1e-12)
            rebuild = delta > self.rebuild_tol

        if rebuild:
            Kff_csr = csr_matrix(Kff)
            try:
                # FIX: Removed fatal 'coarse_solver="pinv"' which freezes on large 3D matrices
                self._ml = pyamg.smoothed_aggregation_solver(Kff_csr)
                self._last_E_e = E_e.copy()
            except Exception as e:
                print(f"[Warning] PyAMG hierarchy build failed: {e}. Falling back to slow spsolve.")
                U[free] = spsolve(Kff, Ff)
                return U

        try:
            x, info = cg(csr_matrix(Kff), Ff,
                         M=self._ml.aspreconditioner(cycle='V'),
                         rtol=self.cg_tol, maxiter=self.cg_maxiter)
            # FIX: Only fallback on breakdown (<0). If it hits maxiter (>0), x is still 
            # a perfectly valid approximate solution. Fallback causes massive unneeded delays.
            if info < 0:
                print(f"[Warning] PyAMG CG solve failed with info={info}. Falling back to slow spsolve.")
                x = spsolve(Kff, Ff)   
        except Exception as e:
            print(f"[Warning] PyAMG CG solve threw exception: {e}. Falling back to slow spsolve.")
            x = spsolve(Kff, Ff)

        U[free] = x
        return U

    def invalidate(self):
        self._ml = None
        self._last_E_e = None


@dataclass
class SIMPParams:
    nelx: int = 60
    nely: int = 30
    nelz: int = 0               
    volfrac: float = 0.5
    penal: float = 3.0
    rmin: float = 1.5
    move: float = 0.2
    max_iter: int = 100
    tol: float = 0.01
    compliance_tol: float = 5e-4    
    compliance_window: int = 5      
    min_iter: int = 50
    tail_default_iters: int = 18
    use_heaviside: bool = True
    beta_init: float = 1.0
    beta_max: float = 32.0
    eta: float = 0.5
    min_penal_for_best: float = 3.0
    max_gray_for_best: float = 0.25
    seed: Optional[int] = None
    checkpoint_dir: Optional[str] = None
    checkpoint_every: int = 10
    amg_ndof_threshold: int = 3000    
    amg_rebuild_tol: float = 0.15     

@dataclass
class StepState:
    iteration: int
    compliance: float
    best_compliance: float
    best_iteration: int
    volume_fraction: float
    grayness: float
    best_grayness: float
    checkerboard: float
    obj_slope: float
    rel_change_1: float
    rel_change_5: float
    stagnation_counter: int
    penal: float
    rmin: float
    move: float
    beta: float
    converged: bool
    best_is_valid: bool
    compliance_history: list[float] = field(default_factory=list)

def _build_ke_unit_2d(nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    gp = 1.0 / np.sqrt(3.0)
    D = E / (1-nu**2) * np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    xn = np.array([-1,1,1,-1], dtype=float)
    yn = np.array([-1,-1,1,1], dtype=float)
    KE = np.zeros((8,8))
    for xi, et in [(-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)]:
        dNdxi  = 0.25*np.array([-(1-et), (1-et), (1+et),-(1+et)])
        dNdeta = 0.25*np.array([-(1-xi),-(1+xi), (1+xi), (1-xi)])
        J = np.array([[dNdxi@xn,dNdxi@yn],[dNdeta@xn,dNdeta@yn]])
        Ji = np.linalg.inv(J)
        dNdx = Ji[0,0]*dNdxi + Ji[0,1]*dNdeta
        dNdy = Ji[1,0]*dNdxi + Ji[1,1]*dNdeta
        B = np.zeros((3,8))
        for i in range(4):
            B[0,2*i]=dNdx[i]; B[1,2*i+1]=dNdy[i]
            B[2,2*i]=dNdy[i]; B[2,2*i+1]=dNdx[i]
        KE += B.T @ D @ B * np.linalg.det(J)
    return KE

def _build_ke_unit_3d(nu: float = 0.3, E: float = 1.0) -> np.ndarray:
    gp = 1.0/np.sqrt(3.0)
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float)
    gauss_pts = signs * gp
    nc = signs.copy()   
    lam = E*nu/((1+nu)*(1-2*nu))
    mu  = E/(2*(1+nu))
    D = np.zeros((6,6))
    D[0,0]=D[1,1]=D[2,2]=lam+2*mu
    D[0,1]=D[0,2]=D[1,0]=D[1,2]=D[2,0]=D[2,1]=lam
    D[3,3]=D[4,4]=D[5,5]=mu
    KE = np.zeros((24,24))
    for r,s,t in gauss_pts:
        dN = 0.125*np.array([[-(1-s)*(1-t),-(1-r)*(1-t),-(1-r)*(1-s)],[ (1-s)*(1-t),-(1+r)*(1-t),-(1+r)*(1-s)],[ (1+s)*(1-t), (1+r)*(1-t),-(1+r)*(1+s)],[-(1+s)*(1-t), (1-r)*(1-t),-(1-r)*(1+s)],[-(1-s)*(1+t),-(1-r)*(1+t), (1-r)*(1-s)],[ (1-s)*(1+t),-(1+r)*(1+t), (1+r)*(1-s)],[ (1+s)*(1+t), (1+r)*(1+t), (1+r)*(1+s)],[-(1+s)*(1+t), (1-r)*(1+t), (1-r)*(1+s)],
        ])
        J = dN.T @ nc
        dNdxyz = dN @ np.linalg.inv(J).T   
        dx,dy,dz = dNdxyz[:,0],dNdxyz[:,1],dNdxyz[:,2]
        B = np.zeros((6,24))
        idx = np.arange(8)
        B[0,3*idx]=dx; B[1,3*idx+1]=dy; B[2,3*idx+2]=dz
        B[3,3*idx]=dy; B[3,3*idx+1]=dx
        B[4,3*idx+1]=dz; B[4,3*idx+2]=dy
        B[5,3*idx]=dz;   B[5,3*idx+2]=dx
        KE += B.T @ D @ B * abs(np.linalg.det(J))
    return KE

KE_UNIT_2D = _build_ke_unit_2d()   
KE_UNIT_3D = _build_ke_unit_3d()   

def _edof_table_2d(nelx: int, nely: int) -> np.ndarray:
    ex = np.repeat(np.arange(nelx), nely)
    ey = np.tile(np.arange(nely), nelx)
    bl = ex*(nely+1) + ey
    br = (ex+1)*(nely+1) + ey
    return np.stack([2*bl,2*bl+1,2*br,2*br+1,
                     2*br+2,2*br+3,2*bl+2,2*bl+3], axis=1).astype(np.int32)

def _edof_table_3d(nelx: int, nely: int, nelz: int) -> np.ndarray:
    nny, nnz = nely+1, nelz+1
    ex,ey,ez = np.meshgrid(np.arange(nelx),np.arange(nely),
                            np.arange(nelz), indexing='ij')
    ex,ey,ez = ex.ravel(),ey.ravel(),ez.ravel()
    def nid(ix,iy,iz): return ix*nny*nnz + iy*nnz + iz
    nodes = np.stack([nid(ex,ey,ez),   nid(ex+1,ey,ez),
                      nid(ex+1,ey+1,ez),nid(ex,ey+1,ez),
                      nid(ex,ey,ez+1), nid(ex+1,ey,ez+1),
                      nid(ex+1,ey+1,ez+1),nid(ex,ey+1,ez+1)], axis=1)  
    edof = (3*nodes[:,:,None] + np.array([0,1,2])[None,None,:]).reshape(len(ex), 24)
    return edof.astype(np.int32)

def _build_sparse_indices(edof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ke_size = edof.shape[1]
    rows = np.repeat(edof, ke_size, axis=1)   
    cols = np.tile(edof, (1, ke_size))         
    return rows.ravel().astype(np.int32), cols.ravel().astype(np.int32)

def _build_density_filter(nelx, nely, rmin, nelz=0):
    if nelz == 0:
        ix = np.repeat(np.arange(nelx), nely)
        iy = np.tile(np.arange(nely), nelx)
        coords = np.stack([ix,iy], axis=1).astype(float)
    else:
        ix,iy,iz = np.meshgrid(np.arange(nelx),np.arange(nely),
                                np.arange(nelz), indexing='ij')
        coords = np.stack([ix.ravel(),iy.ravel(),iz.ravel()], axis=1).astype(float)

    n = len(coords)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(rmin - 1e-10, output_type='ndarray')

    if len(pairs) > 0:
        dist = np.linalg.norm(coords[pairs[:,0]] - coords[pairs[:,1]], axis=1)
        w    = rmin - dist
        H_rows = np.concatenate([pairs[:,0], pairs[:,1], np.arange(n)])
        H_cols = np.concatenate([pairs[:,1], pairs[:,0], np.arange(n)])
        H_vals = np.concatenate([w, w, np.full(n, rmin)])
    else:
        H_rows = np.arange(n, dtype=np.int32)
        H_cols = np.arange(n, dtype=np.int32)
        H_vals = np.ones(n) * rmin

    Hs = np.zeros(n)
    np.add.at(Hs, H_rows, H_vals)
    H_mat = csr_matrix((H_vals / Hs[H_rows], (H_rows, H_cols)), shape=(n, n))
    return H_mat

def _heaviside(rho_f, beta, eta):
    if beta < 1e-6: return rho_f.copy()
    return (np.tanh(beta*eta) + np.tanh(beta*(rho_f-eta))) / \
           (np.tanh(beta*eta) + np.tanh(beta*(1.0-eta)))

def _heaviside_deriv(rho_f, beta, eta):
    if beta < 1e-6: return np.ones_like(rho_f)
    denom = np.tanh(beta*eta) + np.tanh(beta*(1.0-eta))
    return beta*(1.0 - np.tanh(beta*(rho_f-eta))**2) / denom

def _grayness(rho): return float(4.0 * np.mean(rho*(1.0-rho)))

def _checkerboard_2d(rho, nelx, nely):
    R = rho.reshape(nelx, nely)
    if nelx < 2 or nely < 2: return 0.0
    b = (R[:-1:2,:-1:2]+R[1::2,:-1:2]+R[:-1:2,1::2]+R[1::2,1::2])/4
    return float(np.std(b))

def _checkerboard_3d(rho, nelx, nely, nelz):
    R = rho.reshape(nelx, nely, nelz)
    if nelx<2 or nely<2 or nelz<2: return 0.0
    b = (R[:-1:2,:-1:2,:-1:2]+R[1::2,:-1:2,:-1:2]+
         R[:-1:2,1::2,:-1:2]+R[1::2,1::2,:-1:2]+
         R[:-1:2,:-1:2,1::2]+R[1::2,:-1:2,1::2]+
         R[:-1:2,1::2,1::2]+R[1::2,1::2,1::2])/8
    return float(np.std(b))

def _bc_cantilever_2d(nelx, nely):
    ndof = 2*(nelx+1)*(nely+1)
    F = np.zeros(ndof)
    F[2*(nelx*(nely+1)+nely//2)+1] = -1.0
    j = np.arange(nely+1)
    fixed = np.unique(np.concatenate([2*j, 2*j+1]))
    return fixed, np.setdiff1d(np.arange(ndof), fixed), F

def _bc_mbb_2d(nelx, nely):
    ndof = 2*(nelx+1)*(nely+1)
    F = np.zeros(ndof)
    F[2*nely+1] = -1.0          
    j = np.arange(nely+1)
    sym_dofs = 2*j              
    roller = 2*(nelx*(nely+1))+1  
    fixed = np.unique(np.append(sym_dofs, roller))
    return fixed, np.setdiff1d(np.arange(ndof), fixed), F

def _bc_lbracket_2d(nelx, nely):
    ndof = 2*(nelx+1)*(nely+1)
    F = np.zeros(ndof)
    load_node = nelx*(nely+1) + (3*nely//4)
    F[2*load_node+1] = -1.0
    fixed =[]
    for i in range(nelx//2+1):
        n = i*(nely+1)
        fixed.extend([2*n, 2*n+1])
    fixed = np.unique(fixed).astype(int)
    return fixed, np.setdiff1d(np.arange(ndof), fixed), F

def _bc_cantilever_3d(nelx, nely, nelz):
    nny, nnz = nely+1, nelz+1
    ndof = 3*(nelx+1)*nny*nnz
    F = np.zeros(ndof)
    tip = nelx*nny*nnz + (nely//2)*nnz + nelz//2
    F[3*tip+1] = -1.0
    iy,iz = np.meshgrid(np.arange(nny), np.arange(nnz), indexing='ij')
    face = (iy*nnz+iz).ravel()
    fixed = np.unique(np.concatenate([3*face,3*face+1,3*face+2]))
    return fixed, np.setdiff1d(np.arange(ndof), fixed), F

BC_PRESETS = {
    "cantilever": {"2d": _bc_cantilever_2d, "3d": _bc_cantilever_3d},
    "mbb":        {"2d": _bc_mbb_2d},
    "lbracket":   {"2d": _bc_lbracket_2d},
}

def _oc_update(rho, dc, dv, volfrac, move, n_elem, beta, eta, use_heaviside, H_mat):
    dc_s = np.minimum(dc, -1e-12)
    dv_s = np.maximum(dv, 1e-12)  
    l1, l2 = 0.0, 1e9
    
    for _ in range(100):
        lm = 0.5 * (l1 + l2)
        rn = rho * np.sqrt(-dc_s / (lm * dv_s))
        rn = np.clip(rn, rho - move, rho + move)
        rn = np.clip(rn, 1e-3, 1.0)
        
        rn_f = H_mat.dot(rn)
        rn_phys = _heaviside(rn_f, beta, eta) if (use_heaviside and beta > 1e-6) else rn_f
        
        if rn_phys.sum() / n_elem > volfrac:
            l1 = lm
        else:
            l2 = lm
        if l2 - l1 < 1e-9:
            break
            
    return rn

def _fea_and_sensitivity(rho_design, penal, rmin, move, beta, eta,
                          use_heaviside, H_mat,
                          *, KE_UNIT, ke_size, volfrac, n_elem, ndof,
                          edof, free, F, row_idx, col_idx, E0, Emin,
                          amg_solver=None):
    rho_f = H_mat.dot(rho_design)
    rho_phys = _heaviside(rho_f, beta, eta) if (use_heaviside and beta>1e-6) else rho_f

    E_e = Emin + (E0-Emin)*rho_phys**penal          
    KE_vals = (E_e[:,None] * KE_UNIT.ravel()[None,:]).ravel()
    K = csc_matrix((KE_vals, (row_idx, col_idx)), shape=(ndof, ndof))

    if amg_solver is not None:
        U = amg_solver.solve(K, F, free, E_e, ndof)
    else:
        U = np.zeros(ndof)
        U[free] = spsolve(K[free][:, free], F[free])
    compliance = float(F @ U)

    Ue  = U[edof]                              
    KUe = Ue @ KE_UNIT                         
    ce  = np.einsum('ei,ei->e', KUe, Ue)      
    dc_phys = -penal*(E0-Emin)*rho_phys**(penal-1)*ce

    dh = _heaviside_deriv(rho_f, beta, eta) if (use_heaviside and beta>1e-6) else np.ones_like(rho_f)
    dc_f = dc_phys * dh
    dc_design = H_mat.T.dot(dc_f)

    dv_f = 1.0 * dh
    dv_design = H_mat.T.dot(dv_f)

    rho_new = _oc_update(rho_design, dc_design, dv_design, volfrac, move, n_elem, 
                         beta, eta, use_heaviside, H_mat)
    change  = float(np.max(np.abs(rho_new - rho_design)))
    
    return compliance, rho_new, change, rho_phys

def _apply_action(action, penal, rmin, move, beta, rho_new, best_rho, best_is_valid):
    if action.get("restart", False) and best_is_valid:
        rho_new = best_rho.copy()
    if "penal" in action: penal = float(np.clip(action["penal"], 1.0, 5.0))
    if "rmin"  in action: rmin  = float(np.clip(action["rmin"],  1.1, 4.0))
    if "move"  in action: move  = float(np.clip(action["move"],  0.03, 0.4))
    if "beta"  in action: beta  = float(np.clip(action["beta"],  1.0, 64.0))
    return penal, rmin, move, beta, rho_new

def _tail_defaults(callback, params):
    tail = {"enabled": False, "tail_iters": params.tail_default_iters,
            "restart_from_best": True, "penal": 4.5, "rmin": 1.2,
            "move": 0.05, "beta": min(params.beta_max, 32.0)}
    if callback is not None and hasattr(callback, "finalize_tail"):
        custom = callback.finalize_tail(params)
        if custom: tail.update(custom)
    return tail

def _save_checkpoint(path, state_dict):
    arrays  = {k: v for k,v in state_dict.items() if isinstance(v, np.ndarray)}
    scalars = {k: (v.tolist() if isinstance(v,np.ndarray) else v)
               for k,v in state_dict.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path+".tmp", **arrays)
    os.replace(path+".tmp.npz", path+".npz")
    with open(path+".json","w") as f: json.dump(scalars, f)

def _load_checkpoint(path):
    if not (os.path.exists(path+".npz") and os.path.exists(path+".json")):
        return None
    data = dict(np.load(path+".npz"))
    with open(path+".json") as f: data.update(json.load(f))
    return data

def run_simp(
    params: SIMPParams,
    callback=None,
    verbose: bool = False,
    bc_override=None,
    problem: str = "cantilever",
) -> dict:
    if params.seed is not None:
        np.random.seed(params.seed)

    is_3d = params.nelz > 0
    nelx, nely, nelz = params.nelx, params.nely, params.nelz
    n_elem = nelx * nely * (nelz if is_3d else 1)
    E0, Emin = 1.0, 1e-9
    dim_key = "3d" if is_3d else "2d"

    if is_3d:
        edof = _edof_table_3d(nelx, nely, nelz)
        ndof = 3*(nelx+1)*(nely+1)*(nelz+1)
        KE_UNIT, ke_size = KE_UNIT_3D, 24
    else:
        edof = _edof_table_2d(nelx, nely)
        ndof = 2*(nelx+1)*(nely+1)
        KE_UNIT, ke_size = KE_UNIT_2D, 8

    if bc_override is not None:
        bc_fn = bc_override
        _, free, F = (bc_fn(nelx, nely, nelz) if is_3d else bc_fn(nelx, nely))
    else:
        preset = BC_PRESETS.get(problem, BC_PRESETS["cantilever"])
        bc_fn = preset.get(dim_key)
        if bc_fn is None:
            raise ValueError(f"Problem '{problem}' not available for {dim_key.upper()}")
        _, free, F = (bc_fn(nelx, nely, nelz) if is_3d else bc_fn(nelx, nely))

    row_idx, col_idx = _build_sparse_indices(edof)
    H_mat = _build_density_filter(nelx, nely, params.rmin, nelz)

    fea_kw = dict(KE_UNIT=KE_UNIT, ke_size=ke_size,
                  volfrac=params.volfrac, n_elem=n_elem, ndof=ndof,
                  edof=edof, free=free, F=F,
                  row_idx=row_idx, col_idx=col_idx, E0=E0, Emin=Emin)

    amg_solver = _AMGSolver(
        ndof_threshold=params.amg_ndof_threshold,
        rebuild_tol=params.amg_rebuild_tol,
    )
    fea_kw["amg_solver"] = amg_solver

    rho   = params.volfrac * np.ones(n_elem)
    penal = params.penal
    rmin  = params.rmin
    move  = params.move
    beta  = params.beta_init
    eta   = params.eta
    prev_rmin = rmin

    if callback is not None and hasattr(callback, "initial_action"):
        init = callback.initial_action(params)
        if init:
            penal, rmin, move, beta, rho = _apply_action(
                init, penal, rmin, move, beta, rho, rho, False)

    best_rho        = rho.copy()
    best_compliance = np.inf
    best_iteration  = 0
    best_is_valid   = False
    best_grayness   = _grayness(rho)
    # Pre-valid tracking: best snapshot BEFORE gate is satisfied.
    # Kept separate so best_rho / best_compliance / best_grayness / best_iteration
    # are ONLY updated when gate_ok=True.  The tail restart ignores pre_valid_best
    # entirely (it already checks best_is_valid before restarting).
    pre_valid_best_rho        = rho.copy()
    pre_valid_best_compliance = np.inf
    pre_valid_best_iteration  = 0
    pre_valid_best_grayness   = _grayness(rho)
    MIN_ITER_FOR_FALLBACK     = 10   # ignore early compliance spikes
    compliance_hist: list[float] = []
    params_log: list[dict] =[]
    stagnation_counter = 0
    start_iter = 1

    ckpt_path = None
    if params.checkpoint_dir is not None:
        os.makedirs(params.checkpoint_dir, exist_ok=True)
        tag = f"{nelx}x{nely}" + (f"x{nelz}" if is_3d else "") + f"_{problem}"
        ckpt_path = os.path.join(params.checkpoint_dir, f"ckpt_{tag}")
        ckpt = _load_checkpoint(ckpt_path)
        if ckpt is not None:
            rho            = ckpt["rho"]
            best_rho       = ckpt["best_rho"]
            best_compliance= float(ckpt["best_compliance"])
            best_iteration = int(ckpt["best_iteration"])
            best_is_valid  = bool(int(ckpt["best_is_valid"]))
            pre_valid_best_compliance = best_compliance
            pre_valid_best_iteration  = best_iteration
            pre_valid_best_grayness   = best_grayness
            pre_valid_best_rho        = best_rho.copy()
            best_grayness  = float(ckpt["best_grayness"])
            penal = float(ckpt["penal"]); rmin = float(ckpt["rmin"])
            move  = float(ckpt["move"]);  beta = float(ckpt["beta"])
            compliance_hist= list(ckpt.get("compliance_hist", np.array([])))
            start_iter     = int(ckpt["iteration"]) + 1

    iteration = start_iter - 1
    for iteration in range(start_iter, params.max_iter+1):

        if abs(rmin - prev_rmin) > 1e-6:
            H_mat = _build_density_filter(nelx, nely, rmin, nelz)
            amg_solver.invalidate()   
            prev_rmin = rmin

        compliance, rho_new, change, rho_phys = _fea_and_sensitivity(
            rho, penal, rmin, move, beta, eta, params.use_heaviside, H_mat, **fea_kw)
        compliance_hist.append(compliance)

        gray_now = _grayness(rho_phys)
        gate_ok  = penal >= params.min_penal_for_best and gray_now < params.max_gray_for_best
        if compliance < best_compliance and gate_ok:
            # Valid best: gate satisfied — this is what the tail restarts from.
            best_compliance = compliance; best_rho = rho_phys.copy()
            best_iteration  = iteration;  best_grayness = gray_now
            best_is_valid   = True;       stagnation_counter = 0
        else:
            # Track pre-valid best separately (never used by tail restart).
            # Guard against early-iteration compliance spikes corrupting the snapshot.
            if (compliance < pre_valid_best_compliance and
                    not best_is_valid and
                    iteration >= MIN_ITER_FOR_FALLBACK):
                pre_valid_best_compliance = compliance
                pre_valid_best_rho        = rho_phys.copy()
                pre_valid_best_iteration  = iteration
                pre_valid_best_grayness   = gray_now
            if compliance >= best_compliance or not gate_ok:
                stagnation_counter += 1

        conv_rho  = change < params.tol
        win = compliance_hist[-params.compliance_window:]
        conv_C = (len(win) == params.compliance_window and
                  abs(win[-1]-win[0]) / max(abs(win[0]), 1e-10) < params.compliance_tol)
        conv = conv_rho and conv_C

        slope = (win[-1]-win[0]) / max(len(win)-1, 1)
        rel1  = 0.0 if len(compliance_hist)<2 else \
                (compliance_hist[-1]-compliance_hist[-2])/max(abs(compliance_hist[-2]),1e-10)
        rel5  = 0.0 if len(compliance_hist)<6 else \
                (compliance_hist[-1]-compliance_hist[-6])/max(abs(compliance_hist[-6]),1e-10)
        checker = (_checkerboard_3d(rho_phys,nelx,nely,nelz) if is_3d
                   else _checkerboard_2d(rho_phys,nelx,nely))

        state = StepState(
            iteration=iteration, compliance=compliance,
            best_compliance=best_compliance, best_iteration=best_iteration,
            volume_fraction=float(rho_phys.sum()/n_elem),
            grayness=gray_now, best_grayness=best_grayness,
            checkerboard=checker, obj_slope=slope,
            rel_change_1=rel1, rel_change_5=rel5,
            stagnation_counter=stagnation_counter,
            penal=penal, rmin=rmin, move=move, beta=beta,
            converged=conv, best_is_valid=best_is_valid,
            compliance_history=list(compliance_hist),
        )
        params_log.append({"iter": iteration, "phase": "main",
                           "penal": penal, "rmin": rmin, "move": move, "beta": beta,
                           "compliance": compliance, "best_compliance": best_compliance,
                           "grayness": gray_now, "best_is_valid": best_is_valid,
                           "volume_fraction": state.volume_fraction})

        if verbose:
            print(f"  iter {iteration:3d}  C={compliance:8.4f}"
                  f"  gray={gray_now:.3f}  p={penal:.2f}  β={beta:.1f}  r={rmin:.2f}"
                  f"  chg={change:.4f}")

        if callback is not None:
            action = callback(state, rho_new)
            if action:
                penal, rmin, move, beta, rho_new = _apply_action(
                    action, penal, rmin, move, beta, rho_new, best_rho, best_is_valid)
                if action.get("restart", False) and best_is_valid:
                    amg_solver.invalidate()   
                if action.get("stop", False) and iteration >= params.min_iter:
                    rho = rho_new; break

        rho = rho_new

        if ckpt_path is not None and iteration % params.checkpoint_every == 0:
            _save_checkpoint(ckpt_path, dict(
                rho=rho, best_rho=best_rho,
                compliance_hist=np.array(compliance_hist),
                best_compliance=best_compliance, best_iteration=best_iteration,
                best_is_valid=int(best_is_valid), best_grayness=best_grayness,
                penal=penal, rmin=rmin, move=move, beta=beta, iteration=iteration))

        if conv and iteration >= params.min_iter:
            break

    tail = _tail_defaults(callback, params)
    tail_iters_run = 0
    if tail.get("enabled", False) and tail.get("tail_iters", 0) > 0:
        if tail.get("restart_from_best", True) and best_is_valid:
            rho = np.clip(best_rho, 1e-3, 1.0)
        elif tail.get("restart_from_best", True) and not best_is_valid:
            # No valid best exists — reset to uniform density.
            # Without this, the tail gets a free warm-start from the
            # main loop's final rho, which defeats ablation controllers
            # like TailOnlyController that should start from scratch.
            rho = params.volfrac * np.ones(n_elem)
        penal = float(tail.get("penal", penal)); rmin = float(tail.get("rmin", rmin))
        move  = float(tail.get("move",  move));  beta = float(tail.get("beta", beta))
        H_mat = _build_density_filter(nelx, nely, rmin, nelz)
        amg_solver.invalidate()   
        
        for k in range(1, int(tail["tail_iters"])+1):
            compliance, rho_new, change, rho_phys = _fea_and_sensitivity(
                rho, penal, rmin, move, beta, eta, params.use_heaviside, H_mat, **fea_kw)
            compliance_hist.append(compliance)
            params_log.append({"iter": iteration+k, "phase": "tail",
                               "penal": penal, "rmin": rmin, "move": move, "beta": beta,
                               "compliance": compliance, "best_compliance": best_compliance,
                               "grayness": _grayness(rho_phys), "best_is_valid": best_is_valid,
                               "volume_fraction": float(rho_phys.sum()/n_elem)})
            rho = rho_new; tail_iters_run = k

    final_c, _, _, rho_final = _fea_and_sensitivity(
        rho, penal, rmin, move, beta, eta, params.use_heaviside, H_mat, **fea_kw)

    return {
        "compliance_history":       compliance_hist,
        "rho_final":                rho_final,
        "best_rho":                 best_rho,
        "best_compliance":          best_compliance,
        "best_iteration":           best_iteration,
        "final_compliance":         final_c,
        "final_grayness":           _grayness(rho_final),
        "best_grayness":            best_grayness,
        "best_is_valid":            best_is_valid,
        # Pre-valid diagnostics — what the solver tracked before gate was satisfied.
        # These are NEVER used by the tail. Exposed for diagnostic transparency only.
        "pre_valid_best_compliance": pre_valid_best_compliance,
        "pre_valid_best_iteration":  pre_valid_best_iteration,
        "pre_valid_best_grayness":   pre_valid_best_grayness,
        "n_iter":                   iteration + tail_iters_run,
        "params_log":               params_log,
        "tail_config":              tail,
        "is_3d": is_3d, "nelx": nelx, "nely": nely, "nelz": nelz,
    }