"""
simp_r2.py
----------
SIMP driver: standard physics, strict measurement protocol.

The optimization itself is textbook -- Q1 elements, three-field
regularisation, OC update, Jacobi-preconditioned CG, continuation on the
penalty and projection sharpness.  What this driver does differently is
refuse the four conveniences that make a timing or a compliance number
unfalsifiable:

  * A failed linear solve is an error, not a warning.  If a solve is allowed
    to warn and continue, unconverged displacements flow into the compliance
    and the sensitivities, and every number downstream is contaminated
    silently.  The protocol is in `pcg` below.
  * The final iterate is the result.  Returning the best historical iterate
    -- or clamping a reported compliance to the best when it drifts above it,
    or restarting from the best when it rises -- reports something the
    optimizer did not converge to.  There is no selection, no clamp and no
    restart here.
  * Termination is a declared criterion on the physical field, not an
    iteration budget.  A run that stops on the safety guard is recorded as
    not converged rather than reported as a result.
  * The volume constraint is enforced on exactly the field it is written on,
    the projected physical density, and the achieved physical volume fraction
    is reported for every run.

Two distinct regularisation floors are easily conflated, because both are
often written `rho_min`:

    rho_lb      lower bound on the *raw design variable*      (code: 1e-3)
    Emin/E0     stiffness floor in the SIMP interpolation     (code: 1e-9)

They are different regularisations and are now reported separately.  Note
that filtering and projection sit between them, so the minimum *projected*
density is not rho_lb and the minimum element stiffness is not Emin either;
the achieved minima are recorded per run.
"""

from __future__ import annotations

import json
import math
import time


class LinearSolveNotConverged(RuntimeError):
    """Raised when a solve misses tolerance twice.  Invalidates the run.

    This is the fail-closed rule.  A reported result may not contain a linear
    solve that missed tolerance. Log-and-continue is how a run ends up with
    every one of its solves at the iteration cap and still reports a result.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Problem construction
# ─────────────────────────────────────────────────────────────────────────────

def _node_id(ix, iy, iz, nny, nnz):
    return ix * nny * nnz + iy * nnz + iz


def build_cantilever(nelx, nely, nelz, Lx=2.0, Ly=1.0, Lz=0.5,
                     load="patch", patch_frac=None, patch_size=0.1,
                     total_load=-1.0):
    """Cantilever: left face fully fixed, transverse load on the right face.

    load="point"  single node at the right-face midpoint.  A concentrated
                  load in 3D elasticity is singular, so the compliance does
                  not converge under mesh refinement and the benchmark is not
                  physically the same problem at different resolutions.

    load="patch"  the same total force spread over a square patch of *fixed
                  physical area* centred at the same point.  Total load and
                  patch area are both held constant across refinements, so
                  the traction converges and compliance is mesh-comparable.
    """
    import numpy as np

    nnx, nny, nnz = nelx + 1, nely + 1, nelz + 1
    n_node = nnx * nny * nnz
    ndof = 3 * n_node

    # -- supports: x = 0 face, all DOFs -------------------------------------
    iy, iz = np.meshgrid(np.arange(nny), np.arange(nnz), indexing="ij")
    left = _node_id(0, iy.ravel(), iz.ravel(), nny, nnz)
    fixed = np.concatenate([3 * left, 3 * left + 1, 3 * left + 2])
    fixed = np.unique(fixed)

    # -- load ---------------------------------------------------------------
    F = np.zeros(ndof)
    hy, hz = Ly / nely, Lz / nelz
    if load == "point":
        jy = int(round((Ly / 2.0) / hy))
        jz = int(round((Lz / 2.0) / hz))
        n = _node_id(nelx, jy, jz, nny, nnz)
        F[3 * n + 1] = total_load
        patch_nodes = 1
        patch_area = 0.0
    elif load == "patch":
        a = float(patch_size) if patch_frac is None else float(patch_frac) * Lz
        y0, y1 = Ly / 2.0 - a / 2.0, Ly / 2.0 + a / 2.0
        z0, z1 = Lz / 2.0 - a / 2.0, Lz / 2.0 + a / 2.0
        ys = np.arange(nny) * hy
        zs = np.arange(nnz) * hz
        my = (ys >= y0 - 1e-12) & (ys <= y1 + 1e-12)
        mz = (zs >= z0 - 1e-12) & (zs <= z1 + 1e-12)
        jy = np.where(my)[0]
        jz = np.where(mz)[0]
        if jy.size == 0:
            jy = np.array([int(round((Ly / 2.0) / hy))])
        if jz.size == 0:
            jz = np.array([int(round((Lz / 2.0) / hz))])
        gy, gz = np.meshgrid(jy, jz, indexing="ij")
        nodes = _node_id(nelx, gy.ravel(), gz.ravel(), nny, nnz)
        F[3 * nodes + 1] = total_load / nodes.size
        patch_nodes = int(nodes.size)
        patch_area = a * a
    else:
        raise ValueError(f"unknown load model {load!r}")

    free = np.setdiff1d(np.arange(ndof), fixed)
    return {
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "Lx": Lx, "Ly": Ly, "Lz": Lz,
        "ndof": ndof, "n_node": n_node, "n_elem": nelx * nely * nelz,
        "fixed": fixed, "free": free, "F": F,
        "load_model": load, "patch_nodes": patch_nodes,
        "patch_area": patch_area, "total_load": total_load,
    }


def build_torsion(nelx, nely, nelz, Lx=3.0, Ly=1.0, Lz=1.0,
                  load="patch", patch_size=0.15, total_load=1.0):
    """Torsion shaft: left face fixed, equal-and-opposite couple on the right.

    A concentrated nodal couple is singular in 3D elasticity, so the couple
    here is applied over two fixed-area patches with the total force on each
    held constant, which keeps the benchmark the same problem under mesh
    refinement.
    """
    import numpy as np

    nnx, nny, nnz = nelx + 1, nely + 1, nelz + 1
    ndof = 3 * nnx * nny * nnz
    iy, iz = np.meshgrid(np.arange(nny), np.arange(nnz), indexing="ij")
    left = _node_id(0, iy.ravel(), iz.ravel(), nny, nnz)
    fixed = np.unique(np.concatenate([3 * left, 3 * left + 1, 3 * left + 2]))

    F = np.zeros(ndof)
    hy, hz = Ly / nely, Lz / nelz
    zs = np.arange(nnz) * hz
    mz = np.abs(zs - Lz / 2.0) <= patch_size / 2.0 + 1e-12
    jz = np.where(mz)[0]
    if jz.size == 0:
        jz = np.array([int(round((Lz / 2.0) / hz))])

    ys = np.arange(nny) * hy
    n_patch = 0
    for y_target, sy, sz in ((Ly, -0.5, +0.5), (0.0, +0.5, -0.5)):
        my = np.abs(ys - y_target) <= patch_size / 2.0 + 1e-12
        jy = np.where(my)[0]
        if jy.size == 0:
            jy = np.array([int(round(y_target / hy))])
        gy, gz = np.meshgrid(jy, jz, indexing="ij")
        nodes = _node_id(nelx, gy.ravel(), gz.ravel(), nny, nnz)
        F[3 * nodes + 1] += total_load * sy / nodes.size
        F[3 * nodes + 2] += total_load * sz / nodes.size
        n_patch += int(nodes.size)

    free = np.setdiff1d(np.arange(ndof), fixed)
    return {
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "Lx": Lx, "Ly": Ly, "Lz": Lz,
        "ndof": ndof, "n_node": nnx * nny * nnz, "n_elem": nelx * nely * nelz,
        "fixed": fixed, "free": free, "F": F,
        "load_model": load, "patch_nodes": n_patch,
        "patch_area": patch_size * patch_size, "total_load": total_load,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fail-closed Jacobi-preconditioned CG
# ─────────────────────────────────────────────────────────────────────────────

def pcg(matvec, b, M_inv, x0=None, tol=1e-5, maxiter=20000, mask=None,
        check_every=50, residual_history=None):
    """Jacobi-PCG whose stopping test is the TRUE residual.

    Returns (x, iters, converged, rel_resid), where rel_resid is
    ||b - Ax|| / ||b|| formed explicitly, not the recursively updated vector.

    This distinction is the whole reason this function was rewritten. Textbook
    CG carries a residual by recursion, r <- r - alpha*Ap, because that is free
    where recomputing b - Ax costs a matrix-vector product. In double precision
    the two agree to the last digit. In single precision they do not: over
    thousands of iterations the recursion drifts below the true residual by two
    to three orders, so a solve that "reaches" 1e-5 has in fact reached 1e-3.
    A solver that tests and reports the recursive quantity therefore
    overstates its own single-precision convergence by that factor.

    The fix costs one matrix-vector product per check, and it buys two things
    at once. The test is evaluated on b - Ax, so it cannot be fooled. And the
    recursion is *restarted* from that same recomputed vector -- residual
    replacement -- so the drift is corrected rather than merely detected. The
    second is free: the matvec needed for the test is the matvec needed for the
    replacement.

    `check_every` therefore sets both the test cadence and the replacement
    cadence, and trades accuracy of the stopping point against overhead. At 50
    the extra matvec is two per cent of the work.

    It also removes one host synchronisation and one full-vector reduction per
    iteration, which matters here: a device-to-host scalar transfer costs about
    300 us on this hardware, an order of magnitude more than a fused matvec at
    216k elements, so testing convergence every iteration makes the loop
    latency-bound and hides the differences between operator mappings that this
    paper exists to measure. Note what it does *not* do: the breakdown guard
    `float(pAp)` below still synchronises once per iteration, so the loop is not
    synchronisation-free, only synchronised half as often. Removing that guard
    would be a further saving and is deliberately not taken, because every wall
    time in the paper was measured with it in place and changing the solver
    would invalidate them all.

    Single precision cannot reach 1e-5 on these systems by this route or any
    other -- its attainable residual floors near 1e-3 -- so a single-precision
    solve here is expected to exhaust `maxiter` and return converged=False.
    That is a result, not a failure of the routine. Use `pcg_ir` to get the
    tolerance out of a single-precision operator.
    """
    import cupy as cp
    dt = b.dtype
    x = cp.zeros_like(b) if x0 is None else x0.astype(dt, copy=True)
    if mask is not None:
        x *= mask
    r = b - matvec(x)
    if mask is not None:
        r *= mask
    z = M_inv * r
    p = z.copy()
    rz = cp.dot(r, z).astype(dt)

    b_norm = float(cp.linalg.norm(b))
    if b_norm <= 0:
        return x, 0, True, 0.0
    tol_abs = tol * b_norm

    r_norm = float(cp.linalg.norm(r))
    if residual_history is not None:
        residual_history.append(r_norm / b_norm)
    if r_norm <= tol_abs:
        return x, 0, True, r_norm / b_norm

    k = max(1, int(check_every))
    converged = False
    iters = 0
    for _ in range(maxiter):
        Ap = matvec(p)
        if mask is not None:
            Ap *= mask
        pAp = cp.dot(p, Ap).astype(dt)
        if float(pAp) == 0.0:
            break
        alpha = rz / pAp
        x = x + alpha * p
        iters += 1

        if iters % k == 0 or iters >= maxiter:
            # One matvec, used twice: it is the convergence test and it is the
            # replacement that stops the recursion drifting.
            r = b - matvec(x)
            if mask is not None:
                r *= mask
            r_norm = float(cp.linalg.norm(r))
            if residual_history is not None:
                residual_history.append(r_norm / b_norm)
            if r_norm <= tol_abs:
                converged = True
                break
        else:
            r = r - alpha * Ap

        z = M_inv * r
        rz_new = cp.dot(r, z).astype(dt)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    if not converged:
        r = b - matvec(x)
        if mask is not None:
            r *= mask
        r_norm = float(cp.linalg.norm(r))
    return x, iters, converged, r_norm / b_norm


def pcg_ir(matvec64, b64, matvec32, M_inv32, tol=1e-5, mask32=None,
           inner_tol=1e-1, max_outer=40, inner_maxiter=4000,
           check_every=50, trace=None, x0=None):
    """Mixed-precision iterative refinement: fp64 residual, fp32 correction.

    Single precision cannot reach the equilibrium tolerance these systems need
    -- its attainable residual floors two to three orders above 1e-5 -- so the
    fast single-precision operator is unusable on its own. It is not unusable
    in general. Forming the residual in double precision, solving for the
    correction in single with the fast kernel, and applying the update in
    double, recovers the tolerance and keeps most of the speed.

    The correction is solved to a deliberately *loose* inner tolerance. A tight
    inner solve is wasted work: the outer loop only needs a direction good
    enough to cut the residual by an order, and buying more than that costs
    inner iterations that the next outer step would have made unnecessary.
    Measured on this problem, 1e-1 beats 1e-2, and 1e-3 is three to five times
    worse.

    Returns (x64, outer, inner_total, converged, rel_resid) with rel_resid the
    true double-precision residual.
    """
    import cupy as cp
    nb = float(cp.linalg.norm(b64))
    if nb <= 0:
        return cp.zeros_like(b64), 0, 0, True, 0.0
    # Warm start from the previous design iterate, exactly as the direct
    # solver does. Without it the comparison is rigged: the direct path gets
    # to begin near the answer at every design step and the refinement path
    # begins at zero.
    x = cp.zeros_like(b64) if x0 is None else x0.astype(cp.float64, copy=True)
    inner_total = 0
    outer = 0
    # Divergence guard.
    #
    # Refinement is only guaranteed to converge while the inner solve returns a
    # correction that actually points downhill, and a single-precision inner
    # solve on a system with kappa near 1e6 does not always manage it. Applying
    # such a correction unconditionally -- which this loop originally did -- can
    # increase the outer residual, and because the loop then solves for a
    # correction to a worse iterate, it compounds: one 1M run reached a relative
    # residual of 8.6e3 after 397,100 inner iterations before the outer cap
    # stopped it.
    #
    # So: remember the residual, and if a step made it worse, undo that step
    # and re-solve the correction to a tenfold tighter inner tolerance. Undoing
    # costs nothing extra in memory because the correction is still in scope.
    # If tightening runs out of room the routine returns converged=False with
    # the best residual it held, and the caller's fail-closed rule invalidates
    # the run rather than reporting a diverged solution.
    #
    # The guard is inert on a healthy solve: it compares two scalars that were
    # already computed, and never fires when the residual is decreasing.
    tol_in = inner_tol
    rel_prev = None
    d_last = None
    scale_last = 0.0
    for outer in range(1, max_outer + 1):
        r = b64 - matvec64(x)
        rel = float(cp.linalg.norm(r)) / nb
        if trace is not None:
            trace.append(rel)
        if rel <= tol:
            return x, outer - 1, inner_total, True, rel
        if rel_prev is not None and rel > rel_prev and d_last is not None:
            x = x - d_last.astype(cp.float64) * scale_last
            d_last = None
            tol_in *= 0.1
            if tol_in < 1e-5:
                return x, outer, inner_total, False, rel_prev
            continue
        rel_prev = rel
        # Scale the residual to O(1) before dropping to single precision, so
        # the correction solve is not working near the bottom of the exponent
        # range once the outer residual is small.
        scale = float(cp.linalg.norm(r))
        d32, it32, _ok, _res = pcg(matvec32, (r / scale).astype(cp.float32),
                                   M_inv32, tol=tol_in,
                                   maxiter=inner_maxiter, mask=mask32,
                                   check_every=check_every)
        inner_total += it32
        d_last, scale_last = d32, scale
        x = x + d32.astype(cp.float64) * scale
    r = b64 - matvec64(x)
    rel = float(cp.linalg.norm(r)) / nb
    if trace is not None:
        trace.append(rel)
    return x, outer, inner_total, bool(rel <= tol), rel


def project(rho_bar, beta, eta=0.5):
    import cupy as cp
    if beta <= 1e-12:
        return rho_bar
    tb_eta = math.tanh(beta * eta)
    den = tb_eta + math.tanh(beta * (1.0 - eta))
    return (tb_eta + cp.tanh(beta * (rho_bar - eta))) / den


def dproject(rho_bar, beta, eta=0.5):
    import cupy as cp
    if beta <= 1e-12:
        return cp.ones_like(rho_bar)
    tb_eta = math.tanh(beta * eta)
    den = tb_eta + math.tanh(beta * (1.0 - eta))
    ch = cp.cosh(beta * (rho_bar - eta))
    return beta / (den * ch * ch)


# ─────────────────────────────────────────────────────────────────────────────
# Continuation schedule
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SCHEDULE = [
    # (until_iter, penal, beta, move, rmin_target)
    (15,  1.5, 1.0,  0.20, 1.50),
    (40,  3.5, 4.0,  0.15, 1.35),
    (65,  4.5, 16.0, 0.08, 1.25),
    (120, 4.5, 32.0, 0.05, 1.20),
]


def schedule_at(it, schedule=None):
    """Continuation state at outer iteration `it` (1-based).

    A four-phase continuation.  After the last
    phase the parameters are held fixed and the run continues until the
    convergence test fires -- that is the only change, and it is what turns a
    fixed-budget run into a converged optimization.
    """
    sch = schedule or DEFAULT_SCHEDULE
    for until, p, beta, move, rmin in sch:
        if it <= until:
            return p, beta, move, rmin, (until == sch[-1][0])
    p, beta, move, rmin = sch[-1][1:]
    return p, beta, move, rmin, True


# ─────────────────────────────────────────────────────────────────────────────
# Memory instrumentation
# ─────────────────────────────────────────────────────────────────────────────

class MemoryWatch:
    """Peak allocated and peak reserved, plus a device-level high-water mark.

    An end-of-run allocator snapshot is not peak memory: temporaries may
    already have been released, or retained differently by the pool, by the
    time it is taken.  This samples during the run and keeps maxima,
    and separately records the device free/total high-water from
    cudaMemGetInfo, which no allocator bookkeeping can hide.
    """

    def __init__(self):
        import cupy as cp
        self.pool = cp.get_default_memory_pool()
        self.peak_used = 0
        self.peak_reserved = 0
        self.peak_device_used = 0
        free, total = cp.cuda.runtime.memGetInfo()
        self.device_total = total
        self.baseline_device_used = total - free

    def sample(self):
        import cupy as cp
        self.peak_used = max(self.peak_used, self.pool.used_bytes())
        self.peak_reserved = max(self.peak_reserved, self.pool.total_bytes())
        free, total = cp.cuda.runtime.memGetInfo()
        self.peak_device_used = max(self.peak_device_used, total - free)

    def report(self):
        return {
            "peak_allocated_bytes": int(self.peak_used),
            "peak_reserved_bytes": int(self.peak_reserved),
            "peak_device_used_bytes": int(self.peak_device_used),
            "device_total_bytes": int(self.device_total),
            "baseline_device_used_bytes": int(self.baseline_device_used),
            "peak_allocated_GiB": self.peak_used / 2**30,
            "peak_reserved_GiB": self.peak_reserved / 2**30,
            "peak_device_used_GiB": self.peak_device_used / 2**30,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Optimality-criteria update, bisecting on the PHYSICAL volume
# ─────────────────────────────────────────────────────────────────────────────

def oc_update(rho, dc, dv, volfrac, move, filt, beta, eta, rho_lb,
              bisect_tol=1e-8, max_bisect=80, lam_hint=None):
    """One OC step.

    The bisection enforces the volume constraint on the projected physical
    density -- mean(rho_tilde) = V_f -- which is what this code does
    (`pub_simp_solver.py:333`) but not what the stated formulation says.  The
    achieved physical volume is returned so every run can report it.

    Each bisection probe costs a filter application plus a projection, so a
    cold [0, 1e9] bracket costs ~60 filter applications per design step.  On
    a converged, warm-started run that overhead is no longer negligible next
    to the linear solve, and it would dilute exactly the operator comparison
    the paper is making.  The bracket is therefore seeded from the previous
    step's multiplier and expanded only if it fails to bracket, which brings
    the typical cost down to a handful of probes without changing the
    converged result: the returned design still satisfies the same volume
    tolerance.
    """
    import cupy as cp
    dcn = cp.minimum(dc, -1e-30)          # OC needs a descent direction
    dvp = cp.maximum(dv, 1e-30)

    def design(lmid):
        step = cp.sqrt(-dcn / (lmid * dvp))
        rn = cp.clip(cp.clip(rho * step, rho - move, rho + move),
                     rho_lb, 1.0)
        return rn, float(project(filt.forward(rn), beta, eta).mean())

    probes = 0
    if lam_hint is not None and lam_hint > 0:
        l1, l2 = lam_hint / 4.0, lam_hint * 4.0
        for _ in range(30):               # expand until the root is bracketed
            _, v1 = design(l1)
            _, v2 = design(l2)
            probes += 2
            if v1 > volfrac and v2 <= volfrac:
                break
            if v1 <= volfrac:
                l1 /= 8.0
            if v2 > volfrac:
                l2 *= 8.0
            if l1 < 1e-30 or l2 > 1e30:
                l1, l2 = 0.0, 1e9
                break
    else:
        l1, l2 = 0.0, 1e9

    rho_new, achieved = design(0.5 * (l1 + l2))
    probes += 1
    lam = 0.5 * (l1 + l2)
    for _ in range(max_bisect):
        if (l2 - l1) <= bisect_tol * (l1 + l2 + 1e-30):
            break
        lam = 0.5 * (l1 + l2)
        rho_new, achieved = design(lam)
        probes += 1
        if achieved > volfrac:
            l1 = lam
        else:
            l2 = lam
    return rho_new, achieved, lam, probes


# ─────────────────────────────────────────────────────────────────────────────
# The optimization loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simp(problem, path="fused_fp32", volfrac=0.30, rmin=1.5,
             penal=None, E0=1.0, Emin=1e-9, rho_lb=1e-3, eta=0.5,
             cg_tol=1e-5, cg_maxiter=20000, warm_start=True,
             max_outer=400, schedule=None,
             conv_drho=0.01, conv_dc_rel=1e-4, conv_window=10,
             fixed_budget=None, physical_rmin=None,
             log_every=1, verbose=True, record_history=True,
             fused_sensitivity=True,
             ir_inner=None, ir_tol=1e-1, ir_max_outer=40,
             ir_residual_path="fused_fp64"):
    """Run a complete SIMP optimization under the fail-closed protocol.

    Termination.  The run stops when, *after the last continuation
    stage*, both

        max|rho^(k) - rho^(k-1)|                        <= conv_drho
        relative compliance spread over conv_window     <= conv_dc_rel

    hold.  `max_outer` is a guard, not the design of the experiment: a run
    that stops on the guard is reported as not converged.  Passing
    `fixed_budget=120` selects fixed-iteration behaviour so
    the two can be compared directly, and such a run is labelled a
    fixed-work throughput probe rather than a converged optimization.

    Fail-closed.  Every linear solve must reach `cg_tol` *on the true
    residual*.  A warm-started solve that misses is retried from zero and the
    retry's iterations and time are counted in the reported totals.  If the
    retry also misses, LinearSolveNotConverged is raised and the run is
    invalid; there is no log-and-continue path.

    Mixed precision.  Setting `ir_inner` to a single-precision mapping runs
    the equilibrium solve as iterative refinement instead: the residual is
    formed in double precision with `ir_residual_path`, the correction is
    solved in single precision by `ir_inner`, and the update is applied in
    double.  This exists because single precision cannot reach `cg_tol` on
    these systems by any direct route -- its attainable residual floors two to
    three orders above 1e-5 -- so a direct single-precision run is expected to
    raise.  The residual operator is the fastest double-precision mapping
    rather than the double-precision twin of `ir_inner`, because in practice
    one would use the fastest accurate operator available for it.
    """
    import cupy as cp
    from .cuda_operators import OperatorSuite, PATH_SPEC, NEEDS_EDOF
    from .filter_r2 import ConeFilter, physical_rmin_elements
    from .pub_simp_solver import KE_UNIT_3D

    nelx, nely, nelz = problem["nelx"], problem["nely"], problem["nelz"]
    n_elem, ndof = problem["n_elem"], problem["ndof"]
    use_ir = ir_inner is not None
    if use_ir:
        # The outer state is double precision; `path` names the operator that
        # forms the residual, and `ir_inner` the one that solves for the
        # correction.
        path = ir_residual_path
    dt = cp.float64 if PATH_SPEC[path][0] == "fp64" else cp.float32

    mem = MemoryWatch()
    t_start = time.perf_counter()

    suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof,
                          build_edof=(path in NEEDS_EDOF))
    suite32 = mask32 = None
    if use_ir:
        suite32 = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof,
                                build_edof=(ir_inner in NEEDS_EDOF))
        mask32 = cp.zeros(ndof, dtype=cp.float32)
        mask32[cp.asarray(problem["free"])] = 1
    # The filter, the projection and the OC bisection stay in FP64 on every
    # path.  They are O(n_elem) work sitting next to hundreds of O(n_elem)
    # matvecs, so their cost is negligible, and holding them fixed means the
    # precision comparison measures the operator rather than the optimizer.
    fixed_phys_r = physical_rmin is not None
    r0 = (physical_rmin_elements(physical_rmin, problem["Lx"], nelx)
          if fixed_phys_r else rmin)
    filt = ConeFilter(nelx, nely, nelz, r0, dtype=cp.float64)
    adj_err = filt.check_adjoint()

    mask = cp.zeros(ndof, dtype=dt)
    mask[cp.asarray(problem["free"])] = 1
    F = cp.asarray(problem["F"], dtype=dt) * mask
    rho = cp.full(n_elem, volfrac, dtype=cp.float64)
    u = cp.zeros(ndof, dtype=dt)

    state = {"E": cp.ones(n_elem, dtype=dt)}

    def matvec(v):
        return suite.matvec_full(v * mask, state["E"], path=path) * mask

    def matvec32(v):
        return suite32.matvec_full(v * mask32, state["E32"],
                                   path=ir_inner) * mask32

    history, solve_log = [], []
    total_cg_iters = 0
    total_retries = 0
    converged_outer = False
    stop_reason = "guard"
    it = 0
    budget = fixed_budget or max_outer
    rho_phys_prev = None
    lam_prev = None
    t_solve_total = t_adjoint_total = t_oc_total = 0.0

    while it < budget:
        it += 1
        p_it, beta_it, move_it, rmin_it, in_last_phase = schedule_at(
            it, schedule)
        if penal is not None:
            p_it = penal
        if not fixed_phys_r:
            filt.set_rmin(rmin_it)

        rho_bar = filt.forward(rho)
        rho_phys = project(rho_bar, beta_it, eta)
        E_e = Emin + (E0 - Emin) * rho_phys ** p_it
        state["E"] = E_e.astype(dt)

        diag = suite.diagonal(state["E"], path=path)
        M_inv = cp.where(diag > 0, 1.0 / cp.maximum(diag, 1e-300),
                         cp.zeros_like(diag)) * mask

        t0 = time.perf_counter()
        retried = False
        outer_ir = 0
        if use_ir:
            state["E32"] = state["E"].astype(cp.float32)
            d32 = suite32.diagonal(state["E32"], path=ir_inner)
            M_inv32 = cp.where(d32 > 0, 1.0 / cp.maximum(d32, 1e-300),
                               cp.zeros_like(d32)) * mask32
            u_new, outer_ir, iters_total, ok, resid = pcg_ir(
                matvec, F, matvec32, M_inv32, tol=cg_tol, mask32=mask32,
                inner_tol=ir_tol, max_outer=ir_max_outer,
                inner_maxiter=cg_maxiter,
                x0=(u if (warm_start and it > 1) else None))
        else:
            x0 = u if (warm_start and it > 1) else None
            u_new, iters, ok, resid = pcg(matvec, F, M_inv, x0=x0,
                                          tol=cg_tol, maxiter=cg_maxiter,
                                          mask=mask)
            iters_total = iters
            if not ok and x0 is not None:
                retried = True
                total_retries += 1
                u_new, iters2, ok, resid = pcg(matvec, F, M_inv, x0=None,
                                               tol=cg_tol,
                                               maxiter=cg_maxiter, mask=mask)
                iters_total += iters2      # the retry is counted, not hidden
        cp.cuda.Stream.null.synchronize()
        t_solve = time.perf_counter() - t0

        if not ok:
            raise LinearSolveNotConverged(
                f"outer iteration {it}: CG missed tol={cg_tol:g} twice "
                f"(achieved relative residual {resid:.3e} after "
                f"{iters_total} iterations, cap {cg_maxiter}). "
                f"Run invalidated under the fail-closed rule.")

        u = u_new
        total_cg_iters += iters_total
        solve_log.append({
            "iter": it, "cg_iters": iters_total, "rel_resid": resid,
            "ir_outer": outer_ir if use_ir else None,
            "converged": True, "warm": (not use_ir) and (it > 1)
                                       and warm_start,
            "retried_from_zero": retried, "solve_s": t_solve,
            "penal": p_it, "beta": beta_it, "rmin": filt.rmin,
        })

        t_adj0 = time.perf_counter()
        c = float(cp.dot(F, u))
        ce = suite.elem_compliance(u, fused=fused_sensitivity).astype(
            cp.float64)
        dEdrp = p_it * (E0 - Emin) * rho_phys ** (p_it - 1.0)
        dc_drp = -dEdrp * ce
        dprj = dproject(rho_bar, beta_it, eta)
        dc_drho = filt.transpose(dc_drp * dprj)
        dv_drho = filt.transpose(dprj / n_elem)
        cp.cuda.Stream.null.synchronize()
        t_adjoint = time.perf_counter() - t_adj0

        t_oc0 = time.perf_counter()
        rho_old = rho
        rho, achieved_vol, lam, probes = oc_update(
            rho, dc_drho, dv_drho, volfrac, move_it, filt, beta_it, eta,
            rho_lb, lam_hint=lam_prev)
        lam_prev = lam
        cp.cuda.Stream.null.synchronize()
        t_oc = time.perf_counter() - t_oc0
        change = float(cp.abs(rho - rho_old).max())

        # Design change measured on the PHYSICAL field, not the raw variable.
        #
        # Under a strong Heaviside projection the raw density keeps drifting
        # at the move limit in the saturated region while the projected field
        # -- the one that carries the stiffness, the volume and the objective
        # -- is completely frozen.  In the 216k pilot the raw max|d rho| sat
        # at exactly 0.0500 for 330 consecutive iterations while compliance
        # was constant to all printed digits and the warm-started solves
        # needed zero CG iterations.  A termination test on the raw variable
        # therefore never fires, which is why fixed-budget runs stop on an
        # iteration budget instead.  The physical change is the meaningful
        # quantity and is what the convergence test uses; both are reported.
        change_phys = (float(cp.abs(rho_phys - rho_phys_prev).max())
                       if rho_phys_prev is not None else float("inf"))
        rho_phys_prev = rho_phys

        gray = float((4.0 * rho_phys * (1.0 - rho_phys)).mean())
        t_solve_total += t_solve
        t_adjoint_total += t_adjoint
        t_oc_total += t_oc
        mem.sample()

        # Always recorded. The outer convergence test reads its compliance
        # window from `history`, and so does every reported final quantity, so
        # gating this append on `record_history` did not merely omit a log --
        # it disabled termination and returned nan for the compliance. The
        # flag controls what is *returned* (the "history" key below), not
        # whether the optimization works.
        if True:
            history.append({
                "iter": it, "compliance": c, "change": change,
                # `null`, not `Infinity`, on the first iterate where there is
                # no previous physical field to difference against. Python's
                # json module emits a bare `Infinity`, which RFC 8259 does not
                # allow and strict parsers reject -- an avoidable obstacle in
                # a results tree meant to be read by other people. The
                # convergence test above uses the local float, not this.
                "change_phys": (None if math.isinf(change_phys)
                                else change_phys),
                "vol_phys": achieved_vol, "vol_raw": float(rho.mean()),
                "grayness": gray, "cg_iters": iters_total,
                "rel_resid": resid, "penal": p_it, "beta": beta_it,
                "move": move_it, "rmin": filt.rmin,
                "rho_phys_min": float(rho_phys.min()),
                "rho_phys_max": float(rho_phys.max()),
                "E_min_achieved": float(E_e.min()),
                "solve_s": t_solve, "adjoint_s": t_adjoint, "oc_s": t_oc,
                "oc_probes": probes, "retried": retried,
            })

        if verbose and (it % log_every == 0):
            print(f"  it {it:4d}  c={c:.6f}  dr={change:.4f}  "
                  f"drp={change_phys:.4f}  "
                  f"V={achieved_vol:.4f}  g={gray:.4f}  "
                  f"cg={iters_total:5d}  r={resid:.2e}  "
                  f"p={p_it:.1f} b={beta_it:.0f}"
                  + ("  [retry]" if retried else ""), flush=True)

        if fixed_budget is None and in_last_phase and len(history) > conv_window:
            recent = [h["compliance"] for h in history[-(conv_window + 1):]]
            spread = (max(recent) - min(recent)) / max(abs(recent[-1]), 1e-30)
            if change_phys <= conv_drho and spread <= conv_dc_rel:
                converged_outer = True
                stop_reason = "outer_criterion"
                break

    if fixed_budget is not None:
        stop_reason = "fixed_budget"

    wall = time.perf_counter() - t_start
    mem.sample()

    p_end, beta_end = schedule_at(it, schedule)[0], schedule_at(it, schedule)[1]
    rho_bar = filt.forward(rho)
    rho_phys = project(rho_bar, beta_end, eta)
    E_end = Emin + (E0 - Emin) * rho_phys ** p_end

    # One more solve, on the design the run actually ends on.
    #
    # The loop computes the compliance of iterate k and *then* takes the OC
    # step to iterate k+1, so the last entry in `history` belongs to the design
    # before the final update while `rho_phys` above belongs to the design
    # after it. Reporting the first as the compliance of the second mixes two
    # iterates. The discrepancy is small -- of order 1e-5 relative, far below
    # the design-convergence criterion -- but the protocol states that the
    # compliance, the volume fraction, the non-discreteness and the density
    # field are all read from one design, and a claim about internal
    # consistency has to be exactly true or not made at all.
    #
    # Cost is one equilibrium solve on top of roughly seventy, and it is
    # fail-closed like every other.
    E_final = E_end.astype(dt)
    diag_f = suite.diagonal(E_final, path=path)
    M_inv_f = cp.where(diag_f > 0, 1.0 / cp.maximum(diag_f, 1e-300),
                       cp.zeros_like(diag_f)) * mask

    def matvec_final(v):
        return suite.matvec_full(v * mask, E_final, path=path) * mask

    if use_ir:
        state["E32"] = E_final.astype(cp.float32)
        d32f = suite32.diagonal(state["E32"], path=ir_inner)
        M_inv32f = cp.where(d32f > 0, 1.0 / cp.maximum(d32f, 1e-300),
                            cp.zeros_like(d32f)) * mask32
        u_f, _o, it_f, ok_f, res_f = pcg_ir(
            matvec_final, F, matvec32, M_inv32f, tol=cg_tol, mask32=mask32,
            inner_tol=ir_tol, max_outer=ir_max_outer, inner_maxiter=cg_maxiter)
    else:
        u_f, it_f, ok_f, res_f = pcg(matvec_final, F, M_inv_f, x0=None,
                                     tol=cg_tol, maxiter=cg_maxiter, mask=mask)
    if not ok_f:
        raise LinearSolveNotConverged(
            f"final-iterate solve missed tol={cg_tol:g} (achieved relative "
            f"residual {res_f:.3e} after {it_f} iterations). Run invalidated "
            f"under the fail-closed rule.")
    total_cg_iters += it_f
    final_compliance = float(cp.dot(F, u_f))
    # Logged like every other solve, so it counts towards the reported
    # iteration total and towards `max_rel_resid` rather than sitting outside
    # the accounting.
    solve_log.append({
        "iter": it + 1, "cg_iters": it_f, "rel_resid": res_f,
        "ir_outer": None, "converged": True, "warm": False,
        "retried_from_zero": False, "solve_s": None,
        "penal": p_end, "beta": beta_end, "rmin": filt.rmin,
        "final_iterate": True,
    })

    result = {
        "path": path, "nelx": nelx, "nely": nely, "nelz": nelz,
        "n_elem": n_elem, "ndof": ndof,
        "volfrac": volfrac, "rmin_initial": r0, "rmin_final": filt.rmin,
        "fixed_physical_rmin": fixed_phys_r,
        "physical_rmin": physical_rmin,
        "filter_neighbours_per_elem": filt.neighbours_per_element(),
        "filter_adjoint_err": adj_err,
        "E0": E0, "Emin": Emin, "rho_lb": rho_lb, "eta": eta,
        "cg_tol": cg_tol, "cg_maxiter": cg_maxiter, "warm_start": warm_start,
        "load_model": problem["load_model"],
        "patch_nodes": problem["patch_nodes"],
        "patch_area": problem["patch_area"],
        "outer_iters": it,
        "outer_converged": converged_outer,
        "stop_reason": stop_reason,
        "final_compliance": final_compliance,
        "final_compliance_pre_update": (history[-1]["compliance"] if history
                                        else float("nan")),
        "final_change": history[-1]["change"] if history else float("nan"),
        "final_change_phys": (history[-1]["change_phys"] if history
                              else float("nan")),
        "conv_drho": conv_drho, "conv_dc_rel": conv_dc_rel,
        "conv_window": conv_window,
        "final_vol_phys": float(rho_phys.mean()),
        "final_grayness": float((4.0 * rho_phys * (1.0 - rho_phys)).mean()),
        "rho_phys_min": float(rho_phys.min()),
        "E_min_achieved": float(E_end.min()),
        "total_cg_iters": total_cg_iters,
        "warm_start_retries": total_retries,
        "max_cg_iters_single_solve": max((s["cg_iters"] for s in solve_log),
                                         default=0),
        "max_rel_resid": max((s["rel_resid"] for s in solve_log), default=0.0),
        "solves_at_cap": 0,            # zero by construction: fail-closed
        "wall_s": wall,
        # Phase breakdown.  The end-to-end speedup of any operator change is
        # bounded by the share of wall time the operator actually occupies,
        # so that share is reported rather than left for the reader to infer.
        "t_linear_solve_s": t_solve_total,
        "t_adjoint_s": t_adjoint_total,
        "t_oc_s": t_oc_total,
        "t_other_s": wall - t_solve_total - t_adjoint_total - t_oc_total,
        "linear_solve_share": t_solve_total / wall if wall > 0 else float("nan"),
        "memory": mem.report(),
        "history": history if record_history else [],
        "solve_log": solve_log,
    }
    result["rho_final_device"] = rho_phys
    return result


def save_result(result, out_path, include_field=False):
    """Write a run to JSON, optionally alongside the final density field."""
    import os
    import numpy as np
    out = {k: v for k, v in result.items() if k != "rho_final_device"}
    if include_field and result.get("rho_final_device") is not None:
        npy = out_path.replace(".json", "_rho.npy")
        np.save(npy, result["rho_final_device"].get())
        out["rho_field_file"] = os.path.basename(npy)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out_path

