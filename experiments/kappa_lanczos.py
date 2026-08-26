"""
kappa_lanczos.py
----------------
Condition number of the stiffness operator, as a bound that can be trusted.

The estimator this replaces was inverse iteration with a Jacobi-PCG inner
solve.  It converged at 64k and failed everywhere else: at 216k and 512k every
inner solve after the first exhausted a 60,000-iteration cap, so the reported
lambda_min -- and the condition number computed from it -- were the cap rather
than a measurement.  The failure is structural, not a budget problem.  After
the first inverse-iteration step the right-hand side is nearly the smallest
eigenvector, and a diagonally preconditioned Krylov method solving against a
near-null-space right-hand side on a system with kappa near 1e6 does not
converge in any budget worth spending.

The replacement asks for less and gets it.  Run the Lanczos recurrence on K
from a random start, form the tridiagonal T_m, and take its extreme Ritz
values.  For a symmetric positive definite operator the Ritz values are
eigenvalues of an orthogonal projection of K, so they lie inside the spectrum:

    theta_min >= lambda_min      theta_max <= lambda_max

which makes theta_max/theta_min a *lower bound* on kappa(K).  That is the
useful direction here.  The claim the number supports is that
eps_bf16 * kappa is far greater than one, and a lower bound establishes it;
an unconverged estimate of unknown sign does not.

No reorthogonalization is used.  In finite precision Lanczos loses global
orthogonality and produces spurious copies of already-converged Ritz values,
but copies land *on* the spectrum, not outside it, so the bracketing above
survives.  The run is also cheap enough to make the point moot: a few thousand
double-precision operator applications, seconds rather than the six minutes per
mesh the inverse iteration spent failing.

lambda_max is additionally estimated by power iteration and the two are
reported side by side.  Both underestimate lambda_max -- power iteration
converges to it from below, and theta_max is bounded above by it -- but Lanczos
converges faster, so the healthy outcome is theta_max slightly ABOVE the power
estimate, with the ratio close to one.  Measured here it is 1.003 to 1.004,
which says both have resolved the top of the spectrum and neither run was cut
short.  A ratio below one would mean the Lanczos run was the weaker estimate
and should be lengthened.

Usage:
    python experiments/phase3/kappa_lanczos.py --sizes 64k,216k,512k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZES = {"64k": (80, 40, 20), "216k": (120, 60, 30), "512k": (160, 80, 40),
         "1M": (200, 100, 50), "2M": (252, 126, 63)}


def power_iteration(matvec, n, mask, iters, seed, cp):
    rng = cp.random.RandomState(seed)
    v = rng.rand(n).astype(cp.float64) * mask
    v /= cp.linalg.norm(v)
    lam = float("nan")
    for _ in range(iters):
        w = matvec(v)
        nw = float(cp.linalg.norm(w))
        if nw == 0.0:
            break
        v = w / nw
        lam = nw
    return lam


def lanczos_ritz(matvec, n, mask, steps, seed, cp, np):
    """Extreme Ritz values of K from an m-step Lanczos run.

    Returns (theta_min, theta_max, m_used, breakdown).  The recurrence is the
    plain three-term one; `beta` collapsing to zero means the Krylov space is
    exhausted, which for a random start on these operators means the run
    should stop and say so rather than divide by it.
    """
    rng = cp.random.RandomState(seed)
    v = rng.rand(n).astype(cp.float64) * mask
    v /= cp.linalg.norm(v)
    v_prev = cp.zeros_like(v)
    alphas, betas = [], []
    beta = 0.0
    breakdown = False
    for _ in range(steps):
        w = matvec(v)
        alpha = float(cp.dot(w, v))
        w = w - alpha * v - beta * v_prev
        alphas.append(alpha)
        beta = float(cp.linalg.norm(w))
        if beta <= 1e-300:
            breakdown = True
            break
        betas.append(beta)
        v_prev, v = v, w / beta
    m = len(alphas)
    # eigenvalues of the symmetric tridiagonal, on the host: m is a few
    # thousand at most and this costs milliseconds
    T = np.diag(np.array(alphas))
    if m > 1:
        off = np.array(betas[:m - 1])
        T += np.diag(off, 1) + np.diag(off, -1)
    ev = np.linalg.eigvalsh(T)
    return float(ev[0]), float(ev[-1]), m, breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64k,216k,512k")
    ap.add_argument("--penals", default="3.0,5.0")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--emin", type=float, default=1e-9)
    ap.add_argument("--pi-iters", type=int, default=200)
    ap.add_argument("--lanczos-steps", type=int, default=3000)
    ap.add_argument("--gate", default="G6")
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.cuda_operators import OperatorSuite
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.simp_r2 import build_cantilever

    out = os.path.join(_ROOT, "results", args.gate, "kappa_estimation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []

    eps_bf16 = 2.0 ** -8
    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZES[tag]
        prob = build_cantilever(nelx, nely, nelz, load="patch")
        n_elem, ndof = prob["n_elem"], prob["ndof"]
        print(f"\n### {tag}: {n_elem:,} elem, {ndof:,} DOF", flush=True)
        cp.get_default_memory_pool().free_all_blocks()
        suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof)
        mask = cp.zeros(ndof, dtype=cp.float64)
        mask[cp.asarray(prob["free"])] = 1

        for penal in [float(p) for p in args.penals.split(",")]:
            E = cp.full(n_elem,
                        args.emin + (1.0 - args.emin) * args.rho ** penal,
                        dtype=cp.float64)

            def mv(v):
                return suite.matvec_full(v * mask, E,
                                         path="fused_ai_fp64") * mask

            t0 = time.perf_counter()
            lmax_pi = power_iteration(mv, ndof, mask, args.pi_iters, 42, cp)
            th_min, th_max, m, brk = lanczos_ritz(
                mv, ndof, mask, args.lanczos_steps, 123, cp, np)
            dt = time.perf_counter() - t0
            kappa_lb = th_max / th_min if th_min > 0 else float("nan")
            rows.append({
                "size": tag, "n_elem": n_elem, "ndof": ndof,
                "state": f"uniform_p{penal:g}", "rho_mean": args.rho,
                "penal": penal, "Emin": args.emin,
                "estimator": "lanczos_ritz",
                "lam_max_power": lmax_pi,
                "theta_max": th_max, "theta_min": th_min,
                "kappa_lower_bound": kappa_lb,
                "eps_bf16_kappa_lb": eps_bf16 * kappa_lb,
                "bf16_reference_bound": 1.0 / eps_bf16,
                "lanczos_steps": m, "lanczos_breakdown": brk,
                "power_iters": args.pi_iters,
                "theta_max_over_power": th_max / lmax_pi if lmax_pi else None,
                "elapsed_s": dt,
            })
            print(f"  p={penal:g}: theta_max={th_max:.6g} "
                  f"(power {lmax_pi:.6g}, ratio "
                  f"{th_max/lmax_pi:.4f})  theta_min={th_min:.6g}  "
                  f"kappa >= {kappa_lb:.3e}  eps*kappa >= "
                  f"{eps_bf16*kappa_lb:.3e}  ({m} steps, {dt:.1f} s)",
                  flush=True)
        del suite
        cp.get_default_memory_pool().free_all_blocks()

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwritten: {os.path.relpath(out, _ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
