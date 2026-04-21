"""
workflow.py
-----------
GPU-FEM pipeline orchestrator.

Adapted from TO3D workflow.py:
  - Imports GPU modules (simp_gpu, surrogate_gpu) instead of CPU versions
  - Adds `device` and `backend` parameters for GPU control
  - Same retry logic, artefact saving, and report format as TO3D

Pipeline:
  1. NL prompt / spec / preset  →  ProblemSpec   (ConfiguratorAgent, from AUTO/)
  2. ProblemSpec                →  BCs            (AutoSIMP bc_generator)
  3. GPU SIMP loop + surrogate  →  result dict    (simp_gpu + SensitivitySurrogateGPU)
  4. result dict                →  EvalResult     (EvaluatorAgent, from AUTO/)
  5. Retry if failed (up to max_retries)
  6. Save report + 3D artefacts
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np

from .problem_spec import ProblemSpec
from .bc_generator import generate_bc
from .auto_simp import install_passive_patch, uninstall_passive_patch

from .simp_gpu import TO3DParams, run_simp_surrogate_gpu
from .surrogate_gpu import SensitivitySurrogateGPU
from .local_agents import (
    SurrogateRouterAgent,
    AggressiveSurrogateRouter,
    ConservativeSurrogateRouter,
    PureFEMRouter,
    LLMRouterAgent,
    ActiveLearningRouter,
    TemporalSuperResolutionRouter,
    TO3DConfiguratorAgent,
    TO3DEvaluatorAgent,
)
from .fem_gpu import detect_gpu_backend, resolve_backend_choice


# ─────────────────────────────────────────────────────────────────────────────
# Builder helpers (same as TO3D workflow.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_param_controller(controller_name: str, max_iter: int, verbose: bool):
    try:
        if controller_name == "llm":
            return None
        elif controller_name == "schedule":
            from .pub_baseline_controller import ScheduleOnlyController
            return ScheduleOnlyController()
        else:
            return None
    except Exception:
        try:
            from .pub_baseline_controller import ScheduleOnlyController
            return ScheduleOnlyController()
        except Exception:
            return None


def _build_router(router_name: str, max_iter: int, verbose: bool, feature_mode: str = "basic"):
    routers = {
        "rule":         SurrogateRouterAgent,
        "aggressive":   AggressiveSurrogateRouter,
        "conservative": ConservativeSurrogateRouter,
        "fem_only":     PureFEMRouter,
        "llm":          LLMRouterAgent,
        "active":       ActiveLearningRouter,
    }
    if feature_mode == "temporal_sr" and router_name in {"rule", "aggressive", "conservative", "active"}:
        return TemporalSuperResolutionRouter()
    cls = routers.get(router_name, SurrogateRouterAgent)
    return cls()


# ─────────────────────────────────────────────────────────────────────────────
# 3D artefact saving (identical to TO3D workflow.py)
# ─────────────────────────────────────────────────────────────────────────────

def _save_3d_artefacts(rho, nelx, nely, nelz, output_dir, tag=""):
    saved = {}
    np_path = os.path.join(output_dir, f"rho{tag}.npy")
    np.save(np_path, rho)
    saved["density_npy"] = np_path
    _try_save_slices(rho, nelx, nely, nelz, output_dir, tag, saved)
    _try_save_obj(rho, nelx, nely, nelz, output_dir, tag, saved)
    return saved


def _try_save_slices(rho, nelx, nely, nelz, outdir, tag, saved):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        R = rho.reshape(nelx, nely, nelz)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle("Density slices (black = solid, white = void)", fontsize=11)

        for ax, (sl, title, xlabel, ylabel) in zip(axes, [
            (R[:, :, nelz // 2].T, f"XY (z={nelz//2})", "x", "y"),
            (R[:, nely // 2, :].T, f"XZ (y={nely//2})", "x", "z"),
            (R[nelx // 2, :, :].T, f"YZ (x={nelx//2})", "y", "z"),
        ]):
            ax.imshow(1.0 - sl, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

        plt.tight_layout()
        path = os.path.join(outdir, f"density_slices{tag}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["density_slices_png"] = path
    except Exception:
        pass


def _try_save_obj(rho, nelx, nely, nelz, outdir, tag, saved):
    try:
        from skimage import measure
        grid = np.zeros((nelx + 2, nely + 2, nelz + 2), dtype=np.float32)
        grid[1:-1, 1:-1, 1:-1] = rho.reshape(nelx, nely, nelz)
        verts, faces, normals, _ = measure.marching_cubes(grid, level=0.3)
        verts -= 1.0
        lines = ["newmtl topology\nKd 0.17 0.53 0.65\nNs 32.0"]
        for x, y, z in verts:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        for a, b, c in faces:
            lines.append(f"f {a+1} {b+1} {c+1}")
        path = os.path.join(outdir, f"density{tag}.obj")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        saved["density_obj"] = path
    except Exception:
        pass


def _save_compliance_plot(compliance_hist, params_log, output_dir, tag=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters = list(range(1, len(compliance_hist) + 1))
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(iters, compliance_hist, color="#0d5c63", linewidth=1.5)
        axes[0].set_ylabel("Compliance"); axes[0].grid(True, alpha=0.3)
        axes[0].set_title("GPU-FEM SIMP Convergence")

        if params_log:
            used_sur = [int(p.get("used_surrogate", False)) for p in params_log]
            colors = ["#b15e14" if u else "#0d5c63" for u in used_sur]
            axes[1].bar(iters[:len(used_sur)], [1] * len(used_sur), color=colors,
                       width=1.0, edgecolor="none", alpha=0.7)
            axes[1].set_ylabel("FEM (blue) / Sur. (orange)")
            axes[1].set_xlabel("Iteration"); axes[1].set_yticks([])

        plt.tight_layout()
        path = os.path.join(output_dir, f"convergence{tag}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main GPU-FEM orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_gpu_fem(
    prompt: Optional[str] = None,
    spec: Optional[ProblemSpec] = None,
    controller: str = "schedule",
    router: str = "rule",
    max_iter: int = 100,
    max_retries: int = 2,
    output_dir: str = "gpu_fem_output",
    use_surrogate: bool = True,
    use_llm_eval: bool = True,
    physics_loss_weight: float = 0.1,
    surrogate_kwargs: Optional[dict] = None,
    device: str = "auto",
    backend: Optional[str] = None,
    use_large_net: bool = False,
    gpu_batch_size: int = 4096,
    progress_callback=None,
    verbose: bool = False,
) -> dict:
    """
    Full GPU-FEM multi-agent 3D topology optimization pipeline.

    Parameters
    ----------
    device : str
        "auto" (detect GPU), "cuda", "cupy", "torch_cuda", or "cpu".
    use_large_net : bool
        If True, use larger MLP ensemble for 100k+ element problems.
    gpu_batch_size : int
        Surrogate prediction batch size (reduce if VRAM is limited).

    Returns
    -------
    dict — same format as TO3D report, plus 'gpu_backend' field.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # Detect and log GPU backend
    requested_backend = backend or device
    resolved_backend, backend_note = resolve_backend_choice(requested_backend)
    if verbose:
        print(
            f"[GPU-FEM] Requested device/backend={requested_backend} "
            f"Resolved backend={resolved_backend}"
        )
        if backend_note:
            print(f"[GPU-FEM] {backend_note}")

    # ── Step 1: Configure ─────────────────────────────────────────────────
    llm_used = False
    config_warnings: list[str] = []

    if spec is None:
        if prompt is None:
            raise ValueError("Provide either `prompt` or `spec`.")
        if verbose:
            print("[GPU-FEM] Configurator: parsing prompt...")
        configurator = TO3DConfiguratorAgent()
        spec, llm_used, config_warnings = configurator.configure(prompt, verbose=verbose)
        if verbose:
            n_elem = spec.nelx * spec.nely * (spec.nelz if spec.nelz > 0 else 1)
            print(f"[GPU-FEM] Spec: {spec.nelx}×{spec.nely}×{spec.nelz} "
                  f"({n_elem:,} elem)  vf={spec.volfrac}  llm={llm_used}")

    # ── Step 2: Generate BCs ──────────────────────────────────────────────
    if verbose:
        print("[GPU-FEM] Generating boundary conditions...")
    bc = generate_bc(spec)

    # ── Build TO3DParams ──────────────────────────────────────────────────
    to_params = TO3DParams(
        nelx=spec.nelx,
        nely=spec.nely,
        nelz=spec.nelz,
        volfrac=spec.volfrac,
        rmin=spec.rmin if spec.rmin else 1.5,
        max_iter=spec.max_iter if spec.max_iter else max_iter,
    )
    effective_max_iter = to_params.max_iter
    n_elem = spec.nelx * spec.nely * (spec.nelz if spec.nelz > 0 else 1)
    surrogate_feature_mode = None
    if surrogate_kwargs:
        surrogate_feature_mode = surrogate_kwargs.get("feature_mode")
    if surrogate_feature_mode is None:
        surrogate_feature_mode = "temporal_sr" if (use_surrogate and (use_large_net or n_elem >= 50_000)) else "basic"

    # ── Step 3+4+5: Solve → Evaluate → Retry ─────────────────────────────
    param_controller = _build_param_controller(controller, effective_max_iter, verbose)
    router_agent = _build_router(router, effective_max_iter, verbose, surrogate_feature_mode)

    best_report: Optional[dict] = None
    best_compliance = float("inf")

    for attempt in range(1, max_retries + 2):
        if verbose:
            print(f"\n[GPU-FEM] === Attempt {attempt}/{max_retries + 1} "
                  f"(max_iter={effective_max_iter}) ===")

        _sur_kw = {
            "physics_loss_weight": physics_loss_weight,
            "device": "cuda" if resolved_backend in ("cupy", "torch_cuda", "cuda") else "cpu",
            "use_large_net": use_large_net,
            "gpu_batch_size": gpu_batch_size,
            "mesh_shape": (spec.nelx, spec.nely, spec.nelz),
            "feature_mode": surrogate_feature_mode,
        }
        if surrogate_kwargs:
            _sur_kw.update(surrogate_kwargs)

        surrogate = SensitivitySurrogateGPU(**_sur_kw) if use_surrogate else None

        if bc.passive_mask is not None:
            install_passive_patch(bc.passive_mask)

        try:
            result = run_simp_surrogate_gpu(
                params=to_params,
                fixed=bc.fixed_dofs,
                free=bc.free_dofs,
                F=bc.F,
                ndof=bc.ndof,
                surrogate=surrogate,
                router=router_agent,
                device=resolved_backend,
                param_controller=param_controller,
                passive_mask=bc.passive_mask,
                progress_callback=progress_callback,
                verbose=verbose,
            )
        finally:
            if bc.passive_mask is not None:
                uninstall_passive_patch()

        if verbose:
            print(
                f"[GPU-FEM] Done: {result['n_iter']} iters  "
                f"C={result['final_compliance']:.4f}  "
                f"gray={result['final_grayness']:.3f}  "
                f"FEM={result['fem_calls']} / Sur={result['surrogate_calls']}  "
                f"({result['wall_time_sec']:.1f}s)  backend={result.get('gpu_backend')}"
            )

        evaluator = TO3DEvaluatorAgent()
        eval_result = evaluator.evaluate(
            result, spec, max_iter=effective_max_iter,
            use_llm=use_llm_eval, verbose=verbose,
        )

        elapsed = time.time() - t0
        report = _build_report(
            spec, result, eval_result, llm_used, config_warnings,
            attempt, elapsed, router, controller, use_surrogate,
            requested_backend, result.get("gpu_backend", resolved_backend), surrogate_feature_mode,
        )

        fc = result.get("final_compliance", float("inf"))
        if fc < best_compliance:
            best_compliance = fc
            best_report = report

            rho = result.get("rho_final")
            if rho is not None and result.get("is_3d", False):
                artefacts = _save_3d_artefacts(
                    rho, spec.nelx, spec.nely, spec.nelz,
                    output_dir, tag=f"_run{attempt}",
                )
                report.update(artefacts)

            conv_path = _save_compliance_plot(
                result["compliance_history"], result["params_log"],
                output_dir, tag=f"_run{attempt}",
            )
            if conv_path:
                report["convergence_plot"] = conv_path

        if eval_result.passed:
            if verbose:
                print("[GPU-FEM] Quality checks passed — done.")
            break

        if attempt <= max_retries:
            hint = getattr(eval_result, "rerun_hint", {})
            if hint and "max_iter" in hint:
                effective_max_iter = hint["max_iter"]
                to_params.max_iter = effective_max_iter
            else:
                effective_max_iter = int(effective_max_iter * 1.3)
                to_params.max_iter = effective_max_iter

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as fh:
        json.dump(best_report, fh, indent=2, default=str)
    if verbose:
        print(f"\n[GPU-FEM] Report saved: {report_path}")

    return best_report


def _build_report(spec, result, eval_result, llm_used, config_warnings,
                  run_idx, elapsed, router, controller, use_surrogate, requested_backend, gpu_backend,
                  surrogate_feature_mode):
    return {
        "pipeline": "gpu_fem_multiagent",
        "version": "1.0.0",
        "requested_backend": requested_backend,
        "gpu_backend": gpu_backend,
        "run_index": run_idx,
        "elapsed_seconds": round(elapsed, 2),
        "problem_spec": spec.to_dict(),
        "configurator": {"llm_used": llm_used, "warnings": config_warnings},
        "agents": {"router": router, "controller": controller, "use_surrogate": use_surrogate},
        "solver_summary": {
            "nelx": result.get("nelx"), "nely": result.get("nely"), "nelz": result.get("nelz"),
            "is_3d": result.get("is_3d"), "n_iter": result.get("n_iter"),
            "final_compliance": result.get("final_compliance"),
            "best_compliance": result.get("best_compliance"),
            "best_iteration": result.get("best_iteration"),
            "final_grayness": result.get("final_grayness"),
            "best_grayness": result.get("best_grayness"),
            "best_is_valid": result.get("best_is_valid"),
            "compliance_history": result.get("compliance_history", []),
        },
        "surrogate_stats": {
            "fem_calls": result.get("fem_calls"),
            "surrogate_calls": result.get("surrogate_calls"),
            "surrogate_used_fraction": result.get("surrogate_used_fraction"),
            "feature_mode": surrogate_feature_mode,
        },
        "evaluation": {
            "passed": eval_result.passed,
            "summary": eval_result.summary,
            "checks": [
                {"name": c.name, "passed": c.passed, "value": c.value,
                 "threshold": c.threshold, "message": getattr(c, "message", "")}
                for c in getattr(eval_result, "checks", [])
            ],
            "rerun_hint": getattr(eval_result, "rerun_hint", {}),
            "llm_assessment": getattr(eval_result, "llm_assessment", ""),
        },
        "wall_time_sec": result.get("wall_time_sec"),
    }
