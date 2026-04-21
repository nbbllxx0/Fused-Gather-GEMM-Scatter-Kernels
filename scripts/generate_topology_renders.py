"""
generate_topology_renders.py
----------------------------
Two-phase script:

  Phase 1 (--run-simp):  Run SIMP-120 (fused FP32 path) on two problems,
                          save rho_final and rho_best as .npy files.

  Phase 2 (--render):    Load saved .npy density fields and produce
                          publication-quality isosurface PDF figures.

Problems:
  cantilever_216k : 120x60x30  (216 k elements, standard benchmark)
  torsion_500k    : 165x55x55  (499 k elements, hard-problem benchmark)

Usage (run from repo root):
    python scripts/generate_topology_renders.py --run-simp
    python scripts/generate_topology_renders.py --render
    python scripts/generate_topology_renders.py --run-simp --render
    python scripts/generate_topology_renders.py --run-simp --render --prob cantilever_216k
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import subprocess
from pathlib import Path


# Bootstrap to a separate Python env when GPU_FEM_PYTHON is set.
def _bootstrap():
    root        = Path(__file__).resolve().parents[1]
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
    result = subprocess.run(
        [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env=env,
        check=False,
    )
    raise SystemExit(result.returncode)


_bootstrap()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "phase3"))
FIGS_DIR = ROOT / "figs"
DATA_DIR = ROOT / "experiments" / "phase3" / "topology_renders"
FROZEN_FIGS_DIR = FIGS_DIR / "frozen"
FROZEN_DATA_DIR = DATA_DIR / "frozen"
OBSOLETE_FIGS_DIR = FIGS_DIR / "obsolete"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_FIGS_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
OBSOLETE_FIGS_DIR.mkdir(parents=True, exist_ok=True)


# Problem registry
PROBLEMS = {
    "cantilever_216k": {
        "nelx": 120, "nely": 60, "nelz": 30,
        "bvp": "cantilever",
        "label": "Cantilever 216k",
        "n_elem_str": "216,000",
        "rmin": 1.5,
        "volfrac": 0.30,
    },
    "torsion_500k": {
        "nelx": 165, "nely": 55, "nelz": 55,
        "bvp": "torsion",
        "label": "Torsion 499k",
        "n_elem_str": "499,125",
        "rmin": 1.5,
        "volfrac": 0.25,
    },
    "cantilever_1m": {
        "nelx": 200, "nely": 100, "nelz": 50,
        "bvp": "cantilever",
        "label": "Cantilever 1M",
        "n_elem_str": "1,000,000",
        "rmin": 1.5,
        "volfrac": 0.30,
    },
    "bridge_216k": {
        "nelx": 120, "nely": 60, "nelz": 30,
        "bvp": "bridge",
        "label": "Bridge 216k",
        "n_elem_str": "216,000",
        "rmin": 3.0,   # larger filter - smoother topology
    },
    "doubleclamp_216k": {
        "nelx": 120, "nely": 60, "nelz": 30,
        "bvp": "doubleclamp",
        "label": "Double-clamped beam 216k",
        "n_elem_str": "216,000",
        "rmin": 2.0,
        "volfrac": 0.20,
        "figure_id": "F14",
    },
    "cantilever_1m_vf01": {
        "nelx": 200, "nely": 100, "nelz": 50,
        "bvp": "cantilever",
        "label": "Cantilever 1M (low-volume phase-3 rerun)",
        "n_elem_str": "1,000,000",
        "rmin": 2.5,
        "volfrac": 0.10,
        "max_iter": 100,
        "figure_id": "F14",
        "render_style": "reference_single",
    },
    "cantilever_corner_1m_vf01": {
        "nelx": 160, "nely": 80, "nelz": 80,
        "bvp": "cantilever_patch",
        "label": "Cantilever 1M (thick patch-load rerun)",
        "n_elem_str": "1,024,000",
        "rmin": 3.0,
        "volfrac": 0.10,
        "max_iter": 100,
        "figure_id": "F13",
        "render_style": "reference_single",
        "freeze_stem": "F15_centerpatch_y050_z000_rmin3p0",
        "freeze_data_stem": "cantilever_corner_1m_vf01_centerpatch_y050_z000_rmin3p0",
        "load_patch_rel": (1.0, 0.50, 0.00),
        "load_marker_rel": (1.0, 0.50, 0.10),
        "load_marker_style": "face_disc",
        "load_marker_radius_rel": 0.16,
        "traction_patch_radius_rel": 0.24,
        "clamp_marker_x_bounds": (-1.2, -0.10),
        "clamp_marker_opacity": 0.78,
        "obsolete_view_names": ("load_close",),
        "camera_override": {
            "position": (178.0, -142.0, 118.0),
            "focal_point": (98.0, 31.0, 35.0),
            "up": (0.0, 0.0, 1.0),
            "zoom": 0.80,
        },
        "supplemental_views": {
            "xy_side": {
                "position": (80.0, 40.0, 260.0),
                "focal_point": (80.0, 40.0, 40.0),
                "up": (0.0, 1.0, 0.0),
                "zoom": 1.65,
            },
            "iso_back": {
                "position": (58.0, 250.0, 136.0),
                "focal_point": (82.0, 46.0, 34.0),
                "up": (0.0, 0.0, 1.0),
                "zoom": 0.98,
            },
            "yz_load_face": {
                "position": (300.0, 40.0, 40.0),
                "focal_point": (160.0, 40.0, 8.0),
                "up": (0.0, 1.0, 0.0),
                "zoom": 1.90,
            },
        },
    },
}


def _freeze_problem_data(prob_key: str, cfg: dict) -> None:
    freeze_data_stem = cfg.get("freeze_data_stem")
    if not freeze_data_stem:
        return

    for suffix in ("rho_best.npy", "rho_final.npy", "meta.json"):
        src = DATA_DIR / f"{prob_key}_{suffix}"
        if not src.exists():
            continue
        dst = FROZEN_DATA_DIR / f"{freeze_data_stem}_{suffix}"
        shutil.copy2(src, dst)


def _freeze_render_outputs(cfg: dict, saved_pngs: dict[str, Path], out_pdf: Path) -> None:
    freeze_stem = cfg.get("freeze_stem")
    if not freeze_stem:
        return

    for view_name, png_path in saved_pngs.items():
        if not png_path.exists():
            continue
        dst = FROZEN_FIGS_DIR / f"{freeze_stem}_{view_name}.png"
        shutil.copy2(png_path, dst)

    if out_pdf.exists():
        shutil.copy2(out_pdf, FROZEN_FIGS_DIR / f"{freeze_stem}.pdf")


def _quarantine_obsolete_render_outputs(prob_key: str, cfg: dict) -> None:
    legacy_views = cfg.get("obsolete_view_names", ())
    if not legacy_views:
        return

    fig_id = cfg.get("figure_id", "F12")
    for view_name in legacy_views:
        src = FIGS_DIR / f"{fig_id}_{prob_key}_{view_name}.png"
        if not src.exists():
            continue
        dst = OBSOLETE_FIGS_DIR / src.name
        shutil.move(str(src), str(dst))
# -----------------------------------------------------------------------------
# Phase 1: Run SIMP and save density fields
# -----------------------------------------------------------------------------



def run_simp_save(prob_key: str):
    """Run SIMP-120 via simp_gpu.run_simp_surrogate_gpu and save density .npy."""
    from gpu_fem.simp_gpu                import run_simp_surrogate_gpu, TO3DParams
    from gpu_fem.local_agents            import PureFEMRouter
    from gpu_fem.pub_baseline_controller import ScheduleOnlyController
    from gpu_fem.solver_v2               import SolverV2
    from scaling_ladder                  import build_cantilever_problem

    cfg  = PROBLEMS[prob_key]
    nelx, nely, nelz = cfg["nelx"], cfg["nely"], cfg["nelz"]
    n_elem = nelx * nely * nelz

    print(f"\n{'='*60}")
    print(f"  SIMP-120: {cfg['label']}  ({nelx}x{nely}x{nelz} = {n_elem:,} elements)")
    print(f"  Path: fused FP32  Vf={cfg.get('volfrac', 0.30):.2f}  rmin={cfg.get('rmin', 1.5):.2f}")
    print(f"{'='*60}")

    if cfg["bvp"] == "cantilever":
        prob = build_cantilever_problem(nelx, nely, nelz)
    elif cfg["bvp"] == "torsion":
        prob = _build_torsion_problem(nelx, nely, nelz)
    elif cfg["bvp"] == "bridge":
        prob = _build_bridge_problem(nelx, nely, nelz)
    elif cfg["bvp"] == "cantilever_corner":
        prob = _build_corner_cantilever_problem(nelx, nely, nelz)
    elif cfg["bvp"] == "cantilever_patch":
        prob = _build_patchload_cantilever_problem(nelx, nely, nelz, cfg)
    elif cfg["bvp"] == "doubleclamp":
        prob = _build_doubleclamp_problem(nelx, nely, nelz)
    else:
        raise ValueError(f"Unknown BVP: {cfg['bvp']}")

    params = TO3DParams(
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=cfg.get("volfrac", 0.30),
        rmin=cfg.get("rmin", 1.5),
        max_iter=cfg.get("max_iter", 120),
    )

    solver_kwargs = dict(
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=True,
        enable_matrix_free=True,
        enable_mixed_precision=True,
        enable_fused_cuda=True,
        enable_rediscr_gmg=False,
        enable_profiling=False,
    )

    result = run_simp_surrogate_gpu(
        params=params,
        fixed=prob["fixed"], free=prob["free"],
        F=prob["F"], ndof=prob["ndof"],
        surrogate=None,
        router=PureFEMRouter(),
        device="auto",
        param_controller=ScheduleOnlyController(),
        verbose=True,
        solver_class=SolverV2,
        solver_kwargs=solver_kwargs,
    )

    rho_final = result["rho_final"]
    rho_best  = result["rho_best"]
    meta = {
        "prob_key": prob_key,
        "label": cfg["label"],
        "bvp": cfg["bvp"],
        "nelx": nelx,
        "nely": nely,
        "nelz": nelz,
        "n_elem": n_elem,
        "volfrac": cfg.get("volfrac", 0.30),
        "rmin": cfg.get("rmin", 1.5),
        "max_iter": int(cfg.get("max_iter", 120)),
        "source_density": "rho_best",
        "final_compliance": float(result["final_compliance"]),
        "final_grayness": float(result["final_grayness"]),
        "best_compliance": float(result["best_compliance"]),
        "best_grayness": float(result["best_grayness"]),
        "best_iteration": int(result["best_iteration"]),
        "best_is_valid": bool(result["best_is_valid"]),
    }

    np.save(str(DATA_DIR / f"{prob_key}_rho_final.npy"), rho_final)
    np.save(str(DATA_DIR / f"{prob_key}_rho_best.npy"),  rho_best)
    (DATA_DIR / f"{prob_key}_meta.json").write_text(json.dumps(meta, indent=2))
    _freeze_problem_data(prob_key, cfg)

    print(f"\n  Saved: {DATA_DIR.relative_to(ROOT)}/{prob_key}_rho_final.npy")
    print(f"  final compliance = {result['final_compliance']:.6f}   "
          f"grayness = {result['final_grayness']:.4f}")
    print(f"  best  compliance = {result['best_compliance']:.6f}   "
          f"grayness = {result['best_grayness']:.4f}")

    return rho_best  # use best (lowest compliance valid density) for rendering


def _build_bridge_problem(nelx, nely, nelz):
    """Build 3D simply-supported beam (bridge analog).

    Left face: fully fixed (pin wall - provides horizontal + vertical reaction).
    Right face: roller_x (fixes y and z, free in x - vertical reaction only).
    Load:       downward point load at center-top-center of the right half.

    This is statically determinate and well-conditioned for Jacobi-PCG.
    The optimal topology is an arch/truss transferring the center load to both
    supports - distinct from the cantilever and torsion topologies.
    """
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc
    import numpy as np

    Lx, Ly, Lz = float(nelx), float(nely), float(nelz)
    cx, cz = Lx / 2, Lz / 2

    spec = ProblemSpec(
        Lx=Lx, Ly=Ly, Lz=Lz,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=0.25,
        supports=[
            EdgeSupport(edge="left",  constraint="fixed"),    # full pin wall
            EdgeSupport(edge="right", constraint="roller_x"), # roller: fixes y,z
        ],
        loads=[
            PointLoad(x=cx, y=Ly, z=cz, fy=-1.0),  # center-top downward load
        ],
        rmin=1.5,
    )

    bc = generate_bc(spec)
    return dict(
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F,
        ndof=bc.ndof,
    )


def _build_torsion_problem(nelx, nely, nelz):
    """Build torsion BC: fixed left face, torque load on right face."""
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc

    # Torsion: fix left, distributed torque on right
    # We approximate torque as opposing y-loads at top and bottom of right face.
    spec = ProblemSpec(
        Lx=3.0, Ly=1.0, Lz=1.0,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=0.30,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        # Two opposing loads to create torque about x-axis on right face
        loads=[
            PointLoad(x=3.0, y=1.0, z=0.5, fy=-0.5, fz=+0.5),
            PointLoad(x=3.0, y=0.0, z=0.5, fy=+0.5, fz=-0.5),
        ],
        rmin=1.5,
    )
    bc = generate_bc(spec)
    return dict(
        spec=spec,
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F,
        ndof=bc.ndof,
    )


def _build_doubleclamp_problem(nelx, nely, nelz):
    """Build a 3D double-clamped beam with a centered top load."""
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc

    Lx, Ly, Lz = 1.5, 1.0, 0.5
    spec = ProblemSpec(
        Lx=Lx, Ly=Ly, Lz=Lz,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=0.20,
        supports=[
            EdgeSupport(edge="left", constraint="fixed"),
            EdgeSupport(edge="right", constraint="fixed"),
        ],
        loads=[
            PointLoad(x=Lx * 0.5, y=Ly, z=Lz * 0.5, fy=-1.0),
        ],
        rmin=2.0,
    )
    bc = generate_bc(spec)
    return dict(
        spec=spec,
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F,
        ndof=bc.ndof,
    )


def _build_corner_cantilever_problem(nelx, nely, nelz):
    """Build a thicker cantilever with a lower-corner tip load."""
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc

    Lx, Ly, Lz = 2.0, 1.0, 1.0
    spec = ProblemSpec(
        Lx=Lx, Ly=Ly, Lz=Lz,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=0.10,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=Lx, y=0.0, z=0.0, fy=-1.0)],
        rmin=2.5,
    )
    bc = generate_bc(spec)
    return dict(
        spec=spec,
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F,
        ndof=bc.ndof,
    )


def _build_patchload_cantilever_problem(nelx, nely, nelz, cfg):
    """Build a thick cantilever with a small lower-corner traction patch on the right face."""
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc, _face_nodes_3d, _node_coords_3d

    Lx, Ly, Lz = 2.0, 1.0, 1.0
    load_rel = cfg.get("load_patch_rel", cfg.get("load_marker_rel", (1.0, 0.18, 0.18)))
    center_y = float(load_rel[1]) * Ly
    center_z = float(load_rel[2]) * Lz
    spec = ProblemSpec(
        Lx=Lx, Ly=Ly, Lz=Lz,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=cfg.get("volfrac", 0.10),
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        # Dummy point load satisfies ProblemSpec validation; F is overwritten below.
        loads=[PointLoad(x=Lx, y=center_y, z=center_z, fz=-1e-12)],
        rmin=cfg.get("rmin", 2.5),
    )
    bc = generate_bc(spec)

    coords = _node_coords_3d(nelx, nely, nelz, Lx, Ly, Lz)
    face_nodes = _face_nodes_3d("right", nelx, nely, nelz)
    radius = float(cfg.get("traction_patch_radius_rel", cfg.get("load_marker_radius_rel", 0.20))) * min(Ly, Lz)

    yz = coords[face_nodes][:, 1:3]
    d = np.linalg.norm(yz - np.array([center_y, center_z])[None, :], axis=1)
    patch_nodes = face_nodes[d <= radius]
    if patch_nodes.size == 0:
        patch_nodes = face_nodes[np.argsort(d)[:8]]

    F = np.zeros_like(bc.F)
    load_per_node = -1.0 / float(len(patch_nodes))
    for n in patch_nodes:
        F[3 * int(n) + 2] += load_per_node

    return dict(
        spec=spec,
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=F,
        ndof=bc.ndof,
    )
# -----------------------------------------------------------------------------
# Phase 2: Render 3D topology isosurfaces with pyvista
# -----------------------------------------------------------------------------



def render_topology(prob_key: str, rho: np.ndarray, iso_val: float = 0.5, meta: dict | None = None):
    """Render isosurface with pyvista, produce combined PDF."""
    import pyvista as pv
    from skimage.measure import marching_cubes

    cfg  = PROBLEMS[prob_key]
    nelx, nely, nelz = cfg["nelx"], cfg["nely"], cfg["nelz"]
    meta = meta or {}
    label = meta.get("label", cfg["label"])

    # Reshape to a 3D voxel grid.
    # The edof table is built with meshgrid(indexing='ij') + ravel() in C order:
    # element e = ix*(nely*nelz) + iy*nelz + iz  (ix slowest, iz fastest)
    rho_3d = rho.reshape((nelx, nely, nelz), order="C")

    solid_frac = float((rho_3d >= iso_val).mean())
    print(f"  Rendering {label}: shape={rho_3d.shape}  "
          f"iso={iso_val:.2f}  solid_frac={solid_frac:.3f}")

    # Pad to avoid open boundaries at the mesh edge
    rho_pad = np.pad(rho_3d, 1, mode="constant", constant_values=0.0)

    # Marching-cubes isosurface
    verts, faces, normals, _ = marching_cubes(rho_pad, level=iso_val, step_size=1)
    verts -= 1.0  # undo padding shift

    faces_pv = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces]).ravel()
    mesh = pv.PolyData(verts.astype(np.float32), faces_pv)
    # Laplacian smoothing to remove voxelated staircase artifacts
    mesh = mesh.smooth(n_iter=80, relaxation_factor=0.08,
                       boundary_smoothing=False, edge_angle=60)
    mesh.compute_normals(inplace=True)
    bracket_clip = None
    if "bracket" in prob_key:
        bracket_clip = mesh.clip(normal=(0, 1, 0), origin=(0.0, nely * 0.5, 0.0), invert=True)
        bracket_clip.compute_normals(inplace=True)

    # Render two views with problem-specific cameras.
    cx, cy, cz = nelx / 2, nely / 2, nelz / 2

    doubleclamp_clip = None
    if "doubleclamp" in prob_key:
        doubleclamp_clip = mesh.clip(
            normal=(0, 0, 1), origin=(0.0, 0.0, nelz * 0.5), invert=True
        )
        doubleclamp_clip.compute_normals(inplace=True)

    if cfg.get("render_style") == "reference_single":
        hero_cam = cfg.get("camera_override")
        if hero_cam is None:
            hero_cam = dict(
                position=(nelx * 0.88, -nely * 2.15, nelz * 3.65),
                focal_point=(cx * 0.82, cy * 0.62, cz * 0.72),
                up=(0, 0, 1),
                zoom=1.22,
            )
        camera_configs = {
            "hero": hero_cam,
        }
        camera_configs.update(cfg.get("supplemental_views", {}))
    elif "cantilever" in prob_key:
        # Cantilever: X=length (fixed left, tip right), Y=height, Z=thickness
        # (a) Side profile: look along +Z - reveals classic 2D truss structure in XY plane
        # (b) 3/4 isometric: elevated from front-right
        camera_configs = {
            "profile": dict(
                position=(cx, cy, nelz * 5.5),
                focal_point=(cx, cy, cz),
                up=(0, 1, 0),
                zoom=1.2,
            ),
            "iso": dict(
                position=(nelx * 1.6, -nely * 1.8, nelz * 3.0),
                focal_point=(cx, cy * 0.7, cz * 0.8),
                up=(0, 0, 1),
                zoom=1.1,
            ),
        }
    elif "bridge" in prob_key:
        # Bridge: X=span (left=fixed wall, right=roller), Y=height, Z=depth
        # (a) Front elevation: orthographic view along +Z - shows arch/truss in XY plane
        # (b) 3/4 isometric: elevated view showing 3D structural form
        camera_configs = {
            "front": dict(
                position=(cx, cy, nelz * 8.0),
                focal_point=(cx, cy * 0.55, cz),
                up=(0, 1, 0),
                zoom=0.95,
            ),
            "iso": dict(
                position=(nelx * 1.8, -nely * 1.5, nelz * 4.0),
                focal_point=(cx * 0.9, cy * 0.5, cz * 0.7),
                up=(0, 1, 0),
                zoom=0.90,
            ),
        }
    elif "bracket" in prob_key:
        camera_configs = {
            "clip": dict(
                position=(nelx * 2.2, -nely * 1.9, nelz * 3.0),
                focal_point=(cx, cy, cz),
                up=(0, 0, 1),
                zoom=2.05,
            ),
            "iso": dict(
                position=(-nelx * 1.7, -nely * 1.2, nelz * 2.6),
                focal_point=(cx, cy, cz),
                up=(0, 0, 1),
                zoom=1.75,
            ),
        }
    elif "doubleclamp" in prob_key:
        camera_configs = {
            "hero": dict(
                position=(nelx * 1.05, -nely * 1.35, nelz * 4.1),
                focal_point=(cx, cy * 0.48, cz * 0.70),
                up=(0, 0, 1),
                zoom=1.02,
            ),
        }
    else:
        # Torsion: X=length, Y and Z=cross-section
        # (a) End view: look along -X from right - reveals cross-section topology
        # (b) 3/4 isometric: shows the full shaft form with holes
        camera_configs = {
            "end": dict(
                position=(nelx * 2.2, cy, cz),
                focal_point=(cx * 0.85, cy, cz),
                up=(0, 1, 0),
                zoom=1.2,
            ),
            "iso": dict(
                position=(nelx * 1.3, -nely * 2.0, nelz * 2.5),
                focal_point=(cx, cy, cz),
                up=(0, 0, 1),
                zoom=0.92,
            ),
        }

    saved_pngs = {}
    fig_id = cfg.get("figure_id", "F12")
    for view_name, cam in camera_configs.items():
        pl = pv.Plotter(off_screen=True, window_size=(2400, 1350))
        pl.set_background("white")

        if "bracket" in prob_key and bracket_clip is not None and bracket_clip.n_points > 0:
            pl.add_mesh(
                mesh,
                color="#8ea8c0",
                smooth_shading=True,
                opacity=0.12,
                specular=0.2,
                specular_power=18,
                ambient=0.20,
                diffuse=0.70,
            )
            pl.add_mesh(
                bracket_clip,
                color="#34495e",
                smooth_shading=True,
                show_edges=True,
                edge_color="#20364f",
                line_width=0.35,
                specular=0.45,
                specular_power=24,
                ambient=0.24,
                diffuse=0.82,
            )
        elif cfg.get("render_style") == "reference_single":
            pl.add_mesh(
                mesh,
                color="#b89674",
                smooth_shading=True,
                show_edges=True,
                edge_color="#8f755c",
                line_width=0.10,
                specular=0.16,
                specular_power=11,
                ambient=0.29,
                diffuse=0.78,
            )
        elif "doubleclamp" in prob_key:
            pl.add_mesh(
                mesh,
                color="#b58c6c",
                smooth_shading=True,
                show_edges=True,
                edge_color="#8f6b51",
                line_width=0.12,
                specular=0.18,
                specular_power=12,
                ambient=0.28,
                diffuse=0.78,
            )
        else:
            pl.add_mesh(
                mesh,
                color="#34495e",           # slightly lighter steel blue
                smooth_shading=True,
                specular=0.6,
                specular_power=30,
                ambient=0.25,
                diffuse=0.85,
            )

        if "bracket" not in prob_key and "doubleclamp" not in prob_key and cfg.get("render_style") != "reference_single":
            box = pv.Box(bounds=(0, nelx, 0, nely, 0, nelz))
            pl.add_mesh(box, style="wireframe", color="#aaaaaa",
                        line_width=1.2, opacity=0.4)
        elif cfg.get("render_style") == "reference_single":
            box = pv.Box(bounds=(0, nelx, 0, nely, 0, nelz))
            pl.add_mesh(box, style="wireframe", color="#6b6b6b",
                        line_width=0.8, opacity=0.22)
            clamp_x0, clamp_x1 = cfg.get("clamp_marker_x_bounds", (-0.9, -0.08))
            clamp_patch = pv.Box(bounds=(clamp_x0, clamp_x1, 0, nely, 0, nelz))
            load_rel = cfg.get("load_marker_rel", (1.0, 0.5, 0.5))
            load_cx = float(load_rel[0]) * nelx
            load_cy = float(load_rel[1]) * nely
            load_cz = float(load_rel[2]) * nelz
            if cfg.get("load_marker_style") == "face_disc":
                load_patch = pv.Disc(
                    center=(load_cx + 0.15, load_cy, load_cz),
                    inner=0.0,
                    outer=max(1.6, min(nely, nelz) * float(cfg.get("load_marker_radius_rel", 0.20))),
                    normal=(1, 0, 0),
                    r_res=1,
                    c_res=64,
                )
            else:
                load_patch = pv.Cylinder(
                    center=(load_cx + 0.8, load_cy, load_cz),
                    direction=(1, 0, 0),
                    radius=max(1.4, nelz * 0.12),
                    height=1.6,
                    resolution=48,
                )
            pl.add_mesh(
                clamp_patch,
                color="#8a1f1f",
                opacity=float(cfg.get("clamp_marker_opacity", 0.84)),
                smooth_shading=True,
            )
            pl.add_mesh(load_patch, color="#6d9f2f", opacity=0.96, smooth_shading=True)
        elif "doubleclamp" in prob_key:
            box = pv.Box(bounds=(0, nelx, 0, nely, 0, nelz))
            pl.add_mesh(box, style="wireframe", color="#666666",
                        line_width=0.8, opacity=0.25)
            left_patch = pv.Box(bounds=(0, 0.6, 0, nely, 0, nelz))
            right_patch = pv.Box(bounds=(nelx - 0.6, nelx, 0, nely, 0, nelz))
            load_patch = pv.Cylinder(
                center=(cx, nely + 0.9, cz),
                direction=(0, 1, 0),
                radius=max(1.2, nelz * 0.10),
                height=1.2,
                resolution=48,
            )
            pl.add_mesh(left_patch, color="#8a1f1f", opacity=0.92, smooth_shading=True)
            pl.add_mesh(right_patch, color="#8a1f1f", opacity=0.92, smooth_shading=True)
            pl.add_mesh(load_patch, color="#6d9f2f", opacity=0.96, smooth_shading=True)

        pl.camera.position    = cam["position"]
        pl.camera.focal_point = cam["focal_point"]
        pl.camera.up          = cam["up"]
        pl.camera.zoom(cam.get("zoom", 1.05))
        pl.enable_anti_aliasing("ssaa")
        # Key light: upper-front
        pl.add_light(pv.Light(position=(nelx * 1.5, -nely * 2, nelz * 4),
                              focal_point=(cx, cy, cz), intensity=0.75))
        # Fill light: side
        pl.add_light(pv.Light(position=(-nelx * 0.5, nely * 2, nelz * 1),
                              focal_point=(cx, cy, cz), intensity=0.35))

        out_png = FIGS_DIR / f"{fig_id}_{prob_key}_{view_name}.png"
        pl.screenshot(str(out_png), transparent_background=False,
                      window_size=(4800, 2700))
        pl.close()
        saved_pngs[view_name] = out_png
        print(f"  Saved PNG: {out_png.relative_to(ROOT)}")

    # Stitch the rendered views into one combined PDF.
    _make_panel_pdf(prob_key, label, cfg, saved_pngs, meta)
    _quarantine_obsolete_render_outputs(prob_key, cfg)


def _make_panel_pdf(prob_key, label, cfg, saved_pngs, meta=None):
    """Two-panel matplotlib figure: isometric + front view, saved as PDF."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.gridspec as gridspec

    meta = meta or {}
    nelx, nely, nelz = cfg["nelx"], cfg["nely"], cfg["nelz"]
    n_elem = nelx * nely * nelz
    fig_id = cfg.get("figure_id", "F12")
    out_pdf = FIGS_DIR / f"{fig_id}_{prob_key}.pdf"

    if cfg.get("render_style") == "reference_single":
        fig = plt.figure(figsize=(9.4, 6.2))
        gs  = gridspec.GridSpec(1, 1, figure=fig, wspace=0.0)
    elif "doubleclamp" in prob_key:
        fig = plt.figure(figsize=(8.6, 5.8))
        gs  = gridspec.GridSpec(1, 1, figure=fig, wspace=0.0)
    else:
        fig = plt.figure(figsize=(14, 5.5))
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.03)

    if cfg.get("render_style") == "reference_single":
        view_labels = {"hero": ""}
    elif "cantilever" in prob_key:
        view_labels = {"profile": "(a) Side profile (XY plane)", "iso": "(b) Isometric view"}
    elif "bridge" in prob_key:
        view_labels = {"front": "(a) Front elevation (XY plane)", "iso": "(b) Isometric view"}
    elif "bracket" in prob_key:
        view_labels = {"clip": "(a) Clipped isometric view", "iso": "(b) Rotated isometric view"}
    elif "doubleclamp" in prob_key:
        view_labels = {"hero": ""}
    else:
        view_labels = {"end": "(a) End view (YZ projection)", "iso": "(b) Isometric view"}
    for col_i, (view_name, view_lbl) in enumerate(view_labels.items()):
        ax  = fig.add_subplot(gs[0, col_i])
        png = saved_pngs.get(view_name)
        if png and png.exists():
            img = mpimg.imread(str(png))
            ax.imshow(img)
        if view_lbl:
            ax.set_title(view_lbl, fontsize=11, pad=5)
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.08)
    fig.savefig(str(out_pdf), dpi=200, bbox_inches="tight")
    plt.close(fig)
    _freeze_render_outputs(cfg, saved_pngs, out_pdf)
    print(f"  Saved PDF: {out_pdf.relative_to(ROOT)}")
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(
        description="Run SIMP and/or render 3D topology isosurfaces.")
    parser.add_argument("--run-simp", action="store_true",
                        help="Run SIMP-120 (fused FP32) and save .npy density fields")
    parser.add_argument("--render", action="store_true",
                        help="Render saved density fields as isosurface PDF figures")
    parser.add_argument("--prob", default="all",
                        choices=list(PROBLEMS.keys()) + ["all"],
                        help="Which problem to process (default: all)")
    parser.add_argument("--iso", type=float, default=0.5,
                        help="Isosurface threshold (default 0.5)")
    parser.add_argument("--canonical-only", action="store_true",
                        help="Require canonical rho_best + metadata sidecars when rendering")
    args = parser.parse_args()

    if not args.run_simp and not args.render:
        parser.print_help()
        return

    keys = list(PROBLEMS.keys()) if args.prob == "all" else [args.prob]

    for prob_key in keys:
        print(f"\n{'='*60}")
        print(f"  Problem: {prob_key}")
        print(f"{'='*60}")

        rho = None

        if args.run_simp:
            try:
                rho = run_simp_save(prob_key)
            except ValueError as exc:
                print(f"  SKIP --run-simp: {exc}")

        if args.render:
            if rho is None:
                suffixes = ("rho_best",) if args.canonical_only else ("rho_best", "rho_final")
                for suffix in suffixes:
                    npy = DATA_DIR / f"{prob_key}_{suffix}.npy"
                    if npy.exists():
                        rho = np.load(str(npy))
                        print(f"  Loaded: {npy.relative_to(ROOT)}")
                        break
                if rho is None:
                    print(f"  ERROR: no .npy found for {prob_key} - run --run-simp first")
                    continue

            meta_path = DATA_DIR / f"{prob_key}_meta.json"
            meta = None
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            elif args.canonical_only:
                print(f"  ERROR: missing canonical metadata sidecar for {prob_key}")
                continue

            render_topology(prob_key, rho, iso_val=args.iso, meta=meta)

    print("\n  All done.")


if __name__ == "__main__":
    main()
