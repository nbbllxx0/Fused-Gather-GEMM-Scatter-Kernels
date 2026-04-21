"""
bc_generator.py
---------------
Converts a ProblemSpec into the (fixed_dofs, free_dofs, F_vector) tuple and a
passive-element mask that pub_simp_solver.py accepts via bc_override.

Design principles:
  - Pure numpy, no solver modifications needed (except ~10 lines for passive
    region support in OC update — see patch_solver_for_passive()).
  - Coordinates are continuous; loads/supports snap to the nearest mesh node.
  - Returns a bc_override *function* with the signature the solver expects:
        bc_fn(nelx, nely) -> (fixed, free, F)           # 2D
        bc_fn(nelx, nely, nelz) -> (fixed, free, F)     # 3D
  - Passive mask is a separate array (n_elem,) with values:
        0 = free (optimizer controls)
        1 = void (forced to Emin)
        2 = solid (forced to 1.0)

Mesh convention (must match pub_simp_solver.py):
  - 2D: nodes are laid out column-major.
    Node (i, j) has global index  n = i * (nely+1) + j
    where i = 0..nelx, j = 0..nely.
    Element (i, j) has index  e = i * nely + j
    Element (i, j) has corner nodes:
      bottom-left  = (i, j)       →  n = i*(nely+1) + j
      bottom-right = (i+1, j)     →  n = (i+1)*(nely+1) + j
      top-right    = (i+1, j+1)   →  n = (i+1)*(nely+1) + j + 1
      top-left     = (i, j+1)     →  n = i*(nely+1) + j + 1
    DOFs: node n → DOFs [2n, 2n+1] = (ux, uy)

  - 3D: Node (i, j, k):  n = i * (nely+1)*(nelz+1) + j*(nelz+1) + k
    DOFs: node n → DOFs [3n, 3n+1, 3n+2] = (ux, uy, uz)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .problem_spec import (
    ProblemSpec,
    PointSupport, EdgeSupport,
    PointLoad, DistributedLoad,
    CircularRegion, RectangularRegion,
)


# ─────────────────────────────────────────────────────────────────────────────
# Node coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_coords_2d(nelx: int, nely: int,
                     Lx: float, Ly: float) -> np.ndarray:
    """Return (n_nodes, 2) array of node [x, y] positions."""
    ix, iy = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1),
                          indexing='ij')
    x = ix.ravel() * (Lx / nelx)
    y = iy.ravel() * (Ly / nely)
    return np.column_stack([x, y])


def _node_coords_3d(nelx: int, nely: int, nelz: int,
                     Lx: float, Ly: float, Lz: float) -> np.ndarray:
    """Return (n_nodes, 3) array of node [x, y, z] positions."""
    ix, iy, iz = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1),
                              np.arange(nelz + 1), indexing='ij')
    x = ix.ravel() * (Lx / nelx)
    y = iy.ravel() * (Ly / nely)
    z = iz.ravel() * (Lz / nelz)
    return np.column_stack([x, y, z])


def _elem_centroids_2d(nelx: int, nely: int,
                        Lx: float, Ly: float) -> np.ndarray:
    """Return (n_elem, 2) array of element centroid [x, y]."""
    ix, iy = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
    hx, hy = Lx / nelx, Ly / nely
    cx = ix.ravel() * hx + 0.5 * hx
    cy = iy.ravel() * hy + 0.5 * hy
    return np.column_stack([cx, cy])


def _elem_centroids_3d(nelx: int, nely: int, nelz: int,
                        Lx: float, Ly: float, Lz: float) -> np.ndarray:
    """Return (n_elem, 3) array of element centroid [x, y, z]."""
    ix, iy, iz = np.meshgrid(np.arange(nelx), np.arange(nely),
                              np.arange(nelz), indexing='ij')
    hx, hy, hz = Lx / nelx, Ly / nely, Lz / nelz
    cx = ix.ravel() * hx + 0.5 * hx
    cy = iy.ravel() * hy + 0.5 * hy
    cz = iz.ravel() * hz + 0.5 * hz
    return np.column_stack([cx, cy, cz])


# ─────────────────────────────────────────────────────────────────────────────
# Node-index lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_index_2d(i: int, j: int, nely: int) -> int:
    return i * (nely + 1) + j


def _node_index_3d(i: int, j: int, k: int, nely: int, nelz: int) -> int:
    return i * (nely + 1) * (nelz + 1) + j * (nelz + 1) + k


def _nearest_node(coords: np.ndarray, point: np.ndarray) -> int:
    """Index of the node closest to `point`."""
    d = np.linalg.norm(coords - point[None, :], axis=1)
    return int(np.argmin(d))


# ─────────────────────────────────────────────────────────────────────────────
# Edge/face node selection
# ─────────────────────────────────────────────────────────────────────────────

def _edge_nodes_2d(edge: str, nelx: int, nely: int) -> np.ndarray:
    """Return array of node indices on a named edge."""
    if edge == "left":
        return np.array([_node_index_2d(0, j, nely)
                         for j in range(nely + 1)])
    elif edge == "right":
        return np.array([_node_index_2d(nelx, j, nely)
                         for j in range(nely + 1)])
    elif edge == "bottom":
        return np.array([_node_index_2d(i, 0, nely)
                         for i in range(nelx + 1)])
    elif edge == "top":
        return np.array([_node_index_2d(i, nely, nely)
                         for i in range(nelx + 1)])
    else:
        raise ValueError(f"Unknown 2D edge: {edge}")


def _face_nodes_3d(face: str, nelx: int, nely: int,
                    nelz: int) -> np.ndarray:
    """Return array of node indices on a named face."""
    nodes = []
    if face == "left":
        for j in range(nely + 1):
            for k in range(nelz + 1):
                nodes.append(_node_index_3d(0, j, k, nely, nelz))
    elif face == "right":
        for j in range(nely + 1):
            for k in range(nelz + 1):
                nodes.append(_node_index_3d(nelx, j, k, nely, nelz))
    elif face == "bottom":
        for i in range(nelx + 1):
            for k in range(nelz + 1):
                nodes.append(_node_index_3d(i, 0, k, nely, nelz))
    elif face == "top":
        for i in range(nelx + 1):
            for k in range(nelz + 1):
                nodes.append(_node_index_3d(i, nely, k, nely, nelz))
    elif face == "front":
        for i in range(nelx + 1):
            for j in range(nely + 1):
                nodes.append(_node_index_3d(i, j, 0, nely, nelz))
    elif face == "back":
        for i in range(nelx + 1):
            for j in range(nely + 1):
                nodes.append(_node_index_3d(i, j, nelz, nely, nelz))
    else:
        raise ValueError(f"Unknown 3D face: {face}")
    return np.array(nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Core generation: supports → fixed DOFs
# ─────────────────────────────────────────────────────────────────────────────

def _build_fixed_dofs(spec: ProblemSpec, coords: np.ndarray) -> np.ndarray:
    """Collect all DOFs that should be constrained."""
    ndim = spec.ndim
    fixed_set: set[int] = set()

    for sup in spec.supports:
        if isinstance(sup, EdgeSupport):
            if spec.is_3d:
                nodes = _face_nodes_3d(sup.edge, spec.nelx, spec.nely,
                                        spec.nelz)
            else:
                nodes = _edge_nodes_2d(sup.edge, spec.nelx, spec.nely)
            dof_offsets = sup.dof_mask(ndim)
            for n in nodes:
                for off in dof_offsets:
                    fixed_set.add(ndim * n + off)

        elif isinstance(sup, PointSupport):
            pt = np.array([sup.x, sup.y] if ndim == 2
                          else [sup.x, sup.y, sup.z])
            n = _nearest_node(coords, pt)
            for off in sup.dof_mask(ndim):
                fixed_set.add(ndim * n + off)

    return np.sort(np.array(list(fixed_set), dtype=np.int64))


# ─────────────────────────────────────────────────────────────────────────────
# Core generation: loads → force vector
# ─────────────────────────────────────────────────────────────────────────────

def _build_force_vector(spec: ProblemSpec, coords: np.ndarray,
                         ndof: int) -> np.ndarray:
    """Assemble the global force vector F."""
    ndim = spec.ndim
    F = np.zeros(ndof)

    for ld in spec.loads:
        if isinstance(ld, PointLoad):
            pt = np.array([ld.x, ld.y] if ndim == 2
                          else [ld.x, ld.y, ld.z])
            n = _nearest_node(coords, pt)
            F[ndim * n]     += ld.fx
            F[ndim * n + 1] += ld.fy
            if ndim == 3:
                F[ndim * n + 2] += ld.fz

        elif isinstance(ld, DistributedLoad):
            # Consistent nodal forces: pressure × tributary length (2D) or
            # pressure × tributary area (3D).
            # For a uniform mesh, interior nodes get full element spacing,
            # corner/edge nodes get half (trapezoidal rule).
            if spec.is_3d:
                nodes = _face_nodes_3d(ld.edge, spec.nelx, spec.nely,
                                        spec.nelz)
            else:
                nodes = _edge_nodes_2d(ld.edge, spec.nelx, spec.nely)

            # Compute tributary weights (trapezoidal integration)
            node_coords_edge = coords[nodes]
            weights = _tributary_weights(node_coords_edge, spec, ld.edge)

            # Determine load direction and sign from edge normal
            if ld.edge in ("left", "right"):
                dof_offset = 0  # x direction
                sign = 1.0 if ld.edge == "right" else -1.0
            elif ld.edge in ("bottom", "top"):
                dof_offset = 1  # y direction
                sign = 1.0 if ld.edge == "top" else -1.0
            elif ld.edge in ("front", "back") and ndim == 3:
                dof_offset = 2  # z direction
                sign = 1.0 if ld.edge == "back" else -1.0
            else:
                continue

            for ni, n in enumerate(nodes):
                F[ndim * n + dof_offset] += ld.magnitude * sign * weights[ni]

    return F


def _tributary_weights(node_coords: np.ndarray, spec: ProblemSpec,
                        edge: str) -> np.ndarray:
    """
    Compute trapezoidal-rule tributary weights for nodes on an edge/face.

    For 2D edges: weight = tributary length along the edge.
      Interior node: h (element spacing), corner node: h/2.
      Total = edge_length (correct: integrates pressure × length).

    For 3D faces: weight = tributary area.
      Interior: h1×h2, edge: h1×h2/2, corner: h1×h2/4.
      Total = face_area.
    """
    n = len(node_coords)
    if n <= 1:
        return np.ones(n)

    if not spec.is_3d:
        # 2D: nodes along a line.  Use element spacing.
        if edge in ("left", "right"):
            h = spec.Ly / spec.nely
        else:
            h = spec.Lx / spec.nelx
        # Trapezoidal: half weight at ends, full in middle
        w = np.full(n, h)
        # Identify boundary nodes (min and max along the edge direction)
        if edge in ("left", "right"):
            axis_vals = node_coords[:, 1]
        else:
            axis_vals = node_coords[:, 0]
        amin, amax = axis_vals.min(), axis_vals.max()
        eps = h * 0.01
        w[np.abs(axis_vals - amin) < eps] = h / 2
        w[np.abs(axis_vals - amax) < eps] = h / 2
        return w

    else:
        # 3D: nodes on a rectangular face.  Two tributary directions.
        if edge in ("left", "right"):
            h1, h2 = spec.Ly / spec.nely, spec.Lz / spec.nelz
            c1, c2 = node_coords[:, 1], node_coords[:, 2]
        elif edge in ("bottom", "top"):
            h1, h2 = spec.Lx / spec.nelx, spec.Lz / spec.nelz
            c1, c2 = node_coords[:, 0], node_coords[:, 2]
        else:  # front, back
            h1, h2 = spec.Lx / spec.nelx, spec.Ly / spec.nely
            c1, c2 = node_coords[:, 0], node_coords[:, 1]

        # Each direction: half at boundary, full at interior
        eps1 = h1 * 0.01
        eps2 = h2 * 0.01
        w1 = np.full(n, h1)
        w1[np.abs(c1 - c1.min()) < eps1] = h1 / 2
        w1[np.abs(c1 - c1.max()) < eps1] = h1 / 2
        w2 = np.full(n, h2)
        w2[np.abs(c2 - c2.min()) < eps2] = h2 / 2
        w2[np.abs(c2 - c2.max()) < eps2] = h2 / 2
        return w1 * w2


# ─────────────────────────────────────────────────────────────────────────────
# Passive element mask
# ─────────────────────────────────────────────────────────────────────────────

def build_passive_mask(spec: ProblemSpec) -> Optional[np.ndarray]:
    """
    Return an integer array (n_elem,):
        0 = free (optimizable)
        1 = void  (forced to rho_min)
        2 = solid (forced to 1.0)
    Returns None if there are no passive regions.
    """
    if not spec.passive_regions:
        return None

    n_elem = spec.nelx * spec.nely * (spec.nelz if spec.is_3d else 1)
    mask = np.zeros(n_elem, dtype=np.int32)

    if spec.is_3d:
        centroids = _elem_centroids_3d(spec.nelx, spec.nely, spec.nelz,
                                        spec.Lx, spec.Ly, spec.Lz)
    else:
        centroids = _elem_centroids_2d(spec.nelx, spec.nely,
                                        spec.Lx, spec.Ly)

    for region in spec.passive_regions:
        if isinstance(region, CircularRegion):
            center = np.array([region.cx, region.cy])
            if spec.is_3d:
                # Cylinder along z: check 2D distance only
                d = np.linalg.norm(centroids[:, :2] - center[None, :], axis=1)
            else:
                d = np.linalg.norm(centroids - center[None, :], axis=1)
            inside = d <= region.radius
            val = 1 if region.kind == "void" else 2
            mask[inside] = val

        elif isinstance(region, RectangularRegion):
            cx, cy = centroids[:, 0], centroids[:, 1]
            inside = ((cx >= region.x0) & (cx <= region.x1) &
                      (cy >= region.y0) & (cy <= region.y1))
            val = 1 if region.kind == "void" else 2
            mask[inside] = val

    return mask if np.any(mask != 0) else None


# ─────────────────────────────────────────────────────────────────────────────
# Public API: generate bc_override function from ProblemSpec
# ─────────────────────────────────────────────────────────────────────────────

class BCResult:
    """Bundle returned by generate_bc() for easy downstream consumption."""
    __slots__ = ("bc_override", "passive_mask", "fixed_dofs", "free_dofs",
                 "F", "ndof", "node_coords")

    def __init__(self, bc_override, passive_mask, fixed_dofs, free_dofs,
                 F, ndof, node_coords):
        self.bc_override = bc_override
        self.passive_mask = passive_mask
        self.fixed_dofs = fixed_dofs
        self.free_dofs = free_dofs
        self.F = F
        self.ndof = ndof
        self.node_coords = node_coords


def generate_bc(spec: ProblemSpec) -> BCResult:
    """
    Master entry point.  Takes a validated ProblemSpec, returns everything
    the solver needs.

    Usage:
        bc = generate_bc(spec)
        result = run_simp(params, callback=controller, bc_override=bc.bc_override)
        # bc.passive_mask can be passed to a patched OC update
    """
    # Validate first
    errors = spec.validate()
    if errors:
        raise ValueError("ProblemSpec validation failed:\n"
                         + "\n".join(f"  - {e}" for e in errors))

    nelx, nely, nelz = spec.nelx, spec.nely, spec.nelz
    is_3d = spec.is_3d
    ndim = spec.ndim

    # Build node coordinates
    if is_3d:
        coords = _node_coords_3d(nelx, nely, nelz, spec.Lx, spec.Ly, spec.Lz)
        ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    else:
        coords = _node_coords_2d(nelx, nely, spec.Lx, spec.Ly)
        ndof = 2 * (nelx + 1) * (nely + 1)

    # Fixed DOFs
    fixed = _build_fixed_dofs(spec, coords)
    free = np.setdiff1d(np.arange(ndof), fixed)

    # Force vector
    F = _build_force_vector(spec, coords, ndof)

    # Safety: zero out any load on fixed DOFs
    F[fixed] = 0.0

    # Check that F has nonzero entries on free DOFs
    if np.max(np.abs(F[free])) < 1e-30:
        raise ValueError(
            "Force vector is zero on all free DOFs. "
            "Check that loads are not placed on fully-fixed supports.")

    # Passive mask
    passive_mask = build_passive_mask(spec)

    # Build the closure the solver expects
    # Capture fixed/free/F so the signature matches bc_override protocol
    _fixed, _free, _F = fixed, free, F

    if is_3d:
        def bc_override(nelx_arg, nely_arg, nelz_arg):
            return _fixed, _free, _F
    else:
        def bc_override(nelx_arg, nely_arg):
            return _fixed, _free, _F

    return BCResult(
        bc_override=bc_override,
        passive_mask=passive_mask,
        fixed_dofs=fixed,
        free_dofs=free,
        F=F,
        ndof=ndof,
        node_coords=coords,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Solver patch: passive-region support in OC update
# ─────────────────────────────────────────────────────────────────────────────

def apply_passive_mask(rho: np.ndarray, mask: Optional[np.ndarray],
                        rho_min: float = 1e-3) -> np.ndarray:
    """
    Enforce passive regions after OC update.  Call this in the solver loop
    right after _oc_update returns rho_new:

        rho_new = _oc_update(...)
        rho_new = apply_passive_mask(rho_new, passive_mask)

    This is the ~10 lines of solver modification mentioned in the spec.
    """
    if mask is None:
        return rho
    rho = rho.copy()
    rho[mask == 1] = rho_min    # void
    rho[mask == 2] = 1.0        # solid
    return rho


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: ProblemSpec → SIMPParams
# ─────────────────────────────────────────────────────────────────────────────

def spec_to_simp_params(spec: ProblemSpec) -> dict:
    """
    Return a dict of SIMPParams-compatible kwargs derived from the spec.
    Caller does:  SIMPParams(**spec_to_simp_params(spec))
    """
    p = {
        "nelx": spec.nelx,
        "nely": spec.nely,
        "nelz": spec.nelz,
        "volfrac": spec.volfrac,
    }
    if spec.rmin is not None:
        p["rmin"] = spec.rmin
    if spec.max_iter is not None:
        p["max_iter"] = spec.max_iter
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .problem_spec import ProblemSpec, EdgeSupport, PointLoad, CircularRegion

    # Classic cantilever: left edge fixed, point load at mid-right, downward
    spec = ProblemSpec(
        Lx=2.0, Ly=1.0,
        nelx=60, nely=30,
        volfrac=0.5,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, fy=-1.0)],
    )
    bc = generate_bc(spec)
    print(f"Cantilever 2D:")
    print(f"  ndof       = {bc.ndof}")
    print(f"  fixed DOFs = {len(bc.fixed_dofs)}")
    print(f"  free DOFs  = {len(bc.free_dofs)}")
    print(f"  |F|_max    = {np.max(np.abs(bc.F)):.4f}")
    print(f"  passive    = {bc.passive_mask}")
    print()

    # MBB beam: bottom-left roller_x (symmetry), bottom-right pin_y
    spec_mbb = ProblemSpec(
        Lx=3.0, Ly=1.0,
        nelx=90, nely=30,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="pin_x"),
            PointSupport(x=3.0, y=0.0, constraint="pin_y"),
        ],
        loads=[PointLoad(x=0.0, y=1.0, fy=-1.0)],
    )
    bc_mbb = generate_bc(spec_mbb)
    print(f"MBB 2D:")
    print(f"  ndof       = {bc_mbb.ndof}")
    print(f"  fixed DOFs = {len(bc_mbb.fixed_dofs)}")
    print(f"  |F|_max    = {np.max(np.abs(bc_mbb.F)):.4f}")
    print()

    # With passive void (bolt hole)
    spec_hole = ProblemSpec(
        Lx=2.0, Ly=1.0,
        nelx=80, nely=40,
        volfrac=0.4,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, fy=-1.0)],
        passive_regions=[CircularRegion(cx=1.0, cy=0.5, radius=0.15,
                                         kind="void")],
    )
    bc_hole = generate_bc(spec_hole)
    n_void = np.sum(bc_hole.passive_mask == 1) if bc_hole.passive_mask is not None else 0
    print(f"Cantilever with hole:")
    print(f"  ndof       = {bc_hole.ndof}")
    print(f"  void elems = {n_void}")
    print()

    # Distributed load on top edge
    spec_dist = ProblemSpec(
        Lx=2.0, Ly=1.0,
        nelx=60, nely=30,
        volfrac=0.5,
        supports=[
            EdgeSupport(edge="left", constraint="fixed"),
            EdgeSupport(edge="right", constraint="fixed"),
        ],
        loads=[DistributedLoad(edge="top", magnitude=-1.0)],
    )
    bc_dist = generate_bc(spec_dist)
    n_loaded = np.sum(np.abs(bc_dist.F) > 1e-10)
    print(f"Bridge (distributed top load):")
    print(f"  ndof       = {bc_dist.ndof}")
    print(f"  loaded DOFs= {n_loaded}")
    print()

    # Validation: load on fixed support should warn
    spec_bad = ProblemSpec(
        Lx=2.0, Ly=1.0,
        nelx=60, nely=30,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=0.0, y=0.5, fy=-1.0)],  # on fixed edge!
    )
    errs = spec_bad.validate()
    print(f"Validation errors for load-on-support:")
    for e in errs:
        print(f"  - {e}")
