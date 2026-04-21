"""
problem_spec.py
---------------
High-level problem specification for AutoSIMP.

ProblemSpec is the single data structure that captures everything a human would
describe in natural language: domain geometry, loads, supports, passive regions,
material, and solver hints.  It is JSON-serializable (for LLM I/O) and validated
at construction time so that bc_generator.py receives clean input.

Coordinate convention (matches pub_simp_solver.py):
  - 2D: origin at bottom-left of the domain
    x ∈ [0, Lx],  y ∈ [0, Ly]
    Node (i, j) has x = i * (Lx / nelx), y = j * (Ly / nely)
    Element (i, j) occupies [x_i, x_{i+1}] × [y_j, y_{j+1}]
    Solver arrays are indexed  e = i * nely + j  (column-major in y)
  - 3D: same pattern extended to z, e = i*nely*nelz + j*nelz + k
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ── Boundary conditions ─────────────────────────────────────────────────────

@dataclass
class PointSupport:
    """Fix a single node nearest to (x, y [, z])."""
    x: float
    y: float
    z: float = 0.0
    constraint: Literal["fixed", "pin_x", "pin_y", "pin_z",
                         "roller_x", "roller_y", "roller_z"] = "fixed"

    def dof_mask(self, ndim: int) -> list[int]:
        """Return local DOF offsets to fix (0=x, 1=y, 2=z)."""
        m = {
            "fixed":    list(range(ndim)),
            "pin_x":    [0],
            "pin_y":    [1],
            "pin_z":    [2] if ndim == 3 else [],
            "roller_x": [1] + ([2] if ndim == 3 else []),   # free in x
            "roller_y": [0] + ([2] if ndim == 3 else []),   # free in y
            "roller_z": [0, 1] if ndim == 3 else [0, 1],    # free in z
        }
        return m.get(self.constraint, list(range(ndim)))


@dataclass
class EdgeSupport:
    """Fix all nodes along a named edge/face."""
    edge: Literal["left", "right", "top", "bottom",
                   "front", "back"]          # front/back only for 3D
    constraint: Literal["fixed", "pin_x", "pin_y", "pin_z",
                         "roller_x", "roller_y", "roller_z"] = "fixed"

    def dof_mask(self, ndim: int) -> list[int]:
        return PointSupport.dof_mask(
            PointSupport(0, 0, 0, self.constraint), ndim)


@dataclass
class PointLoad:
    """Concentrated force at the node nearest to (x, y [, z])."""
    x: float
    y: float
    z: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.fx**2 + self.fy**2 + self.fz**2)


@dataclass
class DistributedLoad:
    """Uniform pressure along a named edge/face (normal direction)."""
    edge: Literal["left", "right", "top", "bottom", "front", "back"]
    magnitude: float = -1.0          # negative = into the domain


# ── Passive regions ─────────────────────────────────────────────────────────

@dataclass
class CircularRegion:
    """Circular void or solid zone (2D: disc, 3D: cylinder along z)."""
    cx: float
    cy: float
    radius: float
    kind: Literal["void", "solid"] = "void"


@dataclass
class RectangularRegion:
    """Axis-aligned rectangular void or solid zone."""
    x0: float
    y0: float
    x1: float
    y1: float
    kind: Literal["void", "solid"] = "void"


# ── Main specification ──────────────────────────────────────────────────────

@dataclass
class ProblemSpec:
    """
    Complete problem definition for topology optimization.

    All spatial quantities are in consistent units (user's choice).
    The configurator agent fills this from natural language; bc_generator
    converts it to solver-ready arrays.
    """

    # Domain geometry
    Lx: float = 2.0              # domain length (x)
    Ly: float = 1.0              # domain height (y)
    Lz: float = 0.0              # domain depth  (z), 0 → 2D

    # Mesh
    nelx: int = 60               # elements in x
    nely: int = 30               # elements in y
    nelz: int = 0                # elements in z, 0 → 2D

    # Material (SIMP uses normalized E=1, but we keep physical values
    # so the configurator can validate user intent.  bc_generator
    # normalizes before passing to the solver.)
    E: float = 1.0               # Young's modulus
    nu: float = 0.3              # Poisson's ratio

    # Volume fraction
    volfrac: float = 0.5

    # Boundary conditions
    supports: list[PointSupport | EdgeSupport] = field(default_factory=list)
    loads:    list[PointLoad | DistributedLoad] = field(default_factory=list)

    # Passive regions
    passive_regions: list[CircularRegion | RectangularRegion] = field(
        default_factory=list)

    # Solver hints (optional overrides; None → use defaults)
    max_iter: Optional[int] = None
    rmin: Optional[float] = None

    # ── validation ──────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """
        Return a list of error strings.  Empty list → valid.
        Checks physical plausibility, not solver convergence.
        """
        errors: list[str] = []

        # Geometry
        if self.Lx <= 0 or self.Ly <= 0:
            errors.append("Domain dimensions Lx, Ly must be positive.")
        if self.Lz < 0:
            errors.append("Lz must be >= 0 (0 for 2D).")
        if self.nelx < 2 or self.nely < 2:
            errors.append("Mesh must have at least 2 elements in x and y.")
        if self.Lz > 0 and self.nelz < 2:
            errors.append("3D domain requires nelz >= 2.")

        # Material
        if self.E <= 0:
            errors.append("Young's modulus E must be positive.")
        if not (-1.0 < self.nu < 0.5):
            errors.append("Poisson's ratio must be in (-1, 0.5).")

        # Volume fraction
        if not (0.01 <= self.volfrac <= 0.99):
            errors.append("Volume fraction must be in [0.01, 0.99].")

        # Supports
        if not self.supports:
            errors.append("At least one support (fixed DOF) is required.")

        # Loads
        if not self.loads:
            errors.append("At least one load is required.")
        for i, ld in enumerate(self.loads):
            if isinstance(ld, PointLoad) and ld.magnitude < 1e-30:
                errors.append(f"Load[{i}] has zero magnitude.")

        # Check that no point load sits exactly on a fully-fixed support
        for ld in self.loads:
            if not isinstance(ld, PointLoad):
                continue
            for sup in self.supports:
                if isinstance(sup, PointSupport) and sup.constraint == "fixed":
                    d = math.sqrt((ld.x - sup.x)**2 + (ld.y - sup.y)**2
                                  + (ld.z - sup.z)**2)
                    # "close" = within half an element
                    h = max(self.Lx / max(self.nelx, 1),
                            self.Ly / max(self.nely, 1))
                    if d < 0.5 * h:
                        errors.append(
                            f"Point load at ({ld.x},{ld.y}) coincides with "
                            f"a fixed support — load will be zeroed out.")

        # Passive regions inside domain
        for i, pr in enumerate(self.passive_regions):
            if isinstance(pr, CircularRegion):
                if pr.radius <= 0:
                    errors.append(f"Passive region [{i}] radius must be > 0.")
            elif isinstance(pr, RectangularRegion):
                if pr.x0 >= pr.x1 or pr.y0 >= pr.y1:
                    errors.append(
                        f"Passive region [{i}] has inverted bounds.")

        return errors

    @property
    def is_3d(self) -> bool:
        return self.Lz > 0 and self.nelz > 0

    @property
    def ndim(self) -> int:
        return 3 if self.is_3d else 2

    # ── JSON round-trip ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        d = {
            "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz,
            "nelx": self.nelx, "nely": self.nely, "nelz": self.nelz,
            "E": self.E, "nu": self.nu, "volfrac": self.volfrac,
            "max_iter": self.max_iter, "rmin": self.rmin,
            "supports": [], "loads": [], "passive_regions": [],
        }
        for s in self.supports:
            if isinstance(s, EdgeSupport):
                d["supports"].append({
                    "type": "edge", "edge": s.edge,
                    "constraint": s.constraint})
            else:
                d["supports"].append({
                    "type": "point", "x": s.x, "y": s.y, "z": s.z,
                    "constraint": s.constraint})
        for ld in self.loads:
            if isinstance(ld, DistributedLoad):
                d["loads"].append({
                    "type": "distributed", "edge": ld.edge,
                    "magnitude": ld.magnitude})
            else:
                d["loads"].append({
                    "type": "point", "x": ld.x, "y": ld.y, "z": ld.z,
                    "fx": ld.fx, "fy": ld.fy, "fz": ld.fz})
        for pr in self.passive_regions:
            if isinstance(pr, CircularRegion):
                d["passive_regions"].append({
                    "type": "circle", "cx": pr.cx, "cy": pr.cy,
                    "radius": pr.radius, "kind": pr.kind})
            else:
                d["passive_regions"].append({
                    "type": "rect", "x0": pr.x0, "y0": pr.y0,
                    "x1": pr.x1, "y1": pr.y1, "kind": pr.kind})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemSpec":
        """Deserialize from a plain dict (e.g. from LLM JSON output)."""
        supports = []
        for s in d.get("supports", []):
            if s["type"] == "edge":
                supports.append(EdgeSupport(
                    edge=s["edge"], constraint=s.get("constraint", "fixed")))
            else:
                supports.append(PointSupport(
                    x=s["x"], y=s["y"], z=s.get("z", 0.0),
                    constraint=s.get("constraint", "fixed")))

        loads = []
        for ld in d.get("loads", []):
            if ld["type"] == "distributed":
                loads.append(DistributedLoad(
                    edge=ld["edge"], magnitude=ld.get("magnitude", -1.0)))
            else:
                loads.append(PointLoad(
                    x=ld["x"], y=ld["y"], z=ld.get("z", 0.0),
                    fx=ld.get("fx", 0.0), fy=ld.get("fy", 0.0),
                    fz=ld.get("fz", 0.0)))

        passive_regions = []
        for pr in d.get("passive_regions", []):
            if pr["type"] == "circle":
                passive_regions.append(CircularRegion(
                    cx=pr["cx"], cy=pr["cy"], radius=pr["radius"],
                    kind=pr.get("kind", "void")))
            else:
                passive_regions.append(RectangularRegion(
                    x0=pr["x0"], y0=pr["y0"], x1=pr["x1"], y1=pr["y1"],
                    kind=pr.get("kind", "void")))

        return cls(
            Lx=d.get("Lx", 2.0), Ly=d.get("Ly", 1.0), Lz=d.get("Lz", 0.0),
            nelx=d.get("nelx", 60), nely=d.get("nely", 30),
            nelz=d.get("nelz", 0),
            E=d.get("E", 1.0), nu=d.get("nu", 0.3),
            volfrac=d.get("volfrac", 0.5),
            supports=supports, loads=loads,
            passive_regions=passive_regions,
            max_iter=d.get("max_iter"), rmin=d.get("rmin"),
        )
