"""Shared path helpers for the local GPU-FEM package."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalPaths:
    repo_root: Path
    src_dir: Path
    package_dir: Path
    scripts_dir: Path


def resolve_local_paths(start: str | Path | None = None) -> LocalPaths:
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()

    for candidate in [here] + list(here.parents):
        repo_root = candidate if candidate.is_dir() else candidate.parent
        src_dir = repo_root / "src"
        package_dir = src_dir / "gpu_fem"
        scripts_dir = repo_root / "scripts"
        if package_dir.exists():
            return LocalPaths(
                repo_root=repo_root,
                src_dir=src_dir,
                package_dir=package_dir,
                scripts_dir=scripts_dir,
            )

    raise RuntimeError("Unable to locate the local GPU-FEM repo root.")


def ensure_local_paths(start: str | Path | None = None) -> LocalPaths:
    paths = resolve_local_paths(start)
    text = str(paths.src_dir)
    if text not in sys.path:
        sys.path.insert(0, text)
    return paths
