"""
gpu_fem — GPU-accelerated 3D topology optimization with adaptive surrogate routing.

Drop-in replacement for the TO3D pipeline with a GPU FEM backend.
"""

from .fem_gpu import GPUFEMSolver, fea_compute_gpu, detect_gpu_backend
from .surrogate_gpu import SensitivitySurrogateGPU
from .simp_gpu import run_simp_surrogate_gpu, TO3DParams
from .solver_v2 import SolverV2, HexGridGMG

__all__ = [
    "GPUFEMSolver",
    "fea_compute_gpu",
    "detect_gpu_backend",
    "SensitivitySurrogateGPU",
    "run_simp_surrogate_gpu",
    "TO3DParams",
    "SolverV2",
    "HexGridGMG",
]
