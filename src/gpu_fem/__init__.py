"""
gpu_fem -- matrix-free GPU finite-element operators for 3D SIMP topology
optimization.

Import the modules you need directly:

    from gpu_fem.cuda_operators import OperatorSuite
    from gpu_fem.simp_r2 import run_simp, build_cantilever, pcg
    from gpu_fem.filter_r2 import ConeFilter

Nothing is imported here, so that importing one operator path does not
require CuPy, a GPU, or any module other than the one asked for.
"""
