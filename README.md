# Matrix-Free 3D SIMP Topology Optimization with Fused Gather-GEMM-Scatter Kernels

Reference implementation and reproducibility pack for:

> **Matrix-Free 3D SIMP Topology Optimization with Fused Gather-GEMM-Scatter
> Kernels.** Shaoliang Yang, Jun Wang, Yunsheng Wang. Santa Clara University.

The repo ships:

- the `gpu_fem` Python package (solver, presets, public CLI),
- the experiment scripts in `experiments/phase3/` that produce every CSV/JSON
  used by the manuscript tables,
- the post-processing scripts in `scripts/` that turn those CSV/JSON files
  into the manuscript figures and the VRAM / topology-render artifacts.

The reported numbers were generated on a single NVIDIA RTX 4090 (24 GB,
Ada Lovelace). The fused path requires a CUDA-capable NVIDIA GPU; CuPy NVRTC
must be able to call `nvcc`. Rough wall-time budget for the full suite is
2-3 hours of GPU time on an RTX 4090 (excluding the optional 8 M FEA solve
and the optional CPU PyTopo3D rerun).

---

## Contents

```
.
|- src/gpu_fem/             package source (solver + presets + public CLI)
|- experiments/phase3/      one script per paper table / figure data file
|- scripts/                 plotting, rendering, VRAM probe
|- pyproject.toml           editable-install metadata
|- requirements.txt         runtime deps (CuPy and PyTorch noted in comments)
|- repro_environment.txt    pinned versions used for the reported runs
|- CITATION.bib             citation entry
`- LICENSE                  BSD 3-Clause
```

---

## Install

Recommended: use a fresh virtual environment.

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

You then need to install **CuPy** (matched to your CUDA toolkit) and a CUDA
build of **PyTorch** by hand, because the right wheel depends on your local
CUDA version. The reported runs used `cupy-cuda12x==13.6.0` and PyTorch 2.5.1
built against CUDA 12.1:

```bash
pip install cupy-cuda12x==13.6.0
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Optional extras only needed by specific scripts:

```bash
pip install pyvista                  # 3D isosurface renders
pip install pytopo3d pypardiso       # CPU PyTopo3D baseline comparison
```

`experiments/phase3/ncu_profile.py` additionally requires the system
`ncu` CLI from NVIDIA Nsight Compute.

---

## Reproducing the Paper

All experiment scripts write CSV/JSON files into `experiments/phase3/`.
All plotting scripts write PDFs/PNGs into `figs/`. Both directories are
created on demand and are listed in `.gitignore` - readers regenerate
everything from source.

The full reproduction takes **two phases**: (1) run the experiment scripts to
produce the CSV/JSON data, then (2) run the plotting and rendering scripts to
turn the data into figures.

### One-shot script

A complete sequential reproduction on Linux / macOS:

```bash
bash <<'EOF'
set -e
EXP=experiments/phase3
SCR=scripts

# --- Phase 1: experiment data ----------------------------------------------
python $EXP/profile_matfree_hotpath.py
python $EXP/scaling_ladder.py --mode all --sizes 216k,512k,1M --tag mid
python $EXP/scaling_ladder.py --mode simp --sizes 1M           --tag simp_1m
python $EXP/scaling_ladder.py --mode all --sizes 2M,5M         --tag 2m5m
python $EXP/scaling_ladder.py --mode fea --sizes 2M,5M         --tag fea_large
# Optional 8 M cap-limited FEA point (skip if VRAM is tight):
# python $EXP/scaling_ladder.py --mode fea --sizes 8M           --tag fea_8m
python $EXP/benchmark_torsion.py    --preset torsion_gpu_500k --paths fp64,fp32,fused
python $EXP/benchmark_hard_problems.py --mode ABCD
python $EXP/bf16_extended_study.py
python $EXP/kappa_estimation.py     --sizes 64k,216k,512k
python $EXP/statistical_repeats.py  --sizes 216k,512k,1M --paths fp64,fused
python $EXP/determinism_study.py    --sizes 64k,216k
python $EXP/fully_converged_study.py --sizes 216k,512k --paths fp64,fused
python $EXP/energy_benchmark.py     --sizes 216k,1M    --paths fp64,fused
python $EXP/capture_cg_residuals.py
# Optional PyTopo3D CPU comparison (needs pytopo3d + pypardiso):
# python $EXP/cpu_baseline_comparison.py

# --- Phase 2: figures and VRAM table ---------------------------------------
python $SCR/plot_phase3_figures.py
python $SCR/probe_vram.py --preset cantilever_gpu_large
python $SCR/generate_topology_renders.py --run-simp --render
EOF
```

If you want it to keep going after a single failure, drop `set -e`.

### Script -> paper artifact mapping

Each row below tells you which command produces which paper table or figure
data file. The artifact provenance appendix of the paper gives the same
mapping in narrative form.

| Paper artifact                          | Generator script                                       | Output file(s) under `experiments/phase3/`                       |
|-----------------------------------------|--------------------------------------------------------|------------------------------------------------------------------|
| Hot-path profiling table & figure       | `profile_matfree_hotpath.py`                           | `profile_hotpath.csv`, `profile_hotpath.json`                    |
| Cantilever FEA scaling                  | `scaling_ladder.py --mode fea`                         | `scaling_ladder_fea_*.csv|.json`                                 |
| Cantilever SIMP-120 scaling             | `scaling_ladder.py --mode simp`                        | `scaling_ladder_simp_*.csv|.json`                                |
| Torsion table & trajectory              | `benchmark_torsion.py`                                 | `benchmark_torsion_500k.csv`, `torsion_500k_*_history.json`      |
| MBB / bridge hard-problem table         | `benchmark_hard_problems.py`                           | `results_hard_problems.csv`                                      |
| BF16 convergence table                  | `bf16_extended_study.py`                               | `bf16_extended_study.csv|.json`, `bf16_ir_smoke.csv|.json`       |
| Condition-number study                  | `kappa_estimation.py`                                  | `kappa_estimation.csv|.json`                                     |
| Repeat study                            | `statistical_repeats.py`                               | `statistical_repeats.csv|.json`                                  |
| Atomic-scatter determinism              | `determinism_study.py`                                 | `determinism_study.csv|.json`                                    |
| Selected-iterate high-cap validation    | `fully_converged_study.py`                             | `fully_converged_study.csv|.json`                                |
| Board-power energy study                | `energy_benchmark.py`                                  | `energy_benchmark.csv|.json`                                     |
| CG residual figure                      | `capture_cg_residuals.py`                              | `cg_residual_history.json`                                       |
| External CPU PyTopo3D bars (optional)   | `cpu_baseline_comparison.py`                           | `cpu_baseline_augmented.csv|.json`, `pytopo3d_vs_ours.csv`       |

| Plotting / rendering / probe script | Outputs                                                           |
|-------------------------------------|-------------------------------------------------------------------|
| `scripts/plot_phase3_figures.py`    | All scaling, speedup, profiler, residual, torsion-trajectory, and external-bars PDFs under `figs/` |
| `scripts/probe_vram.py`             | Peak VRAM + per-call wall-time print for the requested preset (Table 11 numbers) |
| `scripts/generate_topology_renders.py` | Smoothed marching-cubes isosurface PDFs/PNGs under `figs/` plus density `.npy` arrays under `experiments/phase3/topology_renders/` |
| `scripts/estimate_condition_numbers.py` | Legacy condition-number probe shipped from the prior paper; kept for cross-checks but the paper uses `kappa_estimation.py` |

### Per-table commands (Phase 1)

If you want to run only the experiments behind a specific table or figure:

#### Hot-path profiling (Table 2 + Figure 9)

```bash
python experiments/phase3/profile_matfree_hotpath.py
```

#### Roofline / per-matvec bandwidth (Table 3)

The bandwidth numbers come from CUDA-event timing inside the fused kernel
during the scaling-ladder runs; no separate command is needed. The Nsight
Compute counters quoted in the discussion come from:

```bash
python experiments/phase3/ncu_profile.py        # requires NVIDIA `ncu` CLI
```

#### Cantilever FEA + SIMP scaling (Table 4 + Figures 3-5)

```bash
# Mid-range: 216k, 512k, 1M elements, both FEA-only and SIMP-120
python experiments/phase3/scaling_ladder.py --mode all --sizes 216k,512k,1M --tag mid

# 1M-only SIMP run (deposited as scaling_ladder_simp_1m.*)
python experiments/phase3/scaling_ladder.py --mode simp --sizes 1M --tag simp_1m

# Large representative SIMP-120 rows for 2M and 4.9M
python experiments/phase3/scaling_ladder.py --mode all --sizes 2M,5M --tag 2m5m

# FEA-only large stress-test rows for 2M and 4.9M
python experiments/phase3/scaling_ladder.py --mode fea --sizes 2M,5M --tag fea_large

# Optional 8M FEA cap-limited point (skip if VRAM is tight)
python experiments/phase3/scaling_ladder.py --mode fea --sizes 8M --tag fea_8m
```

#### Torsion benchmark (Table 5 + Figure 10)

```bash
python experiments/phase3/benchmark_torsion.py --preset torsion_gpu_500k --paths fp64,fp32,fused
```

#### MBB / bridge hard-problem table (Table 6)

```bash
python experiments/phase3/benchmark_hard_problems.py --mode ABCD
```

`--mode` accepts any combination of `A` (FEA-only), `B` (SIMP-cold),
`C` (SIMP-warm) and `D` (512k SIMP). `ABCD` runs all four.

#### BF16 convergence (Table 7)

```bash
python experiments/phase3/bf16_extended_study.py
```

Add `--quick` for a fast smoke check.

#### Condition-number study (Section 5.x table)

```bash
python experiments/phase3/kappa_estimation.py --sizes 64k,216k,512k
```

#### Repeat / determinism / high-cap / energy (Tables 8-10)

```bash
python experiments/phase3/statistical_repeats.py  --sizes 216k,512k,1M --paths fp64,fused
python experiments/phase3/determinism_study.py    --sizes 64k,216k
python experiments/phase3/fully_converged_study.py --sizes 216k,512k --paths fp64,fused
python experiments/phase3/energy_benchmark.py     --sizes 216k,1M    --paths fp64,fused
```

#### CG residual figure (Figure 13)

```bash
python experiments/phase3/capture_cg_residuals.py
```

#### External CPU comparison (Figure 8, optional)

```bash
python experiments/phase3/cpu_baseline_comparison.py
```

This step is optional: it requires `pytopo3d` and `pypardiso` and runs a
multi-hour CPU SIMP solve. Skipping it leaves Figure 8 with the deposited
GPU-side bars only.

### Building all figures (Phase 2)

After the experiment data exists under `experiments/phase3/`, regenerate
every paper figure with:

```bash
python scripts/plot_phase3_figures.py
```

Outputs land in `figs/` (created on demand). The script is idempotent and
silently skips figures whose source CSV/JSON is missing, so partial reruns
are fine.

### VRAM table (Table 11)

```bash
python scripts/probe_vram.py --preset cantilever_gpu_large
python scripts/probe_vram.py --preset cantilever_gpu_xlarge
python scripts/probe_vram.py --preset bridge_gpu_large
```

The script prints `peak VRAM (MB)` and per-call wall-time on stdout. The
paper's VRAM numbers come from collecting these prints across presets.

### Topology renders (qualitative figures)

Run the SIMP solves and then the renders in two passes:

```bash
# Run SIMP and save .npy density fields to experiments/phase3/topology_renders/
python scripts/generate_topology_renders.py --run-simp

# Render saved fields as smoothed marching-cubes isosurfaces (PDF + PNG)
python scripts/generate_topology_renders.py --render
```

Restrict to one problem with e.g. `--prob cantilever_216k`. Use
`--canonical-only` to render only the camera views used in the paper.

### Kernel correctness sanity tests

Optional tests of the fused FP32 and BF16 kernels against the three-stage
reference:

```bash
python experiments/phase3/test_fused_kernel.py
python experiments/phase3/test_bf16_kernel.py
python experiments/phase3/test_bf16_ir.py
python experiments/phase3/verify_galerkin_kc.py
```

---

## Public CLI (single SIMP runs)

For interactive use outside the paper-reproduction workflow, the package
ships a small CLI:

```powershell
python -m gpu_fem list-presets
python -m gpu_fem list-presets --gpu-only

python -m gpu_fem run `
  --preset cantilever_gpu_large `
  --mode fem `
  --backend auto `
  --output runs\cantilever_gpu_large_fem `
  --verbose
```

Each run writes `report.json`, density `.npy` arrays, slice PNGs, OBJ mesh
exports, and a convergence plot into the requested output folder.

To run a sweep across presets:

```powershell
python -m gpu_fem suite `
  --presets cantilever_gpu_medium cantilever_gpu_large bridge_gpu_large `
  --modes surrogate fem `
  --backend auto `
  --output-root runs\public_suite `
  --verbose
```

The suite runner aggregates per-case outputs into `summary.json` and
`summary.csv` next to the per-preset folders.

---

## Notes on conda environments

Every experiment script under `experiments/phase3/` checks for an optional
environment variable `GPU_FEM_PYTHON`. If set to a Python interpreter path,
the script re-execs itself under that interpreter - useful when CuPy lives in
a separate conda env. If unset, the scripts run under whatever Python invoked
them. Set it like:

```bash
export GPU_FEM_PYTHON=$HOME/anaconda3/envs/cuda121/bin/python
```

```powershell
$env:GPU_FEM_PYTHON = "$HOME\anaconda3\envs\cuda121\python.exe"
```

---

## Reproducibility scope

- Reported runs used a single RTX 4090 (24 GB, Ada Lovelace) at the
  default 450 W power limit.
- The condition-number, determinism, and BF16 smoke-test workflows use
  fixed seeds (42 for power iteration, 123 for inverse iteration; the
  determinism workflow uses its own seeds - see source).
- SIMP "selected" compliance reports the lowest-compliance iterate among
  iterates that pass the validity gate (`penal >= 3.0`, grayness < 0.25).
- The fused path delivers measured 6.0-6.6x per-matvec speedup over the
  three-stage FP64 baseline (synthetic hot-path) and 4.6-7.3x end-to-end
  SIMP-120 wall-time speedup on the cantilever benchmark; see the paper for
  the full breakdown including the FP32 reference and the energy numbers.
- Exact pinned versions used for the manuscript runs are in
  `repro_environment.txt`.

---

## Citation

```bibtex
@article{yang2026fused,
  author        = {Yang, Shaoliang and Wang, Jun and Wang, Yunsheng},
  title         = {Matrix-Free {3D} {SIMP} Topology Optimization with Fused
                   Gather-{GEMM}-Scatter Kernels},
  year          = {2026},
  eprint        = {2604.18020},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CE},
  url           = {https://arxiv.org/abs/2604.18020}
}
```

See `CITATION.bib` for the canonical entry.

---

## License

BSD 3-Clause. See `LICENSE`.
