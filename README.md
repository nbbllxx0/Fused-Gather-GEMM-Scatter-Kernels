# Fused and node-owned matrix-free finite-element operators for 3D SIMP

Code and measurements accompanying the paper *Fused and Node-Owned
Matrix-Free Finite-Element Operators for 3D SIMP Topology Optimization on a
Single GPU* (Structural and Multidisciplinary Optimization).

Every number the paper reports is produced by the drivers here and written to
`results/`; the tables and figures are generated from those files rather than
typed in.

## Layout

```
src/gpu_fem/
  cuda_operators.py     the matrix-free K*v mappings the paper compares --
                        three-stage gather/GEMM/scatter, element-owned fused,
                        fused with analytic indexing, and node-owned, each in
                        single and double precision -- plus the fused adjoint
                        and the node-owned Jacobi diagonal
  cuda_fused_matvec.py  the element-owned FP32 and BF16 WMMA kernels
  filter_r2.py          structured matrix-free cone filter (nothing assembled)
  simp_r2.py            SIMP driver: fail-closed solves, design-convergence
                        termination, final-iterate reporting, fixed-area load
  pub_simp_solver.py    element stiffness and reference connectivity

experiments/            one driver per study in the paper
tools/                  verification harnesses, and the environment recorder
scripts/                the number-audit generator
results/                every measurement the paper reports, including the
                        per-solve residual log of every run
results/baseline_r1/    the superseded measurements behind the first
                        submitted version, kept for inspection only
```

## Reproducing the numbers

Needs an NVIDIA GPU, CUDA, and the packages in `requirements.txt`. Run the
verification harnesses first; nothing downstream means anything if they fail.

```bash
export PYTHONPATH=src
python tools/verify_operators.py --sizes 64k,216k     # parity + determinism
python tools/verify_simp.py                           # adjoint, FD, volume

python experiments/operator_benchmark.py --sizes 64k,216k,512k,1M,2M

# The ladder, in two passes. The first runs every solver path once with full
# instrumentation and records the direct single-precision paths failing to
# reach tolerance, which is a result and not an omission. The second repeats
# only the paths that converge, because one wall time is not a measurement.
python experiments/run_ladder_v11.py --sizes 216k,512k,1M,2M --tag s1
python experiments/ladder_repeats.py --sizes 216k,512k,1M,2M --reps 3

python experiments/coldstart_ladder.py --sizes 216k,512k,1M,2M
python experiments/floor_sweep.py --size 512k
python experiments/kappa_lanczos.py --sizes 64k,216k,512k
python experiments/kappa_bf16_study.py --sizes 64k,216k,512k

# Mesh comparability: fixed physical filter radius, both the double-precision
# control and the refined path, and no refinement claim is drawn from it.
python experiments/run_ladder_v11.py --sizes 216k,512k,1M     --specs ir_node_fp32,fused_fp64 --physical-rmin 0.025     --max-outer 1200 --tag comparability

python scripts/collect_numbers.py     # -> results/NUMBERS.md
```

`--gate` names the `results/` subdirectory a run writes to. The shipped tree
has four: `G0` the recorded environment, `G1` the verification harnesses, `G3`
the operator microbenchmark, and `G6` every study the paper reports from. The
commands above default to `G6` where it matters.

Re-running a study overwrites the corresponding file in `results/`, so a fresh
measurement can be diffed against the one published here.

`results/NUMBERS.md` maps every quantitative claim in the paper to the
results file it is computed from.

## Two things worth knowing before a long run

`GPU_FEM_ENV_BOOTSTRAPPED=1` stops the drivers re-executing into another
interpreter. `CUDA_PATH` must match the toolkit your CuPy build targets: a
mismatch still lets the kernels compile, but makes CuPy's own reductions fail
at first use, which can be hours into a run.

## Measurement protocol

These rules are enforced in code, not by convention, because each of them is
a way a benchmark can report something that did not happen.

* Every linear solve must reach its stated relative-residual tolerance. A
  warm-started solve that misses is retried from zero and the retry's
  iterations and time are counted; a second failure aborts the run.
* Optimizations terminate on a declared design-convergence criterion, not an
  iteration budget. A run that stops on the safety guard is recorded as not
  converged rather than reported as a result.
* The final iterate is the result. No best-iterate selection, no clamping of
  a reported compliance, no restart from a previous design.
* Operator timings are the median of independent timing blocks, with the
  spread reported and the clock and temperature sampled around each block.
  Single measurements are not reproducible at these timescales.

## Data not included

Most of the final density fields (`*_rho_final_device.npy`, about 180 MB) are
left out to keep the repository clonable. They are reproducible from the
drivers above, and are available from the authors on request.

Six are included, because the paper's topology comparison is computed from
them and would otherwise be unverifiable: the double-precision fused optimum at
each of the four meshes, plus the refined node-owned and three-stage
double-precision runs at one million elements, which are the pair reported as
differing by eight elements out of a million.

## Earlier version

Tag `v0.1.0` is the code released with the first submitted version of the
paper. It is a different implementation: the operator set, the convergence
test and the reporting rules all changed in revision, so its measurements are
not comparable with the ones here. It is kept so the earlier results stay
inspectable.

The three measurement files behind the superseded timing comparison are in
`results/baseline_r1/`, with a note on why each no longer applies.

## Citation

See `CITATION.cff`, and please cite the paper rather than this repository
alone.

## License

See `LICENSE`.
