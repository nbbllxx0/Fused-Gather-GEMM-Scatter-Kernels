# Superseded round-1 measurements

These three files are the measurements behind the **first submitted version**
of the paper. They are retained so that the comparison they were used for stays
inspectable: those timings were measured across solver paths that had not run
equivalent amounts of work, and the present results replace them.

They are **not** current results. Every number in the present paper comes from
`results/G0`, `results/G1`, `results/G3` and `results/G6`, produced under the
corrected protocol described in the manuscript and supplement. The two sets are
not interchangeable:

| File | Role in the submitted version | Why it no longer applies |
|---|---|---|
| `scaling_ladder_simp_mid.csv` | Source of submitted Table 4 | Solver paths terminated on different criteria, so the wall-clock ratios compared unequal amounts of work |
| `statistical_repeats.csv` | Five-repeat timing means | Same comparability defect, and convergence was judged from the recursive conjugate-gradient residual rather than the true residual |
| `fully_converged_study.csv` | Iteration-capped comparison | Runs terminated at the iteration cap rather than at the equilibrium tolerance |

The code that produced them is preserved at tag `v0.1.0`, under
`experiments/phase3/`.
