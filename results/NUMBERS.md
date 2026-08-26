# Number audit

Every quantitative claim the accompanying paper makes, with
the results file it is computed from. Regenerate with
`python scripts/collect_numbers.py`.

## Environment

| item | value |
|---|---|
| os | Windows-10-10.0.26200-SP0 |
| os_version | 10.0.26200 |
| machine | AMD64 |
| python | 3.10.18 |
| python_executable | ~\anaconda3\envs\gpu-fem\python.exe |
| numpy_version | 2.2.6 |
| scipy_version | 1.15.3 |
| cupy_version | 13.6.0 |
| matplotlib_version | 3.10.7 |
| pandas_version | 2.3.3 |
| pyvista_version | 0.46.3 |
| skimage_version | 0.25.2 |
| torch_version | 2.5.1+cu124 |
| cuda_runtime_version | 12090 |
| cuda_driver_version | 13020 |
| gpu_name | NVIDIA GeForce RTX 4090 |
| gpu_compute_capability | 8.9 |
| gpu_sm_count | 128 |
| gpu_total_bytes | 25756696576 |
| gpu_total_GiB | 23.98779296875 |
| gpu_free_at_start_GiB | 22.49609375 |
| gpu_used_by_other_processes_GiB | 1.49169921875 |
| nvidia_driver | 596.21 |
| gpu_power_limit_W | 450.00 |
| gpu_memory_total_MiB | 24564 |
| CUDA_PATH | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1 |
| GPU_FEM_ENV_BOOTSTRAPPED | 1 |
| gpu_clocks_locked | False |
| gpu_drives_display | True |
| note | The GPU also drives the desktop, and its clocks are not locked (locking them is a system-wide setting). Short-kernel measurements therefore carry occasional contention outliers; the microbenchmark reports the median of independent timing blocks with the interquartile range and full range, rather than a single run. |

## Operator parity

| mesh | path | rel L2 vs three-stage FP64 | bar | pass |
|---|---|---|---|---|
| 2k | three_stage_fp64_e32 | 1.026e-16 | 1e-06 | yes |
| 2k | three-stage FP32 | 6.972e-08 | 1e-05 | yes |
| 2k | three-stage FP32 (FP64 scatter) | 5.945e-08 | 1e-05 | yes |
| 2k | fused FP64 | 1.200e-16 | 1e-06 | yes |
| 2k | fused FP32 | 7.318e-08 | 1e-05 | yes |
| 2k | fused FP64 + analytic index | 1.188e-16 | 1e-06 | yes |
| 2k | fused FP32 + analytic index | 7.351e-08 | 1e-05 | yes |
| 2k | node-owned FP64 | 2.617e-16 | 1e-06 | yes |
| 2k | node-owned FP32 | 1.245e-07 | 1e-05 | yes |
| 8k | three_stage_fp64_e32 | 1.049e-16 | 1e-06 | yes |
| 8k | three-stage FP32 | 6.699e-08 | 1e-05 | yes |
| 8k | three-stage FP32 (FP64 scatter) | 5.341e-08 | 1e-05 | yes |
| 8k | fused FP64 | 1.178e-16 | 1e-06 | yes |
| 8k | fused FP32 | 7.438e-08 | 1e-05 | yes |
| 8k | fused FP64 + analytic index | 1.173e-16 | 1e-06 | yes |
| 8k | fused FP32 + analytic index | 7.453e-08 | 1e-05 | yes |
| 8k | node-owned FP64 | 2.595e-16 | 1e-06 | yes |
| 8k | node-owned FP32 | 1.284e-07 | 1e-05 | yes |
| 64k | three_stage_fp64_e32 | 1.107e-16 | 1e-06 | yes |
| 64k | three-stage FP32 | 7.444e-08 | 1e-05 | yes |
| 64k | three-stage FP32 (FP64 scatter) | 6.237e-08 | 1e-05 | yes |
| 64k | fused FP64 | 1.195e-16 | 1e-06 | yes |
| 64k | fused FP32 | 7.445e-08 | 1e-05 | yes |
| 64k | fused FP64 + analytic index | 1.195e-16 | 1e-06 | yes |
| 64k | fused FP32 + analytic index | 7.423e-08 | 1e-05 | yes |
| 64k | node-owned FP64 | 2.646e-16 | 1e-06 | yes |
| 64k | node-owned FP32 | 1.292e-07 | 1e-05 | yes |
| 216k | three_stage_fp64_e32 | 1.132e-16 | 1e-06 | yes |
| 216k | three-stage FP32 | 7.431e-08 | 1e-05 | yes |
| 216k | three-stage FP32 (FP64 scatter) | 6.212e-08 | 1e-05 | yes |
| 216k | fused FP64 | 1.198e-16 | 1e-06 | yes |
| 216k | fused FP32 | 7.441e-08 | 1e-05 | yes |
| 216k | fused FP64 + analytic index | 1.200e-16 | 1e-06 | yes |
| 216k | fused FP32 + analytic index | 7.455e-08 | 1e-05 | yes |
| 216k | node-owned FP64 | 2.647e-16 | 1e-06 | yes |
| 216k | node-owned FP32 | 1.296e-07 | 1e-05 | yes |

### Determinism

| mesh | path | bitwise identical over 5 repeats | max abs spread |
|---|---|---|---|
| 2k | fused FP32 | no | 9.537e-07 |
| 2k | fused FP32 + analytic index | no | 9.537e-07 |
| 2k | node-owned FP32 | yes | 0.000e+00 |
| 2k | node-owned FP64 | yes | 0.000e+00 |
| 8k | fused FP32 | no | 9.537e-07 |
| 8k | fused FP32 + analytic index | no | 1.907e-06 |
| 8k | node-owned FP32 | yes | 0.000e+00 |
| 8k | node-owned FP64 | yes | 0.000e+00 |
| 64k | fused FP32 | no | 1.907e-06 |
| 64k | fused FP32 + analytic index | no | 1.907e-06 |
| 64k | node-owned FP32 | yes | 0.000e+00 |
| 64k | node-owned FP64 | yes | 0.000e+00 |
| 216k | fused FP32 | no | 1.907e-06 |
| 216k | fused FP32 + analytic index | no | 1.907e-06 |
| 216k | node-owned FP32 | yes | 0.000e+00 |
| 216k | node-owned FP64 | yes | 0.000e+00 |

## SIMP driver verification

- filter adjoint identity, relative error: **1.274e-16**
- fused adjoint kernel vs materialised three-stage, rel L2: **2.528e-14**
- finite-difference sensitivity check, worst relative error over 12 probed elements: **9.045e-08** (h = 0.0001)
- achieved physical volume fraction, max deviation from the prescribed value: **1.10e-09**
- cross-path compliance spread: **1.207e-08**

## Traffic model (logical and compulsory bounds)

| path | vectors | K_e | E_e | index | logical B/elem | compulsory B/elem | FLOP/elem | I_logical | I_compulsory |
|---|---|---|---|---|---|---|---|---|---|
| three-stage FP64 | fp64 | fp64 | fp64 | int32 | 1352 | 920 | 1176 | 0.87 | 1.28 |
| three_stage_fp64_e32 | fp64 | fp64 | fp32 | int32 | 1348 | 916 | 1176 | 0.87 | 1.28 |
| three-stage FP32 | fp32 | fp32 | fp32 | int32 | 772 | 508 | 1176 | 1.52 | 2.31 |
| three-stage FP32 (FP64 scatter) | fp32 | fp32 | fp32 | int64 | 964 | 604 | 1176 | 1.22 | 1.95 |
| fused FP64 | fp64 | fp64 | fp64 | int32 | 488 | 152 | 1176 | 2.41 | 7.74 |
| fused FP32 | fp32 | fp32 | fp32 | int32 | 292 | 124 | 1176 | 4.03 | 9.48 |
| fused FP64 + analytic index | fp64 | fp64 | fp64 | none | 392 | 56 | 1176 | 3.00 | 21.00 |
| fused FP32 + analytic index | fp32 | fp32 | fp32 | none | 196 | 28 | 1176 | 6.00 | 42.00 |
| node-owned FP64 | fp64 | fp64 | fp64 | none | 1624 | 56 | 1176 | 0.72 | 21.00 |
| node-owned FP32 | fp32 | fp32 | fp32 | none | 812 | 28 | 1176 | 1.45 | 42.00 |

## Operator microbenchmark

Protocol: median of `blocks` CUDA-event blocks, each auto-sized to ~target_ms of GPU work, after warmup_ms of steady-state warm-up; SM/mem clock, temperature and power sampled around every block; GPU clocks NOT locked (shared with the desktop compositor)

| mesh | path | us/application (median) | IQR % | range % | speedup vs three-stage FP64 |
|---|---|---|---|---|---|
| 64k | three-stage FP64 | 353.1 | 9.4 | 16.1 | 1.00 |
| 64k | three_stage_fp64_e32 | 343.9 | 4.7 | 12.0 | 1.03 |
| 64k | three-stage FP32 | 168.3 | 23.6 | 39.9 | 2.10 |
| 64k | three-stage FP32 (FP64 scatter) | 341.1 | 18.5 | 53.4 | 1.04 |
| 64k | fused FP64 | 67.0 | 1.0 | 8.6 | 5.27 |
| 64k | fused FP32 | 33.5 | 12.8 | 37.9 | 10.54 |
| 64k | fused FP64 + analytic index | 65.0 | 0.3 | 4.8 | 5.44 |
| 64k | fused FP32 + analytic index | 42.2 | 29.7 | 50.0 | 8.37 |
| 64k | node-owned FP64 | 99.4 | 0.6 | 3.9 | 3.55 |
| 64k | node-owned FP32 | 34.3 | 16.0 | 36.6 | 10.29 |
| 216k | three-stage FP64 | 893.4 | 2.1 | 4.2 | 1.00 |
| 216k | three_stage_fp64_e32 | 888.5 | 2.8 | 9.3 | 1.01 |
| 216k | three-stage FP32 | 171.5 | 11.3 | 40.8 | 5.21 |
| 216k | three-stage FP32 (FP64 scatter) | 413.7 | 8.7 | 14.5 | 2.16 |
| 216k | fused FP64 | 213.9 | 0.1 | 4.3 | 4.18 |
| 216k | fused FP32 | 65.4 | 0.5 | 6.3 | 13.66 |
| 216k | fused FP64 + analytic index | 211.6 | 11.8 | 17.0 | 4.22 |
| 216k | fused FP32 + analytic index | 52.5 | 10.8 | 20.2 | 17.01 |
| 216k | node-owned FP64 | 273.1 | 0.4 | 1.0 | 3.27 |
| 216k | node-owned FP32 | 30.2 | 9.9 | 27.2 | 29.54 |
| 512k | three-stage FP64 | 2114.5 | 0.8 | 2.5 | 1.00 |
| 512k | three_stage_fp64_e32 | 2126.2 | 0.7 | 4.2 | 0.99 |
| 512k | three-stage FP32 | 558.1 | 0.7 | 10.9 | 3.79 |
| 512k | three-stage FP32 (FP64 scatter) | 958.6 | 1.5 | 2.9 | 2.21 |
| 512k | fused FP64 | 487.9 | 0.1 | 5.7 | 4.33 |
| 512k | fused FP32 | 161.4 | 5.2 | 22.5 | 13.10 |
| 512k | fused FP64 + analytic index | 476.9 | 0.3 | 4.5 | 4.43 |
| 512k | fused FP32 + analytic index | 106.7 | 0.7 | 1.7 | 19.82 |
| 512k | node-owned FP64 | 640.1 | 0.9 | 5.3 | 3.30 |
| 512k | node-owned FP32 | 52.1 | 0.1 | 5.3 | 40.61 |
| 1M | three-stage FP64 | 4066.2 | 1.0 | 5.5 | 1.00 |
| 1M | three_stage_fp64_e32 | 4071.8 | 0.2 | 2.7 | 1.00 |
| 1M | three-stage FP32 | 1170.2 | 0.6 | 5.2 | 3.47 |
| 1M | three-stage FP32 (FP64 scatter) | 1932.5 | 0.5 | 5.5 | 2.10 |
| 1M | fused FP64 | 932.9 | 0.5 | 4.3 | 4.36 |
| 1M | fused FP32 | 309.0 | 0.5 | 1.3 | 13.16 |
| 1M | fused FP64 + analytic index | 917.9 | 0.5 | 4.8 | 4.43 |
| 1M | fused FP32 + analytic index | 198.6 | 0.5 | 0.9 | 20.47 |
| 1M | node-owned FP64 | 1215.5 | 0.6 | 0.9 | 3.35 |
| 1M | node-owned FP32 | 94.1 | 1.2 | 7.0 | 43.22 |
| 2M | three-stage FP64 | 8371.2 | 0.8 | 1.7 | 1.00 |
| 2M | three_stage_fp64_e32 | 8357.6 | 1.0 | 5.3 | 1.00 |
| 2M | three-stage FP32 | 2370.8 | 0.9 | 5.1 | 3.53 |
| 2M | three-stage FP32 (FP64 scatter) | 3812.5 | 0.5 | 6.5 | 2.20 |
| 2M | fused FP64 | 1854.8 | 1.0 | 4.9 | 4.51 |
| 2M | fused FP32 | 608.2 | 0.7 | 5.9 | 13.76 |
| 2M | fused FP64 + analytic index | 1818.6 | 0.1 | 1.2 | 4.60 |
| 2M | fused FP32 + analytic index | 457.2 | 0.4 | 3.5 | 18.31 |
| 2M | node-owned FP64 | 2402.8 | 0.4 | 4.2 | 3.48 |
| 2M | node-owned FP32 | 251.6 | 0.6 | 5.1 | 33.27 |

## End-to-end optimization runs (`s1`)

20 of 20 valid rows terminated on the declared outer-convergence criterion. Every linear solve in every valid row reached the 1e-5 relative residual tolerance, measured on the true residual ||b-Ax||/||b||. A further 16 run(s) are invalid: their first equilibrium solve could not reach the tolerance before the iteration cap, and they are listed separately below with the residual each one attained.

| mesh | path | outer iters | converged | compliance | V_phys | grayness | CG total | max resid | wall s | solve share | peak GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 216k | fused FP64 | 70 | yes | 1.478193 | 0.3000 | 1.92e-08 | 33800 | 9.91e-06 | 16.9 | 0.94 | 1.65 |
| 216k | refined fused FP32 + analytic index | 67 | yes | 1.478204 | 0.3000 | 2.61e-06 | 57100 | 9.11e-06 | 15.7 | 0.94 | 1.67 |
| 216k | refined fused FP32 | 66 | yes | 1.478236 | 0.3000 | 1.01e-07 | 56850 | 9.86e-06 | 16.8 | 0.95 | 1.69 |
| 216k | refined node-owned FP32 | 67 | yes | 1.478242 | 0.3000 | 5.68e-06 | 58450 | 9.51e-06 | 15.2 | 0.94 | 1.67 |
| 216k | three-stage FP64 | 70 | yes | 1.478193 | 0.3000 | 2.21e-08 | 34050 | 9.84e-06 | 41.5 | 0.97 | 1.83 |
| 512k | fused FP64 | 67 | yes | 1.115408 | 0.3000 | 2.31e-06 | 43250 | 9.64e-06 | 37.2 | 0.96 | 1.83 |
| 512k | refined fused FP32 + analytic index | 67 | yes | 1.115410 | 0.3000 | 2.32e-06 | 79800 | 9.36e-06 | 28.1 | 0.95 | 1.87 |
| 512k | refined fused FP32 | 67 | yes | 1.115410 | 0.3000 | 2.35e-06 | 79750 | 9.41e-06 | 35.1 | 0.96 | 1.92 |
| 512k | refined node-owned FP32 | 67 | yes | 1.115410 | 0.3000 | 2.37e-06 | 81050 | 9.37e-06 | 23.7 | 0.94 | 1.87 |
| 512k | three-stage FP64 | 67 | yes | 1.115408 | 0.3000 | 2.37e-06 | 43300 | 9.83e-06 | 111.8 | 0.99 | 2.28 |
| 1M | fused FP64 | 72 | yes | 0.893160 | 0.3000 | 1.25e-08 | 55950 | 9.73e-06 | 91.6 | 0.97 | 2.14 |
| 1M | refined fused FP32 + analytic index | 72 | yes | 0.893151 | 0.3000 | 1.69e-08 | 99200 | 9.96e-06 | 51.4 | 0.95 | 2.17 |
| 1M | refined fused FP32 | 71 | yes | 0.893157 | 0.3000 | 2.00e-08 | 102650 | 9.97e-06 | 63.6 | 0.96 | 2.27 |
| 1M | refined node-owned FP32 | 71 | yes | 0.893156 | 0.3000 | 1.93e-08 | 101650 | 9.74e-06 | 40.2 | 0.94 | 2.18 |
| 1M | three-stage FP64 | 71 | yes | 0.893158 | 0.3000 | 1.85e-08 | 55300 | 9.93e-06 | 268.9 | 0.99 | 3.00 |
| 2M | fused FP64 | 66 | yes | 0.713228 | 0.3000 | 1.65e-06 | 66100 | 9.92e-06 | 250.7 | 0.98 | 2.77 |
| 2M | refined fused FP32 + analytic index | 76 | yes | 0.713196 | 0.3000 | 3.32e-07 | 144000 | 9.64e-06 | 138.6 | 0.97 | 2.86 |
| 2M | refined fused FP32 | 75 | yes | 0.713200 | 0.3000 | 3.69e-07 | 139600 | 9.91e-06 | 165.0 | 0.97 | 3.07 |
| 2M | refined node-owned FP32 | 73 | yes | 0.713195 | 0.3000 | 3.30e-07 | 142250 | 9.97e-06 | 111.3 | 0.96 | 2.86 |
| 2M | three-stage FP64 | 66 | yes | 0.713228 | 0.3000 | 1.66e-06 | 66100 | 9.96e-06 | 686.9 | 0.99 | 4.49 |

### Runs invalidated by the fail-closed rule

These are not slow results; they are non-results. Each exhausted the 20,000-iteration cap on its first equilibrium solve without reaching 1e-5 on the true residual, so no design iteration was ever taken.

| mesh | path | CG iters | residual attained | wall s |
|---|---|---|---|---|
| 1M | fused FP32 + analytic index | 20000 | 4.528e-03 | 9.2 |
| 1M | fused FP32 | 20000 | 4.542e-03 | 11.9 |
| 1M | node-owned FP32 | 20000 | 4.637e-03 | 7.1 |
| 1M | three-stage FP32 | 20000 | 4.514e-03 | 30.3 |
| 216k | fused FP32 + analytic index | 20000 | 1.234e-03 | 5.2 |
| 216k | fused FP32 | 20000 | 1.251e-03 | 5.6 |
| 216k | node-owned FP32 | 20000 | 1.295e-03 | 5.0 |
| 216k | three-stage FP32 | 20000 | 1.258e-03 | 7.8 |
| 2M | fused FP32 + analytic index | 20000 | 6.000e-03 | 19.3 |
| 2M | fused FP32 | 20000 | 6.098e-03 | 22.9 |
| 2M | node-owned FP32 | 20000 | 6.254e-03 | 14.9 |
| 2M | three-stage FP32 | 20000 | 6.097e-03 | 58.7 |
| 512k | fused FP32 + analytic index | 20000 | 2.669e-03 | 6.3 |
| 512k | fused FP32 | 20000 | 2.661e-03 | 8.0 |
| 512k | node-owned FP32 | 20000 | 2.705e-03 | 5.1 |
| 512k | three-stage FP32 | 20000 | 2.654e-03 | 18.3 |

### Decomposed end-to-end effect (wall-time ratio)

Medians over repeated complete optimizations.

| effect | 216k | 512k | 1M | 2M |
|---|---|---|---|---|
| fusion at matched FP64 | 2.46x | 3.00x | 2.94x | 2.74x |
| mixed precision, by refinement | 1.00x | 1.06x | 1.44x | 1.52x |
| analytic indexing, inside refinement | 1.07x | 1.25x | 1.24x | 1.19x |
| node ownership, inside refinement | 1.03x | 1.19x | 1.28x | 1.25x |
| all four, compounded | 2.73x | 4.72x | 6.69x | 6.17x |

## Converged cold-start FEA ladder

Iterations and achieved residual for every point.

| mesh | path | CG iters | achieved rel. residual | at cap | time to tolerance s | us per application |
|---|---|---|---|---|---|---|
| 216k | three-stage FP64 | 550 | 2.843e-06 | no | 0.64 | 903.1 |
| 216k | three-stage FP32 | 20000 | 1.361e-03 | YES | 8.32 | 150.9 |
| 216k | fused FP64 | 550 | 2.850e-06 | no | 0.25 | 223.6 |
| 216k | fused FP32 | 20000 | 1.384e-03 | YES | 5.54 | 95.4 |
| 216k | fused FP32 + analytic index | 20000 | 1.345e-03 | YES | 5.08 | 53.2 |
| 216k | node-owned FP32 | 20000 | 1.435e-03 | YES | 4.86 | 33.3 |
| 512k | three-stage FP64 | 750 | 9.671e-07 | no | 1.88 | 2157.0 |
| 512k | three-stage FP32 | 20000 | 2.566e-03 | YES | 18.14 | 643.3 |
| 512k | fused FP64 | 750 | 9.678e-07 | no | 0.66 | 539.6 |
| 512k | fused FP32 | 20000 | 2.548e-03 | YES | 8.15 | 164.6 |
| 512k | fused FP32 + analytic index | 20000 | 2.550e-03 | YES | 6.30 | 109.7 |
| 512k | node-owned FP32 | 20000 | 2.594e-03 | YES | 5.20 | 56.3 |
| 1M | three-stage FP64 | 900 | 5.693e-06 | no | 4.39 | 4190.2 |
| 1M | three-stage FP32 | 20000 | 4.524e-03 | YES | 30.51 | 1180.9 |
| 1M | fused FP64 | 900 | 5.698e-06 | no | 1.47 | 1078.4 |
| 1M | fused FP32 | 20000 | 4.540e-03 | YES | 12.22 | 302.9 |
| 1M | fused FP32 + analytic index | 20000 | 4.535e-03 | YES | 10.07 | 203.3 |
| 1M | node-owned FP32 | 20000 | 4.628e-03 | YES | 7.24 | 101.2 |
| 2M | three-stage FP64 | 1150 | 3.091e-06 | no | 11.95 | 8685.6 |
| 2M | three-stage FP32 | 20000 | 6.627e-03 | YES | 58.70 | 2361.9 |
| 2M | fused FP64 | 1150 | 3.092e-06 | no | 4.29 | 2186.2 |
| 2M | fused FP32 | 20000 | 6.666e-03 | YES | 22.78 | 621.6 |
| 2M | fused FP32 + analytic index | 20000 | 6.569e-03 | YES | 19.29 | 451.2 |
| 2M | node-owned FP32 | 20000 | 6.668e-03 | YES | 15.83 | 273.0 |

## Mesh-comparability series

Fixed physical filter radius and patch area, run on both the double-precision control and the refined path. No mesh-refinement claim is drawn from this series: no mesh satisfies the design criterion within the guard.

| mesh | path | elements | rmin (elem) | filter nbrs | compliance | V_phys | outer iters | design converged |
|---|---|---|---|---|---|---|---|---|
| 216k | fused FP64 | 216000 | 1.50 | 19 | 1.4591776428 | 0.300000 | 1200 | no |
| 216k | refined node-owned FP32 | 216000 | 1.50 | 19 | 1.4591776828 | 0.300000 | 1200 | no |
| 512k | fused FP64 | 512000 | 2.00 | 27 | 1.0870530469 | 0.300000 | 1200 | no |
| 512k | refined node-owned FP32 | 512000 | 2.00 | 27 | 1.0870293514 | 0.300000 | 1200 | no |
| 1M | fused FP64 | 1000000 | 2.50 | 81 | 0.8678619098 | 0.300000 | 1200 | no |
| 1M | refined node-owned FP32 | 1000000 | 2.50 | 81 | 0.8678528415 | 0.300000 | 1200 | no |

## Condition-number bounds

Ritz values of a Lanczos run lie inside the spectrum, so the ratio bounds kappa from below. The power-iteration column is an independent check that the run resolved the top of the spectrum.

| mesh | penal | theta_max | lambda_max (power) | ratio | theta_min | kappa >= | eps_bf16*kappa >= | steps |
|---|---|---|---|---|---|---|---|---|
| 64k | 3 | 1.340022 | 1.335409 | 1.0035 | 2.1563e-06 | 6.2145e+05 | 2.4275e+03 | 3000 |
| 64k | 5 | 0.335005 | 0.333852 | 1.0035 | 5.3907e-07 | 6.2145e+05 | 2.4275e+03 | 3000 |
| 216k | 3 | 1.343381 | 1.338845 | 1.0034 | 9.8892e-07 | 1.3584e+06 | 5.3064e+03 | 3000 |
| 216k | 5 | 0.335845 | 0.334711 | 1.0034 | 2.4723e-07 | 1.3584e+06 | 5.3064e+03 | 3000 |
| 512k | 3 | 1.344581 | 1.339164 | 1.0040 | 5.6523e-07 | 2.3788e+06 | 9.2923e+03 | 3000 |
| 512k | 5 | 0.336145 | 0.334791 | 1.0040 | 1.4131e-07 | 2.3788e+06 | 9.2923e+03 | 3000 |

## BF16 precision boundary

Reference is an FP64 solve verified to converge. Note the two error columns disagree by orders of magnitude on the same solve: the residual is what the acceptance rule tests.

| mesh | solver | CG iters | true rel. residual | compliance | rel. error in c |
|---|---|---|---|---|---|
| 64k | FP64 reference (fp64) | 400 | 5.906e-08 | 8.111850 | -- |
| | plain BF16 CG | 4000 | 9.009e+00 | 1.733760 | 0.7863 |
| | BF16 refined, inner 1e-3 | 16000 | 6.396e+00 | 7.546199 | 0.0697 |
| | BF16 refined, inner 1e-5 | 16000 | 7.586e+00 | 8.121841 | 0.0012 |
| 216k | FP64 reference (fp64) | 600 | 5.655e-08 | 5.443752 | -- |
| | plain BF16 CG | 4000 | 8.194e+00 | 0.969779 | 0.8219 |
| | BF16 refined, inner 1e-3 | 16000 | 3.236e+01 | 3.231223 | 0.4064 |
| | BF16 refined, inner 1e-5 | 16000 | 3.589e+01 | 3.540500 | 0.3496 |
| 512k | FP64 reference (fp64) | 750 | 9.673e-07 | 4.097754 | -- |
| | plain BF16 CG | 4000 | 1.002e+01 | 0.769922 | 0.8121 |
| | BF16 refined, inner 1e-3 | 16000 | 6.151e+01 | -0.962541 | 1.2349 |
| | BF16 refined, inner 1e-5 | 16000 | 4.479e+01 | 0.578119 | 0.8589 |

## Stiffness-floor sweep (512k)

| Emin/E0 | outer iters | mean CG/solve | max CG | compliance | V_phys | grayness | binary mismatch vs ref | min projected density | min element stiffness |
|---|---|---|---|---|---|---|---|---|---|
| 1e-09 | 67 | 636.8 | 1450 | 1.115408 | 0.3000 | 2.29e-06 | 0.00000 | 2.837e-14 | 1.000e-09 |
| 1e-06 | 67 | 636.8 | 1450 | 1.115375 | 0.3000 | 2.20e-06 | 0.00006 | 2.848e-14 | 1.000e-06 |
| 0.001 | 67 | 619.1 | 1350 | 1.106306 | 0.3000 | 4.46e-06 | 0.02808 | 2.592e-14 | 1.000e-03 |

Finite-difference sensitivity spot check per floor:

- Emin/E0 = 1e-09: worst relative error 8.522e-06
- Emin/E0 = 1e-06: worst relative error 1.080e-05
- Emin/E0 = 0.001: worst relative error 7.506e-06
