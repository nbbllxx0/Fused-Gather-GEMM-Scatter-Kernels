# Release manifest

66 files, 65.5 MB. Every tracked file except `MANIFEST.md`, which
cannot list its own hash.

23 density fields (`*.npy`, ~180 MB) are not included; the 6 behind the
topology comparison are. `.gitignore` excludes `*.npy`, so those six are
tracked deliberately -- regenerate this file with the same command if you
add more.

| File | Bytes | SHA-256 |
|---|---:|---|
| `.gitattributes` | 115 | `36c7c81f6ec2501e40ea21e6d2c742a9ffcf121fd329c17c0d20b34f81182623` |
| `.gitignore` | 77 | `696f32dd1a899b6bbd9d01b4d9d9693e1d45ced439536878fc0a8484d2303ec4` |
| `CITATION.bib` | 369 | `bc5f256f548236aaf3ee549435f4427edf1f50bb1911030c844da2ae9da74125` |
| `CITATION.cff` | 744 | `3289be00467853966ad7ec0ad96e1d4726f45f2d9040bbe3b30fa37d052a3f9d` |
| `LICENSE` | 1,551 | `2cbf0d0aeb3004fb99958b857f6a7bcc05c512d2a97d514c23ac1ec15dd8b15f` |
| `README.md` | 6,120 | `92c1660a4f84904b6615c3e97c67f8b4efa10b565ed711104dfa9d80ff71e7d7` |
| `experiments/coldstart_ladder.py` | 8,467 | `8558407c87b59896cf9e02aeb8b1a2e4a349352b8c9f29e6dfc55189839fbebb` |
| `experiments/floor_sweep.py` | 9,757 | `a3b6d2024cbe6d8f4a1944dc4b080add59f320d90da62ed2f79b4de74ddcc7fd` |
| `experiments/ir_tuning.py` | 6,484 | `6b21fdd3f18f3e59652ada6a86725cfa43f636666bc028d66a4376115d715c38` |
| `experiments/kappa_bf16_study.py` | 14,356 | `82e86a2fad288f45a044e6a4d51b1acfafb82e5b378ae8c2c710b160fe7e8197` |
| `experiments/kappa_lanczos.py` | 7,949 | `64269b37a036c82a96ae9bae876e2d01662a28cc710acb71346e2a9a620f9f50` |
| `experiments/ladder_repeats.py` | 4,667 | `65e5bc253049e1c47400aa5f47c3f497e7a83681cb7613598e45a63e63417cca` |
| `experiments/ncu_dram_profile.py` | 6,548 | `38707c3487cb42e431a2f2fa1e36c4da0fb4cdd74c7e20e3da1320ea1cc4528e` |
| `experiments/ncu_kernel_probe.py` | 2,568 | `70988b8dc897b6eb87418bd3affb90f531b89e84e26945a2644e90a64cbb6157` |
| `experiments/operator_benchmark.py` | 13,354 | `0a3c38d663a243e5da8f0fd1fba71b720e0562d011e2e20bf016b7360278a1b9` |
| `experiments/precision_pilot.py` | 8,982 | `a450f4e1764613df952ee9b87d54a07aadf6c5b95c7091604252038c2033bd2b` |
| `experiments/run_ladder_v11.py` | 7,521 | `3d62d8c3fd9ec172ae896f008ecdfd55ca48a5f097f70ed62a1e894d85f0dc94` |
| `pyproject.toml` | 518 | `0a4f01ad7502779b17fe897134971c6671b932824aa14ca8ed90dfd72684680d` |
| `requirements.txt` | 206 | `250b6e6f4657fd729985d8e0c1242b15cd5ffeb20639e815357136db4d336f26` |
| `results/G0/environment.json` | 1,380 | `c875cd107bdd17cc478736255ad19d1d2232e7afb78a9fbd6a9e67f6b7df06d9` |
| `results/G0/repro_environment_v2.txt` | 1,784 | `4e68bc2176245c9964dfe9b3a92a6c1603ec2ead37905adb02a9084cd8a78b40` |
| `results/G1/operator_parity.json` | 3,903 | `374a4aa21dfdb11c0e7b6740c5d5e35a1d4790679165692f59d25c4717b21dcd` |
| `results/G1/operator_parity_large.json` | 3,908 | `e65e1b1a985aacf7d6a2cb56e8dbba6d4e4c763a4dae1c0de255de867ffaab4d` |
| `results/G1/simp_verification.json` | 3,416 | `b0a2e50217b19b35f5af5db7cf036e1fd320ac36ecfa6d90dd6c79b0ba6e839c` |
| `results/G1/smoke_operator.csv` | 2,491 | `d1284b742f2feea86b50845c685c70a5e14fbc3d7910a44f76acc170c7f45afc` |
| `results/G1/smoke_operator.json` | 16,453 | `d351354de325c430033a10b5922ab41046b985827341de6e235dc165f960ea16` |
| `results/G3/operator_benchmark.csv` | 14,431 | `6825eea464bf9a9b82b89f16cc5b27f8678a287edec5bc925be4e59553809853` |
| `results/G3/operator_benchmark.json` | 86,983 | `672d1d2496474982921784e867de48ad5e4f59425475890e6437ea41a6916db8` |
| `results/G6/bf16_study.json` | 3,083 | `abe79f81ec50f11300b7c1d7bbd7785039d8106bc58b86492c2d7a752bc19e2b` |
| `results/G6/coldstart_ladder.csv` | 4,926 | `447f14a2d8d89fda940dc02e5315e8c0c4a230fdbfefc0448ebfe5452bf96f3f` |
| `results/G6/coldstart_ladder.json` | 14,656 | `4089d308d7ae0c53038caff0fe3df3e54098d8abbd578c60e2de4d6d00e4ae96` |
| `results/G6/floor_512k_0.001.json` | 70,182 | `30f44e37ef9c9f7f48dd71e801be5a7cc3aff65ea746181ad945f390e7b62af1` |
| `results/G6/floor_512k_1e-06.json` | 70,252 | `725161108ded861be0ccc619791e512cf3e41e5a025046f5715a9e87a853aa1e` |
| `results/G6/floor_512k_1e-09.json` | 70,229 | `6e64d9f16cea55aa16e89abd9ad9742d1eb6d15763cd271b684adce05b16e624` |
| `results/G6/floor_sweep_512k.json` | 5,842 | `4ef57cc0819235589a833679423284a18739cd50213f8a0ab6353b55b551ab57` |
| `results/G6/ir_tuning.json` | 2,643 | `44a09957af86073b5c4503cb162a89216848b704603762b4050fdf1a225fc75a` |
| `results/G6/ir_tuning_2M.json` | 1,283 | `9878e9aa21fb25a1f214388097587b5e51b9f0471773497f4b826e86fe57d34e` |
| `results/G6/kappa_estimation.json` | 3,471 | `0775fef32753671b7ae1450c94d22a640077c4df82428835d51b1dd3bf876ad4` |
| `results/G6/ladder_comparability.json` | 6,780,254 | `a9860c769f9066185f0a1c9906ccb489d52f9a9b4e4bdb01a272782021f8d3bc` |
| `results/G6/ladder_rep.json` | 18,306 | `19ce27a8271f669c1858194c3bb7bdae11aeb3a1c5ead12e09dade71be506b2b` |
| `results/G6/ladder_s1.json` | 1,361,901 | `bad0772b41b25e1581f613f0197b4f505f322031999e2524172b30ea7d36faf5` |
| `results/G6/precision_pilot.json` | 652 | `81171118b01dd6021f702d09802419483e0c4ef07b32226232ca732254178bd8` |
| `results/G6/precision_pilot_big.json` | 1,296 | `13cc9af6718703bf77cfa6bd84acbf16903b181fe22f76359b4e38be0a41e05b` |
| `results/G6/repeat_512k.json` | 826 | `fedbc831cff119d7d378d518e7a16910fc3906869096cb47c240bf0e4c38fa03` |
| `results/G6/run_s1_1M_fused_fp64_rho_final_device.npy` | 8,000,128 | `542cfc01756dccb9f7ae83886e46de2cfb15081ec2bafbaab99e08aa3cf2681d` |
| `results/G6/run_s1_1M_ir_node_fp32_rho_final_device.npy` | 8,000,128 | `9fcf608a6fddcac96a131bd6a7dcf5af93055e965b058fc00a5ee8573a761908` |
| `results/G6/run_s1_1M_three_stage_fp64_rho_final_device.npy` | 8,000,128 | `4312cae32046f7dbbbde556d706517833e06160428be49f4e0d45f62e148fbac` |
| `results/G6/run_s1_216k_fused_fp64_rho_final_device.npy` | 1,728,128 | `533ef55606b18213304db128b6c08e0449c29573b89948a57eaf56452f82b8b3` |
| `results/G6/run_s1_2M_fused_fp64_rho_final_device.npy` | 16,003,136 | `c341f7d6744a5043a8d1d16b100023694fff98c1f8eac2546d48b1dff9029707` |
| `results/G6/run_s1_512k_fused_fp64_rho_final_device.npy` | 4,096,128 | `6705cdd63740173911c400b1eca06ddf8d68fc25794351fe191d143691aed8e6` |
| `results/NUMBERS.json` | 10,858,534 | `bca008fab7a60bb45991fc83a71f07e9b096910214d46d36f771487d02dcfbeb` |
| `results/NUMBERS.md` | 19,275 | `8d60d412622992838b1b15fe3466de843ae597aac6f41056775f88d09bb43ff5` |
| `results/baseline_r1/README.md` | 1,270 | `7e2d9e5a7712cd80095c6326370c502d9c7e81affa0a0bead7c10e2992850e0d` |
| `results/baseline_r1/fully_converged_study.csv` | 646 | `7d3bd333f2e6dd70552185a2bf12ffdd19b6a42247e315f02507d303a5d1f1dd` |
| `results/baseline_r1/scaling_ladder_simp_mid.csv` | 522 | `1953ddac0b06378325be16e150229b76e1cb1956ee90c99db278763cee6f485a` |
| `results/baseline_r1/statistical_repeats.csv` | 2,269 | `f7a2859a676b6ee594f4ab8819f5182ab84e707309f5a2a304626457110af5f1` |
| `scripts/collect_numbers.py` | 22,592 | `c186d365ea1f3fe1e0467571262f279dada34b0bc7f8678763b686499d196581` |
| `src/gpu_fem/__init__.py` | 434 | `fb18a505323cb9e0f771ccf75be3d1a678f7851cf739a03b58c701e016c7f06d` |
| `src/gpu_fem/cuda_fused_matvec.py` | 20,411 | `7e9f7002451b1b118243410ad62a3a6f19e9fc5c099ad2e68ca6af0575f2aa0f` |
| `src/gpu_fem/cuda_operators.py` | 33,599 | `d36cc55cf762e6f701056aeecc18977f8f4e0914be278c6971162fca0db7dc6f` |
| `src/gpu_fem/filter_r2.py` | 6,639 | `2fc74c7a0d0da47adf9ee8154d20682b95c3fd5292f008dbf32cff41a124b88b` |
| `src/gpu_fem/pub_simp_solver.py` | 26,956 | `37263bd5e6dcfee2ab3469bfd4172dcad7208433bf7cd0af6cab392deaf1073d` |
| `src/gpu_fem/simp_r2.py` | 42,887 | `c29840e8739c4e992587794c4b9a009f9a5453a3e8b2c551f870a2e3abf6ca3d` |
| `tools/record_environment.py` | 4,248 | `e7791b0e0fca815e27d52d290df2868497480f8cb7bb1a20103def8ad782b587` |
| `tools/verify_operators.py` | 5,544 | `8ec6d526f46850c9af90b05d694b31270fd1cf23c452599b976a06139171835f` |
| `tools/verify_simp.py` | 11,111 | `974a1eaa3727a2950b6370ca2fe4e9c03ec43cee8f3516266eb7f626b819773e` |
