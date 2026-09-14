# Provisional E7 execution results (Phase 3, plan_2.md §6)

> **Provisional historical evidence — not valid fairness proof.** The takeover's
> independent audit reports queue-accounting/benchmark contamination (seed 1257:
> maker reduction 33 units, credited quantity 13) and action-dependent RNG
> consumption in `envs/order_book_env.py:265–291,427–438`. Identical episode seeds
> therefore do not guarantee common exogenous tapes. These comparisons and CIs
> must not support fairness or superiority claims. The takeover will fix and
> verify the environment, then retrain and reevaluate. Original execution
> provenance and numeric results are retained below; the ITCH study is unaffected.

Training regime **highvol**, hold-outs calm, lowvol, highvol_null, trending, liquidity_shock. 5 training seeds × 5 eval seed families × 20 episodes; PPO 600 iters × 16 episodes/iter. Fees + queue model **on** for every number. Slippage positive = a cost (bps).

`Δ` records the per-episode difference `baseline − PPO` under the same episode seeds (positive = lower measured PPO cost), with a whole-seed-family bootstrap 95% CI. The audit invalidates the common-tape assumption; `sig` is only the historical count of intervals excluding zero, not valid fairness evidence.

The best baseline is selected separately per regime by its smallest baseline mean on these same evaluation families. Therefore the paired `Δ vs best` intervals are descriptive and do not adjust for winner selection. This synthetic study does not revive the retired +50.4% exploratory claim.

## Reproducibility

- Git HEAD at launch: `4b108622d9abb23794858eb534d0d642e3512f99`.
- E7 source SHA-256: `a61b634020768eb0eddafb4ace38ea8225cc17804055d074eac6e82010e81b0e`.
- Exact command: `/tmp/hoplite/workspace/.venv/bin/python python_quant/scripts/rl_fairness_study.py --iters 600 --episodes 16 --train-seeds 5 --eval-seeds 5 --episodes-per-seed 20 --n-boot 2000 --artifacts python_quant/artifacts/fairness/current_e9b4f68_r2 --out docs/results/rl_fairness.json`.
- Training: 5 policies per mode; 600 iterations × 16 episodes/iteration × 4 epochs.
- Evaluation: 5 seed families × 20 episodes, seed0 `24301`, family stride `10000`; exact seed lists are in the JSON.
- Bootstrap: 2000 whole-family replicates per CI.
- Runtime: 889.6 s (2026-09-14T12:00:42.119591+00:00 to 2026-09-14T12:15:31.754908+00:00).
- Environment: Python 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]; NumPy 2.5.3; Gymnasium 1.3.0; thread limits OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1, BLIS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1.
- Policy cache: `python_quant/artifacts/fairness/current_e9b4f68_r2` (fresh, manifest `fairness_policy_cache_manifest.npz`).

## Scope and limitations

- This is a synthetic OrderBookEnv study, not a real NASDAQ ITCH execution result.
- PPO trains only on highvol; the other five regimes are simulated hold-outs.
- The best baseline is selected separately per regime from these same evaluation families; the paired CI does not adjust for that winner selection.
- These results do not revive the retired +50.4% exploratory claim or establish real-tape superiority.

## Mode `novol` — nobody observes the regime flag (obs dim 44)

| regime | PPO shortfall (seed-mean ± seed-std) | best baseline | Δ vs best [95% CI] per training seed | sig better / worse | Δ vs VWAP (mean over seeds) | PPO completion† | PPO fill rate† | PPO max DD (ticks)† | PPO slip vs market VWAP |
|---|---|---|---|---|---|---|---|---|---|
| highvol | 2.612 ± 0.106 | adaptive_pov 2.678 | -0.06 [-0.24,+0.13] +0.01 [-0.10,+0.11] +0.19 [-0.01,+0.38] +0.20 [+0.13,+0.26] -0.01 [-0.24,+0.20] | 1 / 0 | +0.737 bps (+22.0%) | 0.977 ± 0.011 [0.948,0.996] | 0.977 ± 0.011 [0.948,0.996] | 9914.63 ± 186.24 [8994.70,10678.57] | +1.260 |
| calm | 1.622 ± 0.016 | twap 1.471 | -0.18 [-0.25,-0.13] -0.14 [-0.21,-0.08] -0.14 [-0.19,-0.08] -0.15 [-0.19,-0.10] -0.16 [-0.23,-0.09] | 0 / 5 | +0.013 bps (+0.8%) | 1.000 ± 0.000 [1.000,1.000] | 1.000 ± 0.000 [1.000,1.000] | 4771.37 ± 16.34 [4582.92,5041.66] | +1.042 |
| lowvol | 0.972 ± 0.009 | adaptive_pov 0.896 | -0.06 [-0.07,-0.06] -0.08 [-0.08,-0.07] -0.07 [-0.09,-0.06] -0.09 [-0.10,-0.08] -0.08 [-0.08,-0.07] | 0 / 5 | -0.049 bps (-5.3%) | 1.000 ± 0.000 [1.000,1.000] | 1.000 ± 0.000 [1.000,1.000] | 2789.80 ± 17.70 [2708.09,2875.93] | +0.792 |
| highvol_null | 2.441 ± 0.125 | adaptive_pov 2.716 | +0.22 [+0.10,+0.34] +0.16 [-0.03,+0.33] +0.49 [+0.43,+0.55] +0.33 [+0.25,+0.44] +0.17 [+0.03,+0.33] | 4 / 0 | +0.860 bps (+26.1%) | 0.978 ± 0.011 [0.950,0.996] | 0.978 ± 0.011 [0.950,0.996] | 9904.02 ± 119.98 [9393.10,10417.25] | +1.554 |
| trending | 2.903 ± 0.052 | adaptive_pov 3.013 | +0.20 [-0.07,+0.44] +0.07 [-0.12,+0.26] +0.08 [-0.24,+0.31] +0.14 [-0.07,+0.34] +0.06 [-0.09,+0.20] | 0 / 0 | +0.484 bps (+14.3%) | 0.971 ± 0.026 [0.904,0.995] | 0.971 ± 0.026 [0.904,0.995] | 10701.20 ± 188.62 [9624.25,11691.08] | +0.561 |
| liquidity_shock | 3.014 ± 0.131 | adaptive_pov 3.751 | +0.52 [+0.23,+0.75] +0.66 [+0.41,+0.88] +0.85 [+0.38,+1.23] +0.86 [+0.57,+1.14] +0.81 [+0.38,+1.23] | 5 / 0 | +1.604 bps (+34.7%) | 0.932 ± 0.025 [0.869,0.970] | 0.932 ± 0.025 [0.869,0.970] | 11691.73 ± 173.28 [11052.02,12483.15] | +1.689 |

Baselines (shortfall_bps, mean [95% CI]):

| baseline | highvol | calm | lowvol | highvol_null | trending | liquidity_shock |
|---|---|---|---|---|---|---|
| twap | 3.396 [3.128,3.665] | 1.471 [1.409,1.527] | 0.924 [0.908,0.941] | 3.236 [2.903,3.569] | 3.431 [3.216,3.615] | 4.351 [4.081,4.608] |
| vwap | 3.349 [3.170,3.524] | 1.635 [1.560,1.711] | 0.923 [0.904,0.945] | 3.301 [3.167,3.487] | 3.387 [3.179,3.588] | 4.617 [4.363,4.893] |
| pov | 3.256 [3.063,3.411] | 1.617 [1.577,1.646] | 0.923 [0.908,0.941] | 3.085 [2.791,3.324] | 3.530 [3.280,3.775] | 4.014 [3.824,4.225] |
| passive | 3.672 [3.230,4.115] | 1.770 [1.708,1.817] | 1.060 [1.050,1.072] | 3.527 [2.992,3.937] | 3.753 [3.450,4.063] | 4.491 [3.852,5.131] |
| schedule_twap | 2.932 [2.856,3.023] | 1.570 [1.524,1.637] | 0.938 [0.929,0.946] | 2.897 [2.807,2.983] | 3.065 [2.876,3.263] | 4.132 [3.959,4.361] |
| adaptive_pov | 2.678 [2.436,2.861] | 1.541 [1.511,1.569] | 0.896 [0.882,0.910] | 2.716 [2.443,2.962] | 3.013 [2.793,3.344] | 3.751 [3.607,3.911] |
| is_aware | 3.122 [2.853,3.336] | 1.569 [1.518,1.626] | 0.917 [0.904,0.933] | 3.055 [2.814,3.221] | 3.151 [2.705,3.488] | 4.429 [4.286,4.551] |

## Mode `volsym` — regime flag visible to the agent AND every baseline (symmetric)

| regime | PPO shortfall (seed-mean ± seed-std) | best baseline | Δ vs best [95% CI] per training seed | sig better / worse | Δ vs VWAP (mean over seeds) | PPO completion† | PPO fill rate† | PPO max DD (ticks)† | PPO slip vs market VWAP |
|---|---|---|---|---|---|---|---|---|---|
| highvol | 2.670 ± 0.065 | adaptive_pov 2.716 | +0.02 [-0.13,+0.17] -0.05 [-0.51,+0.40] +0.03 [-0.15,+0.22] +0.15 [-0.09,+0.38] +0.07 [-0.29,+0.44] | 0 / 0 | +0.679 bps (+20.3%) | 0.981 ± 0.009 [0.961,0.997] | 0.981 ± 0.009 [0.961,0.997] | 10042.53 ± 151.18 [9384.78,10966.62] | +1.304 |
| calm | 1.625 ± 0.020 | twap 1.471 | -0.14 [-0.19,-0.09] -0.14 [-0.19,-0.09] -0.14 [-0.19,-0.08] -0.18 [-0.24,-0.13] -0.17 [-0.25,-0.11] | 0 / 5 | +0.010 bps (+0.6%) | 1.000 ± 0.000 [1.000,1.000] | 1.000 ± 0.000 [1.000,1.000] | 4773.83 ± 24.92 [4593.34,5012.38] | +1.047 |
| lowvol | 0.971 ± 0.017 | adaptive_pov 0.891 | -0.10 [-0.11,-0.08] -0.08 [-0.08,-0.07] -0.05 [-0.05,-0.05] -0.09 [-0.10,-0.09] -0.08 [-0.09,-0.07] | 0 / 5 | -0.047 bps (-5.1%) | 1.000 ± 0.000 [1.000,1.000] | 1.000 ± 0.000 [1.000,1.000] | 2771.98 ± 13.89 [2702.31,2836.24] | +0.803 |
| highvol_null | 2.564 ± 0.103 | adaptive_pov 2.753 | +0.13 [-0.19,+0.39] +0.20 [-0.14,+0.59] +0.05 [-0.08,+0.30] +0.20 [-0.19,+0.45] +0.36 [+0.06,+0.68] | 1 / 0 | +0.737 bps (+22.3%) | 0.985 ± 0.008 [0.965,0.998] | 0.985 ± 0.008 [0.965,0.998] | 10076.53 ± 259.11 [9357.75,10912.88] | +1.655 |
| trending | 2.962 ± 0.070 | schedule_twap 3.065 | +0.16 [+0.03,+0.30] +0.08 [-0.02,+0.18] +0.02 [-0.20,+0.18] +0.05 [-0.05,+0.14] +0.20 [+0.08,+0.31] | 2 / 0 | +0.425 bps (+12.5%) | 0.975 ± 0.010 [0.956,0.991] | 0.975 ± 0.010 [0.956,0.991] | 10871.67 ± 185.28 [9878.07,11706.68] | +0.586 |
| liquidity_shock | 3.136 ± 0.153 | adaptive_pov 3.992 | +0.75 [+0.51,+1.01] +0.78 [+0.49,+1.06] +0.72 [+0.55,+0.89] +0.89 [+0.65,+1.15] +1.14 [+0.89,+1.35] | 5 / 0 | +1.481 bps (+32.1%) | 0.948 ± 0.010 [0.915,0.979] | 0.948 ± 0.010 [0.915,0.979] | 12205.24 ± 299.64 [11478.97,13361.95] | +1.626 |

Baselines (shortfall_bps, mean [95% CI]):

| baseline | highvol | calm | lowvol | highvol_null | trending | liquidity_shock |
|---|---|---|---|---|---|---|
| twap | 3.396 [3.128,3.665] | 1.471 [1.409,1.527] | 0.924 [0.908,0.941] | 3.236 [2.903,3.569] | 3.431 [3.216,3.615] | 4.351 [4.081,4.608] |
| vwap | 3.349 [3.170,3.524] | 1.635 [1.560,1.711] | 0.923 [0.904,0.945] | 3.301 [3.167,3.487] | 3.387 [3.179,3.588] | 4.617 [4.363,4.893] |
| pov | 3.256 [3.063,3.411] | 1.617 [1.577,1.646] | 0.923 [0.908,0.941] | 3.085 [2.791,3.324] | 3.530 [3.280,3.775] | 4.014 [3.824,4.225] |
| passive | 3.672 [3.230,4.115] | 1.770 [1.708,1.817] | 1.060 [1.050,1.072] | 3.527 [2.992,3.937] | 3.753 [3.450,4.063] | 4.491 [3.852,5.131] |
| schedule_twap | 2.932 [2.856,3.023] | 1.570 [1.524,1.637] | 0.938 [0.929,0.946] | 2.897 [2.807,2.983] | 3.065 [2.876,3.263] | 4.132 [3.959,4.361] |
| adaptive_pov | 2.716 [2.436,2.974] | 1.541 [1.511,1.569] | 0.891 [0.874,0.908] | 2.753 [2.440,3.041] | 3.094 [2.873,3.362] | 3.992 [3.877,4.116] |
| is_aware | 3.122 [2.853,3.336] | 1.569 [1.518,1.626] | 0.917 [0.904,0.933] | 3.055 [2.814,3.221] | 3.151 [2.705,3.488] | 4.429 [4.286,4.551] |

† PPO point estimate is the mean over 5 independently trained policies; `±` is their standard deviation. Brackets are the envelope of the 5 per-policy, whole-evaluation-family bootstrap 95% CIs (not a pooled joint CI); all component CIs are retained in the JSON.

