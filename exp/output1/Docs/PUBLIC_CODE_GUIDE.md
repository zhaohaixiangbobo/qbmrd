# Public code subset and verification commands

All copied Python files and the two frozen parameter files are identical to the current confidential review package. The review-package integrity verifier is not included because it requires the complete review package. `Provenance/PUBLIC_FILE_MANIFEST.csv` authenticates this public subset instead.

## Existing environment

Use Python 3.10+ with the existing packages numpy, pandas, matplotlib, openpyxl, pydantic, scipy, statsmodels and PuLP/CBC. This preparation does not install dependencies. The interpreter below is the author's existing environment; other users should select their own compatible interpreter.

Run the following commands from the repository root (output1), not from Docs:

```powershell
$PY = 'D:\3-anaconda\envs\py312\python.exe'
$C = 'Implementation/S10_Reproduction_Code/submission_revision_202608'
& $PY -B "$C/source_snapshot/exp/run_benchmark_multiseed.py" --help
& $PY -B "$C/source_snapshot/exp/run_lp_marginal_analysis.py" --help
```

## Statistical recheck available from public data

```powershell
& $PY -B "$C/scripts/recheck_month_level_gnae.py" --runs 'R3_Supplementary_Data_English/S4_Benchmark/D2_benchmark_240_run_records.csv' --output-dir 'Local_GNAE_Recheck'
```

The output directory must be new. This performs six-month aggregation and exact paired Wilcoxon/BH calculations, with no optimization. Expected p values are 0.03125, 0.6875, 0.6875 and BH values are 0.09375, 0.6875, 0.6875 for RuleBased, SA and GA versus LP.

## Historical code map

| Material | Entry point under the reproduction-code directory |
|---|---|
| D1 case assembly | source_snapshot/exp/run_lp_case_study.py |
| D2 multi-seed benchmark | source_snapshot/exp/run_benchmark_multiseed.py |
| D2 single-run benchmark/diagnostics | source_snapshot/exp/run_benchmark.py |
| D2 manuscript month-level inference | scripts/recheck_month_level_gnae.py |
| D3 weight analysis | source_snapshot/exp/run_lp_marginal_analysis.py |
| D4 range-control input perturbations | scripts/run_submission.py robustness |
| D5 operational escalation | source_snapshot/exp/run_lp_relaxation_path.py |
| S8 constructed illustration | source_snapshot/exp/run_lp_fairness_example.py |
| D6 reduced-core scale tests | source_snapshot/exp/run_lp_scalability.py |

The original D2 multi-seed generator retains legacy seed-pair statistical output. For the manuscript use the six-month statistical adapter, not that legacy significance file. The revision runner's benchmark command belongs to a separate development protocol and must not be substituted for the historical D2 generator.

## Source-dependent operations

Optimization and the case-building script require additional inputs that are not public in this folder. Authorised editors/referees should use the complete review package and its `Implementation/HISTORICAL_CODE_GUIDE.md`. Other authorised users can supply local source inputs explicitly via the historical entry points' `--input-dir` and `--label-dir` options. Such local private inputs should remain outside any upload folder.

D4 uses a disposable review-package copy, with authentic inputs copied into its source_snapshot/exp/data/input and corresponding references into data/label. The complete review guide specifies this route. Public transformed examples are not supplied here and cannot substitute for authentic source data.

The historical single-run retry2 invocation and all 36 historical final configuration snapshots are not certified by publishing these scripts. Source-copy provenance and re-execution are different claims. Full D2 computation can approach 15 hours of SA/GA budget alone; no long experiment is launched by the commands above.
