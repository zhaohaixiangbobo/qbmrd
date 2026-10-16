# Independent D3 verification — 16 October 2026

All 36 runs completed with optimal status and no heuristic fallback. The run captured 78 internal LP attempts, 36 complete snapshots and 36 final allocation/configuration workbooks. Original source and input hashes remained unchanged.

All 360 non-timing scalar comparisons passed the declared tolerances. For the 18 historically saved allocation matrices (20,904 cells), all comparisons passed; maximum absolute cell difference was 2.273736754432321e-13. None was exactly identical at floating-point representation level. The other 18 historical matrices were not saved, so only scalar comparisons are available for those runs.

Maximum absolute differences: GNAE 8.326673e-17; objective 7.450581e-9; normalised allocation change/stability approximately 2.220446e-16. Scalar tolerances: rtol=1e-9, atol=1e-9; objective atol=1e-3; matrix atol=1e-6 with rtol=1e-9.

Historical solve-time sum was 80.488769 seconds; new solve-time sum was 47.636326 seconds; complete verification elapsed 59.239131 seconds. Runtime is environment-dependent and the new timing does not replace manuscript timing.

The authentic six-month inputs and labels, archived solver core, objective weights (0, 75, 150, 300, 600, 1200), and historical metric conventions were preserved. CBC 2.10.3, PuLP 3.3.2, Python 3.12.12 were used; each attempt retained the original 60-second solver limit without new seed/thread/gap overrides. Exact identity of the historical D3 runtime environment has not been established.

Final constraints: January, May and June used initial constraints (one attempt per run); February and April disabled the c_type constraint (two attempts); March disabled c_type, followed volume thresholds 0.01/0.02/0.05 and disabled the price constraint (six attempts). Full traces and effective configurations are retained privately.

These are newly generated verification records, not contemporaneous historical snapshots. The original submission results have not been overwritten. Full allocations, source inputs and complete configurations remain in the confidential local evidence bundle. This public folder contains scalar results and comparison summaries only.

Three necessary validation tests passed: unchanged source/input hashes; completeness and final-attempt consistency of all 36 snapshots; English text and structural validity of all 36 workbooks. No visual presentation check or broad test suite was performed.
