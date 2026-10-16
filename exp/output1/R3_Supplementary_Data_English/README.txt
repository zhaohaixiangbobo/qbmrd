SUPPLEMENTARY DATA: ENGLISH REVIEW EDITION

Manuscript: From human decision-makers to automated decision aids: a decision-support system for quota-based multi-round distribution

START HERE
Open START_HERE.xlsx for the English file guide and review instructions. Each S3-S9 directory has an English review workbook containing the complete supplied result tables. S4 and S6 also contain English configuration/provenance workbooks. Every supplied setting and value is accessible in Excel. No JSON file or code execution is required.

UTF-8 CSV copies accompany the workbooks. The 25 original result CSV files retain their record order, columns and exact numerical tokens; only Chinese labels or descriptions are translated. All nested D5 applied configurations are additionally expanded into Path, Type and Value tables. The D2 source evidence and all 10 D4 configuration, protocol and provenance records are supplied as complete typed English tables. Numeric tokens longer than Excel's 15 significant-digit limit are retained as exact text where necessary. Display formatting does not change stored values. No optimisation experiments were rerun and no research results were recalculated for this English edition.

DATA GROUPS
D1 / S3_Feb_case: complete anonymised February allocation matrix (277 SKUs, four rounds), representative profiles, round summary and case configuration.
D2 / S4_Benchmark: 240 benchmark records; month-level GNAE; exact paired Wilcoxon inputs/results with Benjamini-Hochberg adjustment; 24 method-month summaries; external review-flag counts; coverage/source evidence.
D3 / S5_Weight: 36 month-by-weight records for 0, 75, 150, 300, 600 and 1200, with aggregate and baseline-effect summaries.
D4 / S6_Perturbation: six baseline and 60 perturbed run records, guards, protocols, summaries, recovered frozen objective, six actual monthly configurations, run metadata and an aggregate of 830 existing diagnostic records (24 baseline and 806 perturbed).
D5 / S7_Operational_Escalation: the 13-step six-month escalation trace, monthly path summary and applied configurations. The two additional configuration sheets fully expand the nested settings from the trace and monthly summary.
Illustration / S8_Temporal_Profile_Illustration: the separate numerical temporal-profile illustration used in Supplementary Table S11.
D6 / S9_Scalability: 24 field-derived instances and aggregate scalability summary.

RELATED IMPLEMENTATION AND EXAMPLES
S10 and S11 are provided under Implementation/ in the non-public R3_Review_Materials_English.zip. They contain the selected English implementation and 13 English transformed demonstration workbooks. These examples support examination of the method on transformed data. They are not the private historical empirical inputs and need not reproduce the reported outcomes. Neither the code nor the examples are required to inspect the result records in this data package.

CONFIDENTIAL SOURCE RECORDS
This public archive contains anonymised derived research records. English, deidentified historical input and executed human-reference data are supplied separately under Source_Data/ in R3_Review_Materials_English.zip for confidential editorial and peer review. The source datasets are not intended for public release.

D4 CONFIGURATION AND PROVENANCE
The recovered source is associated with historical run robustness_20260810T211905 (10 August 2026). Its objective coefficients are maximize_allocation_weight=1000, round_balance_weight=800, round_variance_weight=0, product_balance_weight=0 and smooth_transition_weight=0. This differs from the transition-control benchmark, whose transition weight is 300.

S6_Configuration_Provenance_English.xlsx preserves every field from the supplied D4 base business configuration, the six actual monthly guard configurations, the analysis parameters, seed/quota protocol and run metadata. The monthly configurations were applied identically to each monthly baseline and its ten perturbations. Category, priority and soft adjacent-round balance are enabled; demand-split and price-based SKU rules are disabled. Historical fixed allocations are cleared in memory. The square-pack rule retains its binary extension. The D4 solver wrapper performs no operational-escalation or RuleBased fallback sequence.

The earlier analysis and seed/quota protocols are retained as records and supplemented by the recovered frozen configuration and provenance. The run metadata preserves the recorded Python/platform and snapshot-manifest identity. The historical manifest covers the backend/experiment snapshot; the supplied revision wrapper is separately documented in the related implementation package.

Positive-demand and inventory multipliers are drawn independently from Uniform[0.95,1.05], with month and seed jointly determining the random stream. Demand is rescaled to the original monthly total; zero values remain zero. The recorded demand-multiplier minimum/maximum refer to draws before monthly-total rescaling. The final normalised demand multipliers are therefore not asserted to remain strictly within the original +/-5% interval.

The final guards retain monthly quota and round targets, allow +/-10% round volumes and use a price half-width equal to the larger of the original half-width and 15% of the original price centre. Structure guards follow the preserved analytic construction. Earlier protocols show development changes to price guards and reporting thresholds. The final protocol was applied consistently across the reported monthly baselines and perturbations; it is not described as an unchanged preregistered protocol.

D4 DIAGNOSTIC DEFINITIONS
The diagnostic classifier operates on records emitted by the external ConstraintManager after allocation extraction and small-quantity postprocessing. It does not replace solver status or export every internal model residual.

For an emitted quantity or price record with residual magnitude d and reference scale s, the reporting threshold is max(0.2, 0.00005*s). For ratio or round-count records the threshold is 0.00005*s. A record with d <= threshold is numerical_tolerance_flag. Larger records are hard_constraint_violation if their rule name belongs to the preserved HARD_CONSTRAINTS set, and otherwise post_solution_review_flag. The robustness-specific relative tolerance 0.00005 overrides the generic 0.000001. The value 0.00005 equals 0.005%, or 0.5 basis point; the quantity/price threshold can exceed 0.2.

Scale is max(1, abs(reference bound)); supply-excess records use available supply. For lower/upper price or volume records the triggered boundary is selected. The related implementation preserves exact residual extraction and the fallback for unrecognised record schemas. Model guards, reporting thresholds and CBC's solver tolerances are distinct.

Each emitted record belongs to one classification bucket. Multiple records can concern the same SKU or round. D4_diagnostic_record_summary.csv aggregates the 830 existing detailed records by month, scenario, rule and category. It contains no SKU/product identifiers or record-level private detail.

soft_rule_slack_activations is set to zero by the classifier without computing auxiliary-variable activations. It is a compatibility placeholder and is not evidence that soft-rule slacks were zero.

post_solution_review_flags is structurally zero for the present validator because every rule name it emits is already included in HARD_CONSTRAINTS. It is not a separate set of passed business checks. The hard-classified labels are external diagnostic categories, not a certificate that they are all solver-internal hard constraints.

normalized_review_flag_rate is (hard + review + numerical record counts) divided by applicable_rule_checks. That denominator is the code's size-based estimate: enabled demand and first-round supply contribute one check per SKU; positive fixed cells contribute one each; volume and price contribute one per round; priority contributes one per flagged SKU; price-based contributes one per round; C-category contributes three per round; the total is bounded below by one. It is not a count of solver rows or a population of independent observations. The D4 aggregate reports counts only; no new aggregate flag rate is introduced.

D4 solve_time_sec is wall-clock time around the allocation routine: manager/model construction, optimisation, allocation extraction and postprocessing. It excludes input loading, guard/perturbation preparation, the subsequent constraint-diagnostic extraction loop, external audit/metrics and file export. It is not CBC-only time or full user-interface time. Stored solver status refers to the solver's model result, while diagnostics refer to the returned allocation. No separately solved priority/balance auxiliary-variable table is supplied.

OTHER ARCHIVED FIELD NAMES
For D2, total_violations means external post-solution review flags; feasible/strictly_feasible denotes the archived zero-review-flag condition, not solver infeasibility. In D6, rule_deviations is the corresponding stress-test diagnostic count. smooth_transition_weight denotes transition-control weight. Legacy robustness/relaxation/fallback terms remain in computational field names; the manuscript uses input sensitivity and operational escalation.

LP and RuleBased benchmark seeds identify timing repetitions; SA and GA use stochastic seeds 42-51. The archived method label LP may include active binary rule extensions. D5 times refer to its separately recorded escalation sequence. D6 uses the stated reduced LP-core rule set with several enterprise rule families disabled; it does not establish scaling with every rule enabled.

PROVENANCE
The selected source commit is ce376780a6706d2c9a34e65c04abc1acf7e02849. The materials are supplied directly in the submission files; no repository link is required to inspect them. The commit is provenance for selected source/example files. It is not a retroactively recorded execution commit for every archived experiment or a requirement to download the repository. The selected implementation, language schema adaptation and transformed examples are documented in the separate related implementation package.

LANGUAGE AND NUMERICAL PRESERVATION
All result records, numerical tokens and supplied configuration values were compared with the source package. Only Chinese labels and descriptions were translated. The English Excel guide and UTF-8 CSV copies provide direct access to the preserved records and settings.


D1 ZERO-DEMAND REPORTING CONVENTION
In D1, Allocation rate equals Total allocation divided by Monthly quantity when Monthly quantity is positive. For the 103 rows with zero Monthly quantity, the stored value 1.0 is the implementation reporting convention for the undefined ratio; it does not represent an observed fulfilment percentage.

D3 terminology and saved key solutions
The legacy feasible field and Post-solution feasibility rate in D3 denote the external zero-review-flag condition, not solver feasibility. See S5_Weight/S5_REVIEW_FLAGS_NOTE.txt and the Review_notes worksheet in S5_English_Review.xlsx. Eighteen saved allocation matrices at weights 0, 300 and 1200 are supplied in the confidential review archive; their input configurations are not final per-attempt snapshots.
