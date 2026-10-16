"""English review copy of run_benchmark_multiseed.py.
Historical source logic; see HISTORICAL_CODE_GUIDE.md for execution scope and provenance.
"""
from __future__ import annotations
import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data, standardize_allocation_matrix
from bench.metrics import compute_gnae, compute_objective_value, validate_constraints
from bench.report import write_rows_csv
from bench.solvers.ga import solve_with_ga
from bench.solvers.lp_solver import solve_with_lp
from bench.solvers.rule_based import solve_with_rule_based
from bench.solvers.sa import solve_with_simulated_annealing

def main() -> int:
    parser = argparse.ArgumentParser(description='Historical multi-seed LP/RuleBased/SA/GA benchmark with run records and descriptive summaries. Its legacy seed-pair tests are not the manuscript month-level inference.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR), help='Input directory containing 202501.xlsx through 202506.xlsx; temporary lock files are skipped.')
    parser.add_argument('--label-dir', type=str, default=str(DEFAULT_LABEL_DIR), help='Reference directory with YYYYMM.xlsx files for GNAE; references may be absent.')
    parser.add_argument('--output-dir', type=str, default=str(Path(__file__).resolve().parent / 'output' / 'benchmark_multiseed'), help='Output directory for benchmark_multiseed_*.csv.')
    parser.add_argument('--max-files', type=int, default=0, help='Maximum input files; zero selects all months.')
    parser.add_argument('--n-seeds', type=int, default=10, help='Number of seeds per method.')
    parser.add_argument('--base-seed', type=int, default=42, help='First seed in a consecutive integer seed sequence.')
    parser.add_argument('--lp-time-limit', type=int, default=60, help='LP time limit in seconds.')
    parser.add_argument('--sa-time', type=float, default=300.0, help='SA time limit in seconds.')
    parser.add_argument('--sa-iters', type=int, default=50000, help='SA iteration limit.')
    parser.add_argument('--ga-time', type=float, default=600.0, help='GA time limit in seconds; zero selects the size-dependent budget with a 60-second minimum.')
    parser.add_argument('--ga-pop', type=int, default=0, help='GA population size; zero selects a size-dependent population.')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = default_experiment_configs()
    seeds = [int(args.base_seed) + i for i in range(max(1, int(args.n_seeds)))]
    input_paths = sorted((p for p in input_dir.glob('*.xlsx') if not p.name.startswith('~$')))
    if args.max_files and args.max_files > 0:
        input_paths = input_paths[:args.max_files]
    run_rows: List[Dict[str, Any]] = []
    for input_path in input_paths:
        month = load_month_data(input_path=input_path, label_dir=label_dir)
        rounds = month.rounds
        constraint_config = dict(cfg.constraint_config)
        if 'Price_Based' in month.product_data.columns:
            price_based_cnt = int(month.product_data['Price_Based'].fillna('').astype(str).str.strip().ne('').sum())
            if price_based_cnt <= 1:
                constraint_config['enable_price_based_constraints'] = False
        if month.label_data is not None:
            _calibrate_c_constraints_from_label(constraint_config, month.label_data, rounds)
        print(json.dumps({'month_id': month.month_id, 'n_seeds': len(seeds), 'lp_time_limit_sec': int(args.lp_time_limit), 'sa_time_sec': float(args.sa_time), 'ga_time_sec': float(args.ga_time), 'constraint_config': constraint_config, 'objective_config': cfg.objective_config}, ensure_ascii=False, default=str))
        auto_ga_time, auto_ga_pop = _auto_ga_params(n_products=len(month.product_data), n_rounds=len(rounds))
        for seed in seeds:
            outputs = {'LP': solve_with_lp(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=cfg.objective_config, time_limit_sec=int(args.lp_time_limit)), 'RuleBased': solve_with_rule_based(data_loader=month.data_loader, constraint_config=constraint_config), 'SA': solve_with_simulated_annealing(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=cfg.objective_config, max_time_sec=float(args.sa_time), max_iters=int(args.sa_iters), seed=int(seed)), 'GA': solve_with_ga(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=cfg.objective_config, max_time_sec=float(args.ga_time) if float(args.ga_time) > 0 else float(auto_ga_time), pop_size=int(args.ga_pop) if int(args.ga_pop) > 0 else int(auto_ga_pop), seed=int(seed))}
            for method, out in outputs.items():
                row = _evaluate_method_output(month_id=month.month_id, method=method, seed=int(seed), out=out, rounds=rounds, month=month, constraint_config=constraint_config, objective_config=cfg.objective_config)
                run_rows.append(row)
    write_rows_csv(run_rows, output_dir / 'benchmark_multiseed_runs.csv')
    summary_rows = _build_summary_rows(pd.DataFrame(run_rows), group_cols=['month_id', 'method'])
    overall_rows = _build_summary_rows(pd.DataFrame(run_rows), group_cols=['method'])
    significance_rows = _build_significance_rows(pd.DataFrame(run_rows))
    write_rows_csv(summary_rows, output_dir / 'benchmark_multiseed_summary.csv')
    write_rows_csv(overall_rows, output_dir / 'benchmark_multiseed_overall.csv')
    write_rows_csv(significance_rows, output_dir / 'benchmark_multiseed_significance.csv')
    print(json.dumps({'Output file descriptions': {'benchmark_multiseed_runs.csv': 'Month-by-method-by-seed run records.', 'benchmark_multiseed_summary.csv': 'Month-by-method descriptive mean/SD/95% CI and zero-review-flag rate.', 'benchmark_multiseed_overall.csv': 'Method-level descriptive statistics across all months.', 'benchmark_multiseed_significance.csv': 'Legacy seed-pair Wilcoxon comparisons with LP and BH adjustment; use the separate month-level reanalysis for manuscript inference.'}}, ensure_ascii=False, indent=2))
    return 0

def _evaluate_method_output(month_id: str, method: str, seed: int, out: object, rounds: List[str], month: object, constraint_config: Dict[str, Any], objective_config: Dict[str, Any]) -> Dict[str, Any]:
    has_round_cols = all((r in out.allocation_matrix_full.columns for r in rounds)) if isinstance(out.allocation_matrix_full, pd.DataFrame) else False
    if isinstance(out.allocation_matrix_full, pd.DataFrame) and (not out.allocation_matrix_full.empty) and has_round_cols:
        constraint_check = validate_constraints(data_loader=month.data_loader, allocation_matrix_full=out.allocation_matrix_full, constraint_config=constraint_config)
    else:
        constraint_check = validate_constraints(data_loader=month.data_loader, allocation_matrix_full=pd.DataFrame(), constraint_config=constraint_config)
    objective_value = out.objective_value
    if objective_value is None and isinstance(out.allocation_matrix_full, pd.DataFrame) and (not out.allocation_matrix_full.empty) and has_round_cols:
        objective_value = compute_objective_value(allocation_matrix_full=out.allocation_matrix_full, rounds=rounds, existing_allocations=month.data_loader.get_existing_allocations(), constraint_config=constraint_config, objective_config=objective_config)
    gnae = None
    if month.label_data is not None and isinstance(out.allocation_matrix_full, pd.DataFrame) and (not out.allocation_matrix_full.empty) and has_round_cols:
        alloc_std = standardize_allocation_matrix(out.allocation_matrix_full, rounds)
        label_std = standardize_allocation_matrix(month.label_data, rounds)
        alloc_aligned, label_aligned = _align_by_code(alloc_std, label_std)
        gnae = compute_gnae(alloc_aligned, label_aligned, rounds)
    return {'month_id': str(month_id), 'method': str(method), 'seed': int(seed), 'status': str(out.status), 'feasible': bool(constraint_check.feasible), 'total_violations': int(constraint_check.total_violations), 'solve_time_sec': float(out.solve_time_sec), 'objective_value': float(objective_value) if objective_value is not None else None, 'gnae': float(gnae) if gnae is not None else None}

def _build_summary_rows(df: pd.DataFrame, group_cols: List[str]) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    out_rows: List[Dict[str, Any]] = []
    metric_specs = {'gnae': 'gnae', 'total_violations': 'violations', 'solve_time_sec': 'solve_time', 'objective_value': 'objective'}
    grouped = df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row['n_runs'] = int(len(g))
        row['feasible_rate'] = float(g['feasible'].astype(int).mean()) if 'feasible' in g.columns else None
        for col, prefix in metric_specs.items():
            stats_row = _series_stats(pd.to_numeric(g[col], errors='coerce'))
            row[f'{prefix}_mean'] = stats_row['mean']
            row[f'{prefix}_std'] = stats_row['std']
            row[f'{prefix}_ci95_low'] = stats_row['ci95_low']
            row[f'{prefix}_ci95_high'] = stats_row['ci95_high']
        out_rows.append(row)
    return out_rows

def _series_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return {'mean': None, 'std': None, 'ci95_low': None, 'ci95_high': None}
    values = s.astype(float).tolist()
    n = len(values)
    mean = float(s.mean())
    if n <= 1:
        return {'mean': mean, 'std': 0.0, 'ci95_low': mean, 'ci95_high': mean}
    std = float(s.std(ddof=1))
    if std <= 0:
        return {'mean': mean, 'std': 0.0, 'ci95_low': mean, 'ci95_high': mean}
    sem = std / math.sqrt(n)
    try:
        ci_low, ci_high = stats.t.interval(confidence=0.95, df=n - 1, loc=mean, scale=sem)
    except TypeError:
        ci_low, ci_high = stats.t.interval(alpha=0.95, df=n - 1, loc=mean, scale=sem)
    return {'mean': mean, 'std': std, 'ci95_low': float(ci_low), 'ci95_high': float(ci_high)}

def _build_significance_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    metrics = ['gnae', 'total_violations', 'solve_time_sec', 'objective_value']
    compare_methods = ['RuleBased', 'SA', 'GA']
    raw_rows: List[Dict[str, Any]] = []
    for metric in metrics:
        base_df = df[df['method'] == 'LP'].loc[:, ['month_id', 'seed', metric]].rename(columns={metric: 'baseline_value'})
        for compare_method in compare_methods:
            cmp_df = df[df['method'] == compare_method].loc[:, ['month_id', 'seed', metric]].rename(columns={metric: 'compare_value'})
            merged = base_df.merge(cmp_df, on=['month_id', 'seed'], how='inner')
            merged['baseline_value'] = pd.to_numeric(merged['baseline_value'], errors='coerce')
            merged['compare_value'] = pd.to_numeric(merged['compare_value'], errors='coerce')
            merged = merged.dropna(subset=['baseline_value', 'compare_value'])
            row = {'metric': metric, 'baseline_method': 'LP', 'compare_method': compare_method, 'n_pairs': int(len(merged)), 'test_name': 'Wilcoxon', 'statistic': None, 'p_value': None, 'p_value_bh': None, 'effect_direction': None, 'baseline_mean': float(merged['baseline_value'].mean()) if not merged.empty else None, 'compare_mean': float(merged['compare_value'].mean()) if not merged.empty else None, 'significant_at_0_05': None, 'note': ''}
            if merged.empty:
                row['note'] = 'insufficient pairs'
                raw_rows.append(row)
                continue
            diffs = merged['compare_value'] - merged['baseline_value']
            if bool((diffs.abs() <= 1e-12).all()):
                row['statistic'] = 0.0
                row['p_value'] = 1.0
                row['effect_direction'] = 'equal'
                row['significant_at_0_05'] = False
                row['note'] = 'all paired differences are zero'
                raw_rows.append(row)
                continue
            try:
                test = stats.wilcoxon(merged['compare_value'], merged['baseline_value'], zero_method='wilcox', alternative='two-sided', correction=False, mode='auto')
                row['statistic'] = float(test.statistic)
                row['p_value'] = float(test.pvalue)
                row['effect_direction'] = _effect_direction(float(merged['compare_value'].mean()), float(merged['baseline_value'].mean()), metric)
                row['significant_at_0_05'] = bool(float(test.pvalue) < 0.05)
            except ValueError as exc:
                row['p_value'] = 1.0
                row['effect_direction'] = _effect_direction(float(merged['compare_value'].mean()), float(merged['baseline_value'].mean()), metric)
                row['significant_at_0_05'] = False
                row['note'] = str(exc)
            raw_rows.append(row)
    p_values = [r['p_value'] for r in raw_rows if r.get('p_value') is not None]
    if p_values:
        _, pvals_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        idx = 0
        for row in raw_rows:
            if row.get('p_value') is None:
                continue
            row['p_value_bh'] = float(pvals_bh[idx])
            idx += 1
    return raw_rows

def _effect_direction(compare_mean: float, baseline_mean: float, metric: str) -> str:
    if abs(compare_mean - baseline_mean) <= 1e-12:
        return 'equal'
    lower_better = {'gnae', 'total_violations', 'solve_time_sec', 'objective_value'}
    if metric in lower_better:
        return 'better' if compare_mean < baseline_mean else 'worse'
    return 'higher' if compare_mean > baseline_mean else 'lower'

def _align_by_code(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'Code' not in a.columns or 'Code' not in b.columns:
        return (a.reset_index(drop=True), b.reset_index(drop=True))
    a2 = a.copy()
    b2 = b.copy()
    a2['Code'] = a2['Code'].astype(str)
    b2['Code'] = b2['Code'].astype(str)
    merged = a2.merge(b2, on='Code', how='inner', suffixes=('_a', '_b'))
    rounds = [c[:-2] for c in merged.columns if c.endswith('_a') and c[:-2] not in ['Code', 'Cigarette_Name']]
    cols_a = ['Code'] + [f'{r}_a' for r in rounds]
    cols_b = ['Code'] + [f'{r}_b' for r in rounds]
    out_a = merged.loc[:, cols_a].rename(columns={f'{r}_a': r for r in rounds})
    out_b = merged.loc[:, cols_b].rename(columns={f'{r}_b': r for r in rounds})
    return (out_a, out_b)

def _calibrate_c_constraints_from_label(constraint_config: dict, label_df: pd.DataFrame, rounds: List[str]) -> None:
    if 'C' not in label_df.columns:
        return
    is_c = label_df['C'].fillna('').astype(str).str.strip().ne('')
    c_by_round = label_df.loc[is_c, rounds].fillna(0.0).astype(float).sum(axis=0)
    c_limit = float(c_by_round.max()) + 1.0
    constraint_config['c_type_volume_limit'] = max(float(constraint_config.get('c_type_volume_limit', 0.0) or 0.0), c_limit)
    if 'C_Category' not in label_df.columns:
        return
    c_total = c_by_round.replace(0.0, pd.NA)
    is_chang = label_df['C_Category'].fillna('').astype(str).eq('Long')
    is_xi = label_df['C_Category'].fillna('').astype(str).eq('Slim')
    chang_by_round = label_df.loc[is_chang, rounds].fillna(0.0).astype(float).sum(axis=0)
    xi_by_round = label_df.loc[is_xi, rounds].fillna(0.0).astype(float).sum(axis=0)
    chang_limit = float(chang_by_round.max()) + 1.0
    xi_limit = float(xi_by_round.max()) + 1.0
    constraint_config['chang_type_volume_limit'] = max(float(constraint_config.get('chang_type_volume_limit', 0.0) or 0.0), chang_limit)
    constraint_config['xi_type_volume_limit'] = max(float(constraint_config.get('xi_type_volume_limit', 0.0) or 0.0), xi_limit)
    chang_ratio = (chang_by_round / c_total).dropna().max()
    xi_ratio = (xi_by_round / c_total).dropna().max()
    if pd.notna(chang_ratio):
        constraint_config['chang_type_ratio'] = max(float(constraint_config.get('chang_type_ratio', 0.0) or 0.0), float(chang_ratio) + 0.01)
    if pd.notna(xi_ratio):
        constraint_config['xi_type_ratio'] = max(float(constraint_config.get('xi_type_ratio', 0.0) or 0.0), float(xi_ratio) + 0.01)

def _auto_ga_params(n_products: int, n_rounds: int) -> tuple[float, int]:
    n_products = max(0, int(n_products))
    n_rounds = max(1, int(n_rounds))
    complexity = float(n_products * n_rounds)
    time_sec = max(60.0, min(240.0, 60.0 + 0.04 * complexity))
    pop = int(max(20, min(120, 20 + 0.03 * n_products + 5 * n_rounds)))
    return (float(time_sec), int(pop))
if __name__ == '__main__':
    raise SystemExit(main())
