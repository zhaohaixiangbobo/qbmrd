"""English review copy of run_benchmark.py.
Historical source logic; see HISTORICAL_CODE_GUIDE.md for execution scope and provenance.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data, standardize_allocation_matrix
from bench.metrics import compute_gnae, compute_objective_value, validate_constraints
from bench.report import MonthMethodReportRow, violations_to_count_rows, violations_to_detail_rows, write_allocations_xlsx, write_summary_csv, write_violation_details_csv, write_violation_counts_csv, write_violations_json
from bench.solvers.lp_solver import solve_with_lp
from bench.solvers.ga import solve_with_ga
from bench.solvers.rule_based import solve_with_rule_based
from bench.solvers.sa import solve_with_simulated_annealing

def main() -> int:
    parser = argparse.ArgumentParser(description='Historical single-run monthly LP/RuleBased/SA/GA benchmark and external diagnostic export.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR), help='Input directory containing 202501.xlsx through 202506.xlsx; temporary lock files are skipped.')
    parser.add_argument('--label-dir', type=str, default=str(DEFAULT_LABEL_DIR), help='Reference directory with YYYYMM.xlsx files for GNAE; references may be absent.')
    parser.add_argument('--output-dir', type=str, default=str(Path(__file__).resolve().parent / 'output'), help='Output directory for benchmark summaries and diagnostic CSV files.')
    parser.add_argument('--export-details', action='store_true', help='Export month-by-method allocation workbooks and structured diagnostic TXT files.')
    parser.add_argument('--max-files', type=int, default=0, help='Maximum input files; zero selects all months.')
    parser.add_argument('--lp-time-limit', type=int, default=60, help='LP time limit in seconds.')
    parser.add_argument('--sa-time', type=float, default=30.0, help='SA time limit in seconds; stops when either the time or iteration limit is reached.')
    parser.add_argument('--sa-iters', type=int, default=50000, help='SA iteration limit; reaching this limit may stop the run before the time budget.')
    parser.add_argument('--ga-time', type=float, default=60, help='GA time limit in seconds; zero selects the size-dependent budget with a 60-second minimum.')
    parser.add_argument('--ga-pop', type=int, default=0, help='GA population size; zero selects a size-dependent population.')
    parser.add_argument('--ga-seed', type=int, default=42, help='GA random seed.')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    cfg = default_experiment_configs()
    rows: List[MonthMethodReportRow] = []
    violation_count_rows: List[dict] = []
    violation_detail_rows: List[dict] = []
    input_paths = sorted((p for p in input_dir.glob('*.xlsx') if not p.name.startswith('~$')))
    if args.max_files and args.max_files > 0:
        input_paths = input_paths[:args.max_files]
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
        print(json.dumps({'month_id': month.month_id, 'lp_time_limit_sec': int(args.lp_time_limit), 'constraint_config': constraint_config, 'objective_config': cfg.objective_config}, ensure_ascii=False, default=str))
        lp_out = solve_with_lp(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=cfg.objective_config, time_limit_sec=int(args.lp_time_limit))
        heuristic_out = solve_with_rule_based(data_loader=month.data_loader, constraint_config=constraint_config)
        sa_out = solve_with_simulated_annealing(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=cfg.objective_config, max_time_sec=float(args.sa_time), max_iters=int(args.sa_iters), seed=42)
        auto_ga_time, auto_ga_pop = _auto_ga_params(n_products=len(month.product_data), n_rounds=len(rounds))
        ga_out = solve_with_ga(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=cfg.objective_config, max_time_sec=float(args.ga_time) if float(args.ga_time) > 0 else float(auto_ga_time), pop_size=int(args.ga_pop) if int(args.ga_pop) > 0 else int(auto_ga_pop), seed=int(args.ga_seed))
        for method, out in [('LP', lp_out), ('RuleBased', heuristic_out), ('SA', sa_out), ('GA', ga_out)]:
            has_round_cols = all((r in out.allocation_matrix_full.columns for r in rounds)) if isinstance(out.allocation_matrix_full, pd.DataFrame) else False
            if isinstance(out.allocation_matrix_full, pd.DataFrame) and (not out.allocation_matrix_full.empty) and has_round_cols:
                constraint_check = validate_constraints(data_loader=month.data_loader, allocation_matrix_full=out.allocation_matrix_full, constraint_config=constraint_config)
            else:
                constraint_check = validate_constraints(data_loader=month.data_loader, allocation_matrix_full=pd.DataFrame(), constraint_config=constraint_config)
            objective_value = out.objective_value
            if objective_value is None and isinstance(out.allocation_matrix_full, pd.DataFrame) and (not out.allocation_matrix_full.empty) and has_round_cols:
                objective_value = compute_objective_value(allocation_matrix_full=out.allocation_matrix_full, rounds=rounds, existing_allocations=month.data_loader.get_existing_allocations(), constraint_config=constraint_config, objective_config=cfg.objective_config)
            gnae = None
            if month.label_data is not None and isinstance(out.allocation_matrix_full, pd.DataFrame) and (not out.allocation_matrix_full.empty) and has_round_cols:
                alloc_std = standardize_allocation_matrix(out.allocation_matrix_full, rounds)
                label_std = standardize_allocation_matrix(month.label_data, rounds)
                alloc_aligned, label_aligned = _align_by_code(alloc_std, label_std)
                gnae = compute_gnae(alloc_aligned, label_aligned, rounds)
            rows.append(MonthMethodReportRow(month_id=month.month_id, method=method, status=out.status, feasible=constraint_check.feasible, total_violations=constraint_check.total_violations, solve_time_sec=float(out.solve_time_sec), objective_value=float(objective_value) if objective_value is not None else None, gnae=float(gnae) if gnae is not None else None))
            for r in violations_to_count_rows(constraint_check.violations):
                violation_count_rows.append({'month_id': month.month_id, 'method': method, 'constraint_name': r.get('constraint_name'), 'count': r.get('count')})
            for r in violations_to_detail_rows(constraint_check.violations):
                violation_detail_rows.append({'month_id': month.month_id, 'method': method, 'constraint_name': r.get('constraint_name'), 'violation_index': r.get('violation_index'), 'violation': r.get('violation')})
            if args.export_details:
                detail_name = f'benchmark_details_{month.month_id}_{method}.xlsx'
                extra_sheets = None
                if method == 'LP':

                    def _as_text(v: object) -> str:
                        if isinstance(v, (dict, list, tuple)):
                            return json.dumps(v, ensure_ascii=False, default=str)
                        return str(v)
                    extra_sheets = {'lp_params': pd.DataFrame([{'key': 'month_id', 'value': month.month_id}, {'key': 'lp_time_limit_sec', 'value': int(args.lp_time_limit)}]), 'constraint_config': pd.DataFrame([{'key': k, 'value': _as_text(v)} for k, v in constraint_config.items()]), 'objective_config': pd.DataFrame([{'key': k, 'value': _as_text(v)} for k, v in cfg.objective_config.items()])}
                write_allocations_xlsx(out.allocation_matrix_full, output_dir / detail_name, extra_sheets=extra_sheets)
                vio_name = f'constraint_violations_{month.month_id}_{method}.txt'
                write_violations_json(constraint_check.violations, output_dir / vio_name)
    write_summary_csv(rows, output_dir / 'benchmark_summary.csv')
    if violation_count_rows:
        write_violation_counts_csv(violation_count_rows, output_dir / 'constraint_violation_counts.csv')
    if violation_detail_rows:
        write_violation_details_csv(violation_detail_rows, output_dir / 'constraint_violation_details.csv')
    print(json.dumps({'Output file descriptions': {'benchmark_summary.csv': 'Month-by-method status/feasible/total_violations/solve_time_sec/objective_value/gnae.', 'benchmark_summary.csv field definitions': {'month_id': 'Month identifier, for example 202501.', 'method': 'Method: LP, RuleBased, SA or GA.', 'status': 'Solver status, including auto: operational adjustments or fallback: heuristic substitution where applicable.', 'feasible': 'External validator overall_valid (zero review flags); separate from LP solver status.', 'total_violations': 'Number of external validator review flags.', 'solve_time_sec': 'Method elapsed time in seconds.', 'objective_value': 'Objective evaluated using the backend terms under its minimisation convention.', 'gnae': 'GNAE relative to the reference; lower is closer; empty when references are absent.'}, 'constraint_violation_counts.csv': 'External review-flag counts by month, method and rule.', 'constraint_violation_details.csv': 'Individual external diagnostic records by month, method and rule (structured text).', 'benchmark_details_<YYYYMM>_<Method>.xlsx': 'With --export-details: allocation workbooks; LP also includes parameter/configuration tables.', 'constraint_violations_<YYYYMM>_<Method>.txt': 'With --export-details: complete external diagnostic structure.'}}, ensure_ascii=False, indent=2))
    return 0

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

def _calibrate_c_constraints_from_label(constraint_config: dict, label_df: pd.DataFrame, rounds: list[str]) -> None:
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
