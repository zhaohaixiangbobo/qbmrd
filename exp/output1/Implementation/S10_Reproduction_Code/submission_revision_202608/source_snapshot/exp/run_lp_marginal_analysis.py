"""D3 transition-weight analysis using the recovered allocation procedure."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from bench.configs import default_experiment_configs
from bench.io import load_month_data, standardize_allocation_matrix
from bench.metrics import compute_gnae, compute_normalized_allocation_l1_distance, compute_objective_value, compute_round_volume_deviation, compute_sku_stability_metric, compute_total_stability_metric, validate_constraints
from bench.report import write_allocations_xlsx, write_rows_csv
from bench.solvers.lp_solver import SolverOutput, solve_with_lp
BASELINE_SMOOTH_WEIGHT = 300.0
KEY_SMOOTH_WEIGHTS = {0.0, BASELINE_SMOOTH_WEIGHT, 1200.0}

def main() -> int:
    """Run the supplied monthly inputs over the selected transition weights and export records."""
    parser = argparse.ArgumentParser(description='D3 transition-weight analysis: vary only smooth_transition_weight and inspect changes from the baseline allocation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', type=str, default=str(Path(__file__).resolve().parent / 'data' / 'input'))
    parser.add_argument('--label-dir', type=str, default=str(Path(__file__).resolve().parent / 'data' / 'label'))
    parser.add_argument('--output-dir', type=str, default=str(Path(__file__).resolve().parent / 'output' / 'lp_marginal_analysis'))
    parser.add_argument('--max-files', type=int, default=0, help='Maximum number of input workbooks; 0 processes all available months.')
    parser.add_argument('--lp-time-limit', type=int, default=60, help='Solver time limit in seconds for each weight point.')
    parser.add_argument('--smooth-weights', type=str, default='0,75,150,300,600,1200', help='Comma-separated transition weights; the list must include baseline weight 300.')
    parser.add_argument('--no-export-details', action='store_true', help='Do not export the per-month allocation workbooks for weights 0, 300 and 1200.')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    smooth_weights = _parse_smooth_weights(args.smooth_weights)
    if BASELINE_SMOOTH_WEIGHT not in smooth_weights:
        raise ValueError(f'smooth weights must contain the baseline {BASELINE_SMOOTH_WEIGHT:g}')
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted((p for p in input_dir.glob('*.xlsx') if not p.name.startswith('~$')))
    if args.max_files > 0:
        input_paths = input_paths[:args.max_files]
    cfg = default_experiment_configs()
    run_rows: List[Dict[str, Any]] = []
    elapsed_total = 0
    for month_index, input_path in enumerate(input_paths, start=1):
        month = load_month_data(input_path=input_path, label_dir=label_dir)
        constraint_config = _build_constraint_config(cfg.constraint_config, month)
        reference_objective_config = dict(cfg.objective_config)
        results: List[Tuple[float, Dict[str, Any], SolverOutput]] = []
        for weight_index, smooth_weight in enumerate(smooth_weights, start=1):
            objective_config = dict(reference_objective_config)
            objective_config['smooth_transition_weight'] = float(smooth_weight)
            out = solve_with_lp(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=objective_config, time_limit_sec=int(args.lp_time_limit))
            results.append((smooth_weight, objective_config, out))
            elapsed_total += 1
            print(json.dumps({'stage': 'lp_marginal_analysis', 'month_id': month.month_id, 'month_progress': f'{month_index}/{len(input_paths)}', 'weight': smooth_weight, 'scenario_progress': f'{weight_index}/{len(smooth_weights)}', 'completed_lp_runs': elapsed_total, 'total_lp_runs': len(input_paths) * len(smooth_weights)}, ensure_ascii=False))
        baseline_out = next((out for weight, _, out in results if weight == BASELINE_SMOOTH_WEIGHT))
        for smooth_weight, objective_config, out in results:
            row = _evaluate_marginal_result(month=month, smooth_weight=smooth_weight, objective_config=objective_config, reference_objective_config=reference_objective_config, constraint_config=constraint_config, out=out, baseline_out=baseline_out)
            run_rows.append(row)
            if not args.no_export_details and smooth_weight in KEY_SMOOTH_WEIGHTS:
                _write_key_solution(output_dir=output_dir, month_id=month.month_id, smooth_weight=smooth_weight, allocation=out.allocation_matrix_full, constraint_config=constraint_config, objective_config=objective_config)
    df = pd.DataFrame(run_rows)
    write_rows_csv(run_rows, output_dir / 'lp_marginal_runs.csv')
    write_rows_csv(_build_aggregate_rows(df), output_dir / 'lp_marginal_aggregate.csv')
    write_rows_csv(_build_effect_rows(df), output_dir / 'lp_marginal_effects.csv')
    _write_figures(df, output_dir)
    return 0

def _parse_smooth_weights(raw: str) -> List[float]:
    """Parse, deduplicate and sort the non-negative transition weights."""
    values = [float(item.strip()) for item in str(raw).split(',') if item.strip()]
    if not values or any((value < 0 for value in values)):
        raise ValueError('smooth weights must be a non-empty list of non-negative numbers')
    return sorted(set(values))

def _build_constraint_config(base: Dict[str, Any], month: object) -> Dict[str, Any]:
    """Apply the benchmark price-marker switch and reference-plan category calibration."""
    config = dict(base)
    if 'Price_Based' in month.product_data.columns:
        price_based_count = int(month.product_data['Price_Based'].fillna('').astype(str).str.strip().ne('').sum())
        if price_based_count <= 1:
            config['enable_price_based_constraints'] = False
    if month.label_data is not None:
        _calibrate_c_constraints_from_label(config, month.label_data, month.rounds)
    return config

def _evaluate_marginal_result(month: object, smooth_weight: float, objective_config: Dict[str, Any], reference_objective_config: Dict[str, Any], constraint_config: Dict[str, Any], out: SolverOutput, baseline_out: SolverOutput) -> Dict[str, Any]:
    """Calculate the recorded diagnostics and changes relative to weight 300."""
    rounds = month.rounds
    allocation = out.allocation_matrix_full
    valid_allocation = isinstance(allocation, pd.DataFrame) and (not allocation.empty) and all((r in allocation.columns for r in rounds))
    if not valid_allocation:
        return {'month_id': month.month_id, 'smooth_transition_weight': float(smooth_weight), 'status': out.status, 'feasible': False, 'total_violations': -1, 'solve_time_sec': float(out.solve_time_sec), 'native_objective_value': out.objective_value, 'reference_objective_value': None, 'gnae': None, 'round_volume_deviation': None, 'stability_total': None, 'stability_sku': None, 'solution_l1_to_baseline': None}
    constraint_check = validate_constraints(month.data_loader, allocation, constraint_config)
    native_objective = out.objective_value
    if native_objective is None:
        native_objective = compute_objective_value(allocation, rounds, month.data_loader.get_existing_allocations(), constraint_config, objective_config)
    reference_objective = compute_objective_value(allocation, rounds, month.data_loader.get_existing_allocations(), constraint_config, reference_objective_config)
    gnae = None
    if month.label_data is not None:
        alloc_std = standardize_allocation_matrix(allocation, rounds)
        label_std = standardize_allocation_matrix(month.label_data, rounds)
        aligned_alloc, aligned_label = _align_by_code(alloc_std, label_std)
        gnae = compute_gnae(aligned_alloc, aligned_label, rounds)
    baseline_allocation = baseline_out.allocation_matrix_full
    l1_distance = None
    if isinstance(baseline_allocation, pd.DataFrame) and (not baseline_allocation.empty):
        l1_distance = compute_normalized_allocation_l1_distance(allocation, baseline_allocation, rounds)
    return {'month_id': month.month_id, 'smooth_transition_weight': float(smooth_weight), 'status': out.status, 'feasible': bool(constraint_check.feasible), 'total_violations': int(constraint_check.total_violations), 'solve_time_sec': float(out.solve_time_sec), 'native_objective_value': float(native_objective) if native_objective is not None else None, 'reference_objective_value': float(reference_objective), 'gnae': float(gnae) if gnae is not None else None, 'round_volume_deviation': float(compute_round_volume_deviation(allocation, rounds)), 'stability_total': float(compute_total_stability_metric(allocation, rounds)), 'stability_sku': float(compute_sku_stability_metric(allocation, rounds)), 'solution_l1_to_baseline': float(l1_distance) if l1_distance is not None else None}

def _build_aggregate_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate each weight across months using means and sample standard deviations."""
    if df.empty:
        return []
    numeric_columns = ['solve_time_sec', 'native_objective_value', 'reference_objective_value', 'gnae', 'round_volume_deviation', 'stability_total', 'stability_sku', 'solution_l1_to_baseline', 'total_violations']
    rows: List[Dict[str, Any]] = []
    for weight, group in df.groupby('smooth_transition_weight', dropna=False):
        row: Dict[str, Any] = {'smooth_transition_weight': float(weight), 'n_months': int(len(group)), 'feasible_rate': float(group['feasible'].astype(int).mean())}
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors='coerce')
            row[f'{column}_mean'] = float(values.mean()) if not values.dropna().empty else None
            row[f'{column}_std'] = float(values.std(ddof=1)) if values.dropna().shape[0] > 1 else 0.0
        rows.append(row)
    return rows

def _build_effect_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calculate within-month changes relative to transition weight 300."""
    if df.empty:
        return []
    metrics = ['reference_objective_value', 'gnae', 'total_violations', 'round_volume_deviation', 'stability_total', 'stability_sku', 'solution_l1_to_baseline']
    rows: List[Dict[str, Any]] = []
    for month_id, group in df.groupby('month_id', dropna=False):
        baseline = group[group['smooth_transition_weight'] == BASELINE_SMOOTH_WEIGHT]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        for _, current in group.iterrows():
            row: Dict[str, Any] = {'month_id': str(month_id), 'smooth_transition_weight': float(current['smooth_transition_weight']), 'feasible': bool(current['feasible'])}
            for metric in metrics:
                current_value = pd.to_numeric(pd.Series([current.get(metric)]), errors='coerce').iloc[0]
                baseline_value = pd.to_numeric(pd.Series([baseline_row.get(metric)]), errors='coerce').iloc[0]
                row[f'{metric}_delta_to_baseline'] = float(current_value - baseline_value) if pd.notna(current_value) and pd.notna(baseline_value) else None
            rows.append(row)
    return rows

def _write_key_solution(output_dir: Path, month_id: str, smooth_weight: float, allocation: pd.DataFrame, constraint_config: Dict[str, Any], objective_config: Dict[str, Any]) -> None:
    """Export a key allocation with the calibrated initial rule settings and objective weights. Solver status records any subsequent operational adjustments."""
    suffix = _weight_label(smooth_weight)
    extra_sheets = {'constraint_config': pd.DataFrame([{'key': key, 'value': str(value)} for key, value in constraint_config.items()]), 'objective_config': pd.DataFrame([{'key': key, 'value': str(value)} for key, value in objective_config.items()])}
    write_allocations_xlsx(allocation, output_dir / f'lp_marginal_solution_{month_id}_smooth_{suffix}.xlsx', extra_sheets=extra_sheets)

def _write_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Export the original overall, monthly-change and stability-fit plots with English labels."""
    if df.empty:
        return
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    aggregate = pd.DataFrame(_build_aggregate_rows(df)).sort_values('smooth_transition_weight')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [('reference_objective_value_mean', 'Reference objective (lower better)'), ('gnae_mean', 'GNAE (lower better)'), ('total_violations_mean', 'Constraint violations'), ('solution_l1_to_baseline_mean', 'Solution L1 distance to smooth=300'), ('stability_total_mean', 'Total-volume instability'), ('stability_sku_mean', 'SKU-level instability')]
    for axis, (column, label) in zip(axes.flat, panels):
        axis.plot(aggregate['smooth_transition_weight'], aggregate[column], marker='o', color='#0072B2', linewidth=2)
        axis.set_xlabel('smooth_transition_weight')
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    _save_figure(fig, output_dir / 'lp_marginal_overall_trends')
    months = sorted(df['month_id'].astype(str).unique().tolist())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, month_id in zip(axes.flat, months):
        part = df[df['month_id'].astype(str) == month_id].sort_values('smooth_transition_weight')
        axis.plot(part['smooth_transition_weight'], part['solution_l1_to_baseline'], marker='o', color='#D55E00', linewidth=2)
        axis.set_title(str(month_id))
        axis.set_xlabel('smooth_transition_weight')
        axis.set_ylabel('Normalized L1 distance')
        axis.grid(True, alpha=0.3)
    for axis in axes.flat[len(months):]:
        axis.set_visible(False)
    _save_figure(fig, output_dir / 'lp_marginal_solution_change_by_month')
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = axis.scatter(aggregate['stability_sku_mean'], aggregate['gnae_mean'], c=aggregate['smooth_transition_weight'], cmap='viridis', s=80, edgecolors='black', linewidths=0.5)
    for _, row in aggregate.iterrows():
        axis.annotate(f"{row['smooth_transition_weight']:g}", (row['stability_sku_mean'], row['gnae_mean']), xytext=(4, 4), textcoords='offset points')
    axis.set_xlabel('Mean SKU-level instability')
    axis.set_ylabel('Mean GNAE')
    axis.set_title('Stability–fit trade-off across smooth weights')
    axis.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=axis, label='smooth_transition_weight')
    _save_figure(fig, output_dir / 'lp_marginal_stability_fit_tradeoff')

def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    """Save the figure as PNG and PDF and release its plotting resources."""
    fig.savefig(path_without_suffix.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path_without_suffix.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

def _weight_label(value: float) -> str:
    """Format a transition weight for an output filename."""
    return str(int(value)) if float(value).is_integer() else str(value).replace('.', 'p')

def _align_by_code(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align allocation and reference rows using string-valued SKU identifiers."""
    if 'Code' not in a.columns or 'Code' not in b.columns:
        return (a.reset_index(drop=True), b.reset_index(drop=True))
    a2 = a.copy()
    b2 = b.copy()
    a2['Code'] = a2['Code'].astype(str)
    b2['Code'] = b2['Code'].astype(str)
    merged = a2.merge(b2, on='Code', how='inner', suffixes=('_a', '_b'))
    rounds = [column[:-2] for column in merged.columns if column.endswith('_a') and column[:-2] not in ['Code', 'Cigarette_Name']]
    out_a = merged.loc[:, ['Code'] + [f'{round_name}_a' for round_name in rounds]].rename(columns={f'{round_name}_a': round_name for round_name in rounds})
    out_b = merged.loc[:, ['Code'] + [f'{round_name}_b' for round_name in rounds]].rename(columns={f'{round_name}_b': round_name for round_name in rounds})
    return (out_a, out_b)

def _calibrate_c_constraints_from_label(constraint_config: Dict[str, Any], label_df: pd.DataFrame, rounds: List[str]) -> None:
    """Calibrate C-category capacity and subtype shares from the reference allocation."""
    if 'C' not in label_df.columns:
        return
    is_c = label_df['C'].fillna('').astype(str).str.strip().ne('')
    c_by_round = label_df.loc[is_c, rounds].fillna(0.0).astype(float).sum(axis=0)
    constraint_config['c_type_volume_limit'] = max(float(constraint_config.get('c_type_volume_limit', 0.0) or 0.0), float(c_by_round.max()) + 1.0)
    if 'C_Category' not in label_df.columns:
        return
    c_total = c_by_round.replace(0.0, pd.NA)
    chang_by_round = label_df.loc[label_df['C_Category'].fillna('').astype(str).eq('Long'), rounds].fillna(0.0).astype(float).sum(axis=0)
    xi_by_round = label_df.loc[label_df['C_Category'].fillna('').astype(str).eq('Slim'), rounds].fillna(0.0).astype(float).sum(axis=0)
    constraint_config['chang_type_volume_limit'] = max(float(constraint_config.get('chang_type_volume_limit', 0.0) or 0.0), float(chang_by_round.max()) + 1.0)
    constraint_config['xi_type_volume_limit'] = max(float(constraint_config.get('xi_type_volume_limit', 0.0) or 0.0), float(xi_by_round.max()) + 1.0)
    chang_ratio = (chang_by_round / c_total).dropna().max()
    xi_ratio = (xi_by_round / c_total).dropna().max()
    if pd.notna(chang_ratio):
        constraint_config['chang_type_ratio'] = max(float(constraint_config.get('chang_type_ratio', 0.0) or 0.0), float(chang_ratio) + 0.01)
    if pd.notna(xi_ratio):
        constraint_config['xi_type_ratio'] = max(float(constraint_config.get('xi_type_ratio', 0.0) or 0.0), float(xi_ratio) + 0.01)
if __name__ == '__main__':
    raise SystemExit(main())
