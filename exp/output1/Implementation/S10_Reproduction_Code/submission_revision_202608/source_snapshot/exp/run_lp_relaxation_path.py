"""English review copy of run_lp_relaxation_path.py.
Historical source logic; see HISTORICAL_CODE_GUIDE.md for execution scope and provenance.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data
from bench.metrics import validate_constraints
from bench.report import write_allocations_xlsx, write_rows_csv
from bench.solvers.lp_solver_trace import RelaxationAttempt, solve_with_lp_with_trace

def main() -> int:
    parser = argparse.ArgumentParser(description='Record the implemented operational-escalation sequence; this is not a global minimum-relaxation search.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--label-dir', type=str, default=str(DEFAULT_LABEL_DIR))
    parser.add_argument('--output-dir', type=str, default=str(Path(__file__).resolve().parent / 'output' / 'lp_relaxation_path'))
    parser.add_argument('--max-files', type=int, default=0, help='Maximum input files; zero selects all months.')
    parser.add_argument('--lp-time-limit', type=int, default=60, help='Time limit in seconds for each LP escalation step.')
    parser.add_argument('--no-export-details', action='store_true', help='Skip initial, first-optimal and fallback allocation workbooks.')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted((path for path in input_dir.glob('*.xlsx') if not path.name.startswith('~$')))
    if args.max_files > 0:
        input_paths = input_paths[:args.max_files]
    cfg = default_experiment_configs()
    trace_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    report_sections: List[str] = []
    for month_index, input_path in enumerate(input_paths, start=1):
        month = load_month_data(input_path=input_path, label_dir=label_dir)
        constraint_config = _build_constraint_config(cfg.constraint_config, month)
        print(json.dumps({'stage': 'lp_relaxation_path', 'month_id': month.month_id, 'month_progress': f'{month_index}/{len(input_paths)}', 'message': 'Starting LP operational-escalation analysis'}, ensure_ascii=False))
        trace = solve_with_lp_with_trace(data_loader=month.data_loader, constraint_config=constraint_config, objective_config=dict(cfg.objective_config), time_limit_sec=int(args.lp_time_limit))
        first_optimal_step = _first_optimal_step(trace.attempts)
        month_rows = _build_trace_rows(month, trace.attempts, first_optimal_step)
        trace_rows.extend(month_rows)
        summary = _build_summary_row(month.month_id, trace, first_optimal_step)
        summary_rows.append(summary)
        report_sections.append(_build_report_section(month.month_id, month_rows, summary))
        if not args.no_export_details:
            _export_key_attempts(output_dir, month.month_id, trace.attempts, first_optimal_step)
        print(json.dumps({'stage': 'lp_relaxation_path', 'month_id': month.month_id, 'attempt_count': len(trace.attempts), 'first_optimal_step': first_optimal_step, 'final_status': trace.output.status}, ensure_ascii=False))
    write_rows_csv(trace_rows, output_dir / 'lp_relaxation_trace.csv')
    write_rows_csv(summary_rows, output_dir / 'lp_relaxation_path_summary.csv')
    _write_markdown_report(output_dir / 'lp_relaxation_path_report.md', report_sections)
    _write_path_figure(pd.DataFrame(trace_rows), output_dir)
    return 0

def _build_constraint_config(base: Dict[str, Any], month: object) -> Dict[str, Any]:
    config = dict(base)
    if 'Price_Based' in month.product_data.columns:
        price_based_count = int(month.product_data['Price_Based'].fillna('').astype(str).str.strip().ne('').sum())
        if price_based_count <= 1:
            config['enable_price_based_constraints'] = False
    if month.label_data is not None:
        _calibrate_c_constraints_from_label(config, month.label_data, month.rounds)
    return config

def _first_optimal_step(attempts: List[RelaxationAttempt]) -> Optional[int]:
    for attempt in attempts:
        if attempt.is_lp_attempt and attempt.is_optimal:
            return int(attempt.step_index)
    return None

def _build_trace_rows(month: object, attempts: List[RelaxationAttempt], first_optimal_step: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for attempt in attempts:
        allocation = attempt.allocation_matrix_full
        has_round_columns = isinstance(allocation, pd.DataFrame) and (not allocation.empty) and all((round_name in allocation.columns for round_name in month.rounds))
        feasible: Optional[bool] = None
        total_violations: Optional[int] = None
        if has_round_columns:
            check = validate_constraints(month.data_loader, allocation, attempt.applied_constraint_config)
            feasible = bool(check.feasible)
            total_violations = int(check.total_violations)
        config = attempt.applied_constraint_config
        rows.append({'month_id': str(month.month_id), 'step_index': int(attempt.step_index), 'action': attempt.action, 'action_label': _action_label(attempt.action), 'cumulative_adjustments': json.dumps(attempt.cumulative_adjustments, ensure_ascii=False), 'is_lp_attempt': bool(attempt.is_lp_attempt), 'is_optimal': bool(attempt.is_optimal), 'is_first_optimal': bool(first_optimal_step == attempt.step_index), 'status': attempt.status, 'solve_time_sec': float(attempt.solve_time_sec), 'objective_value': attempt.objective_value, 'has_allocation_matrix': bool(has_round_columns), 'feasible': feasible, 'total_violations': total_violations, 'volume_tolerance': config.get('volume_tolerance'), 'enable_demand_split_constraints': config.get('enable_demand_split_constraints'), 'enable_price_based_constraints': config.get('enable_price_based_constraints'), 'enable_c_type_constraints': config.get('enable_c_type_constraints'), 'enable_price_constraints': config.get('enable_price_constraints'), 'applied_constraint_config': json.dumps(config, ensure_ascii=False, default=str, sort_keys=True)})
    return rows

def _build_summary_row(month_id: str, trace: object, first_optimal_step: Optional[int]) -> Dict[str, Any]:
    attempts = trace.attempts
    final_attempt = attempts[-1] if attempts else None
    return {'month_id': str(month_id), 'attempt_count': int(len(attempts)), 'lp_attempt_count': int(sum((1 for attempt in attempts if attempt.is_lp_attempt))), 'first_optimal_step': first_optimal_step, 'used_rule_based_fallback': bool(trace.used_rule_based_fallback), 'final_status': trace.output.status, 'final_solve_time_sec': float(trace.output.solve_time_sec), 'final_objective_value': trace.output.objective_value, 'final_action': final_attempt.action if final_attempt is not None else None, 'final_constraint_config': json.dumps(trace.final_constraint_config, ensure_ascii=False, default=str, sort_keys=True)}

def _export_key_attempts(output_dir: Path, month_id: str, attempts: List[RelaxationAttempt], first_optimal_step: Optional[int]) -> None:
    selected_steps = {0}
    if first_optimal_step is not None:
        selected_steps.add(first_optimal_step)
    if attempts and (not attempts[-1].is_lp_attempt):
        selected_steps.add(attempts[-1].step_index)
    for attempt in attempts:
        if attempt.step_index not in selected_steps or attempt.allocation_matrix_full.empty:
            continue
        label = 'initial' if attempt.step_index == 0 else 'first_optimal' if attempt.step_index == first_optimal_step else 'fallback_rule_based'
        extra_sheets = {'trace': pd.DataFrame([{'step_index': attempt.step_index, 'action': attempt.action, 'status': attempt.status, 'is_lp_attempt': attempt.is_lp_attempt, 'is_optimal': attempt.is_optimal}]), 'constraint_config': pd.DataFrame([{'key': key, 'value': str(value)} for key, value in attempt.applied_constraint_config.items()])}
        write_allocations_xlsx(attempt.allocation_matrix_full, output_dir / f'lp_relaxation_solution_{month_id}_{label}.xlsx', extra_sheets=extra_sheets)

def _build_report_section(month_id: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines = [f'## {month_id}', '']
    lines.append(f"- Total: {summary['lp_attempt_count']} LP attempts; final status: `{summary['final_status']}`。")
    if summary['first_optimal_step'] is None:
        lines.append('- No optimal LP solution was obtained. The final result is a RuleBased fallback or unavailable LP allocation.')
    else:
        lines.append(f"- First optimal LP step: {summary['first_optimal_step']}。")
    lines.append('')
    lines.append('| Step | Action | Status | First LP optimal | Zero review flags | Review-flag count |')
    lines.append('| --- | --- | --- | --- | --- | --- |')
    for row in rows:
        feasible_text = '' if row['feasible'] is None else str(row['feasible'])
        violations_text = '' if row['total_violations'] is None else str(row['total_violations'])
        lines.append(f"| {row['step_index']} | {row['action_label']} | {row['status']} | {row['is_first_optimal']} | {feasible_text} | {violations_text} |")
    lines.append('')
    return '\n'.join(lines)

def _write_markdown_report(path: Path, report_sections: List[str]) -> None:
    intro = ['# LP operational-escalation report', '', 'This report records the implemented escalation order. It does not enumerate all combinations or establish a globally minimum adjustment cost.', '']
    path.write_text('\n'.join(intro + report_sections), encoding='utf-8')

def _write_path_figure(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    months = sorted(df['month_id'].astype(str).unique().tolist())
    fig, axis = plt.subplots(figsize=(15, max(4.0, 1.25 * len(months))), constrained_layout=True)
    color_map = {'optimal': '#009E73', 'fallback': '#CC79A7', 'infeasible': '#D55E00', 'other': '#999999'}
    for y_value, month_id in enumerate(months):
        part = df[df['month_id'].astype(str) == month_id].sort_values('step_index')
        x_values = part['step_index'].astype(float).tolist()
        axis.plot(x_values, [y_value] * len(x_values), color='#666666', linewidth=1.5, zorder=1)
        for _, row in part.iterrows():
            category = _node_category(row)
            axis.scatter(row['step_index'], y_value, s=130, color=color_map[category], edgecolors='black', linewidths=0.5, zorder=2)
            axis.annotate(_action_figure_label(str(row['action'])), (row['step_index'], y_value), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, rotation=25)
    axis.set_yticks(range(len(months)), months)
    axis.set_xlabel('Relaxation step')
    axis.set_ylabel('Month')
    axis.set_title('LP automatic relaxation path (green: LP optimal; red: infeasible; purple: RuleBased fallback)')
    axis.grid(True, axis='x', alpha=0.25)
    fig.savefig(output_dir / 'lp_relaxation_path.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'lp_relaxation_path.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def _node_category(row: pd.Series) -> str:
    if not bool(row['is_lp_attempt']):
        return 'fallback'
    if bool(row['is_optimal']):
        return 'optimal'
    if str(row['status']) == 'Infeasible':
        return 'infeasible'
    return 'other'

def _action_label(action: str) -> str:
    mapping = {'initial': 'Initial constraints', 'enable_demand_split_constraints=False': 'Disable demand-split constraints', 'enable_price_based_constraints=False': 'Disable price-based share constraints', 'enable_c_type_constraints=False': 'Disable C-category constraints', 'enable_price_constraints=False': 'Disable price constraints', 'fallback:RuleBased': 'RuleBased fallback'}
    if action in mapping:
        return mapping[action]
    if action.startswith('volume_tolerance='):
        return 'Volume tolerance=' + action.split('=', 1)[1]
    return action

def _action_figure_label(action: str) -> str:
    mapping = {'initial': 'Initial constraints', 'enable_demand_split_constraints=False': 'Disable demand concentration', 'enable_price_based_constraints=False': 'Disable price proportion', 'enable_c_type_constraints=False': 'Relax structural constraint', 'enable_price_constraints=False': 'Disable price constraint', 'fallback:RuleBased': 'RuleBased fallback'}
    if action in mapping:
        return mapping[action]
    if action.startswith('volume_tolerance='):
        return 'Volume tolerance=' + action.split('=', 1)[1]
    return action

def _calibrate_c_constraints_from_label(constraint_config: Dict[str, Any], label_df: pd.DataFrame, rounds: List[str]) -> None:
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
