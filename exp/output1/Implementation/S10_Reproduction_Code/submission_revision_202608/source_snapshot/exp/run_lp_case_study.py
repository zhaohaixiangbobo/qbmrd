"""English review copy of run_lp_case_study.py.
Historical source logic; see HISTORICAL_CODE_GUIDE.md for execution scope and provenance.
"""
from __future__ import annotations
import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data
from bench.metrics import validate_constraints
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_MONTH_ID = '202502'
DEFAULT_SMOOTH_WEIGHT = 300.0
EPSILON = 1e-08

def main() -> int:
    parser = argparse.ArgumentParser(description='Build historical case tables and allocation summaries from saved trace outputs.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--month-id', default=DEFAULT_MONTH_ID, help='Case month identifier.')
    parser.add_argument('--smooth-weight', type=float, default=DEFAULT_SMOOTH_WEIGHT, help='Case transition weight.')
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument('--label-dir', type=str, default=str(DEFAULT_LABEL_DIR))
    parser.add_argument('--relaxation-dir', type=str, default=str(EXP_DIR / 'output' / 'lp_relaxation_path_full'), help='Directory containing saved operational-escalation outputs.')
    parser.add_argument('--output-dir', type=str, default=str(EXP_DIR / 'output' / 'lp_case_study_202502'), help='Case output directory.')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    month_id = str(args.month_id)
    smooth_weight = float(args.smooth_weight)
    if not np.isclose(smooth_weight, DEFAULT_SMOOTH_WEIGHT):
        raise ValueError('This historical case requires transition weight 300; other weights require corresponding trace outputs.')
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    supplement_dir = output_dir / 'supplement'
    texts_dir = output_dir / 'manuscript_text'
    for directory in [figures_dir, tables_dir, supplement_dir, texts_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    relaxation_dir = Path(args.relaxation_dir)
    trace = _read_required_csv(relaxation_dir / 'lp_relaxation_trace.csv')
    trace_row = _select_first_optimal_trace_row(trace, month_id)
    source_xlsx = relaxation_dir / f'lp_relaxation_solution_{month_id}_first_optimal.xlsx'
    if not source_xlsx.exists():
        raise FileNotFoundError(f'missing first-optimal allocation workbook: {source_xlsx}')
    allocation = pd.read_excel(source_xlsx, sheet_name='allocation')
    applied_config = _read_config_from_workbook(source_xlsx)
    month = load_month_data(Path(args.input_dir) / f'{month_id}.xlsx', Path(args.label_dir))
    _verify_case_consistency(month, allocation, applied_config, trace_row, smooth_weight)
    columns = _resolve_columns(allocation)
    rounds = _round_columns(allocation.columns)
    ordered_allocation = _sort_full_allocation(allocation, columns)
    round_summary = _build_round_summary(ordered_allocation, columns, rounds)
    selected = _select_representative_skus(ordered_allocation, columns, rounds)
    selected_table = _build_selected_table(selected, columns, rounds)
    audit_table = _build_audit_table(trace_row, applied_config, month_id, smooth_weight, rounds)
    config_table = pd.DataFrame([{'Configuration item': key, 'Value': _display_value(value)} for key, value in sorted(applied_config.items())])
    _write_csv(round_summary, tables_dir / 'Table_3_case_202502_round_summary.csv')
    _write_markdown_table(round_summary, tables_dir / 'Table_3_case_202502_round_summary.md')
    _write_csv(selected_table, tables_dir / 'Table_3_case_202502_representative_skus.csv')
    _write_markdown_table(selected_table, tables_dir / 'Table_3_case_202502_representative_skus.md')
    _write_csv(audit_table, tables_dir / 'Table_3_case_202502_audit.csv')
    _write_csv(ordered_allocation, supplement_dir / 'Table_S5_case_202502_full_sku_allocation.csv')
    _write_csv(round_summary, supplement_dir / 'Table_S5_case_202502_round_summary.csv')
    _write_csv(config_table, supplement_dir / 'Table_S5_case_202502_constraint_config.csv')
    _write_csv(audit_table, supplement_dir / 'Table_S5_case_202502_audit.csv')
    shutil.copy2(source_xlsx, supplement_dir / 'Table_S5_case_202502_full_sku_allocation.xlsx')
    _write_audit_json(supplement_dir / 'Table_S5_case_202502_audit.txt', month_id=month_id, smooth_weight=smooth_weight, trace_row=trace_row, applied_config=applied_config, rounds=rounds)
    _plot_case_figure(allocation=ordered_allocation, selected=selected, columns=columns, rounds=rounds, round_summary=round_summary, output_base=figures_dir / 'Fig_7_case_study_202502_optimal_allocation', month_id=month_id)
    _write_manuscript_text(texts_dir=texts_dir, month_id=month_id, smooth_weight=smooth_weight, rounds=rounds, round_summary=round_summary, trace_row=trace_row)
    _write_readme(output_dir, month_id, smooth_weight, rounds, trace_row)
    print(json.dumps({'stage': 'lp_case_study', 'month_id': month_id, 'smooth_transition_weight': smooth_weight, 'status': str(trace_row['status']), 'post_validation_feasible': True, 'post_validation_violations': 0, 'round_count': len(rounds), 'sku_count': int(len(allocation)), 'representative_sku_count': int(len(selected)), 'output_dir': str(output_dir)}, ensure_ascii=False))
    return 0

def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'missing required artifact: {path}')
    return pd.read_csv(path)

def _select_first_optimal_trace_row(trace: pd.DataFrame, month_id: str) -> pd.Series:
    needed = {'month_id', 'is_first_optimal', 'is_optimal', 'feasible', 'total_violations', 'status'}
    missing = sorted(needed - set(trace.columns))
    if missing:
        raise ValueError(f'trace file is missing required columns: {missing}')
    part = trace[trace['month_id'].astype(str) == str(month_id)].copy()
    selected = part[part['is_first_optimal'].astype(bool) & part['is_optimal'].astype(bool)]
    if len(selected) != 1:
        raise ValueError(f'expected exactly one first LP optimal trace row for {month_id}, found {len(selected)}')
    row = selected.iloc[0]
    if not bool(row['feasible']) or int(row['total_violations']) != 0:
        raise ValueError(f"selected case {month_id} is not post-validated: feasible={row['feasible']}, violations={row['total_violations']}")
    return row

def _read_config_from_workbook(path: Path) -> Dict[str, Any]:
    config_df = pd.read_excel(path, sheet_name='constraint_config')
    if not {'key', 'value'}.issubset(config_df.columns):
        raise ValueError('constraint_config sheet must contain key and value columns')
    return {str(row['key']): _parse_scalar(row['value']) for _, row in config_df.iterrows()}

def _parse_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() == 'true':
        return True
    if text.lower() == 'false':
        return False
    if text.lower() in {'none', 'nan', ''}:
        return None
    try:
        numeric = float(text)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return text

def _verify_case_consistency(month: object, allocation: pd.DataFrame, applied_config: Dict[str, Any], trace_row: pd.Series, smooth_weight: float) -> None:
    expected = default_experiment_configs().objective_config.get('smooth_transition_weight')
    if not np.isclose(float(expected), smooth_weight):
        raise ValueError(f'default smooth weight changed from {smooth_weight} to {expected}; case source is no longer comparable')
    check = validate_constraints(month.data_loader, allocation, applied_config)
    if not check.feasible or check.total_violations != 0:
        raise ValueError(f'case allocation fails fresh validation under its applied configuration: feasible={check.feasible}, violations={check.total_violations}')
    if not str(trace_row['status']).startswith('Optimal'):
        raise ValueError(f"trace row is not an LP optimal result: {trace_row['status']}")
    if applied_config.get('enable_c_type_constraints') is not False:
        raise ValueError('the selected case must explicitly record the planned C-type constraint relaxation')
    for key in ['enable_demand_constraints', 'enable_volume_constraints', 'enable_price_constraints', 'enable_balance_constraints']:
        if applied_config.get(key) is not True:
            raise ValueError(f'case configuration unexpectedly disables {key}')

def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    expected = {'code': 'Code', 'name': 'Cigarette_Name', 'category': 'Class', 'demand': 'Demand', 'supply': 'Available_Supply', 'total': 'Total_Allocation', 'rate': 'Allocation_Rate'}
    missing = [label for label in expected.values() if label not in df.columns]
    if missing:
        raise ValueError(f'allocation matrix is missing columns: {missing}')
    return expected

def _round_columns(columns: Iterable[object]) -> List[str]:
    rounds = [str(column) for column in columns if str(column).startswith('Round_') and str(column)[6:].isdigit()]
    if len(rounds) < 2:
        raise ValueError(f'expected at least two release rounds, found: {rounds}')
    return rounds

def _sort_full_allocation(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out[columns['demand']] = pd.to_numeric(out[columns['demand']], errors='coerce').fillna(0.0)
    out[columns['total']] = pd.to_numeric(out[columns['total']], errors='coerce').fillna(0.0)
    out[columns['category']] = pd.to_numeric(out[columns['category']], errors='coerce').fillna(9999)
    return out.sort_values([columns['category'], columns['demand'], columns['code']], ascending=[True, False, True], kind='mergesort').reset_index(drop=True)

def _build_round_summary(df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> pd.DataFrame:
    categories = sorted(df[columns['category']].dropna().unique().tolist())
    rows: List[Dict[str, Any]] = []
    for round_name in rounds:
        values = pd.to_numeric(df[round_name], errors='coerce').fillna(0.0)
        row: Dict[str, Any] = {'Release round': round_name, 'Total_Allocation': float(values.sum()), 'Active SKU count': int((values > EPSILON).sum())}
        for category in categories:
            row[f'Class {_display_value(category)} allocation'] = float(values[df[columns['category']] == category].sum())
        rows.append(row)
    return pd.DataFrame(rows)

def _select_representative_skus(df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> pd.DataFrame:
    work = df.copy()
    values = work.loc[:, rounds].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    means = values.mean(axis=1)
    work['_round_cv'] = np.divide(values.std(axis=1), means, out=np.zeros_like(means), where=means > EPSILON)
    work['_row_order'] = np.arange(len(work))
    rows: List[pd.DataFrame] = []
    for category in sorted(work[columns['category']].unique().tolist()):
        group = work[(work[columns['category']] == category) & (work[columns['total']] > EPSILON)].copy()
        if group.empty:
            continue
        by_demand = group.sort_values([columns['demand'], columns['code']], ascending=[False, True], kind='mergesort')
        picked = by_demand.head(2).copy()
        picked['_selection_role'] = 'Top two demands' if len(picked) > 1 else 'Only allocated SKU in category'
        picked['_selection_priority'] = 1
        remaining = group[~group.index.isin(picked.index)]
        if not remaining.empty:
            timing = remaining.sort_values(['_round_cv', columns['demand'], columns['code']], ascending=[False, False, True], kind='mergesort').head(1)
            timing['_selection_role'] = 'Largest round variation'
            timing['_selection_priority'] = 2
            picked = pd.concat([picked, timing], axis=0)
        rows.append(picked)
    selected = pd.concat(rows, axis=0, ignore_index=True)
    if len(selected) < 12:
        remaining = work[(work[columns['total']] > EPSILON) & ~work[columns['code']].isin(selected[columns['code']])]
        filler = remaining.sort_values([columns['demand'], columns['code']], ascending=[False, True], kind='mergesort').head(12 - len(selected)).copy()
        filler['_selection_role'] = 'Global demand-based completion'
        filler['_selection_priority'] = 3
        selected = pd.concat([selected, filler], axis=0, ignore_index=True)
    selected = selected.sort_values([columns['category'], '_selection_priority', columns['demand'], columns['code']], ascending=[True, True, False, True], kind='mergesort').reset_index(drop=True)
    if len(selected) != 12 or selected[columns['code']].duplicated().any():
        raise ValueError(f'representative SKU selection must contain 12 unique rows, got {len(selected)}')
    return selected

def _build_selected_table(df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame({'Category': df[columns['category']], 'SKU code': df[columns['code']].astype(str), 'SKU name': df[columns['name']].astype(str), 'Demand': pd.to_numeric(df[columns['demand']], errors='coerce'), 'Available_Supply': pd.to_numeric(df[columns['supply']], errors='coerce')})
    for round_name in rounds:
        out[round_name] = pd.to_numeric(df[round_name], errors='coerce').fillna(0.0)
    out['Total_Allocation'] = pd.to_numeric(df[columns['total']], errors='coerce').fillna(0.0)
    out['Allocation_Rate'] = pd.to_numeric(df[columns['rate']], errors='coerce')
    out['Round coefficient of variation'] = pd.to_numeric(df['_round_cv'], errors='coerce').fillna(0.0)
    out['Selection rule'] = df['_selection_role'].astype(str)
    return out

def _build_audit_table(trace_row: pd.Series, config: Dict[str, Any], month_id: str, smooth_weight: float, rounds: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{'Case month': month_id, 'Transition weight': smooth_weight, 'Round count': len(rounds), 'First optimal LP step': int(trace_row['step_index']), 'Operational adjustments': str(trace_row['action_label']), 'LP status': str(trace_row['status']), 'Zero external review flags': bool(trace_row['feasible']), 'External review-flag count': int(trace_row['total_violations']), 'Applied configuration': 'Automatically adjusted operational configuration'}])

def _plot_case_figure(allocation: pd.DataFrame, selected: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str], round_summary: pd.DataFrame, output_base: Path, month_id: str) -> None:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(17.5, 8.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.02, 1.12])
    ax_summary = fig.add_subplot(grid[0, 0])
    ax_all = fig.add_subplot(grid[0, 1])
    ax_selected = fig.add_subplot(grid[0, 2])
    _plot_round_summary(ax_summary, round_summary, rounds)
    _plot_full_heatmap(ax_all, allocation, columns, rounds)
    _plot_selected_heatmap(ax_selected, selected, columns, rounds)
    fig.suptitle(f'Case Study {month_id}: LP Optimal Allocation Plan Across Release Rounds', fontsize=15, fontweight='bold')
    for suffix, kwargs in [('.png', {'dpi': 300, 'facecolor': 'white'}), ('.pdf', {'facecolor': 'white'})]:
        fig.savefig(output_base.with_suffix(suffix), bbox_inches='tight', **kwargs)
    plt.close(fig)

def _plot_round_summary(axis: plt.Axes, summary: pd.DataFrame, rounds: Sequence[str]) -> None:
    x_values = np.arange(len(rounds))
    bottoms = np.zeros(len(rounds))
    category_columns = [column for column in summary.columns if str(column).endswith(' allocation')]
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    for index, column in enumerate(category_columns):
        values = pd.to_numeric(summary[column], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        axis.bar(x_values, values, bottom=bottoms, width=0.66, label=f'Class {index + 1}', color=colors[index % len(colors)])
        bottoms += values
    axis.set_xticks(x_values, [f'Round {index + 1}' for index in range(len(rounds))])
    axis.set_ylabel('Allocation volume (original unit)')
    axis.set_title('(a) Round-level allocation and active SKUs')
    axis.grid(axis='y', alpha=0.25)
    axis.set_axisbelow(True)
    secondary = axis.twinx()
    active = pd.to_numeric(summary['Active SKU count'], errors='coerce').to_numpy(dtype=float)
    secondary.plot(x_values, active, color='#333333', marker='o', linewidth=2, label='Active SKUs')
    secondary.set_ylabel('Number of active SKUs')
    for x_value, total in zip(x_values, bottoms):
        axis.text(x_value, total, f'{total:,.0f}', ha='center', va='bottom', fontsize=8)
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=8, frameon=False)

def _plot_full_heatmap(axis: plt.Axes, df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> None:
    values = df.loc[:, rounds].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, totals, out=np.zeros_like(values), where=totals > EPSILON)
    normalized[totals[:, 0] <= EPSILON, :] = np.nan
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#E6E6E6')
    image = axis.imshow(normalized, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest', rasterized=True)
    axis.set_xticks(np.arange(len(rounds)), [f'Round {index + 1}' for index in range(len(rounds))])
    axis.set_yticks([])
    axis.set_xlabel('Release round')
    axis.set_ylabel(f'All {len(df)} SKUs (sorted by class and demand)')
    axis.set_title('(b) Allocation shares across all SKUs')
    categories = df[columns['category']].to_numpy()
    starts = np.r_[0, np.flatnonzero(categories[1:] != categories[:-1]) + 1]
    ends = np.r_[starts[1:], len(df)]
    for start, end in zip(starts, ends):
        if start > 0:
            axis.axhline(start - 0.5, color='white', linewidth=1.0, alpha=0.9)
        category = _display_value(categories[start])
        axis.text(-0.65, (start + end - 1) / 2, f'Class {category}', ha='right', va='center', fontsize=8, clip_on=False)
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
    colorbar.set_label("Share of an SKU's allocation across rounds")
    inactive_count = int((totals[:, 0] <= EPSILON).sum())
    axis.text(0.5, -0.1, f'Grey: zero-allocation SKUs (n={inactive_count})', transform=axis.transAxes, ha='center', va='top', fontsize=8)

def _plot_selected_heatmap(axis: plt.Axes, df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> None:
    values = df.loc[:, rounds].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, totals, out=np.zeros_like(values), where=totals > EPSILON)
    image = axis.imshow(normalized, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0, interpolation='nearest')
    axis.set_xticks(np.arange(len(rounds)), [f'Round {index + 1}' for index in range(len(rounds))])
    labels = [f"SKU {row[columns['code']]}" for _, row in df.iterrows()]
    axis.set_yticks(np.arange(len(df)), labels, fontsize=8)
    axis.set_xlabel('Release round')
    axis.set_title('(c) Allocation shares for representative SKUs')
    for row_index in range(normalized.shape[0]):
        for col_index in range(normalized.shape[1]):
            value = normalized[row_index, col_index]
            text_color = 'white' if value >= 0.52 else 'black'
            axis.text(col_index, row_index, f'{value:.0%}', ha='center', va='center', fontsize=7, color=text_color)
    categories = df[columns['category']].to_numpy()
    boundaries = np.flatnonzero(categories[1:] != categories[:-1]) + 1
    for boundary in boundaries:
        axis.axhline(boundary - 0.5, color='white', linewidth=1.1)
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
    colorbar.set_label("Share of an SKU's allocation across rounds")

def _write_manuscript_text(texts_dir: Path, month_id: str, smooth_weight: float, rounds: Sequence[str], round_summary: pd.DataFrame, trace_row: pd.Series) -> None:
    totals = [float(value) for value in round_summary['Total_Allocation'].tolist()]
    active = [int(value) for value in round_summary['Active SKU count'].tolist()]
    rounds_text = '、'.join((f'{value:,.1f}' for value in totals))
    active_text = '、'.join((str(value) for value in active))
    results = f"To illustrate staged release, the historical case uses {month_id} as the representative case. Its first optimal LP occurs at step {int(trace_row['step_index'])} with zero external review flags under the applied configuration. The case has {len(rounds)} rounds and 277 SKUs; transition weight {smooth_weight:g}. Historical Figure 7 shows round totals of {rounds_text}, with active SKU counts {active_text}. The full matrix displays differentiated release profiles under the joint supply, price, demand and category rules, including sustained and concentrated allocations."
    discussion = 'The case provides a SKU-by-round planning matrix. Transition control limits unnecessary changes while allowing differentiated timing under operational constraints. Interpret the results under the saved applied configuration and audit scope.'
    response = 'Historical response text: the February case reports 277 SKUs over four rounds, 12 representative SKUs, and the saved first-optimal allocation and configuration. Historical figure/table numbers differ from the current submission.'
    (texts_dir / 'results_case_study.md').write_text(results + '\n', encoding='utf-8')
    (texts_dir / 'discussion_case_study.md').write_text(discussion + '\n', encoding='utf-8')
    (texts_dir / 'reviewer_response_case_study.md').write_text(response + '\n', encoding='utf-8')

def _write_readme(output_dir: Path, month_id: str, smooth_weight: float, rounds: Sequence[str], trace_row: pd.Series) -> None:
    text = f"# Historical LP case ({month_id})\n\nHistorical case assembly from the first chronologically optimal month with zero external review flags under its applied configuration.\n- Status: `{trace_row['status']}`; first optimal LP step: {int(trace_row['step_index'])}.\n- Objective: `smooth_transition_weight={smooth_weight:g}`.\n- Rounds: {'、'.join(rounds)}.\n- Configuration: saved operational-escalation settings.\n- Representative selection: top demands and strongest round variation by category, completed to 12 SKUs using global demand when necessary.\n- Outputs: historical figures, tables, full allocation, configuration and audit records. These historical labels do not replace the final manuscript numbering.\n"
    (output_dir / 'README.md').write_text(text, encoding='utf-8')

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding='utf-8-sig')

def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = [str(column) for column in df.columns]
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in df.iterrows():
        values = [str(row[column]).replace('|', '\\|').replace('\n', '<br>') for column in df.columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def _write_audit_json(path: Path, month_id: str, smooth_weight: float, trace_row: pd.Series, applied_config: Dict[str, Any], rounds: Sequence[str]) -> None:
    payload = {'month_id': month_id, 'smooth_transition_weight': smooth_weight, 'rounds': list(rounds), 'source': f'lp_relaxation_solution_{month_id}_first_optimal.xlsx', 'trace': {key: _jsonable(value) for key, value in trace_row.to_dict().items()}, 'applied_constraint_config': {key: _jsonable(value) for key, value in applied_config.items()}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value

def _display_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
if __name__ == '__main__':
    raise SystemExit(main())
