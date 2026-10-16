"""HSSC submission-revision command-line entry point.

Each command writes results under a separate run_id without changing source_snapshot, the original exp/backend directories, or Excel data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from revision.common import ensure_business_config, ensure_snapshot, verify_snapshot_integrity
from revision.runner import run_ablation, run_benchmark, run_benchmark_gnae_reanalysis, run_case_study, run_field_table, run_preflight, run_robustness, run_scalability, run_uncertainty


def main() -> int:
    parser = argparse.ArgumentParser(description='HSSC submission revision: isolated snapshot, frozen rules, and reproducible results.')
    parser.add_argument("command", choices=["benchmark", "benchmark-reanalysis", "ablation", "uncertainty", "robustness", "case-study", "field-table", "scalability", "preflight", "verify"])
    parser.add_argument("--max-months", type=int, default=0, help='Small validation only; 0 selects all six months.')
    parser.add_argument("--max-seeds", type=int, default=0, help='Small benchmark validation only; 0 selects all 30 fixed seeds.')
    args = parser.parse_args()

    ensure_snapshot()
    ensure_business_config()
    if args.command == 'verify':
        result = verify_snapshot_integrity()
        print({"ok": result["ok"], "checks": result["checks"]})
        return 0 if result['ok'] else 1
    if args.command == "preflight":
        run_dir = run_preflight(args.max_months)
    elif args.command == "benchmark":
        run_dir = run_benchmark(args.max_months, args.max_seeds)
    elif args.command == "benchmark-reanalysis":
        run_dir = run_benchmark_gnae_reanalysis()
    elif args.command == "ablation":
        run_dir = run_ablation(args.max_months)
    elif args.command == "uncertainty":
        run_dir = run_uncertainty(args.max_months)
    elif args.command == "robustness":
        run_dir = run_robustness(args.max_months)
    elif args.command == "case-study":
        run_dir = run_case_study()
    elif args.command == "field-table":
        run_dir = run_field_table()
    else:
        run_dir = run_scalability(args.max_months)
    print(f"Complete. Results directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



