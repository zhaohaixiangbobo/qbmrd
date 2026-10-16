"""Recompute manuscript month-level GNAE inference from archived CSV records.

This adapter calls the supplied historical statistical function without solving
any allocation model. Outputs are written to a new directory.
"""
from pathlib import Path
import argparse
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    from revision.runner import build_month_level_gnae_reanalysis
    monthly, tests, table = build_month_level_gnae_reanalysis(pd.read_csv(args.runs))
    # Exclusive creation preserves any previous verification outputs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in [('monthly.csv', monthly), ('tests.csv', tests), ('table.csv', table)]:
        pd.DataFrame(records).to_csv(args.output_dir / name, index=False, encoding='utf-8-sig')
    print('Completed six-month statistical recheck; no optimization executed.')


if __name__ == '__main__':
    main()
