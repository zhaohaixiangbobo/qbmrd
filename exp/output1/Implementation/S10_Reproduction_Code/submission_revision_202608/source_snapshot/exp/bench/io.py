from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Review entry points share the authentic, deidentified source-data directories.
REVIEW_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_INPUT_DIR = REVIEW_ROOT / "Source_Data" / "Input"
DEFAULT_LABEL_DIR = REVIEW_ROOT / "Source_Data" / "Reference"
from typing import Dict, List

import pandas as pd

from .bootstrap import ensure_backend_on_path


@dataclass(frozen=True)
class LoadedMonthData:
    """Data loaded for one experimental month.

Attributes:
    month_id: Month identifier, such as 202501.
    input_path: Input workbook path.
    label_path: Optional allocation-label workbook path.
    rounds: Round columns, such as Round_1 through Round_5.
    data_loader: Backend DataLoader instance.
    product_data: Sheet1 product DataFrame.
    label_data: Optional allocation-label DataFrame."""

    month_id: str
    input_path: Path
    label_path: Path | None
    rounds: List[str]
    data_loader: object
    product_data: pd.DataFrame
    label_data: pd.DataFrame | None


def month_id_from_input_path(input_path: Path) -> str:
    """Extract a month identifier from an input path such as 202501.xlsx.

Returns:
    str: Month identifier, such as 202501."""
    return input_path.stem


def label_path_from_month_id(label_dir: Path, month_id: str) -> Path:
    """Prefer an existing YYYYMM.xlsx reference; otherwise retain the Month_N.xlsx example convention.

Args:
    label_dir: Label directory.
    month_id: Month identifier, such as 202501.

Returns:
    Path: Label workbook path."""
    month_num = int(month_id[-2:])
    ascii_month = label_dir / f"{month_id}.xlsx"
    if ascii_month.exists():
        return ascii_month
    return label_dir / f"Month_{month_num}.xlsx"


def load_month_data(input_path: Path, label_dir: Path) -> LoadedMonthData:
    """Load one month of input and label data, using the backend DataLoader to parse rounds and product data.

Args:
    input_path: Input workbook containing Sheet1 and Sheet2.
    label_dir: Directory containing Month_1.xlsx through Month_6.xlsx.

Returns:
    LoadedMonthData: Loaded data."""
    ensure_backend_on_path()
    from data_loader import DataLoader

    month_id = month_id_from_input_path(input_path)
    label_path = label_path_from_month_id(label_dir, month_id)
    label_data = load_label_sheet(label_path) if label_path.exists() else None

    data_loader = DataLoader(str(input_path))
    rounds = data_loader.get_rounds()
    product_data = data_loader.get_product_data()

    return LoadedMonthData(
        month_id=month_id,
        input_path=input_path,
        label_path=label_path if label_path.exists() else None,
        rounds=rounds,
        data_loader=data_loader,
        product_data=product_data,
        label_data=label_data,
    )


def load_label_sheet(label_path: Path) -> pd.DataFrame:
    """Read the first worksheet of a label workbook as allocation-label data.

Args:
    label_path: Label workbook path.

Returns:
    pd.DataFrame: Allocation-label data."""
    xls = pd.ExcelFile(label_path)
    sheet_name = xls.sheet_names[0]
    return pd.read_excel(label_path, sheet_name=sheet_name)


def standardize_allocation_matrix(df: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
    """Trim or complete an allocation matrix to a common schema for comparisons and metrics.

Args:
    df: Matrix containing SKU identifiers and round columns.
    rounds: Round column names.

Returns:
    pd.DataFrame: Standardized allocation matrix."""
    keep_cols = [c for c in ['Code', 'Cigarette_Name'] if c in df.columns] + rounds
    out = df.loc[:, keep_cols].copy()
    for r in rounds:
        if r not in out.columns:
            out[r] = 0.0
    return out


def build_allocation_only_frame(product_data: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
    """Build an empty allocation matrix containing SKU identifiers and round columns.

Args:
    product_data: Product data containing Code, Cigarette_Name, and Demand.
    rounds: Round column names.

Returns:
    pd.DataFrame: Allocation-matrix template."""
    base_cols = [c for c in ['Code', 'Cigarette_Name'] if c in product_data.columns]
    df = product_data.loc[:, base_cols].copy()
    for r in rounds:
        df[r] = 0.0
    return df

