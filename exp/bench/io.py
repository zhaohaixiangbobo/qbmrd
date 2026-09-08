from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .bootstrap import ensure_backend_on_path


# 实验使用的原始业务数据位于本地私有目录，不进入公开版本库。
EXPERIMENT_DATA_DIR = Path(__file__).resolve().parents[1] / "data1"
DEFAULT_INPUT_DIR = EXPERIMENT_DATA_DIR / "input"
DEFAULT_LABEL_DIR = EXPERIMENT_DATA_DIR / "label"


@dataclass(frozen=True)
class LoadedMonthData:
    """
    单个月份实验数据载入结果。

    Attributes:
        month_id: 形如 202501 的月份标识
        input_path: 输入 Excel 路径
        label_path: 真实值 Excel 路径（可选）
        rounds: 轮次列名列表（例如 第一轮..第五轮）
        data_loader: 后端 DataLoader 实例
        product_data: Sheet1 的产品数据（DataFrame）
        label_data: 真实值分配数据（DataFrame，可选）
    """

    month_id: str
    input_path: Path
    label_path: Path | None
    rounds: List[str]
    data_loader: object
    product_data: pd.DataFrame
    label_data: pd.DataFrame | None


def month_id_from_input_path(input_path: Path) -> str:
    """
    从输入文件名提取月份标识。

    Args:
        input_path: 形如 202501.xlsx 的路径

    Returns:
        str: 形如 202501 的月份标识
    """
    return input_path.stem


def label_path_from_month_id(label_dir: Path, month_id: str) -> Path:
    """
    将 202501 形式的 month_id 映射到 label 文件（1月.xlsx..12月.xlsx）。

    Args:
        label_dir: label 目录
        month_id: 形如 202501

    Returns:
        Path: label 文件路径
    """
    month_num = int(month_id[-2:])
    return label_dir / f"{month_num}月.xlsx"


def load_month_data(input_path: Path, label_dir: Path) -> LoadedMonthData:
    """
    载入单个月份输入与真实值，并复用后端 DataLoader 解析轮次与产品数据。

    Args:
        input_path: 输入 Excel（含 Sheet1/Sheet2）
        label_dir: label 目录（含 1月.xlsx..6月.xlsx）

    Returns:
        LoadedMonthData: 载入结果
    """
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
    """
    读取 label Excel 的第一张工作表作为真实值分配数据。

    Args:
        label_path: label Excel 路径

    Returns:
        pd.DataFrame: 真实值 DataFrame
    """
    xls = pd.ExcelFile(label_path)
    sheet_name = xls.sheet_names[0]
    return pd.read_excel(label_path, sheet_name=sheet_name)


def standardize_allocation_matrix(df: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
    """
    将分配矩阵裁剪/补齐到统一字段，以便不同方法做对比与计算指标。

    Args:
        df: 分配矩阵（至少包含 SKU 标识列与轮次列）
        rounds: 轮次列名列表

    Returns:
        pd.DataFrame: 标准化后的分配矩阵
    """
    keep_cols = [c for c in ["代码", "卷烟名称"] if c in df.columns] + rounds
    out = df.loc[:, keep_cols].copy()
    for r in rounds:
        if r not in out.columns:
            out[r] = 0.0
    return out


def build_allocation_only_frame(product_data: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
    """
    构造只包含 SKU 标识列与轮次列的空分配矩阵模板。

    Args:
        product_data: 产品数据（包含 代码/卷烟名称/需求 等列）
        rounds: 轮次列名列表

    Returns:
        pd.DataFrame: 分配矩阵模板
    """
    base_cols = [c for c in ["代码", "卷烟名称"] if c in product_data.columns]
    df = product_data.loc[:, base_cols].copy()
    for r in rounds:
        df[r] = 0.0
    return df
