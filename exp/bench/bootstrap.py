from __future__ import annotations

import sys
from pathlib import Path


def ensure_backend_on_path() -> Path:
    """
    将仓库的 backend 目录加入 sys.path，便于实验脚本复用后端算法实现。

    Returns:
        Path: backend 目录绝对路径
    """
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root / "backend"
    backend_dir_str = str(backend_dir)
    if backend_dir_str not in sys.path:
        sys.path.insert(0, backend_dir_str)
    return backend_dir

