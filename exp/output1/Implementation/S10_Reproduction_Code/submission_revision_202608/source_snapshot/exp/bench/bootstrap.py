from __future__ import annotations

import sys
from pathlib import Path


def ensure_backend_on_path() -> Path:
    """Add the repository backend directory to sys.path for experiment-script reuse.

Returns:
    Path: Absolute backend directory."""
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root / "backend"
    backend_dir_str = str(backend_dir)
    if backend_dir_str not in sys.path:
        sys.path.insert(0, backend_dir_str)
    return backend_dir

