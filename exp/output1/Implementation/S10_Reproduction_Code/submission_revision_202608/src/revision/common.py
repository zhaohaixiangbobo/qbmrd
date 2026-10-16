"""Common snapshot, configuration, run-directory, and loading utilities.

Writes are confined to submission_revision_202608; original root exp and backend directories remain read-only."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys

# Prevent Python from creating __pycache__ inside the read-only source snapshot.
sys.dont_write_bytecode = True
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE = Path(__file__).resolve().parents[2]
REPO_ROOT = WORKSPACE.parent
SNAPSHOT_ROOT = WORKSPACE / "source_snapshot"
SNAPSHOT_EXP = SNAPSHOT_ROOT / "exp"
SNAPSHOT_BACKEND = SNAPSHOT_ROOT / "backend"
RESULTS_ROOT = WORKSPACE / "results"
CONFIG_ROOT = WORKSPACE / "config"


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 to detect replacement of data or snapshot files."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, include: Iterable[str] = ("*.py", "*.xlsx", "*.csv", "*.txt")) -> dict[str, str]:
    """Return deterministic relative paths and hashes for included files without modifying them."""
    paths: set[Path] = set()
    for pattern in include:
        paths.update(path for path in root.rglob(pattern) if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("~$"))
    return {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in sorted(paths)}


def ensure_snapshot() -> None:
    """Create the complete exp/backend snapshot initially; fail rather than overwrite an existing incomplete snapshot."""
    if SNAPSHOT_EXP.exists() or SNAPSHOT_BACKEND.exists():
        if SNAPSHOT_EXP.exists() and SNAPSHOT_BACKEND.exists():
            manifest_path = SNAPSHOT_ROOT / "source_hash_manifest.txt"
            if not manifest_path.exists():
                # If the initial copy stopped before manifest creation, write only the missing manifest and leave existing snapshot files untouched.
                write_json(manifest_path, {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_exp": hash_tree(REPO_ROOT / "exp"),
                    "source_backend": hash_tree(REPO_ROOT / "backend"),
                    "snapshot_exp": hash_tree(SNAPSHOT_EXP),
                    "snapshot_backend": hash_tree(SNAPSHOT_BACKEND),
                })
            return
        raise RuntimeError('source_snapshot is incomplete; inspect it manually before restoration to avoid overwriting evidence.')

    source_exp = REPO_ROOT / "exp"
    source_backend = REPO_ROOT / "backend"
    if not source_exp.is_dir() or not source_backend.is_dir():
        raise FileNotFoundError('Repository-root exp/ or backend/ directory was not found.')

    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source_exp, SNAPSHOT_EXP, ignore=ignore)
    shutil.copytree(source_backend, SNAPSHOT_BACKEND, ignore=ignore)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_exp": hash_tree(source_exp),
        "source_backend": hash_tree(source_backend),
        "snapshot_exp": hash_tree(SNAPSHOT_EXP),
        "snapshot_backend": hash_tree(SNAPSHOT_BACKEND),
    }
    write_json(SNAPSHOT_ROOT / "source_hash_manifest.txt", manifest)


def add_snapshot_import_paths() -> None:
    """Import experiment and backend implementations only from the snapshot."""
    ensure_snapshot()
    for path in (SNAPSHOT_EXP, SNAPSHOT_BACKEND):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON-formatted text with stable indentation within the submission workspace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read UTF-8 JSON-formatted configuration text and require a top-level object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return data


def ensure_business_config() -> Path:
    """Create a frozen configuration from backend defaults once; never overwrite an existing file automatically."""
    ensure_snapshot()
    path = CONFIG_ROOT / "business_frozen.txt"
    if path.exists():
        return path
    add_snapshot_import_paths()
    from models import ConstraintConfig, ObjectiveConfig

    constraints = ConstraintConfig()
    objective = ObjectiveConfig()
    constraints_data = constraints.model_dump() if hasattr(constraints, "model_dump") else constraints.dict()
    objective_data = objective.model_dump() if hasattr(objective, "model_dump") else objective.dict()
    write_json(
        path,
        {
            "description": 'Generated from snapshot backend defaults; business-owner confirmation is required before a formal submission run.',
            "constraints": constraints_data,
            "objective": objective_data,
        },
    )
    return path


def load_frozen_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read frozen business configuration; callers must not alter it in response to labels or solver status."""
    config = read_json(ensure_business_config())
    constraints = config.get("constraints")
    objective = config.get("objective")
    if not isinstance(constraints, dict) or not isinstance(objective, dict):
        raise ValueError('business_frozen.txt must contain constraints and objective objects.')
    return dict(constraints), dict(objective)


def load_protocol() -> dict[str, Any]:
    """Read the fixed budget, seed, and numerical-tolerance protocol."""
    return read_json(CONFIG_ROOT / "protocol.txt")


def create_run(command: str) -> Path:
    """Create a run directory without overwriting historical results and save reproducibility metadata."""
    ensure_snapshot()
    run_id = f"{command}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir = RESULTS_ROOT / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = RESULTS_ROOT / f"{run_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    constraints, objective = load_frozen_config()
    protocol = load_protocol()
    write_json(run_dir / "frozen_config.txt", {"constraints": constraints, "objective": objective})
    write_json(run_dir / "protocol.txt", protocol)
    write_json(
        run_dir / "run_metadata.txt",
        {
            "run_id": run_dir.name,
            "command": command,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "snapshot_manifest_sha256": sha256_file(SNAPSHOT_ROOT / "source_hash_manifest.txt"),
        },
    )
    return run_dir


def load_months(max_months: int = 0):
    """Load six monthly inputs and predeployment manual-plan labels from the snapshot."""
    add_snapshot_import_paths()
    from bench.io import load_month_data

    input_dir = SNAPSHOT_EXP / "data" / "input"
    label_dir = SNAPSHOT_EXP / "data" / "label"
    paths = sorted(path for path in input_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if max_months > 0:
        paths = paths[:max_months]
    return [load_month_data(path, label_dir) for path in paths]


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 CSV with pandas while retaining traceable files for empty outputs."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")




def verify_snapshot_integrity() -> dict[str, Any]:
    """Check original directories and the read-only snapshot against their initial manifest before submission."""
    ensure_snapshot()
    expected = read_json(SNAPSHOT_ROOT / "source_hash_manifest.txt")
    current = {
        "source_exp": hash_tree(REPO_ROOT / "exp"),
        "source_backend": hash_tree(REPO_ROOT / "backend"),
        "snapshot_exp": hash_tree(SNAPSHOT_EXP),
        "snapshot_backend": hash_tree(SNAPSHOT_BACKEND),
    }
    checks = {key: current[key] == expected.get(key) for key in current}
    return {"ok": all(checks.values()), "checks": checks, "current": current}

