"""Configuration loading and path resolution.

The YAML is the single source of truth for every parameter used in a run. Entry
points load it through `load_config`, optionally apply overrides, then resolve
concrete filesystem paths for the active profile with `resolve_paths`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Optional[os.PathLike] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load the YAML config and apply optional deep overrides.

    The active profile can also be forced via the SOCOFING_PROFILE env var,
    which is convenient on Kaggle vs. local without editing the file.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    env_profile = os.environ.get("SOCOFING_PROFILE")
    if env_profile:
        cfg["paths"]["active_profile"] = env_profile
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in upd.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the active profile + subdir config into concrete Paths."""
    profile_name = cfg["paths"]["active_profile"]
    try:
        profile = cfg["paths"]["profiles"][profile_name]
    except KeyError as exc:  # pragma: no cover - config error
        raise KeyError(
            f"active_profile '{profile_name}' not found in paths.profiles"
        ) from exc

    root = Path(profile["dataset_root"])
    return {
        "profile": profile_name,
        "dataset_root": root,
        "results_dir": Path(profile["results_dir"]),
        "logs_dir": Path(profile["logs_dir"]),
        "cache_dir": Path(profile["cache_dir"]),
        "real_dir": root / cfg["paths"]["real_subdir"],
        "level_dirs": {
            level: root / subdir
            for level, subdir in cfg["paths"]["altered_levels"].items()
        },
    }


def dump_run_config(cfg: Dict[str, Any], out_path: os.PathLike) -> None:
    """Write the exact config used for a run, for reproducibility."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
