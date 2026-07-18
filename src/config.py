"""
YAML config loader with validation.
=====================================
Loads patch_tssl_pilot.yaml / patch_tssl_full.yaml
Checks required fields, types, and cross-consistency.
Also provides helper to resolve data paths with --data-root override.
"""

from pathlib import Path
from typing import Any

import yaml

# ── Required top-level sections ──────────────────────
REQUIRED_SECTIONS = ["data", "pretrain", "model", "training", "checkpoint"]

# ── Required keys per section ─────────────────────────
REQUIRED_KEYS = {
    "data":       ["train_csv", "val_csv", "test_csv", "data_dir", "seq_len", "n_channels"],
    "pretrain":   ["patch_len", "stride", "mask_ratio", "patch_num"],
    "model":      ["d_model", "n_heads", "n_layers", "d_ff", "dropout"],
    "training":   ["epochs", "batch_size", "lr", "seed", "device"],
    "checkpoint": ["save_dir", "save_best", "save_last"],
}


def load_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Check sections
    for sec in REQUIRED_SECTIONS:
        if sec not in cfg:
            raise ValueError(f"Missing config section: [{sec}]")

    # Check keys
    for sec, keys in REQUIRED_KEYS.items():
        for k in keys:
            if k not in cfg[sec]:
                raise ValueError(f"Missing key in [{sec}]: {k}")

    # Type / sanity checks
    if cfg["pretrain"]["stride"] != cfg["pretrain"]["patch_len"]:
        raise ValueError("SSL requires stride == patch_len (non-overlapping patches)")

    if cfg["data"]["seq_len"] != 6000:
        raise ValueError(f"seq_len must be 6000 for 30s×200Hz ECG, got {cfg['data']['seq_len']}")

    actual_patches = (cfg["data"]["seq_len"] - cfg["pretrain"]["patch_len"]) // cfg["pretrain"]["stride"] + 1
    if cfg["pretrain"]["patch_num"] != actual_patches:
        raise ValueError(f"patch_num mismatch: config={cfg['pretrain']['patch_num']}, computed={actual_patches}")

    if cfg["model"]["d_model"] % cfg["model"]["n_heads"] != 0:
        raise ValueError(f"d_model ({cfg['model']['d_model']}) must be divisible by n_heads ({cfg['model']['n_heads']})")

    if not (0 < cfg["pretrain"]["mask_ratio"] < 1):
        raise ValueError(f"mask_ratio must be in (0,1), got {cfg['pretrain']['mask_ratio']}")

    return cfg


def _project_root_from_config(config_path: str | Path | None) -> Path | None:
    """Infer project root from a config path, usually <root>/configs/*.yaml."""
    if config_path is None:
        return None

    cfg_path = Path(config_path).resolve()
    cfg_dir = cfg_path.parent
    if cfg_dir.name == "configs":
        return cfg_dir.parent
    return cfg_dir


def _resolve_relative(value: str, roots: list[Path]) -> str:
    """Resolve a relative path against candidate roots without changing intent."""
    path = Path(value)
    if path.is_absolute():
        return str(path)

    for root in roots:
        candidate = root / path
        if candidate.exists():
            return str(candidate)

    if roots:
        return str(roots[0] / path)
    return str(path)


def _resolve_under_root(value: str, root: Path) -> str:
    """Resolve data storage directories under the selected data root."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(root / path)


def resolve_paths(
    cfg: dict,
    data_root: str | None = None,
    output_dir: str | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """Resolve config paths while preserving the split files chosen by YAML.

    ``--data-root`` selects where ECG storage lives (raw_wfdb / records_npy).
    It must not silently replace a Pilot CSV with the full train/val split.
    """
    project_root = _project_root_from_config(config_path)
    data_root_path = Path(data_root).resolve() if data_root else None

    csv_roots: list[Path] = []
    if project_root is not None:
        csv_roots.append(project_root)
    if data_root_path is not None:
        csv_roots.append(data_root_path)

    for key in ["train_csv", "val_csv", "test_csv"]:
        cfg["data"][key] = _resolve_relative(cfg["data"][key], csv_roots)

    if data_root_path is not None:
        cfg["data"]["data_dir"] = _resolve_under_root(cfg["data"]["data_dir"], data_root_path)
        if cfg["data"].get("npy_dir"):
            cfg["data"]["npy_dir"] = _resolve_under_root(cfg["data"]["npy_dir"], data_root_path)
    elif project_root is not None:
        cfg["data"]["data_dir"] = _resolve_relative(cfg["data"]["data_dir"], [project_root])
        if cfg["data"].get("npy_dir"):
            cfg["data"]["npy_dir"] = _resolve_relative(cfg["data"]["npy_dir"], [project_root])

    if output_dir:
        cfg["checkpoint"]["save_dir"] = str(Path(output_dir))
    return cfg


def save_resolved_config(cfg: dict, path: str | Path) -> None:
    """Write the fully resolved config as YAML for reproducibility."""
    out = yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    Path(path).write_text(out, encoding="utf-8")
