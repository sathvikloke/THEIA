"""Config loading: a thin typed wrapper over the YAML in configs/."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


class Config(dict):
    """Dict with attribute access and nested-safe .get, so cfg.train.lr works."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str = "configs/default.yaml", overrides: dict | None = None) -> Config:
    """Load YAML config, apply flat dotted overrides (e.g. {'train.lr': 1e-4})."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw)
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(cfg, dotted, value)
    _expand_paths(cfg)
    return cfg


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _expand_paths(cfg: Config) -> None:
    root = os.environ.get("THEIA_ROOT", ".")
    for k, v in cfg.get("paths", {}).items():
        cfg["paths"][k] = os.path.join(root, v) if not os.path.isabs(v) else v
