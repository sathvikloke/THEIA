"""Config loading: a thin typed wrapper over the YAML in configs/."""
from __future__ import annotations

import os
from typing import Any

import yaml

VALID_SLICE_STRATEGIES = {"tumor_bbox", "max_area", "all"}
VALID_POOLING = {"mean", "max"}


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


def load_config(path: str = "configs/default.yaml", overrides: dict | None = None,
                validate: bool = True) -> Config:
    """Load YAML config, apply flat dotted overrides (e.g. {'train.lr': 1e-4})."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw)
    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(cfg, dotted, value)
    _expand_paths(cfg)
    if validate:
        validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail at load time on settings that would otherwise fail silently or late.

    The expensive failure mode this prevents: `train.monitor` naming a metric that
    `evaluate` never emits. `metrics.get(monitor, nan)` then returns NaN forever,
    no checkpoint is ever selected, and you find out after the run.
    """
    errs: list[str] = []

    genes = [g.lower() for g in cfg.data.target_genes]
    if not genes:
        errs.append("data.target_genes is empty")
    if len(set(genes)) != len(genes):
        errs.append(f"data.target_genes has duplicates: {cfg.data.target_genes}")

    declared = [g.lower() for g in cfg.data.get("headline_genes", [])] + \
               [g.lower() for g in cfg.data.get("exploratory_genes", [])]
    if declared and set(declared) != set(genes):
        errs.append(
            f"data.target_genes {sorted(genes)} != headline + exploratory "
            f"{sorted(set(declared))}; the panel is defined in two places and they disagree"
        )

    if cfg.split.stratify_on.lower() not in genes:
        errs.append(f"split.stratify_on '{cfg.split.stratify_on}' is not in data.target_genes")

    monitor = cfg.train.monitor
    monitor_keys = [monitor] if isinstance(monitor, str) else [
        m if isinstance(m, str) else m[0] for m in monitor]
    allowed = {f"{g}_{suffix}" for g in genes
               for suffix in ("auc", "sens", "spec")} | {
        "grounding_mass", "grounding_pointing", "grounding_iou",
        "grounding_mass_lift", "grounding_pointing_lift", "grounding_iou_lift",
    }
    for mk in monitor_keys:
        if mk not in allowed:
            errs.append(f"train.monitor '{mk}' is not emitted by evaluate(). "
                        f"Valid: {', '.join(sorted(allowed))}")

    if cfg.data.slice_strategy not in VALID_SLICE_STRATEGIES:
        errs.append(f"data.slice_strategy '{cfg.data.slice_strategy}' not in "
                    f"{sorted(VALID_SLICE_STRATEGIES)}")
    pooling = cfg.model.get("region_pooling", "mean")
    if pooling not in VALID_POOLING:
        errs.append(f"model.region_pooling '{pooling}' not in {sorted(VALID_POOLING)}")

    if cfg.split.n_folds < 2:
        errs.append(f"split.n_folds must be >= 2, got {cfg.split.n_folds}")
    if not 0.0 < float(cfg.split.get("inner_val_frac", 0.2)) < 1.0:
        errs.append("split.inner_val_frac must be strictly between 0 and 1")
    if float(cfg.data.get("context_factor", 1.0)) < 1.0:
        errs.append("data.context_factor must be >= 1.0 (it scales the crop outward)")
    if cfg.data.n_slices < 1:
        errs.append(f"data.n_slices must be >= 1, got {cfg.data.n_slices}")
    if cfg.train.grad_accum < 1:
        errs.append(f"train.grad_accum must be >= 1, got {cfg.train.grad_accum}")

    if errs:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(errs))

    _warn_coupled(cfg)


def _warn_coupled(cfg: Config) -> None:
    """Warn on settings that are individually legal but wrong together.

    Not errors — you may legitimately sweep either one. But this exact pair has
    already cost this project a run: backbone_lr_mult was tuned to 0.3 for the
    warm-start experiments, and leaving it there after switching pretraining off
    silently reproduces neither configuration. A wrong-but-plausible AUC is worse
    than a crash, because nothing tells you to look.
    """
    warm = cfg.model.get("grounding_pretrain_ckpt")
    mult = float(cfg.train.get("backbone_lr_mult", 1.0))
    want = 0.3 if warm else 1.0
    if mult != want:
        state = "set" if warm else "null"
        print(f"[config] WARNING: model.grounding_pretrain_ckpt is {state} but "
              f"train.backbone_lr_mult is {mult}, not the measured {want} for that "
              "case. See the comments on both keys; this pair must move together.")


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
