"""Full 5-fold nested run on the real cohort, sized for a 24 GB M4 Pro."""
import sys; sys.path.insert(0, "/Users/sathvikloke/Downloads/THEIA")

def main():
    import json, os
    from pathlib import Path
    from theia.config import load_config
    from theia.data.dataset import nested_kfold_indices
    from theia.engine.train import train_fold, _summarize, _report_pooled
    from theia.runtime import amp_settings, describe, resolve_device

    cfg = load_config("configs/default.yaml")
    # 11.2 GB peak at batch_size=2 on this machine; 4 would swap. grad_accum
    # keeps the effective batch at the configured 16.
    cfg["train"].update(batch_size=2, grad_accum=8, num_workers=2)
    dev = resolve_device("auto")
    amp_on, _ = amp_settings(dev, cfg.train.amp)
    print(f"[run] {describe(dev, amp_on)}", flush=True)

    rows = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    summary = []
    for fold, (tr, va, te) in enumerate(nested_kfold_indices(
            rows, cfg.split.stratify_on, cfg.split.n_folds, cfg.seed,
            cfg.split.inner_val_frac)):
        print(f"\n[run] === fold {fold}: train={len(tr)} val={len(va)} test={len(te)} ===",
              flush=True)
        summary.append(train_fold(cfg, fold, tr, va, te, dev))
        Path(cfg.paths.runs_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.paths.runs_dir, "cv_summary.json"), "w") as fh:
            json.dump({"folds": summary, "pooled": {}}, fh, indent=2)

    print("\n" + _summarize(summary, cfg.train.monitor), flush=True)
    pooled = _report_pooled(summary, cfg)
    with open(os.path.join(cfg.paths.runs_dir, "cv_summary.json"), "w") as fh:
        json.dump({"folds": summary, "pooled": pooled}, fh, indent=2)
    print("\n[run] complete", flush=True)

if __name__ == "__main__":
    main()
