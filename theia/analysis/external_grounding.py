"""Does the grounding result survive on a cohort the model never saw?

This is THEIA's strongest internal result -- attention mass lift +0.391, 11.7x
chance, beating its own per-fold shuffled baseline in 14/14 folds -- and it is
the endpoint the analysis plan promotes to primary. An internal result that
replicates across seeds on ONE cohort is still one cohort. This evaluates the
trained grounding heads on the 420 NSCLC-Radiomics (Lung1, Maastro Clinic,
Netherlands) patients, a different country, different scanners, and a
radiotherapy-planning population rather than a surgical one.

Three controls, because "attention lands on the tumour" is easy to fake:

SPATIAL SHUFFLE (the floor already used internally). The same attention values,
permuted across cells. Destroys structure, preserves the value distribution, so
it isolates *where* the mass went from *how peaked* the map is.

CENTRE PRIOR (the control that actually bites here). These crops are lesion-
centred: measured over the external set, ROI centroids sit at (0.50, 0.51) of the
frame with sd ~0.08, covering 5.8% of the area. A model that learned nothing but
"look at the middle" scores well on mass. Note that pointing and area-matched IoU
are *identical* for any monotonically-decreasing function of distance-to-centre,
so those two columns are sigma-free; only mass depends on the width, and sigma is
set from the mean ROI radius.

WEIGHT RANDOMIZATION. A freshly initialised grounding head, same architecture,
no training. If this produces a non-trivial lift, the metric is measuring the
architecture's inductive bias rather than anything that was learned.

GATE D (pre-specified): external mass lift >= +0.20, beating the shuffle in >=10
of 14 evaluation folds, with the randomized head collapsing to ~0.

Run: python -m theia.analysis.external_grounding --runs ms-s42,ms-s7,peri-s42
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean, stdev

import numpy as np
import torch
from torch.utils.data import DataLoader


def _loader(rows_path: str, genes, batch_size: int, limit: int | None = None):
    from theia.data.dataset import RadiogenomicsDataset, make_collate

    ds = RadiogenomicsDataset(rows_path, genes)
    if limit:
        ds.rows = ds.rows[:limit]
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=make_collate(genes), num_workers=0)


@torch.no_grad()
def attention_over(model, loader, device, randomize: bool = False):
    """Collect (attn_maps, roi) for every external patient.

    Only the vision encoder and grounding head are run -- the classifier and the
    LM are cohort-specific and their outputs are meaningless here, and skipping
    the LM makes this minutes instead of hours.
    """
    if randomize:
        from theia.models.heads import GroundingHead
        n_queries, dim = model.grounding.queries.shape
        model.grounding = GroundingHead(dim, n_queries).to(device)
    model.eval()
    maps, rois = [], []
    for batch in loader:
        images = batch["images"].to(device)
        sm = batch.get("slice_mask")
        tokens, grid = model.vision.encode(images, sm.to(device) if sm is not None else None)
        g = model.grounding(tokens, grid)
        maps.append(g["attn_maps"].detach().float().cpu())
        rois.append(batch["roi"].detach().float().cpu())
    return maps, rois


def centre_prior(attn_shape, roi_frac: float) -> torch.Tensor:
    """A [B,Q,h,w] map that is a decreasing function of distance from centre.

    sigma is set so the Gaussian's half-mass radius matches the radius of a disc
    of area `roi_frac`, i.e. the prior is as tight as the lesions actually are.
    Making it tighter or looser only moves the mass column; pointing and IoU are
    invariant to sigma.
    """
    b, q, h, w = attn_shape
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    d2 = (yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2
    r = (roi_frac * h * w / np.pi) ** 0.5
    m = torch.exp(-d2 / (2 * max(r, 1e-3) ** 2))
    return m.expand(b, q, h, w).clone()


def summarise(maps, rois, seed: int, transform=None) -> dict:
    """Pooled grounding metrics over every external patient."""
    from theia.engine.evaluate import grounding_metrics

    acc: dict[str, list[float]] = {}
    for a, r in zip(maps, rois):
        if transform is not None:
            a = transform(a)
        out = grounding_metrics(a, r, seed=seed)
        for k, v in out.items():
            acc.setdefault(k, []).extend(v)
    res = {k: float(np.mean(v)) for k, v in acc.items() if v}
    for base in ("grounding_mass", "grounding_pointing", "grounding_iou"):
        if base in res and f"{base}_shuffled" in res:
            res[f"{base}_lift"] = res[base] - res[f"{base}_shuffled"]
    res["n"] = int(len(acc.get("grounding_mass", [])))
    return res


def main() -> None:
    from theia.config import load_config
    from theia.models.theia_model import Theia
    from theia.runtime import resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", default="ms-s42,ms-s7,peri-s42,peri-s7",
                    help="comma-separated run_ids under checkpoints/")
    ap.add_argument("--rows", default="data/processed_pretrain/rows.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap patients, for a smoke run")
    ap.add_argument("--out", default="results/external_grounding_canonical.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = resolve_device("auto")
    genes = cfg.data.target_genes
    loader = _loader(a.rows, genes, cfg.train.batch_size, a.limit)
    n_ext = len(loader.dataset)
    print(f"[ext] {n_ext} external patients from {a.rows}")
    print(f"[ext] device={device}\n")

    rows = []
    per_fold_lift = []
    for run in a.runs.split(","):
        for ck in sorted(glob.glob(f"checkpoints/{run}/fold*/best.pt")):
            fold = os.path.basename(os.path.dirname(ck))
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            model = Theia(cfg).to(device)
            # Load ONLY vision + grounding. The classifier's input width depends on
            # model.peritumoral_features and the LM may carry LoRA adapters, so a
            # strict full-model load fails across configs that are irrelevant here
            # -- nothing downstream of the attention maps is used. Restricting the
            # load also means a silently-renamed grounding key shows up as a
            # missing-key count rather than as a quietly untrained head.
            want = {k: v for k, v in blob["model"].items()
                    if k.startswith("vision.") or k.startswith("grounding.")}
            missing, unexpected = model.load_state_dict(want, strict=False)
            got = [k for k in want if k.startswith("grounding.")]
            still_missing = [k for k in missing
                             if k.startswith(("vision.", "grounding."))]
            if not got or still_missing:
                print(f"[ext] skip {run}/{fold}: loaded {len(got)} grounding tensors, "
                      f"{len(still_missing)} vision/grounding keys missing")
                continue
            maps, rois = attention_over(model, loader, device)
            r = summarise(maps, rois, seed=cfg.seed)
            r.update(run=run, fold=fold)
            rows.append(r)
            per_fold_lift.append(r.get("grounding_mass_lift", float("nan")))
            print(f"[ext] {run}/{fold}: mass {r['grounding_mass']:.3f} "
                  f"vs shuffled {r['grounding_mass_shuffled']:.3f} "
                  f"(lift {r['grounding_mass_lift']:+.3f}), "
                  f"pointing {r['grounding_pointing']:.3f}, "
                  f"iou {r['grounding_iou']:.3f}, peak {r['grounding_peak_ratio']:.2f}")
            del model

    if not rows:
        raise SystemExit("[ext] no checkpoint loaded; nothing to report")

    # --- controls, computed once on the last model's geometry -----------------
    roi_frac = float(np.mean([r["grounding_roi_frac"] for r in rows]))
    shape = maps[0].shape
    cp = centre_prior(shape, roi_frac)
    ctrl_centre = summarise([cp[: m.shape[0]] for m in maps], rois, seed=cfg.seed)

    model = Theia(cfg).to(device)
    rmaps, rrois = attention_over(model, loader, device, randomize=True)
    ctrl_random = summarise(rmaps, rrois, seed=cfg.seed)

    lifts = [x for x in per_fold_lift if not np.isnan(x)]
    beat = sum(1 for r in rows if r["grounding_mass"] > r["grounding_mass_shuffled"])
    m_lift = mean(lifts)
    # The plan wrote this threshold as "10 of 14 folds", a count taken from the
    # internal run's fold total. The number of external evaluations is however
    # many run x fold checkpoints exist, so the count is expressed as the same
    # proportion (10/14 = 0.714) to preserve the intent. Recorded as a deviation.
    beat_frac = beat / max(len(rows), 1)
    gate = bool(m_lift >= 0.20 and beat_frac >= 10 / 14
                and abs(ctrl_random.get("grounding_mass_lift", 0)) < 0.10
                and mean(r["grounding_pointing"] for r in rows) > ctrl_centre["grounding_pointing"])

    print(f"\n[ext] CONTROLS")
    print(f"[ext]   centre prior      mass {ctrl_centre['grounding_mass']:.3f} "
          f"(lift {ctrl_centre['grounding_mass_lift']:+.3f}), "
          f"pointing {ctrl_centre['grounding_pointing']:.3f}, iou {ctrl_centre['grounding_iou']:.3f}")
    print(f"[ext]   random-init head  mass {ctrl_random['grounding_mass']:.3f} "
          f"(lift {ctrl_random['grounding_mass_lift']:+.3f}), "
          f"peak {ctrl_random['grounding_peak_ratio']:.2f}")
    print(f"\n[ext] trained mass lift {m_lift:+.3f} +/- {stdev(lifts) if len(lifts)>1 else 0:.3f} "
          f"over {len(lifts)} folds; beats shuffle in {beat}/{len(rows)}")
    for k in ("grounding_mass", "grounding_pointing", "grounding_iou"):
        print(f"[ext] vs centre prior, {k.split('_')[1]:<8} "
              f"{mean(r[k] for r in rows):.3f} vs {ctrl_centre[k]:.3f} "
              f"({mean(r[k] for r in rows) - ctrl_centre[k]:+.3f})")
    print(f"[ext] GATE D (lift >=0.20, beats shuffle in >=71% of folds, "
          f"random head ~0, pointing > centre prior): {'PASS' if gate else 'FAIL'}")

    json.dump({"n_external": n_ext, "rows": rows,
               "control_centre_prior": ctrl_centre, "control_random_init": ctrl_random,
               "mean_mass_lift": m_lift, "folds_beating_shuffle": beat,
               "frac_beating_shuffle": beat_frac,
               "n_folds": len(rows), "gate_d_passed": gate},
              open(a.out, "w"), indent=2)
    print(f"[ext] wrote {a.out}")


if __name__ == "__main__":
    main()
