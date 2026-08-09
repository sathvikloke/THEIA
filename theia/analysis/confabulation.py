"""Does the generated rationale claim things the model cannot know?

A rationale that is fluent, plausible and invented is worse than no rationale.
It is also exactly what this model currently produces: given a patient annotated
*"A spiculated round solid lesion, peripheral, with attachment to pleura"* it
emitted *"A 55-year-old woman was referred to our hospital for a solid nodule in
the right lower lobe."* — an age, a sex, a laterality and a referral history,
none of which are in its input. The model sees a cropped CT and nothing else.

This is a gate, not a metric. It runs before any reader sees a case, because the
failure it catches is the one a reader cannot catch: a confident clinical
sentence is indistinguishable from a correct one unless you hold the source.

Three classes of claim, in descending order of how badly they mislead:

  UNGROUNDABLE   The model has no channel through which it could know this.
                 Age, sex, referral history, prior imaging, symptoms, treatment.
                 Any occurrence is a failure, full stop.
  UNSUPPORTED    Knowable from an image in principle, but absent from THIS
                 patient's annotation. Laterality and lobe are the common ones.
  NUMERIC        A measurement not present in the source. "5 cm" when the
                 annotation says "larger than 5 mm" is a fabricated precision.

Run: python -m theia.analysis.confabulation --ckpt checkpoints/<run>/fold4/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re

# Nothing in a cropped CT tells the model a patient's age or sex, that they were
# referred, or what happened to them afterwards. These are the model reciting
# its language-model prior over case reports.
UNGROUNDABLE = {
    "age": r"\b\d{1,3}[- ]year[- ]old\b|\bin (?:his|her) \d0s\b|\baged \d{1,3}\b",
    "sex": r"\b(?:wo)?man\b|\bfemale\b|\bmale\b|\bgentleman\b|\blady\b|\bboy\b|\bgirl\b",
    "history": r"\breferred\b|\badmitted\b|\bpresented with\b|\bcomplain\w*\b|"
               r"\bhistory of\b|\bunderwent\b|\bfollow[- ]up\b|\bbiopsy\b|"
               r"\bresect\w*\b|\bchemotherap\w*\b|\bsurger\w*\b",
    "symptom": r"\bcough\w*\b|\bdyspn\w*\b|\bhemoptysis\b|\bweight loss\b|\bfever\b|"
               r"\bchest pain\b|\basymptomatic\b",
    "outcome": r"\bsurviv\w*\b|\bdied\b|\bmortality\b|\brecurren\w*\b|\bmetasta\w*\b|"
               r"\bstage (?:I|II|III|IV)\b",
    "identity": r"\bour hospital\b|\bwe report\b|\ba case of\b|\bthis patient\b",
}

# Knowable from imaging in principle, so only a failure when the patient's own
# annotation does not say it.
UNSUPPORTED = {
    "laterality": r"\b(?:right|left)\b",
    "lobe": r"\b(?:upper|middle|lower) lobe\b|\blingula\b",
}

NUMERIC = r"\b\d+(?:\.\d+)?\s?(?:mm|cm)\b"


def _hits(text: str, patterns: dict) -> dict[str, list[str]]:
    out = {}
    for name, pat in patterns.items():
        found = re.findall(pat, text, flags=re.I)
        if found:
            out[name] = sorted({f if isinstance(f, str) else f[0] for f in found})
    return out


def check_one(generated: str, source: str) -> dict:
    """Audit one rationale against the annotation it was supposed to describe."""
    gen = (generated or "").lower()
    src = (source or "").lower()
    report: dict = {"ungroundable": _hits(gen, UNGROUNDABLE), "unsupported": {},
                    "numeric": []}

    for name, pat in UNSUPPORTED.items():
        in_gen = {m.lower() for m in re.findall(pat, gen, flags=re.I)}
        in_src = {m.lower() for m in re.findall(pat, src, flags=re.I)}
        extra = sorted(in_gen - in_src)
        if extra:
            report["unsupported"][name] = extra

    gen_nums = {m.lower().replace(" ", "") for m in re.findall(NUMERIC, gen, flags=re.I)}
    src_nums = {m.lower().replace(" ", "") for m in re.findall(NUMERIC, src, flags=re.I)}
    report["numeric"] = sorted(gen_nums - src_nums)

    report["n_ungroundable"] = sum(len(v) for v in report["ungroundable"].values())
    report["n_unsupported"] = sum(len(v) for v in report["unsupported"].values())
    report["n_numeric"] = len(report["numeric"])
    report["clean"] = (report["n_ungroundable"] == 0 and report["n_unsupported"] == 0
                       and report["n_numeric"] == 0)
    return report


def audit(pairs: list[tuple[str, str]]) -> dict:
    """Audit (generated, source) pairs. `pairs` may be empty."""
    per = [check_one(g, s) for g, s in pairs]
    n = len(per) or 1
    return {
        "n": len(per),
        "clean_rate": sum(p["clean"] for p in per) / n,
        "ungroundable_rate": sum(p["n_ungroundable"] > 0 for p in per) / n,
        "unsupported_rate": sum(p["n_unsupported"] > 0 for p in per) / n,
        "numeric_rate": sum(p["n_numeric"] > 0 for p in per) / n,
        "per_case": per,
    }


def specificity(pairs: list[tuple[str, str]], n_perm: int = 500,
                seed: int = 1337) -> dict:
    """Is the rationale about THIS patient, or the same sentence for everyone?

    A rationale can be entirely free of invented claims and still be worthless:
    if the model emits one generic sentence for every scan, a reader rating it
    "plausible" is rating the template, not the model's reading of the image.

    Permutation test. Token overlap (Jaccard) between each generated text and
    its OWN annotation, against the null built by repeatedly re-pairing
    generations to other patients' annotations. Both sides share whatever
    vocabulary the template imposes, so the comparison isolates patient-specific
    content.

    n_perm matters and a single permutation is not enough: measured on this
    checkpoint, one draw gave a gap of +0.003 and another +0.032. A gate built on
    one draw would flip between PASS and FAIL by seed. The null is now averaged
    over many pairings and reported with an empirical p-value.
    """
    import numpy as np

    def toks(t: str) -> set:
        return set(re.findall(r"[a-z]{3,}", (t or "").lower()))

    n = len(pairs)
    if n < 3:
        return {"n": n, "distinct_frac": float(n > 0), "self_overlap": float("nan"),
                "cross_overlap": float("nan"), "specificity_gap": float("nan"),
                "p_value": float("nan"), "n_perm": 0}

    gen = [toks(g) for g, _ in pairs]
    src = [toks(s) for _, s in pairs]

    def jac(a: set, b: set) -> float:
        return len(a & b) / max(len(a | b), 1)

    observed = float(np.mean([jac(gen[i], src[i]) for i in range(n)]))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        # A derangement is not required, but self-pairs would leak the observed
        # statistic into the null and bias it upward.
        perm = np.array([(p + 1) % n if p == i else p for i, p in enumerate(perm)])
        null.append(np.mean([jac(gen[i], src[perm[i]]) for i in range(n)]))
    null = np.array(null)
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {
        "n": n,
        "distinct_frac": len({g for g, _ in pairs}) / n,
        "self_overlap": observed,
        "cross_overlap": float(null.mean()),
        "cross_overlap_sd": float(null.std()),
        "specificity_gap": observed - float(null.mean()),
        "p_value": p,
        "n_perm": n_perm,
    }


def gate(result: dict, max_ungroundable_rate: float = 0.0,
         spec: dict | None = None, min_specificity_gap: float = 0.02) -> tuple[bool, str]:
    """Is this checkpoint fit to put in front of a reader?

    Two independent ways to fail, because a rationale can be perfectly honest
    and still worthless.

    Ungroundable claims: threshold ZERO, deliberately. There is no acceptable
    rate at which a model invents a patient's age or sex in a clinical
    rationale, and a "small" rate is worse than a large one because it survives
    spot-checking and reaches a reader anyway.

    Specificity: if the same sentence is emitted for every scan, a reader
    scoring it "plausible" is scoring the template, not the model. Measured on
    the current checkpoint, self-overlap 0.547 against cross-overlap 0.543 -- a
    gap of +0.003, i.e. the text carries no patient-specific information at all.
    Passing the honesty check while failing this one is the more dangerous
    outcome, because the study would run and produce a publishable-looking
    number about nothing.
    """
    r = result["ungroundable_rate"]
    if r > max_ungroundable_rate:
        return False, (f"{100*r:.0f}% of rationales contain ungroundable claims "
                       f"(limit {100*max_ungroundable_rate:.0f}%). Not fit for a "
                       "reader study.")
    if spec is not None:
        gap = spec.get("specificity_gap", float("nan"))
        pv = spec.get("p_value", float("nan"))
        # Require both a real gap AND that it survives the permutation null;
        # either alone is too easy to hit by chance at n=31.
        if not (gap > min_specificity_gap and pv < 0.05):
            return False, (
                f"rationales are not patient-specific: overlap with the patient's "
                f"own annotation {spec.get('self_overlap', float('nan')):.3f} vs "
                f"{spec.get('cross_overlap', float('nan')):.3f} with another "
                f"patient's (gap {gap:+.3f}, p={pv:.3f}; need gap > "
                f"{min_specificity_gap:+.3f} at p<0.05); "
                f"{100*spec.get('distinct_frac', 0):.0f}% distinct. A reader would "
                "be scoring the template, not the model.")
    return True, (f"clean {100*result['clean_rate']:.0f}%, "
                  f"ungroundable {100*r:.0f}% — passes the gate.")


def main() -> None:
    import torch

    from theia.config import load_config
    from theia.data.dataset import RadiogenomicsDataset, collate, nested_kfold_indices
    from theia.models.theia_model import Theia
    from theia.runtime import resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results/confabulation.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = resolve_device("auto")
    state = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = Theia(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    rp = os.path.join(cfg.paths.processed_dir, "rows.jsonl")
    rows = [json.loads(l) for l in open(rp)]
    fold = int(state.get("fold", 0))
    seed = int((state.get("cfg") or {}).get("seed", cfg.seed))
    splits = list(nested_kfold_indices(rp, cfg.split.stratify_on, cfg.split.n_folds,
                                       seed, float(cfg.split.inner_val_frac)))
    held = list(splits[fold][2])
    ds = RadiogenomicsDataset(rp, cfg.data.target_genes)
    print(f"[confab] auditing fold {fold}'s {len(held)} held-out patients", flush=True)

    pairs = []
    for i in held:
        batch = collate([ds[i]], genes=model.genes)
        with torch.no_grad():
            out = model(batch, device, generate=True)
        pairs.append((out["text"][0], rows[i]["report"]))

    result = audit(pairs)
    spec = specificity(pairs)
    ok, msg = gate(result, spec=spec)
    print(f"\n[confab] {result['n']} rationales")
    print(f"[confab]   ungroundable (age/sex/history/symptom/outcome): "
          f"{100*result['ungroundable_rate']:.0f}%")
    print(f"[confab]   unsupported laterality or lobe:                 "
          f"{100*result['unsupported_rate']:.0f}%")
    print(f"[confab]   fabricated measurements:                        "
          f"{100*result['numeric_rate']:.0f}%")
    print(f"[confab]   fully clean:                                    "
          f"{100*result['clean_rate']:.0f}%")
    print(f"[confab]   distinct rationales:                            "
          f"{100*spec['distinct_frac']:.0f}%")
    print(f"[confab]   patient-specificity gap:                        "
          f"{spec['specificity_gap']:+.3f}  "
          f"(own {spec['self_overlap']:.3f} vs other {spec['cross_overlap']:.3f})")
    print(f"\n[confab] GATE: {'PASS' if ok else 'FAIL'} — {msg}")

    worst = sorted(result["per_case"], key=lambda p: -p["n_ungroundable"])[:3]
    for p in worst:
        if p["n_ungroundable"]:
            print(f"[confab]   e.g. {p['ungroundable']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({k: v for k, v in result.items() if k != "per_case"}
              | {"specificity": spec, "gate_pass": ok},
              open(a.out, "w"), indent=2)
    print(f"[confab] wrote {a.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
