# THEIA architecture & how it maps to Q1 of the 12-month plan

The week numbers below describe the CT-only core, which is Q1 of the program in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md). The pathology branch, multi-gene panel,
external validation, deferral, and the discovery atlas are Q2–Q3 and are not built
yet.

## The model in one diagram

```
                     ┌───────────────────────────────┐
 CT ROI stack ──────▶│ VisionEncoder (BiomedCLIP ViT) │──▶ patch tokens [B,N,D]
 [B,S,1,H,W]         └───────────────────────────────┘         │
                                                               ▼
                                        ┌──────────────────────────────┐
                                        │ GroundingHead                 │
                                        │  Q learned region queries     │
                                        │  cross-attend over patches    │
                                        └──────────────────────────────┘
                                          │ region_emb [B,Q,D]   │ attn_maps [B,Q,H',W']
                              ┌───────────┘                      └───────────┐
                              ▼                                              ▼
                   ┌────────────────────┐                        supervised vs tumor ROI
                   │ ClassifierHead     │                        (Dice+BCE), the grounding
                   │ EGFR/KRAS logits   │                        signal that anchors attention
                   └────────────────────┘                                    │
                              │ mean-pool region_emb                         │
                              ▼                                              ▼
                   EGFR/KRAS + confidence                    ┌────────────────────────────┐
                                                             │ GenerationHead (LoRA LM)    │
                                             region_emb ────▶│ visual prefix -> rationale  │
                                                             └────────────────────────────┘
```

The classifier pools from the **grounded** region embeddings, so prediction and
localization share one representation. The strength of that coupling is a config
choice: the grounding loss supervises the element-wise **max** over region queries,
so only `model.region_pooling: max` makes the classifier consume exactly what
grounding supervises. Under the default `mean`, an unsupervised query can still
feed the prediction. Report which you used.

## Loss

`L = w_cls · CE(genes) + w_gen · LM(rationale) + w_ground · [Dice+BCE](attn ∪, ROI)`

Unknown labels are masked (`-1`). Grounding uses the union over region queries vs
the tumor mask downsampled to the patch grid.

Two implementation notes that are easy to get wrong:

- The BCE runs in float32 with autocast locally disabled.
  `F.binary_cross_entropy` is on PyTorch's CUDA autocast banned list, so with
  `amp: true` the shipped config used to raise on the first CUDA step.
- `roi_to_grid` marks a cell positive when the tumor covers most of it, and falls
  back to top-k coverage when that erases the mask entirely. Without the fallback,
  any lesion thinner than about ⅔ of a patch cell produced an empty target, and
  the grounding term silently contributed `0.000` while looking healthy.

## Evaluation

Grounding is reported three ways — attention mass in the ROI, a pointing game, and
an area-matched IoU — each beside a `*_shuffled` twin computed on a spatially
permuted copy of the same attention. That twin is the chance level for the crop
geometry actually in use, and it climbs as the tumor fills more of the frame.
`*_lift` is the difference and is the number to report.

The predecessor metric (peak-normalise, threshold at 0.1) scored random and
uniform attention identically to the trained model, because after normalisation
nearly every cell cleared the threshold. It reported the tumor's area fraction,
not localization quality.

## Plan mapping

| Weeks | Plan step | Code |
|------|-----------|------|
| 1–2  | Data access, GPU, baselines | `data/download.py`, `configs/default.yaml` |
| 3–5  | Preprocess, ROI, pseudo-reports, splits | `data/preprocess.py`, `data/dataset.py` |
| 6–9  | Model build, ablations | `models/*`, `engine/train.py`, `engine/losses.py` |
| 10–11| Retrospective validation | `engine/evaluate.py` (AUC+CI, grounding IoU) |
| 12–13| Reader study | `reader_study/build_cases.py`, `reader_study/serve.py` |
| 14–15| Writeup | overlays from `build_cases.py` are your figures |

## Ablations the paper needs (already switch-able in config)

- `loss_weights.ground = 0` → grounding off (isolates the grounding contribution)
- `loss_weights.gen = 0` → classification-only (matches the radiomics baselines)
- `grounding_pretrain.enabled = true` → pretrain grounding on RadGenome-Chest CT
- `model.vision_encoder` swap → BiomedCLIP vs plain ViT

Each ablation is a config diff, not a code change, so a reviewer's "what if you
remove X" is one run away.

## What is real vs still open

Real and wired: config (with load-time validation), download → extract → index →
preprocess as one consistent layout, dataset/splitting including nested outer-test
folds, all three heads, joint forward, the loss, the training loop, evaluation with
bootstrap CIs and chance-baselined grounding, and the blinded reader-study build +
server. 41 tests cover it, including an end-to-end training run on synthetic data
with stubbed backbones (`tests/test_train_smoke.py`) and one regression test per
fixed defect (`tests/test_regressions.py`).

Still open and requiring a human: a TCIA account and accepted Data Usage Policy;
the clinical spreadsheet, which is a manual download from the collection page;
first-run pretrained-weight downloads; and LoRA `target_modules`, which must match
the attention projection names of whichever LM you settle on — a mismatch now
raises `LoRAUnavailable` at construction instead of silently training the full LM
and producing a checkpoint that will not reload.

## Verification

```bash
pytest -q                       # 41 tests, no data / GPU / network required
pytest -q tests/test_train_smoke.py   # end-to-end loop on synthetic data
```
