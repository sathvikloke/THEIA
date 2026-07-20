# THEIA architecture & how it maps to the 16-week plan

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
localization share one representation, so the model can't decide from one place and
point at another. That coupling is the honest version of "grounded prediction."

## Loss

`L = w_cls · CE(EGFR,KRAS) + w_gen · LM(rationale) + w_ground · [Dice+BCE](attn ∪, ROI)`

Unknown labels are masked (`-1`). Grounding uses the union over region queries vs
the tumor mask downsampled to the patch grid.

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

## What is real vs `# TODO(theia)`

Real and wired: config, dataset/splitting, all three heads, joint forward, the
loss, the k-fold training loop, evaluation with bootstrap CIs and grounding IoU,
and the blinded reader-study build + server. Marked TODO: the patient→series path
mapping in preprocess (depends on the TCIA manifest layout), first-run weight
downloads, and LoRA `target_modules` (depends on the LM you settle on).
