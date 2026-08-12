# What was measured, and what it changed

Every setting below was decided by measurement, not by default. Collected here
because they were scattered across config comments and commit messages, and
because a methods section needs them in one place — and because several are
counter-intuitive enough that someone will otherwise re-derive them the hard way.

Two things to read this with:

- **Rerun variance is ±0.025 AUC** on the same configuration (it was ±0.041
  before the stalled seed-1337 run was retrained). Differences below
  that are not differences. Where an ablation was measured on a single fold or a
  single run, it says so, and those should be treated as directional.
- **Several early ablations predate `grad_clip`**, which means a fold could have
  gone NaN and contributed an untrained model. Those are flagged.

---

## Geometry — the settings that decide whether grounding is learnable at all

| setting | tried | result | conclusion |
|---|---|---|---|
| `crop_jitter_frac` | 0.0 | grounding lift ~0.000 across **four** ablation arms | The tumour sat at the exact centre of every crop (measured ROI centroid 6.5 ± 0.1 on a 14×14 grid whose centre is 6.5). The answer was "the middle" for everyone, so there was nothing to learn. |
| | **0.30** | grounding becomes learnable | Adopted. This single change is the difference between a localisation claim and no claim. |
| `context_factor` | 1.0 | tumour fills the frame | Caps what any localisation metric can mean: the shuffled baseline rises with the tumour's area fraction. |
| | **2.5** | tumour is 3.9% of the crop | Adopted. Makes chance ≈ 0.039, so a lift is meaningful. |

## Vision encoder

| setting | tried | result | conclusion |
|---|---|---|---|
| `freeze_vision` | fine-tuned | EGFR 0.426 | At n=153 fine-tuning 88M parameters mostly measures memorisation. |
| | **frozen** | EGFR 0.573 | Adopted. |
| `unfreeze_last_n` | 0 (fully frozen) | lift −0.013, pointing 0.00 *at any loss weight* | A head can reweight fixed features but cannot change what is attended. Grounding is impossible fully frozen. |
| | **2** | lift +0.302, pointing 1.00 | Adopted. *Single fold, directional.* |
| `backbone_lr_mult` | 0.1 | lift +0.060 → −0.001, grounding folds 2/5 → 0/5 | **My hypothesis, and it was wrong.** Intended to stabilise grounding; it starved it. The blocks cannot move enough to learn to localise. |
| | **1.0** (no warm start) | baseline | Adopted. Tracks `grounding_pretrain_ckpt` — 0.3 is right *with* a warm start, and using the wrong one reproduces neither configuration. |

## Loss and optimisation

| setting | tried | result | conclusion |
|---|---|---|---|
| `loss_weights.ground` | 5.0 | lift −0.001 | More pressure is *worse*. Raising it on top of unfreezing breaks grounding. |
| | **0.5** | lift +0.302 | Adopted. |
| grounding BCE form | clamped probabilities | **fixed point**: loss 9.30 with gradient exactly 0.000 | Normalising by the max pins the peak cell on the upper clamp. A map that flattens has every cell clamped at once and can never recover. Run 12 collapsed into this: lift +0.001, peak ratio 1.000, in all five folds. |
| | **BCE-with-logits** | lift +0.325 in 12/15 folds | Adopted. No clamp, so no dead zone; gradient bounded in [−1, 1] by construction. (Was reported as +0.391 in 13/14 on the superseded run set.) |
| `grad_clip` | absent | folds went NaN and trained on garbage to completion | Off CUDA `GradScaler` is disabled, so nothing skipped non-finite steps. Killed folds in runs 7, 8 and 10. |
| | **1.0** + skip on non-finite | a bad batch is skipped | Adopted. |
| `early_stop_min_epochs` | 0 | **2 of 5 folds died before training** | Patience counted from epoch 0, which under OneCycleLR is 4% of peak LR — an untrained model whose validation AUC on ~24 patients is noise. One fold was stopped while grounding was still climbing. |
| | **10** | folds train | Adopted. |
| `train.monitor` | `egfr_auc` alone | lift +0.060 | The reported grounding came from whichever epoch won on AUC. Fold 1 selected ep4 (lift +0.007) while its best grounding sat at ep12. |
| | `+ grounding_mass_lift × 0.5` | lift +0.235 | Adopted, quadrupled grounding at no cost to AUC (0.654 → 0.656). |
| | `+ gen_loss × −0.01` | selection moves ep1 → ep4, generation loss 2.150 → 0.854 at identical AUC | Adopted. Weight measured, not guessed: `gen_loss` swings 2.817 over training against `egfr_auc`'s 0.056, so the intuitive −0.05 would hand selection entirely to the language model. |
| `loss_weights.gen` | 1.0 | folds stall on MPS | |
| | **0.0** | stalls reduced from total to partial | Adopted *for this hardware only*. It is a mitigation, not a fix — see [RESULTS.md](RESULTS.md). |

## Grounding pretraining — a negative result

Pretrained on NSCLC-Radiomics (420 masked patients, no genomic labels).
Pretraining itself works: validation lift **+0.735**, attention mass 16× its
shuffled baseline. It does not transfer usefully.

| transfer | EGFR | grounding | conclusion |
|---|---|---|---|
| off | 0.656 | +0.235, 4/5 | **Adopted.** |
| vision + grounding | 0.528 | +0.749, 5/5 | Buys near-perfect localisation, costs the classification signal. *Confounded: run 8 had 1–2 dead folds (pre-`grad_clip`).* |
| grounding head only | 0.597 | +0.235, **2/5** | Bimodal, not better — two folds carry the mean while three collapse to uniform attention (peak ratio 1.12, 1.13, 3.88). A head fitted against pretrained features fails when handed BiomedCLIP's. |

The encoder and the grounding head must transfer together or not at all, and
together they trade away the classification signal. `grounding_pretrain.lr` also
mattered: 1e-4 plateaued at lift −0.007 while 2e-4 reached +0.352 — a
never-exercised scaffold default that could not learn.

## Baselines

| arm | variant | EGFR AUC |
|---|---|---|
| radiomics | 18 features, L2 | 0.662 |
| | 61 features, L2 | 0.604 |
| | 61 features, L1 (LASSO, as the field does it) | 0.526 |
| clinical | 14 features, L2 / L1 / search | 0.774 / 0.773 / 0.764 |

Two independent attempts to *strengthen* the radiomics arm made it worse. The
estimate spans 0.136 across defensible choices; clinical spans 0.010.

## Evaluation protocol

| choice | effect | conclusion |
|---|---|---|
| flat vs nested CV | **0.683 vs 0.617** on identical features and estimator | 0.066 from the protocol alone. Any comparison between arms must hold it fixed. |
| pooled OOF vs mean of per-fold AUCs | ±0.12 vs ±0.27 CI at the same n | With ~8 positives per fold, per-fold AUCs are too noisy to average. |
| within-fold rank normalisation | required | Each fold is a different model; raw probability scales are not comparable. |
| single-class folds | ranks carry no signal | Now warned; the pooled estimate would drift toward chance with nothing to explain it. |
