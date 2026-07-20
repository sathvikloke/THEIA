<div align="center">

<img src="assets/banner.png" alt="THEIA" width="100%">

<br>

**Predict EGFR / KRAS mutation status from lung CT, localize the evidence, and explain the call.**

One vision-language model that does three jobs at once: classify, ground, generate.

<br>

![Python](https://img.shields.io/badge/python-3.10+-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.44+-fcc419)
![License](https://img.shields.io/badge/license-MIT-2fd4c6)
![Status](https://img.shields.io/badge/status-research%20prototype-eb6834)

</div>

---

## Contents

- [What THEIA does](#what-theia-does)
- [Why it is novel](#why-it-is-novel)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [The pipeline, step by step](#the-pipeline-step-by-step)
- [Configuration](#configuration)
- [Data](#data)
- [Scaling beyond 211](#scaling-beyond-211)
- [Reader study](#reader-study)
- [Interactive demo](#interactive-demo)
- [Citation](#citation)
- [License](#license)

---

## What THEIA does

THEIA takes an axial lung CT and returns three things in one forward pass:

| Output | Head | What you get |
|--------|------|--------------|
| **Prediction** | classifier | EGFR and KRAS mutation status, with a calibrated confidence |
| **Localization** | grounding | attention maps that point at the tumor sub-regions behind the call |
| **Explanation** | generation | a short, structured rationale a clinician can audit |

The name is the Greek titaness of sight and light: the model *sees* (vision) and *illuminates* what it saw (grounding). That linkage between a prediction, the pixels behind it, and a written rationale is the whole point.

---

## Why it is novel

As of mid-2026 the two closest published models each stop one step short:

| Model | Classify | Generate | Ground | Modality |
|-------|:--------:|:--------:|:------:|----------|
| Glio-LLaMA-Vision (npj Digital Medicine, 2026) | yes | yes | **no** | MRI, glioma |
| NEVA (Nature Communications, 2026) | yes | **no** | yes | pathology, neuroblastoma |
| **THEIA (this repo)** | **yes** | **yes** | **yes** | **radiology (CT), NSCLC** |

Nobody has put classification, generation, and grounding in one model, and nobody has done grounding at all for radiology-modality biomarker prediction. Frame the paper around that mechanism, radiology grounding tied to structured semantic-annotation supervision, rather than the general "VLM for radiogenomics" claim, which has a short shelf life.

---

## How it works

```
                       ┌────────────────────────────────┐
  CT ROI stack ───────▶│  Vision encoder (BiomedCLIP ViT)│──▶ patch tokens [B, N, D]
  [B, S, 1, H, W]      └────────────────────────────────┘             │
                                                                       ▼
                                             ┌──────────────────────────────────┐
                                             │  Grounding head                    │
                                             │  Q region queries cross-attend     │
                                             │  over the patch tokens             │
                                             └──────────────────────────────────┘
                                     region_emb [B, Q, D]  │   attn_maps [B, Q, H', W']
                            ┌──────────────────────────────┘             │
                            ▼                                            ▼
                 ┌──────────────────────┐                     supervised against the
                 │  Classifier head      │                    tumor ROI mask (Dice + BCE)
                 │  EGFR / KRAS logits    │                    so attention is anchored
                 └──────────────────────┘                     to real anatomy
                            │  pool region_emb                            │
                            ▼                                            ▼
                EGFR / KRAS + confidence            ┌────────────────────────────────┐
                                                    │  Generation head (LoRA LM)      │
                                    region_emb ────▶│  visual prefix -> rationale     │
                                                    └────────────────────────────────┘
```

The classifier pools from the **grounded** region embeddings, not a separate global token, so the prediction and the localization share one representation. The model cannot decide from one place and point at another.

Loss is a weighted sum:

```
L  =  w_cls · CrossEntropy(EGFR, KRAS)
   +  w_gen · LM_loss(rationale)
   +  w_ground · [Dice + BCE](attention_union, tumor_ROI)
```

Unknown labels are masked out. The grounding term is what makes the attention maps trustworthy instead of decorative.

---

## Repository layout

```
THEIA/
├─ assets/banner.png            this banner
├─ configs/default.yaml         every hyperparameter and path, one file
├─ theia/
│  ├─ config.py                 typed YAML loader with dotted overrides
│  ├─ data/
│  │  ├─ download.py            TCIA fetch helper + integrity checks
│  │  ├─ preprocess.py          DICOM -> ROI arrays, pseudo-report build
│  │  └─ dataset.py             torch Dataset + patient-level k-fold
│  ├─ models/
│  │  ├─ backbone.py            vision encoder (BiomedCLIP / timm ViT)
│  │  ├─ heads.py               classifier, grounding, generation heads
│  │  └─ theia_model.py         the three-head model, joint forward
│  ├─ engine/
│  │  ├─ losses.py              classification + generation + grounding loss
│  │  ├─ train.py               k-fold LoRA fine-tune loop
│  │  └─ evaluate.py            AUC + CIs, grounding IoU, rationale scoring
│  └─ reader_study/
│     ├─ build_cases.py         build blinded, randomized presentations
│     └─ serve.py               Flask app for clinicians to score cases
├─ scripts/                     thin shell entrypoints
├─ tests/                       smoke tests (no data or GPU required)
├─ demo/index.html              interactive front-end demo (mock data)
└─ docs/
   ├─ ARCHITECTURE.md           deeper design notes + week-by-week mapping
   └─ PROJECT_PLAN.md           the full 4-month research plan
```

---

## Quickstart

**Requirements:** Python 3.10+, one CUDA GPU with 24 GB or more for training. CPU is fine for the smoke tests.

```bash
git clone https://github.com/sathvikloke/THEIA.git
cd THEIA
bash scripts/setup.sh
```

Verify the wiring before you touch any data:

```bash
pytest -q
```

Then the three stages:

```bash
bash scripts/run_preprocess.sh     # download + build the dataset
bash scripts/run_train.sh          # k-fold LoRA fine-tune
bash scripts/run_eval.sh           # metrics + build reader-study cases
```

> **Before anything else, confirm GPU access.** It is the single biggest risk to the timeline, not the modeling.

---

## The pipeline, step by step

**1. Download** (`theia/data/download.py`)
Pulls the NSCLC-RADIOGENOMICS series manifest and DICOM images from TCIA. You need a registered TCIA account and, for the API, a token in a `.env` file. Clinical labels and semantic annotations ship as supplementary spreadsheets on the collection page and are dropped into `data/raw/.../clinical/` by hand once.

**2. Preprocess** (`theia/data/preprocess.py`)
For each patient: load the CT and the tumor segmentation, resample the mask onto the CT grid, apply a lung window, take the slices spanning the tumor, crop a margin-dilated ROI, and render the controlled-vocabulary semantic annotation as a one-sentence pseudo-report. Writes one `.npz` per patient plus a `rows.jsonl` index. The pseudo-report is the supervision signal for the generation head, so you never hand-write reports.

**3. Train** (`theia/engine/train.py`)
Patient-level stratified k-fold. Each fold trains the three-head model with LoRA on the language model, AMP, one-cycle schedule, and early stopping on validation EGFR AUC. Checkpoints and per-epoch metrics land in `checkpoints/foldN/`.

**4. Evaluate** (`theia/engine/evaluate.py`)
ROC-AUC per gene with bootstrap confidence intervals, sensitivity and specificity, and grounding IoU (overlap between the attention union and the tumor mask). Note: BLEU and ROUGE on the rationale are a sanity check only. The real rationale evaluation is the reader study, because text-overlap metrics do not measure clinical correctness.

**5. Reader study** (`theia/reader_study/`)
`build_cases.py` produces blinded, randomized presentations across three arms (full THEIA output, prediction only, and ground truth). `serve.py` is a small Flask app where clinicians score plausibility, grounding, and usefulness. Arm identity stays in a held-back key file until scoring is done.

---

## Configuration

Everything lives in `configs/default.yaml`. Override any key from the command line without editing the file:

```bash
python -m theia.engine.train --config configs/default.yaml
```

Ablations that the paper needs are one-line config changes, not code changes:

| Ablation | Change |
|----------|--------|
| Grounding off (isolate its effect) | `train.loss_weights.ground: 0` |
| Classification only (match radiomics baselines) | `train.loss_weights.gen: 0` |
| Pretrain grounding on RadGenome-Chest CT | `grounding_pretrain.enabled: true` |
| Swap the vision encoder | `model.vision_encoder: timm_vit_b16` |

---

## Data

**Primary:** [NSCLC-RADIOGENOMICS](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/) (TCIA). 211 patients, CT and PET/CT, tumor segmentation masks, EGFR/KRAS labels, an RNA-seq subset, and radiologist semantic annotations used as generation supervision. Access is registration plus a data-use agreement, not a lengthy approval.

**Optional grounding pretraining:** [RadGenome-Chest CT](https://www.nature.com/articles/s41597-025-05922-9). 665K grounded reports and 1.2M grounded VQA pairs on chest CT, no genomic labels. Pretrain the grounding head here before the small fine-tune.

> 211 patients is small. This repo defaults to patient-level stratified k-fold, not a single split, and reports confidence intervals. Do not oversell statistical power.

---

## Scaling beyond 211

n=211 is the honest floor, not the ceiling. The bigger, more defensible version scales the **data**, not the framing. Three levers, in rough order of effort:

1. **Pool radiology cohorts.** NSCLC-RADIOGENOMICS + TCGA-LUAD + TCGA-LUSC + NSCLC-Radiomics share CT and EGFR/KRAS calls. Harmonize the labels and you reach low thousands. `dataset.py` indexes from `rows.jsonl`, so a merged index plus a `cohort` field per row is the whole change.
2. **Add a histopathology branch.** TCGA lung whole-slide images with matched mutation status number in the thousands, and mutation-from-H&E is an established large-N task. A second vision encoder feeding the same grounding and generation heads makes THEIA the first grounded, generative, radiology-plus-pathology biomarker model, a materially bigger claim than radiology alone. This is the multi-modal fork, and it adds months.
3. **Grounding pretraining at scale.** Use RadGenome-Chest CT to pretrain grounding before the small fine-tune (already a config flag).

Honest tradeoff: a paradigm-shifting multi-modal build and a 4-month, high-impact timeline are in tension. Ship the tight radiology paper on the single cohort first, or commit to the multi-modal effort on a longer clock. Do not half-do both.

---

## Reader study

The reader study is the actual clinical validation, and the reason THEIA is more than a benchmark number. Clinicians see de-identified, blinded presentations and never learn which arm they are scoring. To run it after training:

```bash
python -m theia.reader_study.build_cases --ckpt checkpoints/fold0/best.pt
python -m theia.reader_study.serve --reader R1
```

Confirm with your IRB office early whether a study on public, de-identified data with no new patient contact needs a determination. Get it in writing. Do not leave it to the week you want to run the study.

---

## Interactive demo

`demo/index.html` is a self-contained front end (open it in any browser) that shows the intended experience on mock data: flip between cases, and hover any clause in the rationale to light up its region on the scan, and the reverse. It is the same layout the reader study adapts. No data or model required to view it.

---

## Citation

If this work is useful, please cite the repository and the underlying datasets per their data-use terms (TCIA NSCLC-RADIOGENOMICS, RadGenome-Chest CT, and any TCGA cohorts).

```bibtex
@software{theia2026,
  title  = {THEIA: Grounded Radiogenomic Inference for NSCLC},
  author = {Loke, Sathvik and collaborators},
  year   = {2026},
  url    = {https://github.com/sathvikloke/THEIA}
}
```

---

## License

MIT, code only. The datasets carry their own terms. See [LICENSE](LICENSE).

<div align="center"><br><sub>THEIA is a research prototype. It is not a medical device and must not be used for clinical decisions.</sub></div>
