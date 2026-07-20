# Grounded Generative Radiogenomics: 4-Month Project Plan

## The idea

Build a vision-language model that predicts EGFR and KRAS mutation status from CT scans in non-small cell lung cancer (NSCLC), and generates a structured, visually-grounded rationale for each prediction instead of just a probability score. The model points to the image regions it used and explains its reasoning in report-style text.

**Updated novelty framing, checked against the two closest papers as of mid-2026.** The right comparison set is no longer MAIRA-2/CheXagent/LLaVA-Rad, those are chest X-ray findings, not radiogenomics. It's these two:

**Glio-LLaMA-Vision** (npj Digital Medicine, 2026): BiomedCLIP vision encoder plus LLaMA 3.1 8B, a classifier head for IDH mutation status in glioma and a generation head for free-text reports, trained jointly. Validated on four cohorts including TCGA (AUC 0.87), reader study with three neuroradiologists (91% clinically acceptable), hallucination-checked (5.1% rate). This is classification plus generation, done well, on MRI/glioma. It does not do spatial grounding, the paper explicitly names that as future work, not something implemented.

**NEVA** (Nature Communications, 2026): vision-language model for neuroblastoma, predicts molecular alterations (NMYC amplification, 1p36 deletion) from histopathology, and does have grounding, interpretable attention maps tied to specific tissue regions. What it doesn't have is free-text generation, it's classification plus attention maps, no report text. It's also pathology imaging, not radiology.

So: generation-plus-classification exists (Glio-LLaMA-Vision, radiology, no grounding). Classification-plus-grounding exists (NEVA, grounding, but pathology, no generation). Nobody has put all three together, classification, free-text generation, and spatial grounding, in the same model, and nobody has done grounding at all in radiology-modality imaging for biomarker prediction. That's the actual open square, and it's narrower and more defensible than "first VLM for radiogenomics." Position the paper against these two by name in related work, don't skip that, a reviewer who's seen either paper will read an unaddressed comparison as either sloppy or evasive.

One practical upside from checking this space: RadGenome-Chest CT (Scientific Data, 2025) is a public dataset of 665K grounded reports and 1.2M grounded VQA pairs tied to segmentation masks on chest CT, no genomic labels, pure radiological findings. Use it to pretrain the grounding mechanism before fine-tuning on the 211-patient NSCLC-RADIOGENOMICS set. That directly offsets the small-N problem for the grounding component specifically.

Also expect more papers like Glio-LLaMA-Vision and NEVA to land during your 4 months, this subfield has a named survey now ("Large language models in radiogenomics," The Visual Computer, 2026) which means it's actively being worked, not a quiet corner. Build the paper's identity around the specific mechanism, radiology-modality grounding tied to structured semantic-annotation supervision, not around being first to combine VLMs and biomarkers in general. That general claim has an expiration date measured in months right now.

Clinical stakes are real too, not synthetic. EGFR and KRAS mutation status directly changes NSCLC treatment choice (targeted therapy eligibility). A model that's wrong here isn't an academic curiosity, so the clinical validation piece has actual weight, not just a checkbox.

## Data

**NSCLC-RADIOGENOMICS** (The Cancer Imaging Archive). 211 patients. CT and PET/CT imaging, tumor segmentation masks, gene mutation labels (EGFR, KRAS, others), an RNA-seq subset (~130 patients), and semantic annotations, radiologist-generated structured descriptions of tumor appearance using a controlled vocabulary. Public, Creative Commons BY 3.0, access is registration plus the TCIA Data Usage Policy, not a lengthy approval process. Get this started day one.

The semantic annotations matter more than they look. They're effectively pre-existing structured "reports" you can use as supervision for the text-generation side, without writing any of them yourself.

**Known limitation, be upfront about it in the paper**: 211 patients is small for deep learning. Plan for cross-validation, not a single train/test split, and don't oversell statistical power in the writeup. Reviewers will check this.

## Model approach

Don't train a foundation model from scratch, not feasible in 4 months. Start from an open-source medical LVLM, **LLaVA-Med** (Microsoft, GitHub) is the obvious base, already biomedical-pretrained, and LoRA fine-tuning it is documented and runs on a single high-end GPU, not a cluster. Confirm GPU access before week 1, this is your single biggest resource risk, not the science.

Two-head setup: a prediction head (EGFR/KRAS mutation status), and a generation head producing grounded rationale text, supervised in part by the semantic annotations. For the grounding mechanism itself, tying generated claims to specific image regions, pretrain on RadGenome-Chest CT (665K grounded reports, 1.2M grounded VQA pairs on chest CT, public, no genomic labels) before fine-tuning on your 211-patient NSCLC-RADIOGENOMICS set. This gives the grounding component a real pretraining signal instead of trying to learn it from 211 patients alone, which is otherwise the weakest part of the plan.

## Baselines and related work to cite (don't skip these)

**Quantitative baselines**: multiple 2025 papers already do classical radiomics/deep learning EGFR and KRAS prediction on CT, some on this exact dataset. Reimplement or cite their reported AUCs (roughly 0.83-0.89 range across the recent literature) as your comparison point. Your paper's contribution is not "better AUC than these," necessarily, it's "same or comparable predictive performance, plus interpretable grounded output these baselines don't have." Chasing a pure AUC win against tuned radiomics pipelines in 4 months is a losing bet, don't make that your headline claim.

**Related work, cite by name, address head-on**: Glio-LLaMA-Vision and NEVA (see above). Don't frame them as beaten or obsolete, frame them accurately, one has generation without grounding, the other has grounding without generation, in different modalities and diseases than yours. Your related-work section needs to make clear you know exactly where those two papers stop and where yours starts.

## The 16-week plan

**Weeks 1-2, setup**
Register for TCIA access, download imaging, mutation labels, semantic annotations, RNA-seq subset. Get GPU access confirmed and working. Pick and stand up LLaVA-Med locally. Freeze your baseline comparison list, pull the exact numbers from the published radiomics/DL papers on EGFR/KRAS prediction you'll cite. Start recruiting reader-study clinicians NOW, not in week 12, this is the step people always leave too late and it kills timelines.

Also this week: your collaborator checks with their IRB office whether a reader study using only public, de-identified data and no new patient contact needs a determination. Get this in writing early. Don't assume it's exempt just because the data's public, confirm it.

**Weeks 3-5, data pipeline**
Preprocess CT/PET, extract tumor ROIs using the provided segmentation masks. Build the paired dataset: image, semantic annotation as pseudo-report, mutation label. Handle class imbalance (mutation-positive cases will be a minority). Set up proper stratified cross-validation splits, leak-check patient-level (make sure the same patient's slices never span train and test).

**Weeks 6-9, model build**
Fine-tune LLaVA-Med with LoRA on the prediction task first, get a working classifier before adding generation complexity. Add the grounded rationale generation head. Run ablations: with vs without semantic-annotation supervision, with vs without grounding. This is the core engineering block, expect it to eat the most time and have the most slippage, buffer accordingly.

**Weeks 10-11, retrospective quantitative validation**
Evaluate mutation prediction (AUC, sensitivity/specificity) against your frozen baseline numbers, same or comparable dataset conditions. Evaluate grounding quality, overlap between the model's attended regions and the ground-truth tumor segmentation masks. Evaluate generated rationale text against the semantic annotations, structured accuracy on specific claims, not just BLEU/ROUGE (reviewers increasingly call out NLG-only evaluation as weak, don't rely on it alone).

**Weeks 12-13, clinical validation (reader study)**
This is the actual clinical validation piece. Present cases to your recruited clinicians blind, model output vs baseline (no explanation, just the label) vs ground truth. Have them score: clinical plausibility of the rationale, whether the grounding makes anatomical sense, trust/usefulness on a Likert scale. Aim for at least 2-3 independent readers, more if you can get them, and report inter-rater agreement. Realistically this needs readers already lined up from week 1, don't start recruiting here.

**Weeks 14-15, writeup**
Draft the manuscript. Lead with the method and the interpretability/grounding contribution, not a raw performance claim. Grounding visualizations (heatmaps over tumor regions tied to specific generated claims) are your strongest figures, invest time here. Get your collaborator's clinical read on the framing before it's "done." Prep a code and data availability statement, most imaging journals in this tier expect or require open-sourcing your model/code.

**Week 16, finalize and submit target**
Polish, format to whichever venue you land on. Post a preprint (arXiv or medRxiv) the same week you submit, don't wait on journal review to make the work citable and public.

## Where this can slip, and what to do about it

GPU access is the most likely hard blocker, not the ML itself. Confirm this in week 1, not week 6.

211 patients is a small N. If your cross-validated results are noisy, that's expected, don't chase a bigger effect size than the data supports, report honestly with confidence intervals.

Reader recruitment is the second most common failure point on plans like this. People agree in principle and then don't show up when it's time to actually score 30 cases. Line up more readers than you need, and start now.

The IRB/exempt determination for the reader study is a paperwork risk, not a science risk, but it can stall you for weeks if you leave it until week 12. Handle it in week 1-2 in parallel with everything else.

## On publication timing, restated plainly since it matters for how you plan

This plan gets you a finished, validated, written paper by end of month 4. It does not get you a MedIA acceptance by month 4, their average review alone runs about 5 months after submission, with roughly 1 in 4 submissions accepted. Treat month 4 as "preprint plus submit," and month 9-14 as the realistic window for an actual acceptance, if it's accepted at all. Plan the next phase of your work assuming that timeline, not a faster one.
