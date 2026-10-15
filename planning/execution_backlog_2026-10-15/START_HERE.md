# THEIA execution handoff — 15 October 2026

**Owner: Sathvik, executing with Claude. There are no active teammate assignments. A paper author roster is a separate matter.**

This is the new entry point for the remaining computational work. It replaces the old Codex-supervised lane/controller workflow for future execution. Historical prompts, sealed results, failures and campaign windows remain provenance; they are not a requirement to restart an old controller or seek hourly Codex review.

Start with [TASKS.md](TASKS.md), its machine-readable counterpart [TASKS.json](TASKS.json), the [dataset register](DATASETS.md), and the [nightly runbook](NIGHTLY_RUNBOOK.md). [AUDIT.md](AUDIT.md) distinguishes work already done from missing execution. [HUMAN_REVIEW.md](HUMAN_REVIEW.md) lists decisions and external inputs that computation cannot supply.

## What “finished” means

A computational task finishes when its stated outputs exist, its technical checks pass, its provenance and failures are retained, and another person can inspect the result. Scientific validation, interpretation, clinical adjudication and publication acceptance are separate. A negative result is a completed experiment; a better score is not a completion criterion.

The queue covers the documented corrected CT study, acquired/planned imaging and molecular cohorts, external evaluation, reader infrastructure, reproduction and submission packaging. The broader pretraining, multimodal, multigene and discovery branches are included explicitly, with scope decisions identified. They are not automatically journal requirements. Unspecified future hypotheses cannot be audited into a finite guaranteed-publication checklist.

The January 20, 2027 preparation target is a planning target. No whole-project completion percentage or promise of acceptance is supported. Report exact experiment and cohort denominators instead.

## First useful work

1. Establish the portable code/artifact baseline (E01–E03). This is necessary because GitHub main and the local research checkout differ substantially.
2. Resume acquisition correctly (D01–D02), repair the LIDC adapter (D05–D07), and prepare the corrected source build (M01) independently where their inputs exist.
3. Run the corrected first fit and remaining fixed run matrix once the required input/protocol records are bound (M02–M06). Implemented code and technical checks do not replace these experiments.
4. Build and execute the separate NLST adaptation/transfer program (N01–N09): six adaptations and 30 matched downstream fits, with settings still needing resolution clearly identified. This work can proceed independently of N08 where its inputs allow.
5. Execute saved-array calculations and packaging against the correct versioned results. Reuse already completed analyses where input and method hashes match.

Do not infer execution authority from this document alone: this request published the backlog. When Sathvik starts a nightly execution session, use that session's explicit window and task scope. Ordinary implementation and computation within that scope do not need a Codex approval loop. Data-access credentials, outside recipients, paid resources, human scientific decisions and new methods remain distinct dependencies.

## Repository and data portability

This repository is public. Publish code, task specifications, aggregate evidence, synthetic fixtures and safe acquisition manifests. Keep clinical images, subject-level predictions, molecular labels, private linkage keys, credentials and unsent outreach messages in controlled local storage. This is a catalog of every dataset identified by the audit, not an upload of the datasets themselves.

The audit observed public main `d70c091a9fa6` and local checkout HEAD `4fc73abb2cf1466d8830de202076e13f855292c7`. The latter is a September 16 commit; October execution packages are separate local artifacts. A clean GitHub clone does **not** contain all cited `revamp/` or October planning sources. E01 explicitly bridges that gap; do not silently run older public code under a newer result label.
