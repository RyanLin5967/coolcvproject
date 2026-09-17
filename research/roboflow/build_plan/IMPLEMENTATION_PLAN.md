# Implementation sequence and evidence gates

Status: plan only, September 17, 2026. Commands/modules below describe the intended interface; they do not exist yet. Account actions, dependency installation, project implementation and training follow this planning phase.

## Deliverable and critical path

Deliver a compiler that preserves annotation coverage, a tested RF-DETR adapter that consumes it, a controlled comparison, and a Roboflow dataset-to-deployment demonstration. The implementation should make the failure visible: a source that labels black pawns can contain unlabeled white pawns; those missing labels must not silently become justified negatives.

The dependency chain is **account capability + data audit → frozen protocol → immutable training views + criterion correctness → GPU smoke test → controlled runs → evaluation → Roboflow deployment/report**. Compiler and numerical work can proceed while a hosted base model trains. The provider's queue, training, checkpoint export and account login are real elapsed-time dependencies. Agent coding speed cannot establish those times in advance.

A one-day mechanism demo is plausible, but not guaranteed. A broad multi-dataset ML claim is outside what one small pilot establishes. Favor complete, reviewable evidence over a large unfinished surface area. If a cloud run is delayed, finish the compiler, tests, platform reconciliation and report with an explicit pending experiment; never manufacture a score.

## Gate 0 — environment and provider branch

1. Create project packaging with Python 3.12, compiler dependencies separated from `train` and `remote` extras, `.env.example` with names only, and ignored credential/data/run paths. Preserve the research directory.
2. Pin RF-DETR 1.10.1 and resolve compatible Torch/torchvision, Lightning and CUDA dependencies. Save the lock and environment manifest. Verify imports locally and in the intended CUDA image. Do not label a dependency range as a tested combination.
3. Check local Roboflow/Modal credential presence without printing values. Complete user browser authentication only where required by the provider. Public dataset downloads need no Hugging Face login.
4. Inspect Roboflow workspace balance, plan, available RF-DETR sizes, train recipe, and raw PyTorch export entitlement. Save a redacted capability record with date and actual SDK version. No speculative paid upgrade.
5. Choose `coverage-chess-hybrid-v1` if hosted export is entitled and a compatible trainable model is available; otherwise choose the original controlled Modal protocol plus a descriptive hosted baseline. If compatibility cannot be proven before a new seed model exists, launch only the bounded seed run after entitlement checks, then run the import gate before committing to continuation.
6. Exercise standard RF-DETR model packaging/upload on a disposable demo model early if the platform permits it, so serialization problems are discovered before the accuracy experiment. Mark the model as a compatibility probe, never as the trained result.

**Exit evidence:** reproducible imports, chosen provider/protocol, actual account capabilities, credential-free manifests, identified checkpoint format and preliminary deployment route.

## Gate 1 — data preparation and compiler

Implement strict schema types for source, ontology mapping, coverage declaration, override, sample identity, bundle manifest, diagnostics and experiment specification. Use explicit schema versions and `extra=forbid`; reject ambiguous booleans/numbers and nonfinite values. An external class ID is never a model index.

Build the source-specific chess importer, decode all images and validate every box against real dimensions. Preserve source files and attribution separately from transformed artifacts. Review scene/near-duplicate groups and annotation overlays before interpreting held-out metrics. Freeze group membership, protocol assignment and initial model-selection policy before outcomes are visible.

Implement deterministic ontology remapping, coverage resolution, exact duplicate/split collision checks, canonical payload hashes and atomic bundle publication. Reject unsafe relative paths, symlinks escaping the input root, unsupported crowd/ignore semantics, conflicting observations and contradictory absence assertions. Diagnostics must identify the offending source/image/class and the corrective action.

Materialize three learner views with the real RF-DETR directory layout. Partial views contain only visible train boxes plus complete validation labels. The reference arm has a separate complete train view. Preserve global image IDs and category tables. Keep final test/hidden-reference files out of the worker input. Roboflow exports are reconciled back to uploaded observations before entering the compiler.

**Required tests:**

- Permuted equivalent inputs produce the same bundle digest and payloads; a semantic annotation/coverage change changes identity.
- Missing declarations default to unknown. Empty labels never imply absence. Verified absence plus an observed box fails.
- Duplicate bytes across splits fail. Identical observations within a split coalesce deterministically; conflicting labels/coverage fail.
- Invalid class mappings, coordinates, dimensions, paths and schema fields fail with stable diagnostic codes.
- Interrupted publication never exposes a valid-looking incomplete bundle; tampered payloads fail verification.
- Training-view projection preserves original ID space, omits prohibited labels/files, and binds parent/view/ontology digests.
- Roboflow import/export reconciliation catches remapped categories, renamed/re-encoded images, changes in boxes, missing images and altered splits using explicit correspondence rather than assumed byte equality.

**Exit evidence:** immutable bundle, source/split/coverage inventory, attribution, scene-audit notes, exact three-view supervision diff and provenance. The hybrid seed/continuation counts are generated, not copied from the original pilot.

## Gate 2 — prove the training semantics

Implement the narrow classes specified in [TRAINER_DESIGN.md](TRAINER_DESIGN.md). Start with synthetic outputs and targets; they isolate loss semantics without expensive training. Then exercise the actual detector and loader. Only the reviewed IoU-aware BCE detection configuration is accepted.

| Test | What it proves |
|---|---|
| Every class exhaustive | Adapter matches stock loss components, logit/box/model gradients and one optimizer update within declared dtype tolerances. |
| Unknown class, unmatched query | Its direct classification gradient is zero; known-class negatives keep stock behavior. |
| Known positive under unknown/positive-only coverage | The full soft-positive BCE cell and its derivative match stock; the negative algebraic term is not incorrectly removed. |
| Empty targets and all-unknown semantic coverage | Finite behavior, stock normalizer and reserved-slot policy; do not assert total parameter gradients are zero. |
| Multiple images with different policies | Batch-row lookup follows image IDs, not batch order; no cross-image broadcasting error. |
| Auxiliary decoder and encoder outputs | Each head uses its own matches; every active classification loss is covered. |
| Training/evaluation group behavior and explicit `num_boxes` | Upstream normalization and caller overrides are preserved. |
| Reserved output highest score | Common postprocessing/evaluation policy is exercised; unmapped semantic output fails, reserved selections are logged. |
| Real sample with K boxes, empty sample, shuffled loader | Coverage identity survives conversion, transforms, collation, packing and transfer without being filtered as a per-box vector. |
| Save/load and wrong-bundle resume | Lookup reconstruction and checkpoint identity are correct; valid-range IDs from another dataset cannot pass silently. |
| Unsupported padded targets, loss modes and augmentation | Unsupported configurations fail before training; no silent fallback. |

Use fixed nontrivial fixtures, including soft IoU targets and deliberately distinct final/aux/encoder predictions. Compare against the pinned stock implementation, not a second copy of our own formula. Test finite-difference/autograd behavior where useful. FP32 CPU establishes numerical correctness first; GPU validation checks the actual execution path. Declare tolerances before examining differences; do not relax a failing test to hide a semantic discrepancy.

For new-ontology stock initialization, explicitly initialize every K+1 main/encoder classification head and save one seeded state. For the compatible hosted two-class base, preserve trained heads. Assert all arms' initial detector parameter digests match. Keep EMA, compile and accumulation beyond one out of the first proof.

**Exit evidence:** saved test results for the real release, full-coverage stock parity, zero forbidden-cell direct gradient, correct positive preservation, real loader-to-loss round trip and reproducible initialization. No accuracy claim is needed to pass this gate.

## Gate 3 — hosted base and CUDA smoke test

For the preferred hybrid, create only the declared fully labeled seed project/view. Verify server-side/exported membership before training. Submit one ordinary RF-DETR hosted job under the actual account's available-credit controls; record the returned training ID and resolved recipe. On a submission timeout, reconcile before retrying. Poll status without creating replacement jobs automatically.

Download the actual trainable checkpoint through the entitled interface. Audit import keys/configuration, semantic mapping, checkpoint flavor, preprocessing and representative prediction parity. Keep its SHA-256 and provenance. If this gate fails, select the documented fallback and generate its views explicitly; do not quietly partially load an incompatible model.

Run a bounded custom-loss GPU smoke test on the chosen complete/partial views. Measure startup time, steps/second, memory, validation time and checkpoint duration. Overfit a tiny fully labeled subset as a wiring diagnostic; this is not a held-out result. Confirm that checkpoints and logs survive worker exit and can be downloaded. Exercise one interruption/recovery path before relying on remote persistence.

Use one GPU at a time, a persisted deadline/attempt ledger, ordinary retries disabled, account-level budgets and a finite attempt timeout. Compute a common step budget from measured throughput with room for validation/checkpoints. The two-hour-per-arm number in setup is only an initial ceiling, never a prediction.

**Exit evidence:** real resource measurement, valid checkpoint retrieval, import/deploy compatibility status, a frozen common continuation schedule, and provider cost limits. Account credits do not establish GPU availability or training duration.

## Gate 4 — controlled experiment and evaluation

Run `naive`, `aware` and `complete_reference` from identical detector state. Pair RNG seeds, sample order, augmentation, optimizer/scheduler, physical batch, precision, updates and evaluation schedule. Record resolved settings and actual completed steps. Select the same checkpoint policy for every arm. Do not choose a longer or more favorable run only for the aware model.

An interrupted run is incomplete. Compare a predeclared common completed checkpoint or rerun the shorter schedule across all arms; do not silently compare unequal update budgets. Preserve completed checkpoints and raw metrics even when the result is disappointing.

Choose operating thresholds and checkpoints using validation. Final test remains unavailable to the learner and is evaluated after choices are fixed. Report AP@[.50:.95], AP50, per-class support/AP/precision/recall, false positives per image, duplicates, and results on reference-negative images. Give the exact model, run, bundle and protocol IDs. Include pre-continuation base scores to expose saturation or degradation.

The complete-reference arm provides context, not an upper bound. The separate Roboflow-hosted baseline is descriptive unless every relevant training factor is controlled. For a one-seed pilot, label the uncertainty; do not invent statistical confidence from 174 correlated boxes in 29 images. A later three-seed or second-domain experiment is separately budgeted.

**Exit evidence:** completed comparable runs, raw predictions, evaluator settings, honest tables and saved failure examples. No minimum AP gain is imposed as a success criterion for software correctness.

## Gate 5 — platform demo and handoff

Upload the selected detector with the tested standard packaging to a Roboflow dataset version. Run fixed validation inputs through the deployed model and compare local/remote predictions with aligned preprocessing/postprocessing. Build a small Workflow that returns pawn counts/predictions. Bind its real model and Workflow IDs to the experiment report. If a platform gate remains blocked, say exactly which part works and which does not.

Generate an offline report with source-by-class coverage, one inspectable image's visible/reference annotations, human-readable supervision explanations, matched prediction montages, aggregate metrics, runtime/cost and provenance. Use fixed examples or identify chosen failure illustrations. Escape untrusted labels; no API keys, signed URLs or third-party tracking in the report.

Package the CLI, documented example, source notices, tests, reproducibility commands and account-free compiler demo. A reviewer should be able to validate/inspect a small bundle without installing CUDA or owning provider accounts. A training reproduction may require the documented GPU environment and checkpoint entitlement; disclose those separately.

**Definition of done:** compiler and adapter contracts pass; experiment result/limitations are preserved; required Roboflow integration is demonstrated or its precise remaining failure is named; artifacts can be retrieved without an active GPU; usage is stopped and accounted for; documentation distinguishes planned features from executed evidence.

## Later work ordered by value

1. Repeat paired seeds and test a less-correlated second domain; compare against pseudo-label completion and specialists with annotation/compute budgets stated.
2. Benchmark compilation speed, peak memory, lookup overhead and artifact transfer on a larger source set. This strengthens the infrastructure/backend story beyond a small chess demo.
3. Exercise checkpoint recovery more broadly, add a second supported RF-DETR release only after parity, and prove EMA/AMP/packing modes before advertising them.
4. Consider region-level coverage or another detector only when its supervision contract and actual user need are specified.

Avoid adding accounts, databases, queues, an always-on dashboard or distributed training merely to make the architecture look larger. The strongest initial evidence is a narrow integration whose data semantics and experimental claims survive inspection.
