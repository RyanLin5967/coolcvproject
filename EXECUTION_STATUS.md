# Execution status

September 17, 2026. The local MVP works: strict compiler, verified learner views, RF-DETR coverage loss, three real 100-update training/evaluation runs, and an offline interactive report at `artifacts/demo/index.html`. **34 tests passed**, ruff passed, Chrome desktop/mobile and tab/slider checks passed. Dataset, checkpoints, logs and full measurements persist under ignored `data/` and `artifacts/`; compact tracked evidence is in `evidence/mvp.json`.

Roboflow project `coveragecv-chess-mvp`, version **2**, contains 201 training images / 451 observed boxes and 58 validation images / 241 complete boxes. Its exported COCO was compared with the local view image-by-image: identities, splits, classes and coordinates match within 0.0001px. Version 1 is superseded because its VOC origin was shifted by one pixel; the connector is fixed. The custom model was trained on the correct LOCAL data and uploaded to version 1. Conversion currently reports `running`; **hosted inference is not yet verified**. Record actual provider status before claiming deployment success. Ordinary hosted training has not launched. Custom recipe API availability reports entitled=false, runtimeEnabled=true, which is not evidence that all hosted training is unavailable.

Naive/aware/complete-reference validation AP50:95: 71.68 / 72.34 / 71.70 percent. Aware improves fixed-threshold recall (86.31 to 97.51 percent) but increases false positives (0.052 to 1.310/image). This is a short single-seed pilot, not a converged benchmark. Test split remains untouched. Full run bindings and limitations are in evidence/mvp.json. Two incorrectly labeled initial runs were excluded under artifacts/invalid_attempts and rerun with explicit arm; report checks reject this failure mode.

## Current authorized next stage

The user now wants a **substantial jump in capability**, not small polish changes. Build an operational local dataset workbench: persisted projects/coverage edits, compiler diagnostics and version differences, arbitrary detection ontologies, background training/comparison jobs with durable state, run controls and a real interactive frontend. Preserve the tested compiler/criterion contracts. Run a longer comparison while building. Keep showing actual results and limitations. The research plan remains context, not a claim that future features exist.

## Budget and credentials

Maximum out-of-pocket budget is $0. Credentials live in ~/.config/coveragecv/credentials.json (0600), outside the repository. Do not print or commit them. Existing Roboflow Public 15 credits and Modal Starter credits are authorized. Modal token verified and installed; user confirmed $0 spend limit. Avoid persistent Modal Volumes (storage can continue charging). No Modal compute or volumes were created; pilot training ran locally on CPU. Roboflow billing usage API is unavailable for this workspace; do not invent a credit balance.

No repeated permission prompts: active session is full access / approval never. Existing research agents retain old settings and must not be resumed for tool work. Use current source and evidence, not earlier planning-only HANDOFF assertions. The persistent autonomous goal is active while the workbench upgrade is underway.
