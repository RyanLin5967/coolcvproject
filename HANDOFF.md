> Historical research handoff. Current implementation, live jobs and billing decisions are in [EXECUTION_STATUS.md](EXECUTION_STATUS.md).

# Restart handoff

Saved September 17, 2026 in `/Users/idide/projects/coolcvproject`. This file is the durable context for the next session. Read [AGENTS.md](AGENTS.md) and [the build-plan index](research/roboflow/build_plan/README.md), then follow the next user instruction.

## Current objective and user decisions

The user wants an impressive project that is relevant to Roboflow and helps them stand out for infrastructure/backend work, while retaining meaningful CV/ML depth. They requested extensive company research, many categories of candidate projects and adversarial review using subagents and Exa. That research is complete. The selected project is **idea #3: Coverage-Aware Dataset Compiler**, provisional package/CLI name `coveragecv`.

The user initially prioritized idea quality over development duration, later asked about a one-day build, and then requested detailed preparation/architecture before implementation/testing. They most recently required **using Roboflow one way or another, preferably for main training**, allowing Modal plus Roboflow. Do not revert to the earlier plan that called Roboflow optional.

Latest instruction: **“about to restart make sure everything is durable.”** This handoff and a local Git checkpoint preserve the work. No external publication/push is required by that request. The next session should not start paid training solely because this handoff contains a training plan.

The user strongly dislikes repeated permission prompts and explicitly asked for autonomous authorized work. Actual permission instructions always govern; do not introduce extra confirmations for routine reads, edits, research or checks.

## What exists and what has not run

Completed and on disk:

- Company/customer/technical research, 44-candidate selection audit, 12 surviving idea cards and standalone HTML explorer.
- Follow-up on the selected feature's public product/source status and prior art.
- Nine detailed build-plan documents, approximately 19,000 words, including two fresh subagent data/design reviews.
- Sixteen pinned RF-DETR source snapshots, with SHA-256 manifest, plus public SDK and Exa research artifacts.
- Blank `.env.example`, `.gitignore`, and project working instructions.
- Documentation local-link/code-fence checks, pinned-source checksum checks and credential-ignore checks passed before restart preparation. These are document checks, not software/training tests.

**Not executed:** application implementation, dependency installation/resolution, Roboflow/Modal account authentication or provisioning, dataset image download/visual audit, numerical tests, model training, GPU measurements, model upload or Workflow deployment. No accuracy improvement has been measured. No remote training jobs were launched by this work.

Public dataset metadata and every one of its 289 small label files were inspected. That is distinct from downloading/decoding/auditing the images. The user has been told which accounts to prepare; no account readiness has been confirmed in chat. Check credential presence locally without displaying values when implementation begins.

## Where to read

| File | Purpose |
|---|---|
| [Build-plan index](research/roboflow/build_plan/README.md) | Start here for the final selected plan |
| [Setup](research/roboflow/build_plan/SETUP.md) | Accounts, local credentials, proposed compute/budgets |
| [Roboflow integration](research/roboflow/build_plan/ROBOFLOW_INTEGRATION.md) | Authoritative provider decision, hybrid data isolation, export/upload gates |
| [Architecture](research/roboflow/build_plan/ARCHITECTURE.md) | Compiler, identities, coverage states, trainer boundary, runtime, CLI |
| [Trainer design](research/roboflow/build_plan/TRAINER_DESIGN.md) | Exact source pin, criterion algebra, matching, initialization and constraints |
| [Data protocol](research/roboflow/build_plan/DATA_PROTOCOL.md) | Verified inventory, deterministic withholding, split/evaluation limits |
| [Implementation plan](research/roboflow/build_plan/IMPLEMENTATION_PLAN.md) | Ordered gates and acceptance tests |
| [Reuse map](research/roboflow/build_plan/REUSE_MAP.md) | Existing implementations, licenses and prior-art distinctions |
| [Adversarial review](research/roboflow/build_plan/ADVERSARIAL_REVIEW.md) | Critical source-specific failures and hybrid experiment review |
| [Research index](research/roboflow/README.md) | Original company research, shortlist/report and explorer |
| [Feature follow-up](research/roboflow/PARTIAL_LABEL_FOLLOWUP.md) | Prior discussion of public RF-DETR/product feature evidence |

The final build plan supersedes older shortlist text on compute/provider choices. Pricing and hosted account capabilities are time-sensitive; use actual account state and recheck official docs when executing.

## Required provider decision

Preferred: **Roboflow versioned datasets → hosted ordinary RF-DETR training on a separate fully annotated seed subset → trainable checkpoint export → three matched Modal continuation arms → final model upload and Roboflow Inference/Workflow demo.**

The reviewed hosted training interfaces expose configuration/recipe tuning but no documented arbitrary custom-loss hook. Therefore the novel coverage-aware stage runs where we control Python, initially Modal. Do not claim a sidecar upload enables the feature in hosted Train. Do not claim most compute runs on Roboflow until measured.

Raw PyTorch model download is documented for Core/some Enterprise accounts; Public training credits or a grant do not by themselves establish export entitlement. Check actual balance/plan/recipe/architecture before using credits. A hosted export also needs faithful import into the pinned trainer. ONNX inference export alone is not the intended fine-tuning artifact. No paid upgrade is required by the plan.

If export is unavailable/incompatible: use required Roboflow source projects/versions, a descriptive hosted baseline and final model deployment, while all controlled custom-loss arms run on Modal from one stock initialization. A hosted-only versus Modal comparison is not causal evidence for our loss because other training factors differ.

## Critical technical decisions to preserve

- RF-DETR **1.10.1**, commit `e3fc28795f2a4303069c6b72e80431e5ed716030`; start with ordinary Nano detection, 384 resolution. Current develop differs. Exact dependency locks remain unverified.
- Four image/class coverage states: exhaustive and verified-absent allow negatives; positive-only and unknown do not. Every observed matched positive keeps its **whole original soft-positive loss cell**. Unknown is default; missing boxes/class lists do not prove absence or completeness.
- Default IoU-aware BCE: mask the entire unreduced classification cell with `negative_allowed OR matched_positive`. Preserve detach operations, denominator, box losses, matcher and final/auxiliary/encoder dispatch. Do not mask just the negative algebraic term or renormalize by active classes.
- Retain exact method signatures, including `matched_targets=None` in `loss_labels` and `num_boxes=None` in `forward`.
- Model has **K+1 outputs** for K semantic classes. Preserve reserved-slot stock supervision and audit prediction mapping; postprocessing can select that slot and the evaluator can skip unmapped outputs silently.
- Release 1.10.1 does not support padded ground-truth filler validity. Reject padded GT. Packed variable-length targets and image padding are different features.
- Coverage lookup uses globally dense image IDs at the criterion boundary, bound to bundle/view/ontology digests. Do not send a class vector through box-filtering transforms. Preserve IDs in train-only projections.
- Materialize the actual `train/_annotations.coco.json` and `valid/_annotations.coco.json` RF-DETR layout. Canonical bundle `splits/*.coco.json` is not directly that layout. Keep test/hidden oracle files out of learner views.
- Stock new-ontology initialization requires explicit seeded K+1 main/encoder head initialization; upstream resize can tile/truncate rows. **Compatible hosted task-specific heads must instead be preserved.** All arms share identical detector weights and fresh identical optimizers.
- Initial proof uses one GPU, accumulation one, compile off, EMA off, FP32 numerical checks and explicit label-preserving transforms. EMA wraps the whole Lightning module, so coverage-buffer lifecycle needs testing before enabling it.
- Immutable files/run manifests first; no database, queue, permanent endpoint or large trainer fork. Preserve source notices for adapted code.

## Dataset and experiment

Public source: `LibreYOLO/chess-pieces-mjzgj` on Hugging Face, revision `17e0d3e7c76bea701ad623b0f7b13bec8859ff80`, from RF100 Chess Pieces v1. Metadata: 583 files, 13,776,642 bytes; 289 images split 202 train / 58 valid / 29 test. Source declares CC BY 4.0 and credits Joseph Nelson and Brad Dwyer.

Start with black/white pawn detection. Original YOLO IDs 4/10 map to COCO category IDs 1/2 and model indices 0/1. Original train has 496 black and 474 white pawn boxes. All inspected labels were syntactically/geometrically valid relative to normalized coordinates. Exact published image hashes were distinct; scene/near-duplicate independence and actual annotation completeness remain unverified.

Original `coverage-chess-v1` protocol: deterministic alternating source assignment across all 202 training images, seed `20260917`; 480 visible / 490 withheld boxes. Naive and aware receive the same partial labels; complete-reference receives all 970. These counts apply only to this original protocol.

Preferred `coverage-chess-hybrid-v1`: first reserve an audited group-preserving deterministic seed subset targeting 40 original training images. The integration/review documents define exact group hashing and minimum support gates. Hosted base training must not see the hidden labels on continuation images. Recompute all continuation counts; never reuse 480/490. Preserve final test for evaluation only. Shared validation is allowed under a fixed policy.

Measure naive/aware/complete-reference with matched initialization, continuation images, updates, transforms and evaluator. Complete-reference is context, not a guaranteed upper bound. Include precision/recall and false positives, not AP alone. Chess/RF100 may have upstream pretraining or model-selection exposure; a small pilot is not proof of unseen-domain generalization. Existing partial-label methods mean the novelty is integration/engineering, not inventing selective supervision.

## Next execution steps once the user resumes implementation

1. Follow Gate 0 in IMPLEMENTATION_PLAN: Python 3.12/uv packaging and locked extras, imports, private credential-presence checks, actual Roboflow capabilities and provider branch. A global default Python 3.9.6 was too old at inspection.
2. Download/audit the tiny pinned image set, freeze scene groups and protocol, implement compiler and reconcile Roboflow versions.
3. Prove stock all-exhaustive loss/gradient parity, zero forbidden-cell direct gradient, observed-positive preservation and real loader identity/class mapping before long training.
4. Obtain/validate hosted seed checkpoint if the preferred branch is viable; retain a useful descriptive hosted baseline if import fails. Measure a small CUDA probe, retrieve checkpoints, then freeze the common update budget.
5. Execute/evaluate the matched arms, test Roboflow upload/inference compatibility and generate the report. Preserve negative findings and explicitly name uncompleted gates.

Account signup/login can require user interaction; it is not a reason to stop independent compiler/test work. Proposed Modal budgets in SETUP are planning suggestions, not already configured account limits or measured training prices.

## Credentials, permissions and agent state

- Root global config was verified as `approval_policy = "never"` and `sandbox_mode = "danger-full-access"` in `/Users/idide/.codex/config.toml`. Network access is enabled in this session. Follow the next session's actual instructions; under approval-never do not set `sandbox_permissions` or request escalation.
- Old agents `/root/data`, `/root/cv_models`, `/root/inference_edge`, `/root/attack_technical`, `/root/attack_novelty` retained stale workspace-write settings and caused prompts. They were stopped. Reuse their files, not their tool sessions. Fresh `/root/plan_data_v2` and `/root/plan_review_v2` completed under full access/never; all useful findings are saved in documents. Do not depend on agent memory after restart.
- Exa research helper: `research/roboflow/tools/exa_research.py`. It reads keys from `~/.config/exa/keys`, discovered from the user's `~/projects/dbresearch` setup. No key values belong in this repository/chat. Some keys exhausted credits; helper tries the next on 401/402/429. Exa is research-only, not an application dependency.
- `.env.local` is ignored; `.env.example` is deliberately blank. No provider credentials were created during planning. Modal normally authenticates via its local credential store; do not copy credentials into the GPU image or report.
- Persisted local files survive process restart. The local Git checkpoint and `RESTART_MANIFEST.json` preserve/recheck the handoff and research files. No remote backup has been claimed or pushed.
