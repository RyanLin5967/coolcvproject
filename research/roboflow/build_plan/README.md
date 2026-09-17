# Coverage-Aware Dataset Compiler: build preparation

Prepared September 17, 2026. **Planning is complete; implementation, authentication, training and deployment have not started.** Working CLI/package name: `coveragecv`.

The project addresses a concrete training failure: combining datasets that annotated different classes can treat unlabeled objects as negatives. We will compile explicit coverage declarations into an immutable dataset artifact and enforce them in RF-DETR's classification loss. The infrastructure/backend value is traceability, validation, reliable execution and reproducibility around a real CV integration.

## Roboflow is required

Preferred lifecycle: **Roboflow versioned datasets → hosted base training on a separate seed subset → trainable checkpoint export → controlled Modal continuation → model upload and Roboflow Workflow**.

The reviewed hosted training interfaces do not document an arbitrary custom-criterion hook. The coverage-aware stage therefore runs under our Python code on Modal. Raw weight export is also a plan-specific feature. If the actual account cannot export a compatible checkpoint, use Roboflow datasets, a separate hosted baseline and final deployment while running the controlled experiment on Modal from shared stock initialization. This preserves a substantial Roboflow integration without requiring a paid upgrade. Details and official sources: [ROBOFLOW_INTEGRATION.md](ROBOFLOW_INTEGRATION.md).

## What to prepare

1. **Roboflow:** create/use a dedicated public demo workspace; note its workspace slug, actual credits and raw PyTorch weight-export entitlement. Copy the blank [`.env.example`](../../../.env.example) to ignored `.env.local` and fill `ROBOFLOW_API_KEY` / `ROBOFLOW_WORKSPACE` locally, or use the documented browser authentication route. Do not paste credentials into chat.
2. **Modal:** create a Starter account if no suitable CUDA machine is available; its GPU service requires a payment method. Set usage/spend limits and authenticate locally when the project environment is installed. Current credits and a concrete resource/cost plan are in [SETUP.md](SETUP.md).
3. **Nothing else needs a new account:** the initial public dataset needs no Hugging Face token; local logs replace a hosted experiment tracker; no AWS/GCP/database signup is needed. Existing Exa credentials remain research-only.

## Read the plan

| Document | Contains |
|---|---|
| [Setup](SETUP.md) | Accounts, local credentials, verified environment, GPU choices, current costs and readiness checklist |
| [Roboflow integration](ROBOFLOW_INTEGRATION.md) | Hosted/custom training boundary, checkpoint gates, seed isolation, no-export fallback, version reconciliation and deployment |
| [Architecture](ARCHITECTURE.md) | Component diagram, coverage semantics, immutable schema/identities, compiler behavior, runtime and CLI contract |
| [Trainer design](TRAINER_DESIGN.md) | Exact RF-DETR pin, integration seam, classification mask, K+1 outputs, initialization, normalization and transform constraints |
| [Data protocol](DATA_PROTOCOL.md) | Pinned 289-image dataset, verified label inventory, controlled withholding, split limitations and evaluation |
| [Reuse map](REUSE_MAP.md) | Existing implementations, what to adapt, prior-art distinctions and source/license notices |
| [Implementation plan](IMPLEMENTATION_PLAN.md) | Ordered execution gates, behavior-based tests, provider checks, experiment and deliverable criteria |
| [Adversarial review](ADVERSARIAL_REVIEW.md) | Independent failure analysis, six critical gates, hybrid confounds and bounded claims |

## Evidence and unresolved checks

Completed during preparation: public provider/API research; RF-DETR release-source inspection; public dataset metadata and all 289 label-file inspection; two fresh subagent reviews; architecture and experiment design. The dataset snapshot has 202 train, 58 validation and 29 test images. The original stock-init withholding protocol exposes 480 pawn boxes and hides 490; the hybrid variant must recompute counts after reserving its seed subset.

Still requiring execution: account entitlements, exact dependency locks/imports, image/scene audit, stock-loss/gradient parity, hosted checkpoint import, GPU throughput, model upload parity and empirical accuracy. No model has trained and no improvement is claimed. A small chess experiment is a mechanism pilot; it does not establish broad-domain generalization or unique research novelty.

## First execution milestone

Build the environment and compiler, inspect account capabilities, freeze the data protocol, then prove the custom criterion preserves fully annotated behavior and removes only unjustified classification gradients. Start long training after those checks. A one-day demonstration may be feasible, but cloud queues and actual training throughput must be measured.

The repeated local approval issue was traced to old research agents retaining an earlier permission mode. Those threads were stopped; replacements confirmed full access and approval-never. The effective global config already contained those settings. The project [working instructions](../../../AGENTS.md) record that handling so old threads are not reused for tool work.
