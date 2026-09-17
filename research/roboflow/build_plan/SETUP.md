# Accounts, credentials, and compute preparation

Checked September 17, 2026. This is a setup guide; no account was created, credential changed, dependency installed, GPU provisioned, or training job launched during planning.

## What the user needs

| Item | Needed for initial experiment? | Action |
|---|---|---|
| [Roboflow account](https://app.roboflow.com/) | **Required** for the requested platform integration; preferred for hosted base training | Use a dedicated public demo workspace, check actual credits and raw checkpoint export entitlement, and prepare local API authentication. Do not buy an upgrade yet. |
| [Modal account](https://modal.com/) | Selected provider for the custom-loss stage; unnecessary if a suitable existing CUDA machine is used | Choose Starter, add a payment method, configure the workspace limits below, and authenticate locally when the environment is installed. |
| GitHub account | Only for publishing/sharing the repository | Existing account is sufficient. Local development and public source downloads need no new token. |
| Hugging Face account/token | No | The chosen dataset mirror and metadata are public and were fetched without authentication. |
| Weights & Biases / MLflow service | No | Save JSON/CSV and optional local TensorBoard logs. |
| AWS/GCP account, hosted database, paid Roboflow upgrade | No | A paid upgrade is not required; the architecture includes a path without hosted checkpoint export. |
| Exa credentials | Already available | Existing research helper reads the configured keys privately; not needed by the delivered application. |

The first dataset is `LibreYOLO/chess-pieces-mjzgj`, pinned to `17e0d3e7c76bea701ad623b0f7b13bec8859ff80`. Its 289 images and labels total approximately 13.8 MB according to the published file metadata. Details, source attribution, and split limitations are in [DATA_PROTOCOL.md](DATA_PROTOCOL.md).

## Roboflow first, with Modal for the custom training code

Roboflow's [hosted training interface](https://docs.roboflow.com/models/train/train-a-model) and newer SDK recipes expose model/run settings and tuning parameters. The reviewed public interfaces do not document arbitrary Python/custom-loss injection. Our feature changes training supervision, so uploading an annotation sidecar would not establish that hosted training consumes it.

The preferred route is **Roboflow hosted base training → exported trainable checkpoint → matched Modal continuation arms → final model back to Roboflow**. Base training uses a separate seed subset so it cannot see the hidden labels in the continuation experiment. If export is unavailable/incompatible, use Roboflow for versioned datasets, a separate hosted baseline and final deployment, while Modal runs the controlled experiment from stock initialization. See [ROBOFLOW_INTEGRATION.md](ROBOFLOW_INTEGRATION.md) for the complete decision and compatibility gates.

Roboflow Public currently advertises 15 monthly credits, though your grant/account may have more. Raw weights download is a separate Core/select-Enterprise entitlement. Inspect the actual account before choosing a path; extra credits alone do not prove export access. [Pricing](https://roboflow.com/pricing), [weights download](https://docs.roboflow.com/models/model-weights/download-roboflow-model-weights).

## Modal: recommended setup

1. Sign up for **Starter** at [Modal](https://modal.com/). Its current plan advertises $30/month of compute credits. GPU use still requires a valid payment method. Check the actual remaining credits in your account; do not assume all advertised credits remain available. [Pricing](https://modal.com/pricing), [GPU requirements](https://modal.com/docs/guide/gpu).
2. In **Usage & Billing**, set a Workspace usage budget and a Workspace spend limit. Proposed starting limits: **$15 gross usage** and **$10 maximum out-of-pocket spend**. These are suggested settings for this experiment, not permission to launch jobs now. A smaller limit also works, with fewer completed experiments.
3. The usage budget is before credits; the spend limit is after credits. Without an explicit spend limit, the default can be the usage limit minus credits. Starter has workspace controls; separate environment budgets require Team/Enterprise. [Budget behavior](https://modal.com/docs/guide/budgets).
4. During implementation, install the locked Modal SDK in the project environment and run `uv run modal setup`. This opens the documented authentication flow. The upstream command is `modal setup`; `uv run` simply selects the project's environment. [Setup](https://modal.com/docs/guide), [CLI reference](https://modal.com/docs/cli/latest/setup).
5. Keep Modal authentication in its local credential store. The SDK ordinarily uses `~/.modal.toml`; environment alternatives are `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. They do not belong in the repository or the remote training image. [Configuration reference](https://modal.com/docs/sdk/py/latest/config).

No token needs to be pasted into this conversation. Account authentication can be completed locally and reported as done.

### Proposed resources and cost

Start with one **A10 24 GB**, two physical CPU cores, and 16 GiB host memory. Use a Modal **Function**, not a persistent notebook or web endpoint. Probe a representative batch before deciding the experiment's update budget. An L4 or A100 is a fallback after comparing measured throughput and the displayed cost, not an automatic upgrade.

| Resource | Current GPU-only rate | Rationale |
|---|---:|---|
| A10 24 GB | $1.1016/hour | Initial CUDA target |
| L4 24 GB | $0.7992/hour | Lower price; project-specific speed unmeasured |
| A100 40 GB | $2.0988/hour | More VRAM; use only if measured benefit justifies it |

Modal Function CPU is $0.0000131 per physical-core-second, and RAM is $0.00000222 per GiB-second. The nominal A10 + two-core + 16 GiB request therefore costs:

```text
3600 × (0.000306 + 2 × 0.0000131 + 16 × 0.00000222)
= $1.323792 per billable hour
```

This is a resource-rate calculation, not a training-duration estimate or a total quote. Actual CPU/RAM usage, attempts, image setup, persistence, transfer, and applicable taxes can affect the bill. Prices were checked on the [official pricing page](https://modal.com/pricing); actual billing is authoritative.

The worker will have an explicit per-attempt timeout, a persisted experiment deadline, an attempt ledger, ordinary retries disabled, and one concurrent GPU. The proposed maximum initial long-run allocation is three sequential arms of at most two hours each after a short smoke test. **Two hours is a ceiling, not predicted training time.** Throughput measurements determine a common smaller step budget if needed. Modal preemption can restart an input even when normal retries are disabled, so a per-attempt timeout alone does not cap total usage. [Timeouts](https://modal.com/docs/guide/timeouts), [preemption](https://modal.com/docs/guide/preemption).

Use a named Volume for checkpoints, metrics, and run manifests; commit completed checkpoints explicitly. Download artifacts before deleting remote copies. Confirm no jobs remain active after the experiment. Storage can remain billed independently of GPU execution; check current volume pricing. [Volumes](https://modal.com/docs/guide/volumes).

### Alternatives if Modal is unavailable

| Option | Use when | Limitation |
|---|---|---|
| Existing NVIDIA machine | Already available | No new signup; still needs a reproducible environment and artifact capture. |
| [Runpod](https://www.runpod.io/) Secure Cloud Pod | Prefer a conventional rented GPU with SSH/Jupyter | More lifecycle/storage management. The official rate guide lists A40 48 GB at $0.49/hour, but current inventory/console price wins. Stopping GPU compute does not remove storage charges. |
| [Google Colab](https://colab.research.google.com/) | Need a quick free fallback | GPU type, access, and runtime are not guaranteed; unsuitable for promising completion by a deadline. |
| Local Mac | No cloud budget/account | CPU correctness tests and small MPS experiments; full training time remains unmeasured. |

Sources: [Runpod rates](https://www.runpod.io/articles/guides/ai-server-cost), [Pod pricing/lifecycle](https://docs.runpod.io/pods/pricing), [Colab FAQ](https://research.google.com/colaboratory/faq.html). Do not sign up for every provider; Modal is the selected initial path.

## Required Roboflow preparation

Use a workspace dedicated to public demonstration data. Record its URL/slug, plan, remaining credits, available RF-DETR sizes and raw PyTorch export entitlement. Public is sufficient for the fallback integration. A free workspace makes the demo data/models public, which fits this already-public dataset. We will create the source/seed/deployment projects during execution with preserved split membership.

The current key instructions are [Find Your Roboflow API Key](https://docs.roboflow.com/reference/authentication/authentication/find-your-roboflow-api-key), with [workspace API settings](https://app.roboflow.com/settings/api). The private API key is workspace-bound. Fine-grained scoped and multiple keys are documented Enterprise capabilities; do not assume a free workspace has them. [Scoped keys](https://docs.roboflow.com/reference/authentication/authentication/scoped-api-keys).

Store `ROBOFLOW_API_KEY` locally in a git-ignored `.env.local` or a local secret manager. Workspace/project/version identifiers are separate configuration. No key needs to be pasted into chat. The first connection checks workspace/recipe capabilities and records them; later writes target the dedicated demo projects. Export capabilities remain subject to the workspace's plan and terms. An optional alternative is Roboflow CLI browser authentication (`roboflow login`) or its [OAuth MCP connection](https://docs.roboflow.com/agents/mcp-server); only one working authentication route is needed.

The repository now includes a blank [`.env.example`](../../../.env.example) and [ignore rules](../../../.gitignore). Copy the example to `.env.local` and fill it locally. This prepares credentials without installing the project; the application/env-file loader has not been implemented yet.

The final platform demo includes [custom model upload](https://docs.roboflow.com/models/model-weights/upload-custom-weights) and an Inference/Workflow smoke test. A custom loss leaves the detector architecture intact, but the exact checkpoint format, class mapping, preprocessing and predictions must pass a compatibility check. We will validate that early, before a long run. The standalone compiler proof remains useful if a deployment importer is incompatible; the report must name that unfinished integration explicitly.

## Local environment and install plan

Observed during this planning session:

- Apple M5 Pro, 48 GiB unified memory (earlier hardware check).
- Approximately 145 GiB free disk.
- `uv`, `git`, `gh`, Docker CLI, and Node are installed.
- Modal CLI is not installed.
- System `python3` is 3.9.6, below RF-DETR 1.10.1's Python >=3.10 requirement.

Use **uv-managed Python 3.12** in the project. Resolve and lock the compiler dependencies separately from the optional training/remote extras so dataset validation does not require GPU packages. Use a tested Torch/torchvision pair for macOS and an explicitly selected CUDA wheel channel remotely. Exact Torch/Lightning patch versions will be locked after compatibility resolution and imports; no untested combination is being called verified.

Initial training configuration: RF-DETR Nano at its 384 resolution; one device; integer physical batch selected by a probe; accumulation one; full precision for correctness; compile off; EMA policy explicit; deterministic allowed transforms; fixed update budget. Add CUDA mixed precision only after the numerical tests pass. The Mac's unified memory is not a GPU throughput measurement.

No user-side installation is required now beyond account preparation. During the next execution phase, the environment installation, credential-presence checks, GPU probe, data validation, and run setup can be handled from this workspace.

## Readiness checklist

- [ ] Roboflow demo workspace and local authentication prepared; actual credits and export entitlement checked.
- [ ] Preferred hybrid or fallback protocol selected before training; no paid upgrade assumed.
- [ ] Modal account created, payment method accepted, budget/spend limits visible.
- [ ] Local Modal authentication completed once the project environment exists.
- [ ] Dependency locks resolve and imports pass on local CPU and remote CUDA.
- [ ] Dataset license/attribution, class counts, image decoding, and scene leakage audit complete.
- [ ] Correctness tests pass before a paid experiment starts.
- [ ] Smoke-test throughput determines a matched three-arm run budget.
- [ ] Saved checkpoints and metrics can be retrieved independently of the training process.

The user-side preparation is a Roboflow account/workspace plus local credentials, and Modal signup/payment setup if using its GPUs. The project will handle environment setup, capability checks, dataset preparation and run orchestration in the execution phase. No Hugging Face token, W&B account or additional cloud account is required.
