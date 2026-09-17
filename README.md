# CoverageCV

Merge object-detection datasets without teaching a model that unannotated classes are absent. CoverageCV compiles explicit annotation policies into verified dataset artifacts, then carries those policies into RF-DETR's classification loss.

**Working MVP:** [interactive local demo](artifacts/demo/index.html). Open that file in a browser; it contains real images, observed/reference labels, predictions, three measured runs and provenance. No server is required. Generated artifacts stay on this machine and are ignored by Git.

## Run

```sh
uv sync --extra train --extra remote --group dev
uv run --no-sync pytest -q
uv run --no-sync coveragecv demo
open artifacts/demo/index.html
```

`demo` downloads the pinned public chess dataset and RF-DETR Nano weights on its first run, groups repeated board configurations, prepares partial/complete views, trains three CPU runs for 100 updates each, evaluates complete validation labels, and generates the report. It reuses matching completed runs and rejects stale/mismatched results. Use `--output artifacts/another-attempt` for a new experiment. This command does not launch cloud workloads.

For the already generated report:

```sh
uv run --no-sync coveragecv report
```

## What works

- A COCO compiler with explicit ontology mapping and four coverage states: `unknown`, `positive_only`, `exhaustive`, `verified_absent`. Missing declarations default to unknown. The compiler supports arbitrary class names; the bundled training/evaluation recipe is deliberately restricted to black-pawn / white-pawn.
- Deterministic SHA-256 manifests, verified image bytes and dimensions, provenance, atomic artifact publication and run ledgers. Contradictory absence, conflicting duplicates, split collisions, unsupported annotation types and modified artifacts fail validation.
- Learner views contain only observed train labels and complete validation labels. Test images and the withheld training-label reference are excluded from those views.
- RF-DETR **1.10.1** integration through the upstream Lightning loop and a small criterion adapter. It masks unjustified negative classification supervision while preserving matched positives, box regression, the reserved class, auxiliary decoder heads and encoder supervision.
- Shared detector initialization, explicit experiment arms, checkpoint/data bindings, a common complete-reference COCO evaluator and an offline interactive report.
- A resumable Roboflow dataset uploader and export verifier. Version 2 of the [demo project](https://app.roboflow.com/ryan-lin-khj4s/coveragecv-chess-mvp/2) was downloaded again and checked image-by-image: 201 train images / 451 boxes and 58 validation images / 241 boxes, with matching classes, splits and coordinates.

The local tests check stock loss and gradient parity for complete coverage, unknown-class gradients, matched positives, empty targets, decoder/encoder branches, artifact tampering, bad comparisons and Roboflow's VOC coordinate conversion. Saved evidence is in [`evidence/mvp.json`](evidence/mvp.json).

## Pilot result

These are **100-update, single-seed validation measurements**, not a converged benchmark or a broad accuracy claim.

| Run | AP50:95 | Recall at 0.25 | False positives/image at 0.25 |
| --- | ---: | ---: | ---: |
| Naive partial-label merge | 71.68% | 86.31% | 0.052 |
| Coverage-aware | 72.34% | 97.51% | 1.310 |
| Complete-label reference | 71.70% | 97.51% | 0.448 |

The aware model finds more objects at this fixed threshold and also makes more false positives. The complete-label reference is not an accuracy ceiling after only 100 updates. The separate 30-image test split remains unevaluated. Board/camera domain is shared, repeated-position grouping is heuristic, and the pretrained detector's exposure to this public benchmark is not established. The pilot demonstrates execution and the supervision tradeoff; a larger, independent evaluation is the next ML step.

## Compile your own data

Supply a JSON spec with `classes` and `sources`. Each source declares its COCO annotations, image directory, split, category-name mapping, coverage, revision, evidence and attribution. Paths resolve relative to the spec; image filenames must remain within the source image directory.

```json
{
  "classes": ["cat", "dog"],
  "sources": [{
    "id": "cat-dataset", "revision": "v1", "split": "train",
    "annotations": "annotations.coco.json", "images": "images",
    "class_map": {"cat": "cat"}, "coverage": {"cat": "exhaustive"},
    "evidence": "This source labels every cat; dog coverage is unknown.",
    "attribution": "Dataset owner and license"
  }]
}
```

```sh
uv run --no-sync coveragecv compile spec.json
uv run --no-sync coveragecv inspect artifacts/bundles/<digest>
uv run --no-sync coveragecv view artifacts/bundles/<digest>
```

`exhaustive` and `verified_absent` permit negative supervision. `unknown` and `positive_only` preserve observed positives without inventing negative evidence. Coverage declarations are assertions by a data owner, not facts inferred by the compiler. Conflicting duplicate observations are rejected rather than automatically reconciled.

## Roboflow and budget

Roboflow hosts the versioned observed dataset. The custom-loss experiment ran locally, with no Modal compute or volumes. The account API returned `availability.entitled=false` for custom training recipes; that result does not establish that all ordinary hosted training is unavailable. Ordinary hosted training does not consume our coverage sidecar or custom criterion automatically.

A model archive was also submitted to Roboflow version 1 for conversion. Its current saved status is shown in the report/execution record; uploading alone is not proof of working hosted inference. Version 1's dataset has a superseded VOC-origin issue, fixed in version 2; the uploaded model itself was trained on the correct local coordinates.

Credentials come from environment variables or `~/.config/coveragecv/credentials.json`, outside the repository. The user budget is $0 out of pocket. Do not start cloud training or provision persistent storage just to run the demo.

See [execution state](EXECUTION_STATUS.md), [research and architecture](research/roboflow/build_plan/README.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Earlier research plans describe a larger future system; they are not a claim that every planned feature is implemented.
