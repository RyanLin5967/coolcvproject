# CoverageCV

**Merge partially annotated detection datasets without teaching the model that every missing box is background.** CoverageCV carries explicit annotation coverage from source data through compilation, training, evaluation and deployment.

The working product is a local dataset workbench: **http://127.0.0.1:8765** when running. It includes editable coverage policies, immutable dataset revisions, image/provenance inspection, durable training jobs, live comparisons, multi-seed research results and verified Roboflow integration.

## Start the workbench

```sh
uv sync --extra train --extra remote --extra web --extra analysis --group dev
uv run --no-sync coveragecv serve
```

On this workspace, prepared datasets and measured runs already exist under ignored `data/` and `artifacts/`. On a fresh checkout, import your own dataset ZIP containing a `coverage.json` spec, COCO files and images, or prepare the public chess example:

```sh
uv run --no-sync coveragecv demo
uv run --no-sync coveragecv serve
```

`demo` downloads the pinned public chess data and official RF-DETR Nano weights, repairs recognized repeated-position splits, trains three 100-update CPU pilot runs and generates the historical offline report. It does not start cloud compute. The workbench job queue also runs locally; Modal experiments use the explicit research scripts.

## What is implemented

- **Coverage contracts:** `unknown`, `positive_only`, `exhaustive`, and `verified_absent`, with explicit ontology mappings, source revisions, evidence and attribution. Undeclared classes default to unknown.
- **Verified artifacts:** canonical SHA-256 identities, full file inventories, decoded image dimensions, split checks, duplicate/conflict detection, immutable learner views and atomic publication. Test images and hidden reference training labels are excluded from learner views.
- **Arbitrary detection ontologies:** the compiler, RF-DETR trainer and evaluator support general class names/counts. Integration tests train, save, reload and evaluate a three-class non-chess fixture.
- **Coverage-aware RF-DETR 1.10.1:** the unreduced IoU-aware classification loss masks unjustified negatives while preserving observed positives, box losses, reserved channels, decoder auxiliaries and encoder supervision. Complete coverage matches the upstream loss and gradients.
- **Operational application:** ZIP/local import, compiler diagnostics, source-by-class policy editing, version history/diffs, coverage visualization, image inspection, training recipes, queue progress, logs, cancellation and predictions from actual checkpoints.
- **Durable execution:** SQLite WAL, exclusive transactional claims, process-isolated workers that survive webserver restarts, restart reconciliation, explicit cancellation and artifact-bound results.
- **Teacher experiments:** train-only predictions must agree across original/mirrored images before becoming provisional labels. Unknown coverage stays unknown; human labels win conflicts. Teacher provenance is preserved through crop/flip transforms. Experimental regression weights distinguish predicted coordinates from human boxes.
- **Reproducible research:** shared initial parameters, paired seeds, matching update budgets, returned cloud checkpoints, run/data/hash checks, per-class metrics, error analysis and transparent reporting of unsuccessful methods.
- **Real Roboflow integration:** resumable exact-label dataset upload/export verification, an actual Roboflow-trained baseline and a custom model whose hosted class/box mapping has been checked against native predictions.

## Measured results

Coverage-aware training recovers supervision lost when independently annotated sources are merged. The original matched pawn experiment gained **12.56 AP points** over a naive merge across three paired seeds. With a stronger augmentation recipe, the aware model reaches **78.82 AP50:95**, within **0.11 points** of the complete-label reference while training on 451 observed boxes instead of 896.

All figures below are complete-validation COCO AP50:95, multiplied by 100; uncertainty is sample standard deviation across training seeds. Arms within a recipe share initial parameters, batch size and update budget.

| Task and recipe | Naive partial labels | Coverage-aware | Complete labels |
| --- | ---: | ---: | ---: |
| Pawns · 384px, 2,000 updates | 63.79 ± 2.07 (n=3) | 76.35 ± 0.70 (n=3) | 78.04 ± 0.53 (n=3) |
| Pawns · 512px augmentation, 4,000 total updates | 76.25 ± 1.55 (n=3) | **78.82 ± 0.42 (n=3)** | 78.94 ± 0.21 (n=3) |
| All 13 chess classes · 384px, 2,000 updates | 58.23 ± 2.40 (n=3) | 69.64 ± 1.00 (n=3) | 72.54 ± 0.09 (n=3) |
| All 13 chess classes · 512px augmentation, 4,000 total updates | 69.03 ± 0.69 (n=3) | **72.91 ± 0.11 (n=3)** | 74.16 ± 0.13 (n=2) |
| All 13 chess classes · Large/704px/EMA, 2,000 updates | 72.78 (n=1) | **73.68 (n=1)** | 74.97 (n=1) |

The augmented recipe also improves the naive baseline: its paired coverage benefit is **+2.57 points on pawns** and **+3.88 on all chess classes**. The completed Large pilot adds **0.69 points** over the same-seed Nano/512px aware model (73.68 versus 73.00), with a **0.91-point** matched Large coverage benefit. This is a modest improvement, not the hoped-for breakthrough; architecture, resolution, EMA and update budget change together, and it has only one seed. The augmented pawn models achieve 100% recall at score 0.25, averaging 0.075 false positives per image. These are 58-image chess validation results; the two tasks share a board/camera domain. The full ontology retains a generic `bishop` class with no training positives. The complete-label reference is contextual, not a proven upper bound.

Construction safety supplies a separate industrial dataset: 995 train / 120 validation / 91 test images after group-based split repair. Partial training retains 2,119 boxes and withholds 4,261. Its **first matched seed** scores 46.34 naive, 48.14 aware and 52.76 complete: a preliminary **+1.80-point** gain. Only four of nine planned runs completed (naive n=2, aware n=1, complete n=1); its planned test evaluation has **not run**. The two completed naive seeds average 45.85 ± 0.69; that aggregate is not substituted into the one-seed paired comparison.

A separate four-case construction continuation tested class-balanced object crops, using one frozen plan derived only from partial human labels. All arms added 2,000 updates, reaching 6,000 total. Ordinary aware continuation scored **48.00**; crop-aware scored **47.28**, crop-naive **43.17**, and crop-complete **52.45**. The matched primary crop effect is **−0.72 points**, so crop training is not promoted. Repeated sampling enlarged rare objects but added no annotations.

Inference experiments on the original construction checkpoints produced the following **single-seed validation** results:

| Inference policy | Naive | Coverage-aware | Complete labels |
| --- | ---: | ---: | ---: |
| Fresh CPU full frame | 46.34 | 48.14 | 52.75 |
| Full frame + four tiles, all classes | 45.29 | 48.96 | 52.35 |
| Tiles for classes with small training boxes; full frame for other classes | 46.89 | **49.84** | 53.99 |

The last policy routes `helmet` and `no-helmet` through tiled inference because their observed training-box median longest side at 512px is ≤96px; other classes retain raw full-frame predictions. The rule uses training statistics, but its design followed validation error inspection: **this is secondary, validation-guided inference engineering**, not the primary training result. It requires five image passes and gives the original aware detector +1.70 AP points over its fresh full-frame control.

The same fixed tiling was declared for all four continuation checkpoints before their primary scores. All four are now evaluated; full tiling lowers each full-frame score. Applying the training-size routing rule gives **49.88 ordinary-aware, 48.06 crop-aware, 43.87 crop-naive and 53.87 crop-complete**. The best aware result is only **0.04 points** above the original detector with the same routing: the extra training did not produce a breakthrough. Crop training remains worse than its matched ordinary-aware control under both inference policies. CPU and MPS outputs did not pass exact/raw parity, so these diagnostic comparisons use CPU throughout.

[The construction label audit](docs/CONSTRUCTION_LABEL_AUDIT.md) inspected all 33 observed no-helmet training boxes and 11 validation references, plus selected errors. It found a training/validation scale shift, inconsistent head geometry and specific reference defects, alongside real model errors. One scene contributes six of the eleven rare-class references. Labels and splits remain unchanged; these findings do not explain the entire accuracy gap or justify a 95-AP claim. A proposed annotation-review/acquisition workflow is **deferred, not implemented or launched**.

**Unsuccessful experiments stay visible.** Longer 384px training averaged 75.36 on pawns; teacher supervision averaged 77.92, below augmentation alone. Teacher supervision also trailed augmentation on all chess classes (72.75 versus 72.91). A cautious pseudo-box pilot scored 78.70 on pawns, below its matched 78.90 augmentation control. Fixed three-model box fusion scored 78.69 on pawns and 72.48 on all chess classes; it did not improve the aware models and is not promoted.

**Experiment snapshot:** 55 of the original 64 planned cloud runs collected; nine remain interrupted. Independent local rescoring reproduced all three completed Large metric dictionaries exactly. An additional crop refiner trained on the 994 observed human boxes has also finished: applied to the same-seed Nano/512px models, it changed AP from **69.21 → 69.33 naive, 73.00 → 72.55 aware, and 74.06 → 73.29 complete**. The primary aware result regressed by 0.44 points, so **the refiner is not promoted**. Its aware evaluation added 23.19 seconds for 58 images on local CPU, excluding detector inference and COCO scoring. See [decisions and difficulties](docs/DECISIONS.md) for the lost first checkpoint, zero-area-box fix, MPS incompatibility and recovery without further cloud training.

Open **Research lab** for collected measurements, actual seed counts, per-class results and localization diagnostics. [The portable results snapshot](docs/RESULTS.json) retains metrics and checkpoint/data identities without datasets or credentials. Full local ledgers live under ignored `artifacts/`. Recipe development uses validation feedback; these are not untouched-test discoveries.

## Bring your own data

A spec contains an ordered class ontology and source declarations. Paths resolve relative to the spec. A ZIP import must contain exactly one `coverage.json`, with all referenced files inside the archive.

```json
{
  "classes": ["person", "helmet"],
  "sources": [{
    "id": "people-source", "revision": "v1", "split": "train",
    "annotations": "train.coco.json", "images": "images",
    "class_map": {"person": "person"},
    "coverage": {"person": "exhaustive", "helmet": "unknown"},
    "evidence": "This source labels every person; it makes no claim about helmets.",
    "attribution": "Dataset owner and license"
  }]
}
```

Add a validation source with complete reference coverage before training comparisons. A declaration is an assertion by the data owner: consistency checks cannot prove that annotators found every object. `verified_absent` contradicting an observed positive fails compilation.

```sh
uv run --no-sync coveragecv compile coverage.json
uv run --no-sync coveragecv inspect artifacts/bundles/<digest>
uv run --no-sync coveragecv view artifacts/bundles/<digest>
uv run --no-sync coveragecv initialize artifacts/views/<digest> artifacts/my-initialization
uv run --no-sync coveragecv train artifacts/views/<digest> \
  artifacts/my-initialization/initialization.pt artifacts/my-run \
  --arm aware --recipe augmented_fresh --max-steps 2000
```

Teacher supervision is explicit and auditable:

```sh
uv run --no-sync coveragecv pseudo-label artifacts/views/<digest> \
  artifacts/my-run/detector.pt artifacts/teacher-views
uv run --no-sync coveragecv train artifacts/teacher-views/<digest> \
  artifacts/my-initialization/initialization.pt artifacts/my-student \
  --arm aware --recipe augmented --warm-start artifacts/my-run/detector.pt \
  --max-steps 2000 --pseudo-box-weight 0.1
```

`0.1` is an experimental geometry weight, not an established winning setting. Classification still uses accepted pseudo positives. Use `1` for the original teacher recipe, or `0` to test classification-only pseudo supervision. Weighted pseudo-box experiments currently require one training process. CLI/device settings do not implicitly provision cloud resources.

The CLI also supports official RF-DETR Large initialization with `initialize --variant large` and training with `--recipe large_fresh`. That recipe uses 704px inputs and EMA. Use `--device mps` on compatible Apple hardware or `--device cuda` on an existing CUDA machine. The workbench offers local CPU or Apple-MPS execution and Nano/Large recipes, checks device availability, and keeps these jobs separate from cloud submissions.

## Roboflow integration

The observed [chess dataset, version 2](https://app.roboflow.com/ryan-lin-khj4s/coveragecv-chess-mvp/2) was uploaded, downloaded again and compared image-by-image: 201 train images / 451 boxes and 58 validation images / 241 boxes, with identical class/split identities and coordinates within 0.0001px.

A separate complete-label project, `coveragecv-chess-hosted/1`, completed **20 epochs of Roboflow-hosted RF-DETR Nano training**. Its recipe and reported metrics differ from the matched custom-loss experiments. That baseline used the original chess test split, so the chess test set is not globally untouched.

The custom 2,000-update aware model is deployed as `ryan-lin-khj4s/coveragecv-chess-mvp-3-rfdetr-nano-t3`. Hosted inference passed semantic/geometry checks on three validation images: 15 black-pawn and 13 white-pawn detections, minimum matched same-class IoU 0.936. This is a deployment smoke test, not a full hosted evaluation or bitwise equivalence claim.

The connector fixes two measured compatibility issues: Roboflow VOC coordinates need a one-based origin, and the observed hosted weight importer expects the reserved classifier row first. The export explicitly reorders every main/encoder classifier; native checkpoints remain unchanged. Earlier failed attempts remain recorded, not mislabeled as successful.

Coverage-aware loss runs in CoverageCV on local hardware or Modal. Ordinary Roboflow-hosted training does **not** automatically consume the coverage sidecar. Roboflow stores the versioned data, trains the separate baseline and serves the custom exported model.

## Verification and cloud reproduction

```sh
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts
node --check src/coveragecv/workbench/static/app.js
```

The final full suite passed **122 tests**. Public-SDK export and real CPU/MPS training have separate execution evidence. Browser checks covered research views on desktop/mobile and the actual construction-study panel without mocked results; training-form checks intercepted submissions. The real browser/process lifecycle test, `scripts/qa_workbench.py`, checks import, policy failure/recovery, CPU training, restart survival and cancellation in an isolated state directory on port 8766. Generated evidence/screenshots remain outside Git. At handoff, every project Modal app reports zero containers; no further experiments are being launched.

Plain Nano and Large exports load through the public `RFDETR.from_checkpoint(...)` API with exact parameter equality, including the actual trained 73.68-AP Large aware checkpoint. An export metadata fix supplies the upstream top-level model identifier and architecture overrides, including the Nano continuation's positional-encoding size; earlier source checkpoints remain unchanged.

Cloud deployment uses frozen source and dataset snapshots with pinned requirements. **Deploy by module name** (`modal deploy -m coveragecv.training.modal_improve`), not by file path. This preserves the import path inside the image. See [architecture and reproduction](docs/ARCHITECTURE.md) for the complete preparation sequence; research scripts intentionally do not silently create account credentials or billing limits.

Credentials are read from environment variables or `~/.config/coveragecv/credentials.json`, outside the repository. Cloud jobs have bounded resources/timeouts, zero minimum containers and no persistent Modal Volumes. The requested budget is **$0 out of pocket**, but accounting is unresolved: Modal's API reported $32.68 metered / $2.68 billed after $30 of credits, while the user reports $12.91 credits remaining, $25.41 workspace usage and no visible card charge. Neither a cash charge nor a zero-cost outcome is confirmed. After the user authorized continued work, reservations were $3.30 for Large, $1.70 for two refinement attempts and $4.40 for four construction continuations: **$9.40 maximum reserved** against the user-reported remaining credits. Failed work remains counted; local tiled evaluation adds no cloud calls. Reservations are not actual usage figures or a reconciled balance.

See [decisions and difficulties](docs/DECISIONS.md), [execution state](EXECUTION_STATUS.md), [historical research](research/roboflow/build_plan/README.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Historical planning documents describe possibilities, not implemented capabilities.
