# Coverage-aware dataset compiler: data and experiment protocol

Planning snapshot: 2026-09-17. This document specifies the first implementation; no training has run and no accuracy result is claimed. Public metadata and all 289 tiny YOLO label files were inspected without authentication. Images have not yet been downloaded or visually audited. [ARCHITECTURE.md](ARCHITECTURE.md) is the authoritative integration contract; this document supplies the dataset inventory and experimental procedure.

## Decision

Use **black-pawn and white-pawn detection** from one pinned, small public chess dataset for the first controlled experiment. Simulate two disjoint source projects: one exhaustively annotated for black pawns, one exhaustively annotated for white pawns. Train ordinary and coverage-aware RF-DETR on exactly the same partially labeled images. Evaluate both on the same untouched, reference-labeled validation/test partitions.

This experiment isolates an annotation-policy mismatch. It does not establish robustness across different camera domains, imperfect annotators, conflicting ontologies, or missing instances within a class. The word “exhaustive” in the controlled experiment means exhaustive relative to this public reference annotation set; actual annotation completeness requires a visual audit.

The 13-class version is not the primary experiment. One generic `bishop` annotation coexists with color-specific bishop categories, and that generic class is absent from validation/test. Expanding to the 12 explicitly colored piece categories is a later stress test, after a documented decision about the generic bishop. Do not silently guess its color or report meaningful AP for a class with no evaluation instances.

## Public source, access, and attribution

| Item | Pinned choice |
| --- | --- |
| Actual Hugging Face repository | `LibreYOLO/chess-pieces-mjzgj` |
| Revision | `17e0d3e7c76bea701ad623b0f7b13bec8859ff80` |
| Upstream dataset | Roboflow 100 / `roboflow-100/chess-pieces-mjzgj`, version 1 |
| Declared license | CC BY 4.0 in the source YAML and dataset README |
| Creator attribution | Source README credits Joseph Nelson and Brad Dwyer; retain that attribution and the RF100 citation |
| Original export | April 3, 2023; auto-oriented and stretched to 640×640; README states no image augmentations were applied |
| Images / labels | 289 images and 289 YOLO label files |
| Complete repository files | 583 files, 13,776,642 bytes; 9 directory entries are not files |
| Label content size | 120,877 bytes |
| Access | Public, ungated files were read successfully without a Hugging Face token or Roboflow API key |

Primary sources: [pinned repository](https://huggingface.co/datasets/LibreYOLO/chess-pieces-mjzgj/tree/17e0d3e7c76bea701ad623b0f7b13bec8859ff80), [pinned class/config file](https://huggingface.co/datasets/LibreYOLO/chess-pieces-mjzgj/blob/17e0d3e7c76bea701ad623b0f7b13bec8859ff80/data.yaml), [pinned attribution README](https://huggingface.co/datasets/LibreYOLO/chess-pieces-mjzgj/blob/17e0d3e7c76bea701ad623b0f7b13bec8859ff80/README.dataset.txt), [pinned export README](https://huggingface.co/datasets/LibreYOLO/chess-pieces-mjzgj/blob/17e0d3e7c76bea701ad623b0f7b13bec8859ff80/README.roboflow.txt), [original version](https://universe.roboflow.com/roboflow-100/chess-pieces-mjzgj/dataset/1).

Use the actual repository name above: the mirror README's code example spells the owner `Libre-YOLO`, which differs from the working repository. Do not copy that example blindly. Fetch at the fixed revision, follow all metadata pagination links, and verify the expected inventory before materializing artifacts. A single API page is not a complete tree even if a large `limit` was requested.

Retain the dataset attribution, CC BY 4.0 notice/link, RF100 citation, fixed revision, original export notes, and a transformations note in every published derived artifact. The transformations note must disclose the two-class projection, format conversion, source partitioning, and deliberate annotation withholding. Do not present withheld annotations as annotation mistakes in the original dataset.

No signup is needed to read this public dataset mirror. A Roboflow account is nevertheless **required for the requested project integration**, including versioned source projects, hosted training where feasible, and the final deployment demo. It does not by itself enable a custom loss in hosted training. [ROBOFLOW_INTEGRATION.md](ROBOFLOW_INTEGRATION.md) defines the preferred hosted-seed/Modal-continuation path and its fallback.

The exact withholding counts below describe the original `coverage-chess-v1` **stock-initialization protocol** on all 202 training images. The preferred `coverage-chess-hybrid-v1` protocol first reserves a disjoint fully labeled seed subset for Roboflow training; it must recompute every continuation count and cannot reuse the 480/490 figures. Original source inventory and validation/test counts remain applicable unless the scene audit requires a split revision.

## Verified label inventory

The original YOLO class indices must not be confused with compiled COCO categories or RF-DETR logits.

| Original YOLO ID | Class | Train boxes | Validation boxes | Test boxes |
| --- | --- | ---: | ---: | ---: |
| 0 | bishop | 1 | 0 | 0 |
| 1 | black-bishop | 97 | 22 | 21 |
| 2 | black-king | 101 | 29 | 17 |
| 3 | black-knight | 141 | 30 | 25 |
| 4 | **black-pawn** | **496** | **77** | **86** |
| 5 | black-queen | 62 | 11 | 14 |
| 6 | black-rook | 147 | 28 | 26 |
| 7 | white-bishop | 125 | 22 | 25 |
| 8 | white-king | 105 | 29 | 15 |
| 9 | white-knight | 140 | 19 | 25 |
| 10 | **white-pawn** | **474** | **77** | **88** |
| 11 | white-queen | 81 | 16 | 14 |
| 12 | white-rook | 138 | 26 | 20 |

| Split | Images | Both pawn classes | Black pawn only | White pawn only | Neither pawn class |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 202 | 97 | 8 | 8 | 89 |
| Validation | 58 | 25 | 0 | 1 | 32 |
| Test | 29 | 16 | 0 | 2 | 11 |

“Neither” means neither class has a reference annotation, not that the image is literally empty. Keep these images: they supply meaningful negatives for the two-class task and make false-positive measurement possible. The original full label set contains one empty test label file and no empty train/validation files.

All 289 label files parsed as five-column YOLO detection rows with integer class IDs in `[0,12]`, finite normalized coordinates in `[0,1]`, positive widths/heights, and boxes within the normalized image frame. No invalid row was found in this label-only audit. Actual image decodes/dimensions and box overlays remain an execution-time check.

For reproducibility, SHA-256 of the concatenation of each sorted label path, a NUL byte, and its original file bytes is `5c60590b94784af11ffca089dd8a505b06201658aef66ed32a1b5d162e48eeb0`. This is an audit checksum with an explicitly stated concatenation convention, not the final artifact-addressing format.

## Split integrity and limits of the evidence

Preserve the supplied train/validation/test image membership: 202/58/29. Do not repartition until evaluating duplicate/scene groups. All 289 image content identifiers in the complete public file metadata are distinct, and no original filename stem before `.rf.` repeats across the inventory. This rules out the exact duplicates detected by those checks; it does **not** establish independence of scenes, near-duplicates, camera sessions, or model pretraining. **Near-duplicate/scene independence is currently unverified. Completing and documenting that audit is a blocker before interpreting pilot accuracy as held-out evidence.** It does not block synthetic correctness tests or a throughput smoke test.

Before the first expensive run:

1. Download the small pinned dataset and verify bytes, decodability, dimensions, label/image one-to-one correspondence, and per-file SHA-256 digests.
2. Produce a contact sheet for each split plus a cross-split nearest-neighbor shortlist using an existing image-hash implementation. Review the strongest matches and record the outcome. At 289 images a human-readable audit is practical; similarity thresholds must not be treated as proof.
3. Inspect pawn overlays on all 29 test images and a reproducible train/validation sample; record any missing/wrong labels. This is annotation-quality review before inspecting model predictions, not model-specific cherry-picking. If editing labels is necessary, version the correction and apply the same reference to all arms.
4. If same-scene groups visibly cross splits, build one fixed group-level repartition before training and reissue every count/manifest. Do not alter only the losing arm or silently retain the original counts in a report.

Separate two notions of leakage explicitly. The held-out test must remain outside **this project's fine-tuning and model selection**. RF-DETR/checkpoint/backbone authors may have used COCO, RF100, or related data for pretraining, architecture selection, or benchmarking. Chess is an RF100 dataset. A held-out fine-tuning split is therefore not evidence that the model family has never encountered these images or this domain. Use identical checkpoint hashes across arms and report the checkpoint's documented provenance. This demonstration is a paired supervision experiment, not a claim of uncontaminated novel-domain generalization.

## Ontology and identifier contract

Use an explicit, versioned mapping rather than deriving class order independently for each split:

| Semantic index / coverage column | Canonical key | Source YOLO ID | Compiled COCO category ID | Expected RF-DETR label/logit index |
| ---: | --- | ---: | ---: | ---: |
| 0 | `black-pawn` | 4 | 1 | 0 |
| 1 | `white-pawn` | 10 | 2 | 1 |

The custom COCO loader in RF-DETR **1.10.1**, commit `e3fc28795f2a4303069c6b72e80431e5ed716030`, supports category remapping and sharing the training split's mapping with validation/test. Reuse that implementation and assert `cat2label == {1: 0, 2: 1}` and the inverse evaluator mapping at that pinned commit. Do not assume a `category_id` is a logit index. Reference: [pinned RF-DETR COCO loader](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/datasets/coco.py). A local checksum-indexed source snapshot is in `../raw/build_rfdetr_1_10_1/sources.json`.

Assign **dense, global image indices `0..N-1`** across the compiled artifact, with deterministic split order `train`, `valid`, `test` and lexical original-path order inside each split. Store these as COCO `image_id` values and sidecar row indices. Also retain a separate stable image key and content SHA-256 for provenance. For the unmodified split inventory: train IDs `0..201`, validation `202..259`, test `260..288`. No image ID is recycled inside an artifact; no original external ID is trusted to be dense.

At the criterion boundary, gather coverage rows using each preserved `target['image_id']`, not batch position, dataloader order, source-local ID, file basename, or a hash modulo table size. Verify integer dtype, range, uniqueness of declared rows, device placement, and ontology hash. Repeated sampling of the same image is legal and must gather the same row. Shuffling, multithreaded loading, DDP, and validation reordering must not change coverage semantics.

The full identity tuple is `(artifact_digest, image_id)`. An index has no meaning across different artifacts. Evaluation outputs must retain that tuple or the equivalent immutable run-to-artifact binding. Train/validation projections preserve the parent image IDs and class order; they have a separate view digest and explicit parent/index-space binding. Removing evaluator-only files must never silently renumber coverage rows.

## Coverage schema and consumer compatibility

The compiler owns source declarations; the trainer never infers coverage from the absence of boxes. Represent these semantic states per image and canonical class:

| State | Meaning | Negative classification supervision | First adapter |
| --- | --- | --- | --- |
| `exhaustive` | Source asserts all instances of this class were labeled | Allowed | Supported |
| `verified_absent` | Source explicitly asserts there is no instance of this class | Allowed; boxes for this class are a contradiction | Supported |
| `positive_only` | Some positive instances are known; others may be missing | Suppressed except the full original expression at known matched positive cells | Selective supervision supported; general partial-instance accuracy not established |
| `unknown` | No reliable coverage assertion for this class | Suppressed except at any known matched positive cells | Supported |

Never silently promote a source with positive boxes but no exhaustiveness declaration to exhaustive. `positive_only` makes that situation explicit; if observations arrive under an `unknown` declaration, they remain valid positive observations without establishing completeness. Every observed positive used for training must retain its entire original matched-cell expression; coverage is permission for negative supervision, not permission to discard labels. The first accuracy experiment withholds entire classes per image. Supporting these state semantics does not establish that selective masking solves every missing-instance problem.

Coverage state names remain semantic in JSON; a documented compact `uint8[N,C]` encoding may be used for tensor loading. Derive a Boolean negative-supervision permission table from the validated states. Its artifact manifest must bind shape, dtype, class order, row order, ontology digest, and file checksum. Keep placeholder/background/padded logits separate from these `C` foreground coverage columns; the pinned criterion decides their treatment, and an explicit mapping test must cover them.

Coverage declarations include the policy ID, source revision, scope, author/origin, and evidence type. For controlled withholding, the evidence is the experiment controller plus original reference-label revision. For real user imports, a declaration is a user-supplied policy assertion; a compiler cannot certify that humans actually labeled every object.

Default coverage is `unknown`. A dataset's class list says which categories can appear in annotations; it does not establish that every image was exhaustively labeled for every listed category. Empty labels and a successful COCO import do not change that rule.

## Deterministic partial-label construction

Project the reference dataset onto the two target classes before generating experimental arms. Other piece categories are outside this task's ontology; excluding them is consistent in every arm and every evaluation split.

Freeze source assignment without inspecting model outputs:

1. Set `assignment_version = coverage-chess-v1` and `assignment_seed = 20260917`.
2. For each original training **label path**, compute `SHA256(UTF8('coverage-chess-v1' + NUL + '20260917' + NUL + path))`.
3. Sort by digest, breaking any digest tie by path. Alternately assign rows to source A (even ranks) and source B (odd ranks).
4. A covers black pawns only: retain every reference black-pawn box, withhold every white-pawn box, declare coverage `[exhaustive, unknown]`.
5. B covers white pawns only: retain every reference white-pawn box, withhold every black-pawn box, declare coverage `[unknown, exhaustive]`.
6. Keep all training images, including images with zero visible target boxes. Zero-visible-box batches are a required adapter test.

Verified outcome of this exact assignment at the pinned revision:

| Source | Images | Visible target boxes | Withheld target boxes | Images with both pawn classes in reference | Images with zero visible target boxes |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: black coverage | 101 | 243 black | 237 white | 49 | 48 |
| B: white coverage | 101 | 237 white | 253 black | 48 | 49 |
| Total | 202 | 480 | 490 | 97 | 97 |

This produces two disjoint image sets. Do not duplicate every image into both sources with complementary labels: those duplicates could let ordinary training indirectly recover all labels and would answer a different question. The controller may inspect reference annotations to construct the benchmark; training code may not read withheld labels.

Keep a deterministic `withholding_audit.jsonl` with original annotation identities, source assignment, and kept/withheld decisions in a separate evaluation/controller artifact. It is useful for proving what the benchmark does, but must not be present on the training data path. Public readers may inspect it after the experiment; it is not confidential data, just withheld experimental information.

## Matched arms and evaluation access

| Arm | Train image set | Visible training annotations | Coverage behavior |
| --- | --- | --- | --- |
| Naive merge | Same 202 | Same 480 partial annotations | Treat every target class as exhaustive, matching ordinary training semantics |
| Coverage-aware merge | Same 202 | Same 480 partial annotations | Respect A/B declarations for negative supervision |
| Full-label reference | Same 202 | All 970 target annotations | Exhaustive for both target classes |

The full-label arm is a reference, not a guaranteed upper bound: finite training and stochastic optimization can reverse individual scores. Keep initialization/checkpoint bytes, optimizer settings, augmentation, batch composition, iteration budget, evaluator, and checkpoint-selection rule matched. The full-label reference naturally uses more annotations and is not an annotation-cost-matched competitor.

Use validation only for a prespecified checkpoint-selection rule and any necessary debugging of ordinary training behavior. Evaluate the test set after freezing implementation and comparison settings. The trainer may use validation labels for standard validation; it must not see test annotations or train withholding records. Never choose the best seed or demo examples by test improvement. If multiple candidate settings are tried, disclose them and use one validation-based choice consistently.

First run: one fixed seed as a smoke experiment, labeled preliminary. Follow-up: at least three paired seeds and per-seed results. Report COCO AP@[.50:.95], AP50, per-class AP, recall, and false positives on the “neither pawn” images under a prespecified/validation-chosen score threshold. Report both-class co-occurrence results descriptively, with denominators. The test has only 29 images and 174 target boxes; those boxes are not 174 independent scenes. Image/group bootstrap intervals, if used, must state the resampling unit and cannot repair unknown scene dependence.

Also publish training curves, coverage-per-class counters, actual wall time, iteration count, device, peak memory, model/checkpoint digests, and the exact loss configuration. The mechanistic claim (removing invalid negative gradients) can be proven by tests even if the small benchmark shows no accuracy benefit. Do not turn an inconclusive metric into a positive result.

Pseudo-label completion, separate specialists, partial-label prior-art methods, controlled coverage rates, and independent real multi-source datasets are valuable follow-ups. They are not prerequisites for stating what this first matched experiment measured, but they are prerequisites for broad superiority claims. Report a failure honestly if selective negative masking increases false positives or underperforms naive training.

## Compiler architecture and artifact layout

Keep the first implementation a deterministic Python library/CLI with typed schemas and local content-addressed artifacts. Reuse existing image decoders, JSON/array libraries, COCO evaluation, and the pinned RF-DETR dataset loader. The product input is COCO plus explicit coverage declarations; a benchmark preparation script handles this dataset's simple YOLO-to-COCO conversion. Do not build a general format-conversion framework, database, distributed scheduler, or hosted annotation system to support 289 images.

Processing boundaries:

1. **Acquire:** download fixed source revisions into an immutable raw cache; checksum and inventory bytes.
2. **Import:** parse COCO or the small YOLO benchmark adapter into one canonical detection representation. Resolve paths relative to the source root and reject traversal/collisions. Verify dimensions from decoded images. Convert YOLO boxes to absolute COCO `[x,y,width,height]` without applying a second resize.
3. **Declare:** load the explicit ontology mapping and coverage-policy declarations. Preserve source image/annotation identities and original split memberships.
4. **Validate:** reject contradictory coverage, unmapped classes, invalid geometry, ambiguous duplicate policies, cross-split byte duplicates, unknown image references, and consumer-incompatible states. Warnings such as uncertain near-duplicates remain separately reviewable.
5. **Compile:** assign deterministic global image IDs and canonical category IDs; produce sorted COCO annotations, coverage sidecar, lineage, and compatibility manifest.
6. **Verify:** read back compiled bytes with the actual pinned consumer, assert class/image identity and coverage joins, and write a validation report.
7. **Consume:** the training adapter accepts an artifact only when its digests, schema, ontology, and supported-consumer contract match. It gathers coverage at the criterion boundary by preserved image ID.

Proposed artifact shape:

```text
bundle/<bundle-digest>/
  manifest.json                 # schema/ontology/source/consumer contract + checksums
  ontology.json                 # semantic column -> COCO ID -> model label mapping
  provenance.jsonl              # sources, image IDs, byte hashes, lineage, policy origin
  coverage.jsonl                # semantic declarations + evidence/policy references
  coverage.npy                  # exact [N,C] state matrix; checksum-bound, no pickle
  splits/train.coco.json
  splits/valid.coco.json
  splits/test.coco.json
  images/<sha256>.<extension>
  diagnostics.json
  ATTRIBUTION.md

benchmark-controller/<benchmark-digest>/
  full_reference_annotations.json
  withholding_audit.jsonl
  source_assignment.json
  experiment_protocol.json
```

Execution materializes a verified RF-DETR-compatible train/validation view (`train/_annotations.coco.json`, `valid/_annotations.coco.json`, and resolvable image paths), preserving parent image IDs and category mappings while keeping test/controller labels off the training worker. This consumer view is derived from the canonical layout above and has its own digest. Large images can use verified local links in a development cache; a portable exported bundle must resolve all paths and verify bytes without relying on those links. Avoid embedding machine-specific absolute paths in canonical manifests.

Canonical identity excludes incidental timestamps, download URLs with expiring query strings, local paths, and JSON whitespace. Include source byte hashes, ontology, sorted labels, split assignment, coverage declarations, compiler/schema versions, and transformation parameters. Define canonical serialization once; retain a separate human-friendly run timestamp. Never hash a manifest that recursively contains its own final digest.

For the first release, byte-identical images with identical observations and policies may coalesce within one split, with explicit provenance and a diagnostic. Differing observations or policies block compilation; byte duplicates across splits are always an error. Do not silently OR coverage masks and union annotations: overlapping boxes, contradictory labels, and different labeling policies make that unsafe. The first controlled benchmark has no exact byte duplicates, so this resolution policy does not affect its counts.

## Transform contract

Start with verified resize, suitable horizontal flip, and normalization using RF-DETR's existing transforms, with `image_id` preserved. Unknown coverage remains unknown. Disable crop-based box removal initially: although image-level coverage can survive a correctly implemented crop, filtering/minimum-area rules add semantics to validate separately. A crop that removes all visible boxes must never make an unknown class verified absent when later supported.

Disable multi-image composition transforms such as mosaic, MixUp, or CutMix for the first adapter. A single image ID no longer identifies a single coverage policy after such composition; supporting it requires explicit transformed coverage semantics, potentially region-level handling. Do not pretend that taking a union of class coverage makes the composed image exhaustively annotated.

Box clipping/removal rules, `iscrowd`, zero-area annotations, and class remapping must match the pinned consumer. The first compiler rejects crowd/ignore/segmentation-only cases it does not implement instead of silently converting them into ordinary boxes. State these limitations in the artifact's consumer contract.

## Required data tests and acceptance gates

These are planned verification cases, not completed implementation tests.

| Test | What a passing result proves |
| --- | --- |
| Tiny handcrafted image with black and white boxes | A/B compilation preserves the covered box, withholds the other, and attaches the correct policy |
| Exhaustive class with zero boxes | Its negatives remain supervised; an empty target is not automatically “unknown” |
| Unknown class, with and without observed positives | No general negative permission is inferred; any observed positive retains its entire matched-cell loss |
| `verified_absent` plus positive box | Contradiction rejected before training |
| `positive_only` input | Its known positives remain supervised while nonpositive cells lack negative permission |
| Class IDs 4/10 -> COCO 1/2 -> labels 0/1 | Actual pinned loader and evaluator preserve meaning across every split |
| Shuffled dataloader and repeated image sampling | Coverage joins by preserved global ID, independent of batch/row order |
| Image IDs near 0 and N-1, missing/out-of-range IDs | Both valid boundaries work; invalid references fail closed |
| Single-image resize/flip | IDs/coverage survive, geometry follows the existing transform contract |
| Crop-based box dropping or multi-image augmentation enabled | Consumer refuses unsupported semantics |
| Same semantic input in different import order | Canonical artifact digest and compiled contents remain identical |
| Sidecar removed, wrong shape/class order/hash | Coverage-aware training refuses to run rather than quietly becoming naive |
| Exact duplicate crosses splits | Compile fails with affected image identities |
| Test/controller annotations on train data path | Packaging/access audit catches an experiment setup error |
| Plain COCO/YOLO export without sidecar | Export explicitly reports lost coverage semantics; no “safe training” claim |

The architecture/loss plan must separately test zero forbidden classification gradients, preserved positives and valid negatives, every enabled auxiliary/encoder branch, and full-coverage equivalence with stock RF-DETR. A correct data compiler alone does not prove a correct loss patch.

## Demo and honest claims

Show one reference image containing both pawn colors, its source-specific visible labels, the explicit coverage row, and the compiled artifact identity. Let the viewer inspect why a missing white-pawn box is unknown in source A. Show the naive/aware/full-reference models on the same fixed evaluation images and publish aggregate metrics alongside them. Use a predefined montage or random seed for examples; label any hand-selected failure illustration as such.

Defensible first-release claim: “A coverage-preserving dataset compiler and RF-DETR integration, verified by gradient/identity tests and a controlled experiment on explicitly withheld class labels.” A positive benchmark result can add its exact measured delta and uncertainty. Do not claim invented partial-label learning, native hosted Roboflow support, guaranteed accuracy improvement, or a proven solution to arbitrary incomplete annotation.
