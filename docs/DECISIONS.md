# Decisions and difficulties

Updated September 17, 2026. This is a continuing engineering record: decisions, observed failures, resulting changes and unresolved questions. A saved checkpoint means durable work and a status report; it does not end autonomous execution. [RESULTS.json](RESULTS.json) captures the current measurements. Full receipts under ignored `artifacts/` remain local.

## Coverage is an explicit training contract

**Problem:** merging datasets that annotate different classes makes unlabeled objects look like negatives. A class list or a missing box does not prove exhaustive annotation.

**Decision:** preserve `unknown`, `positive_only`, `exhaustive` and `verified_absent` for every image/class. Only exhaustive or verified-absent declarations permit negative classification supervision; observed positives remain supervised. The adapter preserves upstream RF-DETR matching, reserved output, normalization and auxiliary/encoder dispatch. Fully exhaustive loss and gradient parity are tested. Coverage never suppresses evaluation false positives.

**Limit:** the compiler can detect contradictory declarations and corrupt artifacts, but cannot prove annotator completeness. Claims about exhaustive coverage require source evidence. This project contributes a reproducible engineering workflow around established partial-label ideas, not a claim to invent selective supervision.

## Training inputs must not contain the hidden answer

The compiler publishes hashed bundles, then separate observed learner views. Test data and deliberately withheld training labels are excluded from those views. Source mappings, image identities and split membership remain auditable. A cloud experiment archive contains a separate evaluation reference because its worker also evaluates the completed model; the selected training view remains explicit. Teacher mining only reads the partial training view.

The complete-label arm is a reference with more labels, not a guaranteed numerical upper bound. All comparisons bind the checkpoint, ontology, seed, initialization and update budget. Refactoring the scorer reproduced the complete saved metric dictionary for a checked reference run. New metrics must not silently change the established resize/postprocessing protocol.

## Roboflow needed semantic verification, not just a successful upload

Three concrete integration failures shaped the connector:

- **Coordinate origin:** the first dataset round-trip exposed a one-pixel VOC mismatch. Explicit one-based export fixed it. Observed version 2 preserves 201 train images / 451 boxes and 58 validation images / 241 boxes, with round-trip coordinate error below 0.0001px.
- **Classifier row order:** the first hosted custom model mislabeled classes because the observed importer expected its reserved row first. A separate export reorders every main and encoder classifier, preserving the native checkpoint. Version 3 passed a three-image class/geometry smoke test: 28 confident detections and minimum matched same-class IoU 0.936. The earlier failed semantic check remains recorded.
- **Transfer reliability:** repeated local uploads of the large checkpoint broke. A hash-checked cloud-to-cloud transfer completed the deployment.

A separate 20-epoch Roboflow-hosted RF-DETR Nano baseline also finished. Its recipe, label access and provider AP50 differ from our matched AP50:95 experiments. It used the original chess test split, so that test is not globally untouched. Ordinary hosted training does not consume our custom criterion; the coverage-aware stage runs in CoverageCV, while Roboflow versions data, trains the separate baseline and serves the exported custom model.

A later portability check found a distinct native-export issue: the public `RFDETR.from_checkpoint(...)` loader needed a top-level upstream model identifier and faithful architecture metadata. The export now provides them. Nano/512px and Large/704px checkpoints loaded through that public API with exactly identical parameters, without retraining or modifying the original source files. Two focused regression tests cover the public loader. Successful loading through our internal adapter had not been sufficient evidence of public-SDK compatibility.

The actual trained Large aware checkpoint subsequently passed the same public-SDK tensor/class-name check. Independent local rescoring of all three completed Large arms also matched their full saved metric dictionaries exactly. Neither check adds another training seed, but both address whether the reported model and scores can be reproduced outside the original cloud call.

## Bigger changes need stronger controls

The first three-seed pawn study gained **12.56 AP points** from coverage (63.79 naive → 76.35 aware). On all 13 chess classes it gained **11.42 points** (58.23 → 69.64). These demonstrate the missing-negative-supervision mechanism on one small domain.

The next change combined 512px inputs, geometric augmentation, a restarted optimizer and 2,000 additional updates. It raised aware results to **78.82 ± 0.42** on pawns and **72.91 ± 0.11** on all chess classes. The same stronger recipe also raised naive results to 76.25 and 69.03; the matched coverage benefits are therefore **+2.57** and **+3.88 points**, not the larger original gaps. Augmented complete-label scores are 78.94 on pawns (n=3) and 74.16 on all chess classes (n=2; third run interrupted). Report the actual counts rather than pretending every row is a finished three-seed study.

## More training, pseudo labels and ensembles did not reliably win

Longer 384px training averaged 75.36 on pawns, below the original 76.35; all-class performance barely changed (69.72 versus 69.64). Extending a weak recipe was not an effective accuracy strategy.

Teacher labels require original/flip agreement, coverage eligibility and human-box precedence. Even then, teacher-plus-augmentation averaged **77.92 on pawns** and **72.75 on all classes**, below augmentation alone. Plausible boxes can inject localization errors. Follow-on code preserves pseudo-label identity through real crop/flip filtering and separates classification evidence from box regression. The completed 0.1-weight pawn pilot reached 78.70, below its matched 78.90 augmentation control. The other cautious/classification-only pilots were interrupted. None is promoted as a demonstrated improvement.

A fixed, equal-weight three-seed weighted-box-fusion protocol used saved predictions, with IoU 0.55 and minimum input score 0.001; it used no ground truth during fusion. Aware AP was **78.69 on pawns** and **72.48 on all classes**, below their single-model seed means. This adds three inference passes and did not justify replacing the single model. An ensemble is one prediction system, not another three-seed mean; its result is reported separately.

## Localization motivated the Large/704px/EMA experiment

At the fixed score threshold, the first-seed coverage-aware pawn model finds 237 of 241 validation objects. Its AP50 is 98.93, but AP90 is 16.39; the complete-label reference also has weak AP90 (20.33). Missing detections no longer explain most remaining strict-IoU AP loss.

We therefore implemented a materially different recipe: official RF-DETR Large 2026 weights, 704px inputs, four decoder layers and upstream EMA. The one-seed cloud pilot uses matched naive/aware/complete arms, 2,000 updates each. This is a combined recipe comparison, not an isolated estimate of model size or EMA. Nano augmentation results used 4,000 total updates, so that cross-recipe comparison has a different training budget.

All three restarted cloud arms completed: **72.78 naive, 73.68 aware and 74.97 complete AP50:95**. Coverage adds **0.91 points** within the matched Large recipe. The aware score improves by **0.69 points** over the same-seed Nano/512px model (73.00); the corresponding Nano naive/complete scores are 69.21/74.06. This was a substantial implementation change but a modest measured accuracy gain, not a breakthrough. One seed and unequal cross-recipe update budgets limit the inference; larger capacity alone did not solve strict localization.

Real CPU train/save/reload and 20-update Apple-MPS smoke checks also passed. Separate local MPS execution remains distinct from CUDA measurements. The workbench exposes local device availability and Nano/Large recipes; focused tests and desktop/mobile browser checks passed with intercepted submissions, rather than launching extra training through the browser.

A requested 95-point target is an ambition, not a forecast: pawn AP50 already reaches 100, while the stricter AP50:95 is 78.82. Reporting the easier metric as though it met the strict target would be misleading.

## A direct box-refinement pilot follows the modest capacity gain

The next experiment targets coordinates directly with one class-agnostic crop refiner. It trains on the all-class learner view's **994 observed human boxes** from 201 images, with no hidden reference boxes or pseudo labels. An ImageNet ResNet18 encoder sees 224px crops with 1.5× context, predicts coordinate corrections and keeps BatchNorm frozen. The crop's integer bounds/padding and inverse transform are explicit and tested.

The predeclared final model uses **1,000 updates, batch 32**, center/scale proposal jitter, horizontal flips and **25% identity proposals**. Identity examples teach it to leave already-correct boxes alone. The objective is `5 × SmoothL1(beta=1/9) + 2 × GIoU`; no validation checkpoint selection is permitted. This exact recipe is frozen before cloud execution.

Apply the **same partial-trained refiner** to all three seed-20260917 Nano/512px arms, preserving their classes and confidence scores. Only eligible positive-area proposals (confidence ≥0.05, top 100 eligible boxes per image) are corrected. The complete-label detector does not receive a separately complete-trained refiner. The primary question is whether aware AP improves against its original predictions under the unchanged evaluator. If it does not, do not promote the added inference stage.

A real two-update CPU train/save/reload/inference smoke check passed, including preserved classes and scores. The first cloud attempt then completed 1,000 training updates but failed during postprocessing: RF-DETR emitted finite zero-area boxes that our refiner rejected. Because training and evaluation shared one temporary cloud workspace and the function failed before returning, that trained checkpoint was lost. The original attempt and failure remain recorded.

The fix preserves zero-area boxes unchanged in evaluation and excludes them before selecting the top 100 correction candidates. Twenty-three focused checks passed. More importantly, the workflow now **collects and verifies the trained checkpoint first**, writes its durable receipt, and evaluates controls separately. Evaluation can restart from the same checkpoint instead of repeating paid training.

The retry checkpoint was safely collected before a second incompatibility appeared: Apple MPS does not support this refiner's 7×7-to-4×4 adaptive pooling. This time the checkpoint survived. Local evaluation now defaults to CPU with the same architecture and weights, without another cloud call. Device, inference seconds, evaluated images and changed-box counts accompany the result so extra inference work remains visible. Both cloud attempts count against the conservative reservation: $0.85 each, $1.70 total.

The completed comparison is a negative result for the primary hypothesis:

| Seed-20260917 Nano/512px detector | Original AP50:95 | With shared refiner | Change | CPU refinement seconds / 58 images |
| --- | ---: | ---: | ---: | ---: |
| Naive | 69.2098 | 69.3292 | +0.1194 | 42.73 |
| Coverage-aware | 72.9954 | 72.5534 | **−0.4420** | 23.19 |
| Complete labels | 74.0649 | 73.2851 | −0.7798 | 30.86 |

The added stage is **not promoted**: it failed the predeclared aware-AP criterion despite a small naive gain. Timing includes checkpoint loading and crop preparation, but excludes detector inference and COCO scoring. The aware pass changed 1,026 boxes while keeping classes/scores unchanged. Retry training took 98.45 seconds for all 1,000 updates; the collected checkpoint is `d201b55a8a1f498d8afcfed9bd0348c9ef06680c21600a66570ea5824ceff364`. Synthetic proposal jitter did not establish useful correction behavior on real detector errors; that distribution mismatch remains a possible explanation, not a demonstrated cause.

## A separate dataset exposed a smaller effect

The two chess tasks share images and domain. Construction safety adds five industrial classes from 1,206 original images. Pinned source hashes, filename grouping, exact bytes and normal/mirrored perceptual hashes found two groups crossing the published splits; whole-group reassignment repairs that known leakage. Final splits are 995 train / 120 validation / 91 test. Synthetic annotation-source families expose 2,119 of 6,380 training boxes.

The first matched seed scores **46.34 naive, 48.14 aware, 52.76 complete**: a preliminary +1.80-point benefit, substantially smaller than the original chess gain. Only four of nine planned runs completed; naive has a second seed, aware and complete do not. The predeclared all-nine-model test evaluation has not run, and test labels remain unevaluated. Incomplete cohorts cannot support a three-seed generalization claim. Grouping is heuristic, annotation completeness is inherited, and public-data pretraining exposure is unknown.

## Durable work required process isolation and immutable deployments

Web requests do not own training processes. SQLite WAL, transactional job claims, a scheduler lock and separate worker process groups let training survive a webserver restart. Cancellation verifies process identity before signaling. Actual browser import, invalid-policy repair, training, restart survival and cancellation checks passed. Desktop/mobile research views across all three tasks also passed. The current full suite has **154 passing tests**. The completed acquisition view also passed actual desktop/mobile browser checks; independent rescoring matched all four primary metric dictionaries exactly. The construction panel was checked against real server results without mocked metrics. The latest post-training provider check reports zero containers across all project apps; remaining inference analysis is local.

Modal call IDs are saved before collection so reconnecting retrieves existing work. Deploying by file path failed inside the image because the module import name differed; deploying with `modal deploy -m ...` fixed it. Each deployment retains a frozen source snapshot and hashed inputs. Local fixes do not retroactively change experiments already run. GPU account concurrency also caused queues: queued work must be distinguished from failed or stalled work.

## Billing discrepancy: what happened, what is known, what changed

The user requires **$0 out of pocket** and reported a configured Modal $0 spend limit. An earlier API check showed $19.72 metered and $0 billed. A later check returned **$32.68 metered / $2.68 billed after $30 credits**. We stopped all four owned apps and verified zero containers. Fifty-two of 64 planned cloud runs had completed; 12 were interrupted. The API's billed balance is not evidence that a card payment was actually collected.

The user subsequently reported **$12.91 credits remaining**, **$25.41 workspace usage** and **no visible card charge**, and explicitly authorized continuing with those credits. Workspace identity was checked, but the API/dashboard accounting conflict remains unresolved. No confirmed cash charge or guaranteed zero-cost outcome is asserted. The environment-budget API was unavailable on this plan; the reported spend-limit configuration/effect remains unverified.

The engineering mistake was relying too heavily on delayed usage while several jobs had already committed compute. Provider timeouts and zero idle containers bound individual jobs but do not reserve credit for concurrent work.

The first bounded restart reserved **$3.30** for **three Large/704px/EMA runs** under a $3.50 cap. All three completed, bringing the original study to **55 of 64 runs**; nine remain interrupted. The refinement pilot required **two $0.85 attempts**, four construction crop continuations reserve **$4.40**, and three newly authorized acquisition calls reserve **$3.30**. The maximum post-restart reservation is now **$12.70** against the user's reported $12.91 remaining credits. Failed work remains counted; local inference studies add no cloud calls. These are commitment ceilings, not actual metered costs or a reconciled balance.

A persistent local cloud policy limits allowed apps; locked reservation ledgers count commitments independently of delayed metering. Failed or interrupted attempts remain recorded, and stopped cohorts are not automatically resumed. Local work remains separate. These controls bound this client's new commitments; they do not reconcile provider accounting or guarantee an account-wide cash outcome. Implementation checkpoints `a0131e2` and `2c32f1c` preserve earlier work while iteration continues.

## Maintaining this record

Update this file when a result changes the next experiment, a provider incompatibility is discovered, a run fails, or accounting is reconciled. Preserve negative findings, distinguish smoke checks from accuracy evidence, keep exact seed counts, and link the supporting receipt in the portable results snapshot. Report meaningful decisions and difficulties in chat while work continues; saving a checkpoint must not be treated as a request to stop.


## Construction: change object scale and sampling together

After the crop refiner regressed, an independent audit found that `no-helmet` has only 33 observed training boxes in 19 images. Its validation objects are about 22×26 pixels at the existing model input size; this class contributes 64.37% of the aware-to-complete AP gap. Training loss improves while validation geometry plateaus. That evidence motivates targeted enlargement and class-balanced sampling, rather than another undirected extension.

A new four-case pilot continues seed-20260917 Nano512 detectors for 2,000 fixed updates: ordinary aware, crop-aware, crop-naive and crop-complete. All crop arms share an immutable plan derived exclusively from partial human training labels: 4,000 full-frame samples plus 4,000 class-balanced anchor crops. The plan gives no-helmet 775 anchor samples (repeated observations, **not new labels**), with a median longest side near 96 pixels in those crops. Extra complete labels affect supervision only, never crop selection. The primary comparison is crop-aware versus ordinary aware at the same starting weights and added update budget.

Upstream JPEG draft decoding had to be disabled before applying rectangles defined in original pixels. We reuse upstream crop/flip/resize to transform every observed target, preserve image identity for coverage lookup, and replay a sequential plan with no dependence on model RNG. Seven focused checks and a real two-update training smoke passed before execution. Checkpoints were collected before scoring, following the refiner failure lesson.

All four fixed final checkpoints now have full-frame validation results:

| Continuation, seed 20260917 | AP50:95 after 6,000 total updates |
| --- | ---: |
| Aware · ordinary augmentation | 48.0026 |
| Aware · observed-object crops | 47.2780 |
| Naive · identical crop plan | 43.1655 |
| Complete labels · identical partial-derived crop plan | 52.4492 |

The primary crop effect is **−0.7246 AP points** against the matched aware continuation. This tested a substantial sampling/scale change, but it did not improve the predeclared metric and is not promoted. It does not establish that balancing or larger crops independently fail, because the intervention changed them together and the ordinary control retained stock augmentation. These one-seed continuations are not pooled with the original construction seeds. Their conservative commitment is $4.40, included in the post-restart reservation ledger.

## Slicing improves small-class inference, with controls and added cost

We separately tested one fixed full-frame-plus-four-tiles inference policy on all three original construction checkpoints. Tiles are 384px with stride 256, internal-edge rejection is two pixels, and class-aware NMS uses IoU 0.5. It processes all 120 validation images and all five classes. Raw full-frame and full-frame NMS-only controls prevent attributing suppression changes to slicing.

| Original 4,000-update checkpoint | Fresh CPU full frame | Full frame + NMS | Full frame + four tiles | Training-size class routing |
| --- | ---: | ---: | ---: | ---: |
| Naive | 46.3351 | 45.6044 | 45.2922 | 46.8873 |
| Aware | 48.1386 | 47.4357 | 48.9618 | **49.8418** |
| Complete labels | 52.7535 | 52.0037 | 52.3549 | 53.9942 |

Full tiling helps the aware detector but harms some larger-object classes and both other aggregate controls. A secondary rule uses the tiled output only for classes whose median human-observed training-box longest side at 512px is ≤96px. This selects `helmet` and `no-helmet`; all other classes retain raw full-frame output. Its threshold matches the crop-plan scale target and was not swept, but the decision to add routing followed validation diagnostics, so the result is explicitly **validation-guided exploration**. The aware gain is **+1.7032 points** over its fresh CPU full-frame baseline. It costs five image passes; cached routing adds no additional model passes. The full-label control also benefits, so the entire gain cannot be attributed to coverage.

CPU/MPS checks exposed nonmatching raw outputs and prediction counts despite several close matched boxes. We use CPU throughout these diagnostic comparisons and make no exact backend-parity claim. Fresh CPU scores differ slightly from historical scores; both remain recorded. This is distinct from the earlier exact metric rescoring audit, which reused saved predictions rather than rerunning model inference.

The fixed four-way crop-training/tiled-inference interaction was declared before primary continuation scores, with all four cases and unchanged slicing settings. All four cases have completed:

| 6,000-update continuation | Fresh CPU full frame | Full frame + tiles | Training-size class routing |
| --- | ---: | ---: | ---: |
| Aware · ordinary augmentation | 48.0026 | 47.8177 | **49.8821** |
| Aware · observed-object crops | 47.2780 | 45.8360 | 48.0587 |
| Naive · observed-object crops | 43.1655 | 42.3117 | 43.8702 |
| Complete · observed-object crops | 52.4492 | 52.3630 | 53.8681 |

Full tiling trails every continuation's own full-frame score. The training-size rule improves all four full-frame aggregates, but crop-aware still trails ordinary-aware by 1.8234 points with the same routing. Thus slicing does not rescue the negative primary crop-training result. The best 49.8821 is only **0.0403 points** above the original aware checkpoint's 49.8418 under the same routing. It is a one-seed, validation-guided, five-pass inference result, not a new multi-seed training benchmark or a substantial extra-training gain.

## Annotation inspection found concrete issues, not an explanation for every error

[The completed annotation audit](CONSTRUCTION_LABEL_AUDIT.md) inspected all **33 observed no-helmet training boxes in 19 images**, all **11 validation no-helmet references in six images**, 12 selected localization residuals, nine fixed-threshold no-helmet false positives and seven full-frame overlays. It verified the partial learner manifest and read no hidden training labels or test data. The residual selection deliberately focuses on failures and does not estimate annotation-error prevalence.

No-helmet validation boxes are smaller than the observed training examples (median 22.0×26.4 versus 36.4×52.8 pixels at 512px), and one image contains six of eleven validation references. Head boxes vary between upper cranium and whole head/face. Specific defects include a full-person region labeled `helmet` and two differently sized helmet references around one visible helmet. Three of nine no-helmet false positives occur on unannotated background faces; whether a source policy intended to exclude them is unknown. There are also real detector errors, including hardhat-like false positives, duplicates and overextended boxes.

These findings justify review of the data policy; they do **not** show that annotation defects explain the full accuracy gap, that correction would achieve 95 AP, or that all remaining errors are label noise. The audit changed no annotations, classes, splits or reported scores. The user subsequently approved a separate acquisition simulation, described below; the original benchmark remains preserved. No new human adjudication is claimed.

## Acquisition tests whether review selection yields useful new supervision

Repeating 33 no-helmet examples did not supply new appearance diversity. The implemented next experiment spends an explicit **150 image/class review units per arm**, with 30 reviews per class, to compare maximum-confidence guided selection against seeded random selection. Both plans were frozen before reading any published reference training labels. Selection only sees the existing partial training view, coverage and detector predictions; confidence ranks queries and does not itself create annotations or establish absence.

Applying the frozen requests to the published training oracle adds **265 guided boxes** versus **198 random boxes**. Guided no-helmet support grows **33 → 84**; random grows **33 → 38**. The total observed labels become 2,384 versus 2,317. Guided queries touch 141 unique images and random queries 143, illustrating why 150 image/class reviews must not be described as 150 distinct images. These are newly revealed published annotations, not new human work, and their correctness inherits source limitations.

The compiler preserves observed labels and records every queried pair, coverage transition, acquired annotation and source digest in a new immutable view. Unrequested reference labels do not enter training. The original learner and its benchmark scores remain unchanged; validation is copied unchanged and test labels remain unused. Equal review budgets intentionally produce different acquired box counts, so this is **not** a same-label-budget comparison or a new isolated loss-function claim.

Reviews also change negative supervision: guided acquisition records 140 exhaustive and 10 verified-absent pairs, while random records 86 exhaustive and 64 verified-absent pairs. These states describe the published-oracle simulation, not independently established annotation truth. Any downstream effect therefore combines newly revealed positives with newly trusted negatives. Manifest comparisons confirm all 121 validation payloads remain byte-identical in both acquired views.

Guided and random arms start from the identical original 4,000-update aware checkpoint and add **2,000 fixed updates**. A complete-standard arm starts from its matching complete checkpoint; all use Nano/512px, batch 4, the same continuation recipe and final-step evaluation. The existing 6,000-update ordinary-aware continuation is reused as the **zero-review control**, avoiding duplicate training. The primary comparison is guided minus random AP50:95 on the unchanged full-frame validation set; all four controls will be reported together. One seed and validation-guided design remain explicit limitations.

Twenty-one focused checks and a real two-update CPU smoke passed, followed by a 154-test full suite. Three cloud calls reserve $3.30, bringing the conservative post-restart total to $12.70; provider accounting is still unresolved. All three checkpoints were collected and verified before local scoring.

The unchanged full-frame primary comparison completed:

| Arm, seed 20260917 · 6,000 updates | Review units | Observed training boxes | AP50:95 | No-helmet AP |
| --- | ---: | ---: | ---: | ---: |
| Zero review · reused ordinary-aware control | 0 | 2,119 | 48.0026 | 20.0247 |
| Seeded random acquisition | 150 | 2,317 | 47.3432 | 15.1395 |
| Maximum-confidence guided acquisition | 150 | 2,384 | **49.8486** | 26.6056 |
| Complete-label standard continuation | — | 6,380 | 52.4241 | 34.1330 |

The primary guided-minus-random effect is **+2.5054 AP points**; guided gains **+1.8460** over zero review. Random loses **0.6594** versus zero review, so revealing more published labels did not automatically help. This supports the guided acquisition strategy in this one controlled seed; it does not establish a universal annotation-efficiency gain, independent human review quality or equal-box-budget superiority. Review units are not actual human time or cost: drawing 265 boxes may take more effort than drawing 198. Added positives and changed negative coverage both contribute to the intervention.

A bounded local interaction applied the exact existing CPU tiling and original training-size class-routing rule to all three new checkpoints. Its declaration preceded secondary scores, with no parameter search or further GPU training. All cases completed:

| Acquisition arm | Single-pass primary AP50:95 | Original five-pass class-routing AP50:95 |
| --- | ---: | ---: |
| Zero review | 48.0026 | 49.8821 |
| Random | 47.3432 | 49.3254 |
| Guided | **49.8486** | **51.6204** |
| Complete-standard | 52.4241 | 53.7707 |

The best guided configuration has AP50 **92.2047** and strict AP50:95 **51.6204**. Its routed gain is **+1.7384 points** over the matched zero-review routed control and **+2.2951** over random with the same routing. The recorded guided CPU run took **39.45 seconds for 600 image passes**, versus **8.02 seconds for 120 full-frame passes** across the same 120 images, approximately 4.92× the measured time. This is a secondary one-seed interaction between added published training labels and five-pass inference; it does not replace the single-pass acquisition primary result or establish human-cost efficiency.

Final integrity checks confirmed all 120 validation images, all 717 reference boxes, all five classes, exact class-wise selection from raw/tiled outputs and per-class AP consistency. All newly authorized experiments and audits are complete. The original benchmark remains 55/64 collected with nine historical interruptions; the acquisition/crop/refinement studies stay separate. No further experiment is pending in this delivery.

[The acquired-label audit](ACQUISITION_LABEL_AUDIT.md) inspected all 51 guided and five random new no-helmet entries after selection, without reading model outcomes or changing the running study. It found no obvious hardhat-versus-no-helmet or whole-body labeling error in that subset, while noting variable upper-head geometry. Guided supplies 11 boxes with both dimensions ≤32px at model scale; random supplies none. But nine guided boxes come from five frames of one rooftop camera sequence, and all five random head boxes are already contained in guided. The 21 guided image hashes therefore do not establish 21 independent scenes. These are useful checks on acquired supervision, not new human adjudication or evidence that the trained model improved.

## 2026-09-17 — Present the merge problem, not a generic training dashboard

The first UI exposed almost every training recipe and made CoverageCV resemble a general model-training product. The default path now explains the actual contribution: source coverage declarations prevent missing annotations from becoming unjustified negative supervision. A recorded construction image illustrates two explicitly synthetic source policies; the separate predictions page shows actual checkpoint outputs. Training, import, provider receipts and policy editing remain in an expandable local workbench.

Benchmarks show one matched three-arm recipe per dataset. A client-side cohort check intersects seed IDs and rejects mismatched initial parameters, update counts, batches, resolution or device. This matters for construction: ordinary training has two historical seeds but the aware/reference arms have one, so the displayed control is the matching 46.34 run, not the unmatched 45.85 aggregate. Full labels are a controlled reference, never described as a frontier model. Single-run prediction scores are labeled separately from benchmark averages.

The flicker had two causes. A fresh SSE connection replayed historical events, and full-page refreshes replaced canvases even when completed results were unchanged. New streams sync to the current event cursor; reconnects preserve Last-Event-ID recovery. Completed prediction DOM is retained, image and prediction requests are cached, async drawing rejects stale responses, and complete frames are copied from an offscreen canvas. Actual browser tests dispatch refresh events and verify both canvas identity and pixels remain unchanged.

The public site is a committed static export: 32 read routes, three verified comparisons, six evenly spaced validation images per task (12 distinct image files), and recorded predictions. Image choice is positional, not score-based. Nine checkpoint/data/evaluation contracts are checked before export; credentials, machine paths, checkpoints and cloud billing records are excluded. It needs no training server or public write API. The real workbench and static presentation share the same frontend. A captioned, silent walkthrough records the actual public interface.

Validation: 162 Python tests passed, Ruff and JavaScript syntax checks passed, and read-only desktop/mobile checks passed for both local and standalone static sites. The UI check verifies displayed metric values, distinct dataset names, no layout overflow or browser errors, no mutations, no static-site API dependency, stable canvases during refresh, and rapid image/confidence changes. Presentation changes do not alter any training or evaluation result.
