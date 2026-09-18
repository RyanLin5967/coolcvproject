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

Real CPU train/save/reload and 20-update Apple-MPS smoke checks passed. Those checks establish that the path executes and preserves its checkpoint, **not** that its accuracy is higher. A separate local MPS aware run is in progress without cloud charges. It is kept distinct from CUDA measurements; local matching controls are conditional on the predeclared local pilot result. No Large accuracy result is available in this snapshot.

The workbench now exposes local device availability and Nano/Large recipes. Eleven focused tests and desktop/mobile browser checks passed; browser job submissions were intercepted rather than launching extra training. A class-agnostic crop refiner is also being prototyped as a localization hypothesis, with no measured result yet. A requested 95-point target is an ambition, not a forecast: pawn AP50 already reaches 100, while the stricter AP50:95 is 78.82. Reporting the easier metric as though it met the strict target would be misleading.

## A separate dataset exposed a smaller effect

The two chess tasks share images and domain. Construction safety adds five industrial classes from 1,206 original images. Pinned source hashes, filename grouping, exact bytes and normal/mirrored perceptual hashes found two groups crossing the published splits; whole-group reassignment repairs that known leakage. Final splits are 995 train / 120 validation / 91 test. Synthetic annotation-source families expose 2,119 of 6,380 training boxes.

The first matched seed scores **46.34 naive, 48.14 aware, 52.76 complete**: a preliminary +1.80-point benefit, substantially smaller than the original chess gain. Only four of nine planned runs completed; naive has a second seed, aware and complete do not. The predeclared all-nine-model test evaluation has not run, and test labels remain unevaluated. Incomplete cohorts cannot support a three-seed generalization claim. Grouping is heuristic, annotation completeness is inherited, and public-data pretraining exposure is unknown.

## Durable work required process isolation and immutable deployments

Web requests do not own training processes. SQLite WAL, transactional job claims, a scheduler lock and separate worker process groups let training survive a webserver restart. Cancellation verifies process identity before signaling. Actual browser import, invalid-policy repair, training, restart survival and cancellation checks passed. Desktop/mobile research views across all three tasks also passed; the latest recorded full suite has 56 passing tests, before subsequent in-progress edits.

Modal call IDs are saved before collection so reconnecting retrieves existing work. Deploying by file path failed inside the image because the module import name differed; deploying with `modal deploy -m ...` fixed it. Each deployment retains a frozen source snapshot and hashed inputs. Local fixes do not retroactively change experiments already run. GPU account concurrency also caused queues: queued work must be distinguished from failed or stalled work.

## Billing discrepancy: what happened, what is known, what changed

The user requires **$0 out of pocket** and reported a configured Modal $0 spend limit. An earlier API check showed $19.72 metered and $0 billed. A later check returned **$32.68 metered / $2.68 billed after $30 credits**. We stopped all four owned apps and verified zero containers. Fifty-two of 64 planned cloud runs had completed; 12 were interrupted. The API's billed balance is not evidence that a card payment was actually collected.

The user subsequently reported **$12.91 credits remaining**, **$25.41 workspace usage** and **no visible card charge**, and explicitly authorized continuing with those credits. Workspace identity was checked, but the API/dashboard accounting conflict remains unresolved. No confirmed cash charge or guaranteed zero-cost outcome is asserted. The environment-budget API was unavailable on this plan; the reported spend-limit configuration/effect remains unverified.

The engineering mistake was relying too heavily on delayed usage while several jobs had already committed compute. Provider timeouts and zero idle containers bound individual jobs but do not reserve credit for concurrent work.

The restart is now restricted to **three Large/704px/EMA runs**, reserving **$3.30** under a **$3.50** cap before submission. A persistent local cloud policy blocks other apps; a locked reservation ledger counts commitments independently of delayed metering. Failed or interrupted attempts remain recorded. Other stopped cohorts are not automatically resumed. Local MPS work continues separately. These controls bound this client's new commitments; they do not reconcile provider accounting or guarantee an account-wide cash outcome.

## Maintaining this record

Update this file when a result changes the next experiment, a provider incompatibility is discovered, a run fails, or accounting is reconciled. Preserve negative findings, distinguish smoke checks from accuracy evidence, keep exact seed counts, and link the supporting receipt in the portable results snapshot. Report meaningful decisions and difficulties in chat while work continues; saving a checkpoint must not be treated as a request to stop.
