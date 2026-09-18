# Construction annotation and error audit

The clearest next intervention is **targeted training-data review and acquisition with an explicit head-box policy**, plus a review queue that makes annotation defects visible. Larger models, more epochs and global cropping have not addressed this bottleneck. This audit found real reference defects, but they do **not** establish that annotation quality explains the entire accuracy gap or that correcting them would produce 95 AP.

## Scope and reproducibility

Inspected every **33 observed human no-helmet training boxes across 19 images**, every **11 validation no-helmet boxes across six images**, 12 deliberately selected high-confidence localization residuals, all nine no-helmet false positives at confidence 0.25 / IoU 0.50, and seven full-frame reference overlays. Selection of localization residuals is diagnostic and explicitly biased toward failures; it is not a prevalence estimate.

Only the immutable **partial learner training view** and its complete-reference validation annotations were read. No hidden training annotations, test annotations, test images, training runs or cloud services were accessed. No labels or splits were changed. The annotation files and inspected images were checked against the learner-view SHA-256 manifest.

- View: `d388715244900aee03fe206c0259a759d71b74eeea1e595c45691ae3c87813a1`.
- Baseline: original construction aware seed 20260917, Nano 512, 4,000 updates; fixed full-frame evaluation. Tiled predictions are a separate existing diagnostic.
- [Reproducible evidence and exact source hashes](../artifacts/construction-label-audit/evidence.json), [crop generator](../artifacts/construction-label-audit/audit.py), [full-frame and nesting audit](../artifacts/construction-label-audit/details.py).
- Crops use nearest-neighbor enlargement. Green marks reference boxes; magenta marks predictions. Enlarged pixels do not add visual information. Predictions matched by maximum same-class IoU are a GT-conditioned diagnostic, never an inference procedure or substitute metric.

## What is directly supported

| Finding | Evidence | Implication |
|---|---|---|
| No-helmet supervision is sparse | 33 boxes / 19 of 995 training images, versus helmet 689 / 313 images, person 749 / 325 | Replaying 33 examples adds exposure but no new appearance or annotation-policy diversity. |
| Validation heads are smaller | Median no-helmet dimensions at 512: train 36.4 × 52.8 pixels; validation 22.0 × 26.4. Both dimensions ≤32: 6/33 train versus 8/11 validation | There is a measured scale shift; this does not alone prove the reason for all errors. |
| One scene dominates the rare-class result | Image 1108 contributes 6/11 no-helmet references | More than half that class's recall depends on one scene. No-helmet still contributes 20% of five-class mAP despite only 1.53% of validation boxes. |
| The heads are largely found already | Each of 11 no-helmet GT boxes has a same-class baseline prediction with IoU >0.50 at score ≥0.05. At score ≥0.25, greedy IoU 0.50 matching gives 9 TP, 9 FP, 2 FN | Strict localization, confidence and false positives matter more here than simply finding previously unseen head locations. The maximum-IoU diagnostic permits multiple candidate predictions and is not ordinary recall. |
| Head geometry varies | Train ann1409 and ann1855 mostly enclose hair/forehead, while ann1808 encloses the whole head/face. Validation ann2168 encloses hair/forehead; model extends into the face | A globally applied box-height change cannot consistently match all these references. All 44 no-helmet crops concern head regions; there is no wholesale head-versus-full-body no-helmet class mix. |

[All train crops, page 1](../artifacts/construction-label-audit/train-nohelmet-1.png), [page 2](../artifacts/construction-label-audit/train-nohelmet-2.png), [page 3](../artifacts/construction-label-audit/train-nohelmet-3.png), [all validation crops](../artifacts/construction-label-audit/valid-nohelmet-1.png), [localization residuals](../artifacts/construction-label-audit/valid-localization-residuals-1.png).

## Specific annotation defects and ambiguities

These are inspectable cases, not a license to rewrite evaluation labels to suit predictions.

1. **Two helmet references on one visible helmet:** validation image 1024, annotations 2275 and 2276. The tight red-helmet box is wholly inside a second `helmet` box covering the helmet, full head and neck. Their IoU is **0.451589** and area ratio **2.2144**. The baseline tight helmet prediction has IoU 0.448 with the larger reference. [Full image](../artifacts/construction-label-audit/full-validation-1024.png).
2. **Full-person box labeled helmet:** image 998, ann2132 encloses the standing person's body and is labeled `helmet`; ann2134 tightly encloses that same person's blue headwear, also `helmet`. Their area ratio is **10.2174** and mutual IoU **0.097872**. [Full image](../artifacts/construction-label-audit/full-validation-998.png). This is strong visual evidence of a category/geometry annotation defect.
3. **Incomplete composite-scene annotation:** image 1002 contains five visible people, while references cover only the foreground man and the hardhat wearer behind him. The three other visible background faces have no person or no-helmet references. Predictions on those three faces account for **three of nine no-helmet false positives** at the fixed threshold. One is visibly bareheaded; the others wear a straw-like hat and a chef-style hat. The absent references are a fact; whether a documented policy intentionally excludes these background people is unknown. The current COCO labels contain no ignore regions for them. [Full image](../artifacts/construction-label-audit/full-validation-1002.png), [all nine false positives](../artifacts/construction-label-audit/valid-nohelmet-false-positives.png).
4. **Geometric and headwear ambiguities in the dominant scene:** image 1108's person ann2801 is visibly offset left relative to its apparent subject; the best high-confidence same-class prediction overlaps at only 0.285. Six heads are `no-helmet`, including baseball-cap-like headwear. Two other soft-looking head coverings are labeled `helmet`. The headwear material cannot be established confidently from this image; this warrants a policy review rather than an automatic relabel. [Full image](../artifacts/construction-label-audit/full-validation-1108.png).
5. **Additional review candidates:** deterministic same-class nesting flags five validation pairs, including tiny `person` boxes inside larger person boxes in image 1025 and two person references around the bent worker in image 1013. These are candidates, not five independently proven errors. [Candidate coordinates](../artifacts/construction-label-audit/nested-same-class-candidates.json), [image 1025](../artifacts/construction-label-audit/full-validation-1025.png), [image 1013](../artifacts/construction-label-audit/full-validation-1013.png).

There are also genuine model errors: no-helmet false positives on visibly hardhat-like headwear, duplicate predictions, overextended boxes and poor small-object localization. The audited defects do not erase those failures. Two clear helmet reference defects out of 233 helmet annotations cannot by themselves explain a roughly 50-point overall AP shortfall.

## Corrective action with the strongest evidence

Implement an **annotation acquisition and adjudication workflow** as the next substantial product feature:

1. Freeze the original benchmark and retain every existing score. Add reviewable candidates, source-image crops, current coverage state and full provenance; do not silently change validation labels or drop difficult classes. Structural nesting, scale anomalies and model disagreement should rank review items, not automatically label them as wrong.
2. Define the annotation policy explicitly: no-helmet means head region with no certified hardhat; decide whether the box encloses the entire visible head or the upper cranium, how caps/hoods are handled, how background people are treated, and when occlusion/blur is unjudgeable. Existing snapshots lack a consistently evident geometric policy.
3. Acquire **new verified training supervision** for small no-helmet examples, prioritizing additional scenes over repeated crops of the same 19 images. Selection must use training images/coverage and model uncertainty only. Compare a fixed annotation budget against random selection; use newly adjudicated labels or explicitly revealed training annotations with logged acquisition cost. This tests the compiler's actual value: more accuracy per verified annotation, without hidden labels leaking into ordinary training.
4. If producing a clean benchmark, have an independent reviewer adjudicate it under that policy and version it as a separate dataset **before** evaluating a new candidate. Publish both original-reference and revised-reference results. Never advertise a score increase caused by reference revision as a model-only gain. Do not use the inspected validation cases as additional training data.

This is a more defensible path toward a compelling Roboflow project than another undirected model or epoch sweep. It does not promise an immediate large AP gain: the acquisition experiment still has to run and succeed.
