# Continuous research results

## Native resolution and supervision exposure

All six predeclared construction continuations completed on Modal, followed by six GPU evaluations against the unchanged 120-image, 717-box validation reference. Every run added 2,000 updates. Partial and full-label controls replayed the same partial-derived image sequence. The complete results, checkpoint identities and cloud call IDs are in [CONTINUOUS_SCALE_RESULTS.json](CONTINUOUS_SCALE_RESULTS.json).

| Learner | Nominal training / inference size | Sampling | AP50:95 | AP50 |
| --- | --- | --- | ---: | ---: |
| Coverage-aware control | 512 / 512 | Uniform | 50.32 | 89.87 |
| Coverage-aware | 512 / 512 | Repeat factor | 50.73 | 88.96 |
| Coverage-aware | 640 / 640 | Uniform | **52.17** | 90.07 |
| Coverage-aware | 640 / 640 | Repeat factor | 51.57 | 90.02 |
| Full-label control | 512 / 512 | Uniform | 52.07 | 92.04 |
| Full-label | 640 / 640 | Repeat factor | **54.01** | 93.51 |

Native-resolution aware training gains **1.86 AP points** over its equally trained control, and 1.50 over its preceding aware parent. Sampling alone adds 0.42 points; its combination with native resolution is worse than resolution alone. Full labels with the combined recipe gain 1.93 points over their own control. These are one-seed exploratory results; parent histories differ between label arms.

The gain is concentrated. No-helmet AP increases from 31.65 to 40.31, contributing 1.73 of the 1.86 aggregate points. Only eleven validation objects belong to that class. Overall AP75 decreases from 54.01 to 53.01. The result supports investigating small-object scale, not claiming broad localization improvements across classes.

All ten frozen cross-resolution GPU evaluations completed. The 512px aware control gains **0.90 points** from inference at 640 alone (50.32 → 51.22). At that same 640 inference size, the new native-resolution training adds **0.96 points** (51.22 → 52.17). At 512 inference, the new training instead loses 0.25 points. The interaction is 1.21 points: training and runtime resolution contribute jointly. All weights, scores and reference labels remain unchanged. See [CROSS_RESOLUTION_RESULTS.json](CROSS_RESOLUTION_RESULTS.json).

Pawns do not benefit from the same runtime change. Fresh CUDA results are **79.20 → 77.11** for the strongest aware checkpoint and **79.17 → 77.68** for its same-seed full-label control. The 640 override is rejected for pawns. The 0.0045-point difference between the old recorded pawn score and fresh CUDA512 scoring is device-level variation, not a model improvement.

Four selected/control construction recipes are now frozen for the reserved vendor TEST split, with no test-driven recipe changes permitted. The same four checkpoints also receive the original validation tiling policy and original partial-TRAIN class-size gate. This tests interaction with the existing deployment adapter; it does not introduce another tile-size search. Test scoring is single-pass; the secondary tile results remain validation results.

## Geometry-only original/flip consistency

A separate fixed method paired predictions from original and horizontally flipped images, within the same image and category, and averaged bounded corner displacements. It preserved original confidence scores, classes, ordering and detection count. No unmatched prediction was removed or invented.

The cloud pilot reused saved TRAIN predictions from an older all-pieces detector. On 235 image-grouped holdout matches, mean IoU changed from **0.91486 to 0.91535**; strict 0.90-IoU matches fell from **159 to 157**. This failed both the declared minimum 0.003 mean-IoU improvement and the nondecreasing strict-match requirement. **The method was rejected before validation.** The detector had already seen these training images, so even a passing gate would not establish generalization.

Source review identified that the unused evaluation branch rejected zero-area, low-confidence stock predictions. The working implementation now preserves those rows unchanged and excludes them from pairing, with a synthetic regression test. This fix was made after the frozen pilot; the original source snapshot and negative result remain intact. No evaluation was rerun or attributed to the newer source.

## Execution and accounting

The persistent goal remains active after each completed batch. Paid call IDs, immutable protocols, checkpoints, source snapshots and failures are saved before moving to the next hypothesis. Training and inference run only on cloud GPUs; local checks cover source and small synthetic metadata/geometry contracts.

The six scale runs consumed 2,686.83 recorded training seconds. At maximum requested resource rates, including an additional 300 seconds per call, their computed bound is 2.906 credits; 3.50 remains allocated against their original 6.00 reservation. Prior ledgers are retained. The ten cross-resolution evaluations reserved 1.20 credits. The next contextual zoom pilot and four construction confirmations reserve 0.60 and 0.48, bringing the retained cumulative ceiling to 19.90 against the user's confirmed 20.31 starting balance. The latest observed monthly provider bill is still zero. These bounds are conservative allocations, not claims about a settled remaining account balance.

## Wide-context zoom pilot

A separate fixed chess experiment uses the same detector on 320–512px square crops, retaining board context around each original predicted object. It pairs only same-class, high-overlap crop detections, rejects artificial crop-edge boxes, and blends bounded coordinate displacements while preserving the original classes, confidence, ordering and detection count. This changes the feature-grid allocation; 640px source images contain no additional pixels to recover.

The policy is inspired by [Learning to Zoom and Unzoom (CVPR 2023)](https://arxiv.org/abs/2303.15390) and [Cascaded Zoom-In Detector (CVPR Workshops 2023)](https://arxiv.org/abs/2303.08747). Those papers train different architectures; this is an exploratory inference adapter, not a reproduction of their results.

Fresh candidate predictions on an image-hashed 20% subset of observed TRAIN labels must improve mean IoU by at least 0.003 across at least 100 matches, preserve strict 0.90 matches, and introduce no new misses at 0.50. Only a passing gate permits validation scoring of both aware and full-label checkpoints. Eleven synthetic tests cover geometry, preservation and rejection before reference access. No local model computation occurred.

## Reserved construction test confirmation

All four frozen cases completed on CUDA. TEST has 91 images and 627 boxes; the unchanged VALID reference has 120 images and 717 boxes. [SCALE_CONFIRMATION_RESULTS.json](SCALE_CONFIRMATION_RESULTS.json) retains both cohorts, every checkpoint and every paid call.

| Learner | Single-pass TEST AP50:95 | Fixed-policy tiled VALID AP50:95 |
| --- | ---: | ---: |
| Aware 512 control | 41.95 | 51.39 |
| Aware native640 | **43.15** | **52.64** |
| Full-label 512 control | 44.39 | 53.64 |
| Full-label native640 + repeat sampling | **45.55** | **55.10** |

The aware change gains **1.20 TEST points** and **1.25 tiled VALID points** over its equally continued control. The full-label change gains 1.17 TEST points. Both gains remain heavily concentrated in no-helmet. Aware TEST AP50 decreases 79.56 → 77.97 despite AP50:95 increasing; this is not improvement across every metric. The lower TEST absolute score reveals a cohort gap that validation alone hid. Full-label training remains stronger overall. Further construction choices must not use this opened test to tune parameters.

The new aware tiled VALID score exceeds the earlier recorded 52.09 by 0.54 points. It is an additional-cost deployment recipe, with 265 acquired training boxes and five passes, not an isolated coverage-mask ablation.

## Context zoom outcome and next mechanism

The wide-context pilot was **rejected before validation**. Across 235 observed TRAIN matches, mean IoU fell **0.91979 → 0.90658**, and strict 0.90 matches fell **164 → 150**. Actual CUDA inference processed 33 original images and 513 crops. No validation predictions or scores were produced. See [CONTEXT_ZOOM_RESULT.json](CONTEXT_ZOOM_RESULT.json).

The next experiment changes query assignment and confidence learning during detector training, inspired by [Detection Transformer with Stable Matching](https://arxiv.org/abs/2304.04742). It separates a position-only positive target from geometry-modulated matching in a pawn 2×2 study, with full-label control and combined-treatment runs. This uses the RF-DETR positional-target branch and new coverage-preserving integration; it does not claim reproduction of the paper's entire architecture. The purpose is to align supervision with accurate boxes, rather than rescue failed crop/flip adapters through threshold selection.

An independent source audit found no material checkpoint, reference, split or crop-coordinate error in the completed confirmation/zoom work. It identified a collector restart weakness: terminal JSON markers were written nonatomically. Current collectors now publish those JSON artifacts atomically. Frozen cloud source and actual experiment receipts remain unchanged.

Provider closed-hour reports show 4.1876 credits for the completed earlier four research apps and 1.8909 for the completed scale app. All owned apps had zero active tasks after collection. The next allocation retains 6.00 and 3.00 against those two completed groups, keeps 2.68 for the remaining categories, and reserves 6.92 for the stable-assignment matrix: **18.60 cumulative against 20.31 starting credits**, with 1.71 uncommitted. Original ledgers remain unchanged; no cash budget is added.
