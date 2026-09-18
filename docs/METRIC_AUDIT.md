# Independent metric and localization audit

Audited September 17, 2026. This report records 52 completed cloud runs before the new Large/704/EMA results. The independent numeric evidence is `artifacts/metric_audit_v2.json`. No training or cloud execution was performed by this audit.

## Verified numbers

Independent pycocotools reconstruction from every saved prediction file reproduced all 52 reported COCO AP values exactly: maximum absolute difference **0**. The main metric remains AP averaged over IoU thresholds 0.50 through 0.95; AP50, AP75 and recall are different metrics.

| Task and recipe | Seeds | Naive AP | Aware AP | Complete-label AP | Aware AP50 | Aware AP75 | Aware recall at 0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pawns, initial 384 | 3 | 63.789 | 76.347 | 78.041 | 99.643 | 96.913 | 99.308 |
| Pawns, 512 continuation | 3 | 76.251 | 78.822 | 78.936 | 100.000 | 98.646 | 100.000 |
| All 13 chess classes, initial 384 | 3 | 58.226 | 69.642 | 72.537 | 88.225 | 86.313 | 96.953 |
| All 13 chess classes, 512 continuation | 3 aware/naive; 2 complete | 69.032 | 72.911 | 74.156 | 89.694 | 88.062 | 98.632 |
| Construction, matched first seed | 1 | 46.339 | 48.139 | 52.756 | 87.086 | 49.313 | 92.887 |

The initial paired gains are +12.558 AP points for pawns and +11.416 for all 13 classes. The corresponding 512 gains are +2.571 and +3.878. Construction has only one complete three-arm seed; its paired gain is +1.800. A second naive construction result exists, but must not be pooled against the single aware/control run as though seed counts matched.

Descriptive comparisons to complete-label controls, computed on exactly matched three-arm seed sets:

| Cohort | Aware AP / complete AP | Fraction of naive-to-complete AP gap recovered |
|---|---:|---:|
| Pawns initial, 3 seeds | 97.830% | 88.118% |
| Pawns 512, 3 seeds | 99.855% | 95.748% |
| All 13 initial, 3 seeds | 96.009% | 79.771% |
| All 13 512, 2 matched seeds | 98.297% | 76.673% |
| Construction, 1 matched seed | 91.249% | 28.051% |

These ratios are **not accuracy, AP, or substitutes for the headline metric**. For the matched two-seed all-13 comparison, aware AP is 72.893 and naive AP 68.742; those differ from their three-seed means.

## Correctness and fairness

- Every paired arm checked shares its seed, initialization parameter digest and requested update count. Model resolution and evaluation split match. The 512 runs use 4,000 total updates; Large uses 2,000 and is a separate combined-recipe pilot, not a clean model-size ablation.
- The evaluator correctly maps normalized boxes using original `[height, width]`, then maps zero-based semantic labels to canonical COCO category IDs. The reserved output is deliberately omitted. COCO AP consumes all retained predictions, while the separate operating point uses the fixed 0.25 confidence threshold.
- No hidden-label reads were identified in training or teacher-mining paths. The training view excludes hidden/test annotations. Complete controls and evaluation reference files coexist in the broader cloud archive; this is code-path/data-contract isolation, not a filesystem security sandbox.
- Large initialization was checked directly against the official cached pretrained checkpoint: **481 nonclassification tensors are bitwise equal, zero differ, zero change shape**. The only missing tensor is the upstream deterministic `_kp_active_mask` buffer. All task classification heads are intentionally reset. Warnings emitted while constructing an empty model do not mean the subsequent strict initialization load trains from scratch.
- Upstream `RFDETREMACallback.on_train_end` copies the averaged state into the live module. Saving the detector after `trainer.fit` therefore exports EMA weights. Existing smoke evidence verifies saved/reloaded parameter parity. This is source and artifact evidence, not a new full training run.
- Native SDK portability has a separate metadata issue: `save_detector` currently places `model_name` inside `args`, but public `RFDETR.from_checkpoint` reads the top-level `model_name`. With `args.pretrain_weights=None` and filename `detector.pt`, its variant inference fails. Future exports should include top-level `model_name="RFDETRLarge"` or `"RFDETRNano"`; historical files can be rewrapped without changing weights. The project evaluator reconstructs the explicit saved variant and is unaffected. Large hosted serving has not been established by the existing Nano deployment parity check.
- There is a preprocessing distinction: this project's evaluator uses PIL's default bicubic resize; stock torchvision validation uses bilinear with antialiasing, while public RF-DETR prediction uses tensor bilinear without antialiasing. It affects image pixels, not the coordinate scale. All controlled results share the project evaluator. Do not silently replace it and mix new scores into the old protocol. It is a worthwhile separately named parity experiment, not evidence of a large metric bug.

## What limits strict AP

Pawns at 512 have 100% AP50 and recall, yet mean AP90 is **23.644%** and AP95 **1.213%**. All-13 aware AP90 is **34.642%**, AP95 **2.631%**. Construction aware AP90 is **6.076%**, AP95 **0.794%**. The corresponding complete-label controls also fall sharply at high IoU.

On first-seed complete-label pawn predictions, median matched IoU is 0.893; median width ratio is 0.987 and height ratio 0.997. Median center errors are only -0.10 px horizontally and +0.01 px vertically. Thus there is no large common origin/scale error to fix. Pawn boxes are about 33 px wide in the original 640 px images, making small boundary differences important at IoU 0.95.

More tellingly, matched-object residual correlations across two independently trained aware seeds are **0.95–0.97** for pawn centers and sizes. Aware-versus-complete correlations are **0.93–0.95**. All-13 gives similarly high correlations, approximately **0.94–0.96**. More seed averaging is unlikely to remove a shared error pattern. These observations are consistent with shared model bias, ambiguous annotation boundaries, or both; predictions alone cannot establish label noise or a hard accuracy ceiling.

The all-13 dataset also preserves a generic `bishop` category with zero training positives and one validation positive. Its AP is essentially zero. With that category remaining at zero, even perfect performance on the other 12 categories yields 92.31 macro AP. Removing it after seeing results would change the benchmark, not improve the detector.

## Assessment of a crop-based box refiner

A second-stage image-conditioned geometry regressor is a defensible architectural experiment. It can focus spatial features on a single object and learn residual boundary corrections that the global detector shares across seeds. It cannot recover absent information, guarantee 95 AP, or resolve intrinsically inconsistent labels.

The relevant established mechanism is refinement with proposal distributions matched to inference. [Cascade R-CNN](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html) explicitly identifies proposal-quality mismatch and trains progressively higher-quality stages. A single crop-only regressor with fixed scores is an adaptation, not a reproduction of that full method. [Detectron2's CascadeROIHeads](https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/cascade_rcnn.py) uses class-agnostic geometry heads; [its Fast R-CNN head](https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py) and [box transforms](https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/box_regression.py) are the appropriate coordinate/loss references. [Torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) supplies the small ImageNet-pretrained backbone.

One compact, predeclared protocol:

1. Start with **all-13 chess only**, fixed seed 20260917. Read human annotations from its verified partial **training** view only, excluding all pseudo-label views. Train one partial-label refiner and apply that exact checkpoint to both naive and aware detectors; train a separate matched complete-label refiner for the descriptive complete control. Do not use the all-13 refiner to claim unchanged annotation budget on the pawn-only task.
2. Generate deterministic proposals around each observed box. Mix identity proposals and modest center/log-size jitter; for example 25% identity, otherwise center offsets uniform within ±10% of width/height and log-size offsets within ±0.15. Reject proposals with IoU below 0.5 against their generating **training** box. This prior is fixed before validation evaluation; it is not estimated from validation residuals.
3. Expand each candidate by 1.5×, sample a 224×224 RGB crop, and normalize using the pretrained backbone's documented channel statistics. Preserve an explicit floating-point crop transform and padding outside the image. Avoid integer crop rounding. Retain spatial features, such as adaptive 4×4 pooling followed by a small MLP; global average pooling alone discards useful location information.
4. Predict four center/log-size residuals with an exactly zero-initialized final head. Freeze batch-normalization running statistics. Use Smooth L1 on encoded residuals plus GIoU on decoded boxes, with fixed unit coefficients. A compact initial budget is 1,000 optimizer updates, batch 32, AdamW with backbone LR 1e-5 and head LR 1e-3; no checkpoint selection or validation-driven schedule changes.
5. Apply the refiner once to each image's existing predictions with confidence ≥0.05, at most the top 100. Keep every other prediction, class ID and score unchanged. Apply the same deterministic correction to identical proposal geometry. Clip only final boxes to valid image bounds; leave invalid/nonfinite corrections unchanged and count them. Do not introduce NMS, score editing or per-class threshold tuning.
6. Before training, verify identity export gives exactly unchanged boxes, crop/decode coordinate round trips near image borders, finite gradients, and rejection of non-training/pseudo/unknown manifest inputs. Evaluate final refiner weights with the existing full COCO scorer and report AP/AP50/AP75/AP90, per-class regressions, recall, and latency. Apply the one fixed partial refiner to all three detector seeds if available; those seed statistics measure detector variation, not refiner-training uncertainty.
7. Reserve at most two 900-second L40S jobs for partial/full refinement if cloud execution is separately authorized and budgeted. At the already recorded rate with 4 CPU cores and 24 GiB RAM, the runtime ceiling is about **$1.17 total**, before any separate build costs. Actual throughput is unmeasured; do not present this budget as a duration guarantee. Local execution is also possible after avoiding contention with the active MPS experiment.

Main failure modes: jitter proposals may differ from real detector errors; a crop containing adjacent pieces may regress to the wrong instance; independently annotated object extent may be irreducibly ambiguous; tiny datasets can overfit; refinement may damage already correct boxes without changing scores. Construction is riskier because person, helmet and torso/vest boxes are nested and use different extent conventions. Class-agnostic refinement there needs separate evidence. Keeping proposal identity, a spatial head, modest jitter and fixed score/class behavior makes the initial chess test interpretable without fitting validation annotation style.

If the refiner fails, retain the negative result. A second blinded human annotation pass on a small randomly chosen sample would distinguish annotation variability from model error more directly; the present audit cannot make that attribution on its own.
