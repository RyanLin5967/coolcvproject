# Next accuracy experiment: observed-object crop continuation

This is an independent diagnosis and a proposed experiment, not a completed result. Numeric evidence and prediction-file hashes are in `artifacts/next_accuracy_leap.json`. No new training, cloud execution or test-label access occurred during this audit.

## Why construction is the strongest target

The completed seed-20260917 construction models score 46.339 naive, 48.139 aware and 52.756 complete-label COCO AP50:95. The aware model already finds 666 of 717 validation objects at confidence 0.25; naive finds 563. The coverage mechanism improves recall substantially, but geometry and the rare class limit strict AP.

| Class | Human boxes visible in train | Train images with visible class | Validation boxes | Median box at 512px | Aware AP | Complete AP |
|---|---:|---:|---:|---:|---:|---:|
| helmet | 689 | 313 | 233 | 59.6 × 61.6 | 55.02 | 56.43 |
| no-helmet | 33 | 19 | 11 | 22.0 × 26.4 | 19.09 | 33.95 |
| no-vest | 261 | 133 | 90 | 88.0 × 144.2 | 44.52 | 47.22 |
| person | 749 | 325 | 242 | 110.6 × 354.2 | 67.48 | 68.84 |
| vest | 387 | 168 | 141 | 100.8 × 163.6 | 54.58 | 57.33 |

The no-helmet class alone contributes **2.972 of the 4.617 AP-point aware-to-complete gap: 64.37%**. Eight of its eleven validation boxes have original area below 32². Its aware AP50 is 68.79, AP75 is 3.47 and AP90 is zero. Across classes, aware COCO small-object AP is **10.71**, medium AP36.59 and large AP59.25. These are descriptive size partitions, not separate accuracy claims.

More identical full-frame training is poorly supported by the recorded loss curves. From the first to last complete epoch, aware training GIoU loss falls 0.221→0.152 while validation GIoU remains approximately 0.199–0.204. Validation box L1 moves 0.0596→0.0629. The complete control also has essentially flat validation localization loss. That can reflect generalization, supervision ambiguity or insufficient small-object detail; it does not establish annotation noise or a hard ceiling.

The default training crop is relatively broad: pinned RF-DETR 1.10.1 resizes short edge to 400/500/600, then randomly crops a square with side 384–600. A no-helmet head often remains only one or two 16px patches wide. The proposed intervention changes which real human observations receive training exposure and the spatial scale at which they are learned, without manufacturing annotations.

## One bounded experiment

Run four seed-20260917 continuations, each **1,500 updates, batch4, Nano512, original PE24, LR2e-5 / encoder2e-6, cosine to zero, no EMA**. Each starts from its corresponding completed 4,000-update construction model. The two aware continuations start from the exact same checkpoint.

| Run | Training data access | Continuation sampling | Purpose |
|---|---|---|---|
| aware_uniform | Original partial view | Uniform full frames | Matched extra-training control |
| aware_crop | Original partial view | 50% full frame; 50% class-balanced observed-object crops | Primary new treatment |
| naive_crop | Identical partial view | Identical crop/full-frame plan to aware_crop | Coverage control under new treatment |
| complete_crop | Complete training view | Identical crop/full-frame plan derived only from partial labels | Complete-label reference under new treatment |

Freeze one 6,000-slot plan. For 3,000 slots draw images uniformly. For the other 3,000, draw one of the five classes uniformly, then one **observed human annotation** uniformly from that class. This provides an expected 600 directly targeted no-helmet crops, versus approximately 199 no-helmet annotations encountered across 6,000 uniform full-image draws. Incidental objects in either case remain supervised when observed, so these exposure counts are not total loss contributions.

A square crop has side `min(image_short_edge, max(64, ceil(max(box_width, box_height) * U[2,3])))`. Jitter its center by up to ±10% of the crop side, subject to keeping the selected observed anchor completely inside; clamp the crop into image bounds. Resize to512 and apply the same fixed-seed 0.5 horizontal-flip policy. The full-frame branch uses resize512 plus that flip. Disable a second random crop in both branches and in the uniform control, to make the declared input policy exact. No new image sources or pseudo labels.

All crop arms must use the *same partial-label-derived plan*, including the complete control. That avoids giving the complete control a different rare-class sampling schedule. It still sees more annotations inside the selected images/crops, which is the intended complete-label advantage. Confirm dense image IDs, source-image hashes and dimensions match across views before sharing the plan.

This is a combined exposure/scale intervention. It does not isolate balancing from magnification; that decomposition would be a follow-up only if the combined treatment succeeds.

## Implementation seam and correctness constraints

Use a small wrapper around the pinned `RFDETRDataModule._dataset_train`, after `dm.setup('fit')` and the existing `assert_data_contract`. Keep the underlying dataset's verified COCO mapping and `prepare` routine. Disable its transform chain and JPEG draft decoding (`_transforms=None`, `_draft_size=None`) so its `__getitem__` returns the original pixels and prepared absolute-xyxy targets. Let wrapper index select the frozen source index/crop plan, then apply crop, resize, flip and Normalize. A wrapper length of6,000 is divisible by batch4 and gives1,500 updates without alignment padding. Keep `num_workers=0` for the first fixed-seed pilot. Preserve or explicitly forward dataset attributes needed by upstream Lightning.

Reuse **`rfdetr.datasets._torchvision.crop`**, then `Resize`, `RandomHorizontalFlip`, torchvision `ToImage`/`ToDtype`, and the pinned RF-DETR `Normalize`, rather than inventing different box transforms. The upstream crop clips every intersecting box, filters labels/area/iscrowd with the same keep mask, and preserves image-level fields. It removes only boxes with zero visible area; do not add a minimum-area or minimum-visible-fraction filter that silently turns visible annotated fragments into unlabeled negatives. Do not retain only the selected anchor: retain **all** visible annotations in that view.

Keep the original dense `image_id` and its coverage vector. Cropping does not establish absent classes: unknown stays unknown, positive-only stays positive-only, exhaustive remains exhaustive. An empty crop must never upgrade unknown to verified-absent. Existing coverage lookup can therefore remain unchanged. Crop planning reads only the verified partial training view; it must reject pseudo-labeled views and must not read hidden, complete-reference, validation or test annotations. The complete learner only reads its own annotations when materializing a plan entry.

Make sample and flip choices a deterministic function of plan digest/index/seed, rather than depending on how many annotations an arm contains. Serialize the plan, source-view identity, actual anchor/class exposure counts and a sample of geometric traces. Assert source image content binding in addition to integer IDs.

Minimum tests before execution:

1. Known crop geometries with inside, intersecting, outside and zero-area boxes; selected anchor remains intact and all intersecting targets stay aligned with labels.
2. Same plan generates identical image pixels/IDs across partial and complete views, while their extra annotations differ as expected.
3. Unknown-class negative gradients remain zero after crops; retained positive classification and box gradients are present. No new completeness inferred from empty targets.
4. Fixed seed reproduces image/crop/flip traces; uniform branch equals the declared full-frame transform.
5. One real CPU/MPS or bounded GPU update through wrapper, export, reload and scorer. Save/download the checkpoint before scoring to avoid repeating the refiner's original postprocessing failure mode.

Implementation can remain in a dedicated training-data module and an explicitly named runner recipe. Preserve historic runs, views and metrics; do not patch installed RF-DETR or mutate the original source artifacts.

## Evaluation, budget and limits

Primary comparison: `aware_crop AP − aware_uniform AP` under the **unchanged complete-validation full-frame evaluator**. Preserve all five classes, AP50:95 as headline, the fixed0.25 operating point, and all negative results. Report AP50/AP75/AP90, small/medium/large AP, per-class AP, and all four arms. A practical success target is at least **+3 AP points** over the matched uniform continuation; this is a target, not a prediction. No sliced inference, confidence tuning or best-checkpoint selection in this experiment. Test labels remain untouched. Four one-seed runs do not establish multi-seed robustness, and eleven no-helmet validation objects make that class's estimate especially unstable.

Measured existing Nano512 construction training: **801 seconds for4,000 updates**, approximately0.200s/update. The new1,500-update runs should be roughly300s of training each; allow extra decode/startup/export overhead. Use four separately reserved L40S calls, CPU4, RAM limit24GiB, timeout1,200s, max4, min0, retries0, no persistent Volumes. Mirroring the existing bounded refiner allocation, reserve **$0.85 per call, $3.40 total**. The reservation is a conservative client commitment estimate, not an actual bill or proof that the provider credit discrepancy has been reconciled. Parent must record the new cohort in the persistent budget policy before execution.

For chess, the already-running crop refiner is the more direct geometry experiment. The all-13 generic bishop has no observed training examples at all: class balancing cannot invent those, and excluding the class would change the benchmark. Neither this construction intervention nor current evidence justifies a 95 AP50:95 promise.

## Established mechanisms, distinct proposal

[SAHI's primary paper](https://arxiv.org/abs/2202.06934) provides evidence that training/inference slices can help small-object detectors; its reported gains on aerial datasets are not predictions for this dataset. [Detectron2's training sampler implementation](https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/build.html) provides a concrete precedent for increasing rare-class exposure. This proposed uniform-class crop sampler is deliberately simpler than its repeat-factor algorithm and should not be described as that algorithm. The project-specific contribution is preserving audited partial-label coverage and geometric provenance through object-centric training and evaluating it with matched controls.


## Executed protocol update

The accepted implementation uses **2,000 continuation updates, batch 4, 8,000 planned samples**, with a $1.10 conservative reservation per call ($4.40 across four calls). This supersedes the 1,500-update planning estimate above. The plan contains exactly 4,000 uniform full-frame slots and 4,000 class-balanced observed-anchor slots. All three crop arms replay identical image pixels, integer crop rectangles and flips. The ordinary aware continuation is the fixed-budget primary control. Its stock augmentation remains intact, so this tests the combined sampling/augmentation intervention. Each arm starts from its own existing seed-20260917 4,000-update checkpoint. Total detector updates are 6,000.

Seven geometry/provenance/dataloader checks and an actual two-update aware CPU training smoke passed before deployment. Inputs and source were frozen; four cloud calls were launched after deployment. Checkpoints return locally before complete-validation CPU scoring. No test evaluation or checkpoint selection is allowed. The plan and run bindings are in `artifacts/object-crops/`; results are pending at this update.
