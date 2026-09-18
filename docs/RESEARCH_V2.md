# Ontology and localization follow-up

This experiment tested two separate limitations of the existing coverage-aware detector: lost negative evidence at known objects, and high-IoU box accuracy. All twelve training runs, eighteen GPU evaluations and six fixed-policy tiled comparisons completed. Improvements were modest; no large advantage over the fully labeled reference was demonstrated. The complete portable evidence is in [RESEARCH_V2_RESULTS.json](RESEARCH_V2_RESULTS.json).

| Dataset · single-pass GPU AP50:95 | Starting aware model | More training only | Ontology | Ontology + precision | Full-label + precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pawns | 78.90 | 78.32 | 78.77 | 78.22 | 78.61 |
| All chess pieces | 73.71 | 73.76 | **74.10** | 73.67 | 75.20 |
| Construction | 49.72 | 50.64 | 50.67 | **50.95** | 52.82 |

Ontology alone gained 0.45, 0.34 and 0.03 points over the corresponding continued-training controls. Its pawn result still regressed against the starting model. Adding the precision objective hurt both chess tasks. Extra training explains most of the construction single-pass improvement.

With the unchanged construction tile policy and original training-derived class gate, the starting aware model scored 51.34 on the GPU evaluator, the continued-training control 51.94, ontology 52.09, ontology plus precision 51.24, and the new complete-label reference 54.24. The starting complete-label reference scored 53.75. The ontology gain over equally trained, equally tiled coverage-aware control is therefore **0.15 points**, not a large causal effect. The all-pieces and construction maxima replace their older published 73.68 and 51.62 results; pawns retain their older 79.20 maximum. Historical CPU scores and freshly rescored CUDA parent values differ slightly and are retained separately.

## Hypotheses and relevant research

The existing image/class coverage mask correctly avoids supervising unknown classes as absent. It also omits a narrower source of legitimate negative evidence: a query matched to a labeled black pawn represents an object that cannot also be a white pawn. The new optional ontology contract restores only those explicitly declared object-level negatives. It never changes image coverage, creates boxes, or treats an unmatched query as background for an unknown class.

The ontology groups are application declarations, not assumptions inferred from arbitrary class names:

- Pawn colors are mutually exclusive for one object.
- The twelve color-specific chess identities are mutually exclusive. The generic `bishop` class is excluded because it overlaps the more specific bishop labels.
- `helmet`/`no-helmet` and `vest`/`no-vest` are separate exclusive groups. `person` remains independent.

[Alpha-IoU (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8f15eda80c50adb0e71943adc8015cf-Abstract.html) motivates testing a power IoU objective for stricter localization. The original implementation here replaces only GIoU with Alpha-GIoU at alpha=3; matching, L1 loss, classification, group normalization, auxiliary heads and encoder supervision retain their existing behavior. Alpha=1 is checked against upstream values and gradients.

[Cascade R-CNN](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html) motivates training refiners on real detector proposals. [IoU-Net](https://openaccess.thecvf.com/content_ECCV_2018/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.html) motivates separating classification confidence from localization quality. These papers justify hypotheses; their published gains are not expected or claimed for this dataset.

[PseDet (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/3c40417b8dca30c08cc361df5b33ad7e-Paper-Conference.pdf) identifies teacher quality, class-dependent thresholds and confidence/localization mismatch as problems in pseudo-label learning. That supports caution about repeating the project's already negative teacher-label experiments. The current matrix uses observed human labels only. No PseDet reproduction is claimed.

## Fixed comparison

Each dataset starts from its existing aware/full parent pair and adds 2,000 updates, batch 4, seed 20260918, learning rates 2e-5 and 2e-6, augmentation and final-step EMA. No validation occurs during training. The architecture, resolution and annotation budget remain fixed within each dataset/arm.

| Case | Coverage loss | Object exclusivity | Box objective |
| --- | --- | --- | --- |
| Continued-training control | Yes | No | Original GIoU |
| Ontology | Yes | Yes | Original GIoU |
| Ontology + precision | Yes | Yes | Alpha-GIoU, alpha=3 |
| Complete-label precision reference | Full coverage | Already supervised | Alpha-GIoU, alpha=3 |

Pawns use Nano/512px; all pieces use Large/704px. Construction uses the existing guided-acquisition learner: 2,384 observed training boxes versus 6,380 in its complete reference. The 265 previously acquired boxes are the same for all three coverage-aware continuation cases. This study does not claim that acquisition used the original annotation budget.

All twelve resulting checkpoints and their six parents are scored with the same cloud GPU evaluator on unchanged complete validation references. This avoids attributing CPU/GPU numerical differences to training changes. Validation reuse and single-seed continuation remain limitations; these are not independent held-out-test confirmations.

## Rejected localization cascade

The proposal-based refiner used 2,082 matched predictions from 201 training images, including original and horizontal-flip views. A deterministic image-content split reserved 534 proposals for selecting the refiner; duplicate image bytes and both views stayed together. The parent detector and encoder had already seen these training images, so this is a refiner holdout, not a wholly unseen detector split.

A frozen human-label-trained encoder supplied visual features. Ridge strengths and bounded correction strengths were selected using only observed training boxes. Unchanged boxes achieved mean holdout IoU 0.90933; the best nonzero correction achieved 0.90609. The identity correction won. The cascade was rejected before validation scoring.

## Interrupted local pilot

Before the cloud matrix, a 1,000-update pawn Alpha-GIoU pilot scored 78.6806 AP50:95, below the existing best 79.1987. It was not promoted. The local alpha=1 control was interrupted at the user's request, and the local complete-label case never started. The 2,000-update cloud matrix replaces those unfinished comparisons; do not combine their update budgets into a matched result.

## Execution and provenance

Training and scoring use separate Modal images. Training receives only verified learner views and parent weights; complete reference bundles are packaged separately for evaluation. Checkpoints are downloaded and hash-verified before scoring, so an evaluation failure does not require training again. Persistent reservations and saved call IDs prevent automatic duplicate paid training after a client interruption. Evaluation likewise resumes existing calls and validates the checkpoint, reference, split, source snapshot and device.

The user confirmed $20.31 remaining credits. Twelve training calls reserve at most $14.40 and eighteen evaluation calls reserve at most $3.60, a combined $18.00 ceiling. These are conservative reservations, not measured spending. No new persistent volumes or automatic retries are configured. All subsequent model compute runs on cloud GPUs because the user prohibited local training and inference.

The subsequent six-case tiled interaction reserved $1.20, and the one-call segmentation pilot below reserved $1.00. The combined conservative ceiling was $20.20, within the confirmed balance. The billing API after the main training batch reported $25.16 monthly metered usage, entirely offset by credits, versus $20.31 before it. The $4.85 difference is an observed usage change, not a final settled bill or a remaining-balance reading.

Frozen protocols and detailed predictions live in ignored `artifacts/research_v2/`; portable result summaries are committed separately. New gallery predictions come from the exact newly selected checkpoints and inference policies. The UI retains its disclosure that independent minima/maxima are not matched comparisons.

The GPU evaluator initially failed at container import because deployment by filesystem path registered an unqualified module name. Redeploying with `modal deploy -m coveragecv.training.modal_research_evaluation` fixed it; the original queued calls completed without resubmission or duplicate evaluation reservations. Deployment commands for these isolated images must use the package module name.

## Foundation-model boundary refinement

A different architecture used frozen [SAM2.1 Large](https://github.com/facebookresearch/sam2) to turn detector box prompts into object masks, then proposed mask-derived box boundaries. The [Transformers SAM2 interface](https://huggingface.co/docs/transformers/model_doc/sam2) supplied the pretrained model and prompt processing. Model revision `665f8e2ad61cf5f53d65644ff27c8ee525124610` and weight hashes are recorded. No new project annotations or SAM fine-tuning were used.

The pilot used 1,042 real original-view proposals matched to observed partial training boxes: 779 fit and 263 image-grouped holdout proposals. Global and class-specific box-padding calibration used fit labels only. Nine fixed combinations of calibration and blending were evaluated on the refiner holdout. Acceptance required at least 0.003 improvement in mean IoU before any validation scoring.

Unchanged boxes scored **0.91315**. The best nonzero refinement scored **0.90686**, even after calibration for the tighter segmentation boundaries. Identity won, so the method was rejected before validation. It is not part of the published predictor. The result reinforces that replacing learned box geometry with visually plausible masks does not necessarily match a dataset's annotation convention. The detector had previously seen the refiner holdout images; this is not an independent detector test split.
