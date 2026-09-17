# RF-DETR adapter: pinned implementation design

Status: source-inspected design, not executed. RF-DETR **1.10.1**, release September 7, 2026, tag resolves to `e3fc28795f2a4303069c6b72e80431e5ed716030`. Sixteen relevant files are saved under `../raw/build_rfdetr_1_10_1/`; `sources.json` records immutable URLs and content hashes. Current `develop` contains changes absent from this release. All claims below refer to the pin.

## Integration seam

| Proposed component | Responsibility | Existing code reused |
|---|---|---|
| `CoverageTrainingSpec` | Validated project configuration, bundle identity, enabled adapter/loss/transform modes | Separate from upstream `TrainConfig`, so unsupported fields are never silently ignored |
| `CoverageArtifact` | Verified ontology, globally dense image IDs, semantic coverage table, source/contract digests | Canonical compiler bundle |
| `CoverageDataModule` | Check category remapping, transforms, dataset split custody and preserved image identity | `RFDETRDataModule` loaders/collation/device transfer |
| `CoverageSetCriterion` | Gather eligibility by image ID, copy targets, apply the exact loss mask | `SetCriterion.forward`, matcher, box losses, group/aux/encoder dispatch |
| `CoverageModelModule` | Replace the criterion after upstream construction; bind checkpoints to the contract | `RFDETRModelModule` optimizer, training step, callbacks, checkpoint hooks |

The three training primitives are [public exports](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/training/__init__.py#L6). The [model module constructs `self.criterion`](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/training/module_model.py#L392) after network/weight initialization. Replacing that attribute in a subclass is the selected seam; a constructor-level criterion injection argument was not verified. Treat this as a version-coupled adapter, not an upstream compatibility promise.

Construct the custom criterion from the stock criterion's actual constructor fields. Reuse its matcher, weight dictionary, loss list and flags. Do not mutate `__class__`, replace global functions, or duplicate the whole training loop. Use an explicit version/contract check, a narrowly attributed loss implementation, and stock parity fixtures to detect upstream changes.

Keep the exact override signature `loss_labels(outputs, targets, indices, num_boxes, log=True, matched_targets=None)`. Inherited dispatch supplies `matched_targets`; dropping it breaks the integration even if a hand-written direct unit call passes. Reject padded filler targets in this release.

## Why coverage is injected at the loss boundary

[`ConvertCoco`](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/datasets/coco.py#L649) rebuilds a target dictionary; arbitrary source metadata is lost. Geometric filters also infer instance fields from dimensions: a coverage vector with length equal to the box count can be incorrectly filtered. Both [torchvision](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/datasets/_torchvision.py#L97) and [alternate transform](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/datasets/transforms.py#L661) code make this worth avoiding.

The compiler assigns globally dense IDs. `CoverageSetCriterion.forward(outputs, targets, num_boxes=None)` gathers the corresponding coverage rows from an immutable registered tensor lookup. Enriched target dictionaries are new objects; existing targets/tensors stay unchanged. The upstream dispatch receives those enriched copies and the original optional normalization override.

An image ID is trusted only after verifying that each actual dataset maps it to the bundle's image content and split. Batch positions or filenames alone are not identities. Missing/invalid IDs fail. Globally unique IDs avoid selecting the training row for a validation image. The table can be a nonpersistent module buffer; checkpoint metadata stores its digest and the matching ontology/bundle identity. Resume must load the correct artifact before restoring state. Digest strings belong in checkpoint/config metadata, not the upstream tensor target dictionary.

No `.item()` loop on CUDA: concatenate/stack image-ID tensors and gather all rows once. The execution test must demonstrate identity survives loader workers, shuffling, preprocessing, packing and device transfer. This remains required even though custom coverage itself bypasses those steps.

## Loss behavior

The default is `ia_bce_loss=True`, not plain focal loss. The [pinned implementation](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/criterion.py#L557) uses:

```text
p = sigmoid(z)
positive_weight = 0
negative_weight = p²

at a matched annotation cell:
  t = clamp(p^alpha × IoU^(1-alpha), min=0.01).detach()
  positive_weight = t
  negative_weight = 1-t

ell = negative_weight × z
      - logsigmoid(z) × (positive_weight + negative_weight)
```

The eligibility mask is `E[image,class] OR P[image,query,class]`, where E means exhaustive/verified absent and P means matched to an observed positive. Multiply the **complete unreduced `ell`** by eligibility, then keep upstream summation and normalization.

This preserves the `(1-t)` term at matched positives. Do not mask the algebraic negative term independently. Preserve the existing detach on `t`, detached IoU target boxes, and the derivative through `p²` for unmatched negative cells. A valid soft positive may reduce an overconfident score; the test oracle is stock equality, not an assumed gradient sign.

The adapter supports only detection IoU-aware BCE initially. Reject alternate focal, varifocal, position-supervised, segmentation, keypoint and unreviewed loss configurations. Loss masking removes local classification supervision; it does not freeze class parameters or remove indirect effects through shared representations.

## Class layout and initialization

The model and criterion receive **K+1 channels** for K semantic classes in the [builder](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/lwdetr.py#L777). Keep the last reserved channel eligible under the stock rule; never use it as a dataset class or as the unknown-coverage state. For the two-pawn experiment, semantic model indices are 0 and 1 and the reserved index is 2. Source YOLO IDs and exported COCO category IDs are separate mappings.

An all-unknown, zero-target sample has no direct semantic-class or box gradient, but the reserved output and optimizer/shared parameters can still change. Test gradients with respect to the relevant logits, not total parameter updates. Verify the upstream postprocessor/evaluator's reserved-channel treatment under the same settings for every branch; this is an integration test, not an assumption about a special softmax background.

Source review found the postprocessor includes the reserved slot in top-k and the COCO evaluator skips unmapped labels. Record reserved selections and assert every semantic label maps back correctly; never reinterpret the reserved slot as a pawn. Changing pre-top-k filtering would be a separate inference intervention.

[`_resize_linear`](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/lwdetr.py#L58) tiles/truncates pretrained rows. The method named `reinitialize_detection_head` uses it; that name does not imply random initialization. For a new ontology, explicitly create/reset the semantic rows of `model.class_embed` and every module in `model.transformer.enc_out_class_embed`. Preserve/document the reserved-row initialization policy and stock classifier bias convention. Save one seeded initialized model state, then load it identically for each arm before optimizer/EMA creation. Verify hashes and parameters instead of relying on `strict=False` loading.

A classifier reset does not erase backbone, decoder, query, or box-head pretraining. RF100 Chess is also associated with Roboflow's benchmark ecosystem. No claim that the dataset is unseen by all pretraining or model-selection activity follows from using a different label vocabulary. The experiment measures this fine-tuning intervention under a shared initialization.

**Hosted-base exception:** when a compatible exported Roboflow model already has the exact two-pawn ontology, preserve its main and encoder classifier heads. Explicitly validate/reorder class rows if needed; do not run the new-ontology reset. Create fresh identical optimizers for each Modal continuation arm. The base's fully labeled seed subset is disjoint from continuation images. See [ROBOFLOW_INTEGRATION.md](ROBOFLOW_INTEGRATION.md).

## Matching and denominator

The matcher compares predictions against observed annotation classes and boxes. Keep it unchanged. Its [classification cost](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/matcher.py#L179) is distinct from the training loss. Its reserved `use_pos_only` parameter is not an implemented partial-label solution.

Inherited [criterion forward](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/criterion.py#L937) matches final, auxiliary decoder, and encoder outputs independently. Each call to the overridden `loss_labels` constructs P from that call's indices. Reusing final-layer matches would be wrong. Preserve box losses, coefficients and classification diagnostics.

Group DETR defaults to 13 groups and 300 queries per group in training; evaluation uses one group. Preserve the [annotated-box/group/distributed denominator](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/criterion.py#L480), its minimum clamp, `sum_group_losses` behavior and explicit override. Do not normalize by active classes, eligible cells, or imagined hidden objects. Multi-GPU is outside the initial validated envelope; no early-return shortcut should be introduced that would break its future collective requirements.

Two-stage encoder proposal ranking still depends on class scores. Masking unknown classification cells does not make those scores irrelevant to proposal selection. Leave [proposal selection](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/transformer.py#L397) unchanged and disclose the boundary.

## Configuration for the first proof

| Setting | Proposed value/policy |
|---|---|
| Model | `RFDETRNanoConfig`, K=2 semantic classes, 384 resolution |
| Initialization | Identical imported hosted-base state with trained task heads preserved; fallback uses stock pretrained components plus seeded new-ontology K+1 heads |
| Group/query counts | Stock 13/300 for experiment arms; smaller synthetic test cases are separate |
| Device | CPU for loss tests; one NVIDIA A10 for measured runs |
| Precision | Full precision for correctness; one common AMP mode only after parity passes |
| Batch | Start probe at integer 2; choose a common physical batch that fits |
| Accumulation | 1 initially |
| Compile | Disabled |
| EMA | Disabled for first proof; later enable identically across arms after common-state initialization |
| Evaluation/checkpoints | Common schedule, fixed update budget, no early stopping for comparison |
| Augmentation backend | Explicit `torchvision` |
| Geometry | `multi_scale=False`, `expanded_scales=False`, `scale_jitter=False`, `aug_config={}`, `square_resize_div_64=True` |
| Mixed images/destructive crops | Rejected |

The upstream EMA callback wraps the entire Lightning module with `AveragedModel`, so even a nonpersistent coverage buffer can be copied into the EMA object. Disabling EMA avoids this lifecycle ambiguity initially. Enabling it later requires coverage-table device/copy/reconstruction tests and a declared base-versus-EMA checkpoint policy.

These are proposed adapter-supported settings, to be validated against the actual configuration objects. `aug_config={}` alone does not disable the stock crop/resize-jitter branch; `scale_jitter=False` is required. [Transforms](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/datasets/coco.py#L974).

The pinned criterion has **no padded-ground-truth `valid` mask support** from later develop changes. Do not backport that implicitly. Packed target transport is a different feature and can be tested through [`PackedTargets`](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/utilities/tensors.py#L495); begin unpacked and expand only after parity. Image padding via `NestedTensor.mask` is also a distinct concept.

Stock conversion drops crowd and degenerate boxes. The compiler must reject/exclude unsupported crowd/ignore cases or explicitly weaken coverage; no silent label dropping under an exhaustive assertion. The first chess experiment has ordinary boxes but must validate actual imported records.

The inspected training step contains accumulation scaling whose complete interaction with Lightning was not audited here. This is not a reported upstream bug. Accumulation one removes that unresolved dimension from the initial experiment; later accumulation needs a dedicated equivalence test.

## Dependencies and test boundary

The [pinned package metadata](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/pyproject.toml) requires Python >=3.10, Torch >=2.2, torchvision >=0.17, transformers >=5.1,<6, Pydantic >=2,<3 and supervision >=0.29. Its train extra requires Lightning >=2.6,<3 excluding 2.6.2/2.6.3, TorchMetrics >=1.8.2,<1.9, faster-coco-eval >=1.7.2, pycocotools, SciPy, and torch-hungarian 0.1.0rc0 among other packages.

These ranges are upstream metadata, not a verified lock. Use Python 3.12 and resolve compatible macOS and CUDA environments at execution time. Avoid optional TensorRT/ONNX/augmentation extras until needed. Capture exact wheels/lockfiles and run imports before claiming support.

CPU/MPS and CUDA matcher backends can differ; cross-device bitwise training equality is not promised. MPS can be used for small local experiments with AMP off initially, but its training throughput was not measured. Required correctness and experiment tests are enumerated in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and [ADVERSARIAL_REVIEW.md](ADVERSARIAL_REVIEW.md).
