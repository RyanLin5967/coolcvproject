# Existing implementations to reuse

Checked September 17, 2026. The feature's value is a reliable coverage contract and a verified integration into Roboflow's ecosystem. Partial-label learning itself is established prior work. This map distinguishes production dependencies, small adapted code, and conceptual references.

## Implement as little new machinery as possible

| Component | Reuse | Project-specific work | Boundary |
|---|---|---|---|
| Detector, backbone, matcher, box losses | [RF-DETR 1.10.1](https://github.com/roboflow/rf-detr/tree/e3fc28795f2a4303069c6b72e80431e5ed716030) | Version/configuration checks and initialization policy | Do not rebuild a detector or claim compatibility with every release. |
| Training lifecycle | RF-DETR's exported data/model modules and Lightning loop | Thin criterion/data subclasses, artifact binding, run manifests | No global monkey patches or copied training step. |
| Classification loss | Pinned [IoU-aware BCE implementation](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/src/rfdetr/models/criterion.py) | Small attributed adaptation that masks whole unreduced cells | Match stock loss/gradients when all classes are exhaustive; preserve aux/encoder paths. |
| COCO evaluation | Upstream evaluator and [maintained pycocotools](https://github.com/ppwwyyxx/cocoapi) | Common metrics/prediction export, semantic mapping assertions and reserved-output diagnostics | No custom metric that excuses unknown-class false positives on complete evaluation labels. |
| Data validation | [Pydantic v2 strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/) | Coverage/ontology/provenance schemas and cross-field invariants | Types alone cannot prove a completeness declaration true. |
| Canonical JSON | [Trail of Bits `rfc8785.py`](https://github.com/trailofbits/rfc8785.py) | Semantic ordering, finite bounds, payload manifests and root-digest policy | Canonical serialization does not supply semantic canonicalization. |
| Hashes/filesystem | Python `hashlib`, streaming reads and atomic same-filesystem rename | Artifact identity, staged publication, corruption detection | No custom hash algorithm, database or object store needed. |
| Dataset/model platform | [Roboflow Python SDK](https://github.com/roboflow/roboflow-python) | Version reconciliation, training operation ledger, explicit model identity and checkpoint gates | Recipes and exporters remain service/version dependent. Pin the selected SDK release at execution. |
| Remote execution | [Modal Functions](https://modal.com/docs/guide) and [Volumes](https://modal.com/docs/guide/volumes) | Thin launcher, finite attempts/deadline, state retrieval | No custom GPU scheduler or permanent endpoint. |
| Reproducible environment | [uv](https://docs.astral.sh/uv/concepts/projects/) | Locked extras, explicit Torch/CUDA selection, environment manifests | A lock resolves versions; imports/steps establish compatibility. |
| Verification | pytest, optional Hypothesis for compiler invariants; PyTorch autograd | Behavior-based fixtures and upstream parity | Do not test only that our formula equals a copy of itself. |
| Visual report | Ordinary static HTML, saved JSON and image overlays | Coverage explanations, fixed comparisons, raw-metric links | No hosted frontend account or live training service necessary. |

## Prior art to study, not paste into the wrong architecture

**LVIS / Detectron2.** The [LVIS loader](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/lvis.py) handles per-image negative-category and non-exhaustive-category metadata. This is a concrete reference for distinguishing unobserved labels from verified negatives. Detectron2's [Fast R-CNN federated-loss implementation](https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py) selects a class subset using positives and sampled classes; it is not our per-image coverage mask and is not a drop-in DETR criterion. Use it to understand established approaches and to design a later comparison.

**Class-aware selective loss.** [Alibaba-MIIL/PartialLabelingCSL](https://github.com/Alibaba-MIIL/PartialLabelingCSL) is an official implementation for multi-label classification with partial annotations. Its task is classification, not object detection with Hungarian matching. The transferable idea is selective supervision under incomplete labels; its loss and empirical results do not establish correctness for this detector.

**Multi-dataset detection.** [Object Detection with a Unified Label Space from Multiple Datasets](https://arxiv.org/abs/2008.06614) directly studies conflicting supervision across dataset label spaces. [UniDet](https://github.com/xingyizhou/UniDet) is a different project associated with *Simple Multi-Dataset Detection*, not the implementation of that 2020 paper. Keep those citations separate. These references prevent an inflated novelty claim and motivate later stronger baselines; neither removes the need for a Roboflow-specific data/trainer integration.

**Dataset tooling.** [Datumaro](https://github.com/open-edge-platform/datumaro) already offers extensive dataset conversion and merge tooling. Study its import/validation boundaries; a future adapter can extend format coverage. Adding its full dependency surface to a single-format first release is unnecessary. The differentiator is preserving and enforcing coverage semantics through training, not claiming to invent dataset merging.

## License and attribution decisions

| Material | Reviewed status | Planned handling |
|---|---|---|
| RF-DETR pinned core source / selected Nano family | [Apache-2.0 source license](https://github.com/roboflow/rf-detr/blob/e3fc28795f2a4303069c6b72e80431e5ed716030/LICENSE); model-family distinctions documented upstream | Retain notices for any adapted function. Avoid assuming all larger/plus families share the same terms. Check the chosen checkpoint's own provenance. |
| Detectron2 | [Apache-2.0](https://github.com/facebookresearch/detectron2/blob/main/LICENSE) | Reference initially; preserve attribution if adapting code later. |
| PartialLabelingCSL | [MIT](https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/LICENSE) | Conceptual reference initially; no detection compatibility implied. |
| Datumaro | Current [MIT license](https://github.com/open-edge-platform/datumaro/blob/develop/LICENSE) | Optional future dependency; freeze a release/license if selected. |
| Maintained pycocotools | [FreeBSD-style two-clause license](https://github.com/ppwwyyxx/cocoapi/blob/ac87f5077ad6b8864c2dc5e93d14cae62d1db05a/license.txt) | Preserve its actual notice; do not relabel it MIT. |
| `rfc8785.py` | [Apache-2.0](https://github.com/trailofbits/rfc8785.py/blob/main/LICENSE) | Use package and retain dependency notice. |
| Chess images/labels | Source declares CC BY 4.0 | Preserve creator attribution, source revision and explicit transformation/withholding notes; see [DATA_PROTOCOL.md](DATA_PROTOCOL.md). |

Source-code licenses, checkpoint terms, platform access and dataset licenses are distinct. This is a record of reviewed materials and implementation choices, not a claim that any arbitrary model download grants every possible right. Unverified optional LVIS/UniDet code licenses are not a reason to block the selected implementation: neither is being vendored.

The final repository should contain a third-party notice identifying each actually copied fragment, its upstream path/commit, its license and our modification. Use dependency packages for ordinary libraries. Pinned research snapshots are source-review evidence, not a request to vendor the entire RF-DETR repository into the delivered package.
