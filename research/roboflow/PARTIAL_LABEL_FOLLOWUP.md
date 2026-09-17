# Follow-up: does Roboflow already have coverage-aware composition?

Rechecked September 17, 2026 after the user narrowed interest to idea 03 and a one-day build.

**Conclusion:** The inspected public RF-DETR classification-loss path does not expose the proposed per-image/class annotation-coverage handling. Roboflow already provides ordinary project merging. The research supports a plausible public integration gap; it does not establish absence in private hosted training, internal experiments or an undisclosed roadmap.

## Fresh checks

- [Merge Projects documentation](https://docs.roboflow.com/datasets/manage/merge-datasets) describes combining unique images and existing annotations. It does not document coverage-aware training semantics.
- Re-fetched `develop/src/rfdetr/models/criterion.py`; saved as `raw/followup_current_criterion.py`. The default IoU-aware classification branch initializes negative weights over logits and sums its classification loss without a per-image/class coverage mask. Existing matched-target validity handling concerns padding and must not be mistaken for annotation-coverage support. This is bounded code inspection, not an exhaustive audit of every product.
- GitHub API confirms [issue 135](https://github.com/roboflow/rf-detr/issues/135) was opened and closed April 9, 2025; the latest recorded update remains that date. Although GitHub labels its state reason completed, the comments show a scope-based closure, not a feature implementation.
- [Maintainer response](https://github.com/roboflow/rf-detr/issues/135#issuecomment-2790054522) describes the proposal as outside the scope of that work. This establishes awareness and a historical scope decision. It does not establish current priority or willingness to adopt an external implementation.
- Fresh issue/PR searches for partial annotations, selective loss and federated loss found the original discussion and an additional incomplete-annotation discussion, but no matching new implementation in those searches. Search absence is not proof of feature absence.
- A focused Exa query returned current training/format documentation and the same feature request; raw results are `raw/followup_partial_label_current.json`. This is an additional follow-up query beyond the initial report's 71-query research snapshot.

## Is the concept new?

No. [Object Detection with a Unified Label Space from Multiple Datasets, ECCV 2020](https://arxiv.org/abs/2008.06614) addresses inconsistent/partial annotations across datasets. LVIS/Detectron2 also has relevant coverage metadata. The opportunity is an accessible, correct RF-DETR data/training integration and evidence, not a claim to invent partial-label learning.

## Would anyone use it?

There is concrete user evidence: [the stamps/coins forum report](https://discuss.roboflow.com/t/merging-two-separate-models-datasets-after-training/7951), December 2024, describes the exact mixed-label issue; issue 135 independently requests selective supervision. These are examples of demand, not proof of broad prevalence, enterprise urgency or future adoption by Roboflow.

Direct beneficiaries are users combining partially labeled datasets. Roboflow's model/data team could value a demonstrated extension or contribution, but its earlier scope decision makes adoption uncertain. No maintainer outreach has occurred.

The one-day project's defensible claim is: a tested extension addressing a documented RF-DETR user problem, with a controlled experiment and explicit limitations. It should be compared against completing annotations/pseudo-labeling, preserve known positives and applicable auxiliary losses, and avoid claiming universal multi-dataset support or a guaranteed accuracy improvement.

## Hosted product versus open source: further clarification

Rechecked September 17, 2026 after the user asked specifically about the commercial product. RF-DETR's public model/training library is not the complete hosted Roboflow product. The official [hosted training documentation](https://docs.roboflow.com/models/train/train-a-model) describes multiple architectures, cloud orchestration, web configuration, HTTP API, SDK, CLI and MCP. The inspected configuration and merge documentation do not expose per-image/class annotation-coverage declarations or selective partial-label supervision. This is evidence of no documented customer-facing implementation, not proof of the private backend's behavior. Hosted implementation status remains unverified; the confirmed public RF-DETR gap must not be presented as proof that all Roboflow products lack it.

The old `/train/train` URL now returns a page-not-found document; use `/models/train/train-a-model` for current hosted documentation. An additional Exa check is saved in `raw/followup_hosted_partial_annotations.json`.

The issue's April 9, 2025 comments explicitly explain closure as outside the scope of that work. The maintainer said a future implementation would constitute a main result rather than a side feature. No deeper technical or commercial rationale was supplied, and the comments do not establish that the team now wants to adopt an external implementation.
