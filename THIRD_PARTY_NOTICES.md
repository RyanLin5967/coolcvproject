# Third-party notices

`src/coveragecv/training/criterion.py` adapts the unreduced IoU-aware BCE expression
from `src/rfdetr/models/criterion.py` in RF-DETR 1.10.1, commit
`e3fc28795f2a4303069c6b72e80431e5ed716030`.

RF-DETR copyright (c) 2025 Roboflow. Original RF-DETR source also attributes
LW-DETR (Baidu), Conditional DETR (Microsoft), DETR (Facebook) and Deformable DETR
(SenseTime). Licensed under Apache License 2.0. The preserved upstream license is
[here](research/roboflow/raw/build_rfdetr_1_10_1/LICENSE).

Modification: retain the complete upstream cell expression and apply explicit
image/class coverage eligibility before reduction. The matcher, box losses,
normalizer and auxiliary/encoder dispatch are reused from the installed package.

The chess dataset is separate from the code: Joseph Nelson and Brad Dwyer,
RF100 Chess Pieces / Roboflow version 1, declared CC BY 4.0. Derived pilot data
projects black/white pawns, converts annotations to COCO and deliberately
withholds one class per training source. Preserve the source attribution and
transformation notes when sharing derived data. Full provenance is in
`research/roboflow/build_plan/DATA_PROTOCOL.md` and generated manifests.
