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

- Optional inference ensembles use [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion), `ensemble-boxes==1.0.9`, MIT license. Solovyev, Wang and Gabruseva, *Weighted boxes fusion: Ensembling boxes from different object detection models* (2021), https://arxiv.org/abs/1910.13302.


## SAM2.1 experimental boundary refinement

The rejected segmentation-refinement experiment uses Meta’s SAM2.1 Large pretrained weights through Hugging Face Transformers, pinned to revision `665f8e2ad61cf5f53d65644ff27c8ee525124610`. Meta licenses the model checkpoints under Apache-2.0: https://github.com/facebookresearch/sam2. The checkpoint is downloaded into the disposable cloud image and is not redistributed in this repository. The refinement/calibration implementation in `segment_refinement.py` is original. SAM2 is not part of the published predictor because its training-only acceptance test failed.
