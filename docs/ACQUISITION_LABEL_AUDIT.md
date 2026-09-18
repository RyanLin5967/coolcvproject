# Acquired training-label visual audit

**The newly acquired no-helmet boxes contain useful-looking supervision, with no obvious hardhat-versus-no-helmet or whole-body labeling error found in this bounded visual review.** That is not an accuracy result or expert adjudication. The main limitation is correlated scene content: nine guided boxes come from five frames of one visually recognizable camera sequence.

This audit occurred after the 150 image-class queries per arm had been frozen and applied, while training was running. Selection, labels, splits and training were not changed. No model results were read for this audit.

## Exact scope and method

Read only the acquired guided/random learner views' training annotation files, their manifests and `acquisition-audit.json`, plus the frozen protocol. Selected the intersection of `acquired_annotation_ids` and the category named `no-helmet`; sorted by image ID and annotation ID. This is a **census of all 56 arm-specific entries**, not a favorable sample. All six crop sheets were opened and visually inspected, followed by a full-frame sheet for the repeated-scene candidates.

Random's five head boxes also occur in guided, with identical image hashes and coordinates, so the audit covers **51 unique boxes**. IDs are local to each acquired learner view and differ between arms. The overlap mapping is recorded in the evidence.

Training annotation hashes and all inspected image hashes were verified against the respective immutable view manifests. No hidden complete training reference, validation labels, test labels or test images were read. No cloud calls were made.

| Property | Guided | Random |
|---|---:|---:|
| Image-class review units, all classes | 150 | 150 |
| Newly revealed no-helmet boxes | 51 | 5 |
| Images containing those acquired boxes | 21 | 3 |
| Median box width × height at 512 | 34.4 × 39.2 px | 40.0 × 47.2 px |
| Boxes with both dimensions ≤32 px at 512 | 11 | 0 |

These are published training labels revealed by a simulation, **not newly obtained or independently verified human annotations**. More boxes and smaller boxes do not by themselves establish a better trained model.

## Visual findings

- **Geometry generally targets the upper head.** Guided image 175 / ann409–410, image 180 / ann420–423, image 592 / ann1385–1390 and image 734 / ann1743–1746 primarily enclose hair and upper cranium, often excluding much of the face. Several profile or rear views extend farther down the head, for example image 599 / ann1421–1422 and image 656 / ann1542–1544. There is variation in tightness and head extent; this does not look like a uniform whole-face-box policy. The review does not establish the source annotator's intended policy.
- **No clear helmet conflict observed.** Image 487 / ann1141 and image 656 / ann1543 show cap-like headwear labeled no-helmet, consistent with treating ordinary caps separately from hardhats. Image 571 / ann1337 and image 362 / ann853–855 are dark or small, limiting visual confidence. No acquired box was clearly centered on a visible rigid hardhat, an unrelated object or a complete body. This is a visual finding, not a guarantee that every class label is correct.
- **Distinct bytes do not establish distinct scenes.** Guided images **168, 362, 479, 571 and 598** share the same rooftop, buildings, sky and recognizable group of people. Visible timestamp overlays are dated 2018/05/16 and lie around 16:45–16:48. They contribute **9/51 boxes (17.6%)**. The full-frame sheet supports a repeated camera-sequence interpretation; this is not a formal near-duplicate distance study. Count these as five images, not five demonstrated independent scenes. Other guided images visibly include classroom, office, industrial-group, vehicle and nighttime settings, but no formal scene-clustering count was produced.
- **Random shares its entire rare-class acquisition with guided.** Its image 487 / ann1112–1113 correspond to guided ann1140–1141; image 569 / ann1291 corresponds to guided ann1333; image 599 / ann1367–1368 correspond to guided ann1421–1422. Thus the five random head examples are not an independent qualitative comparison of label quality.

No concrete error severe enough to recommend changing this already-running experiment was found in the acquired no-helmet subset. Preserve the frozen cohort and interpret its measured outcome when available. Future acquisition can explicitly diversify scene groups and record box-policy adjudication; neither improvement was retroactively applied here.

## Evidence and identities

[Machine-readable evidence, full hashes and all 56 entries](../artifacts/acquisition-label-audit/evidence.json), [reproduction script](../artifacts/acquisition-label-audit/audit.py), [similar-scene full-frame sheet](../artifacts/acquisition-label-audit/guided-similar-scene-review.png).

Guided sheets: [1](../artifacts/acquisition-label-audit/guided-nohelmet-1.png), [2](../artifacts/acquisition-label-audit/guided-nohelmet-2.png), [3](../artifacts/acquisition-label-audit/guided-nohelmet-3.png), [4](../artifacts/acquisition-label-audit/guided-nohelmet-4.png), [5](../artifacts/acquisition-label-audit/guided-nohelmet-5.png). [Random sheet](../artifacts/acquisition-label-audit/random-nohelmet-1.png). Green boxes are the published acquired references; nearest-neighbor enlargement does not create missing image detail.

- Guided view: `875b4c33d3aa027317b31bd94a970bf955844be1fa025f4e18aca891e2f91467`; training annotations SHA-256 `cad9d21120324e7610fd8380d8d4b135bf61f3aa8d38b5e39a89f4d2fa90b801`.
- Random view: `33655e4f14384bb9c5ca541a7b478da61d174d748c53ed7f11600e3169d10d20`; training annotations SHA-256 `844612db828897ceb371cec8e8c85ac3d78c3aa8b632d9808bbbd1462058612f`.
- Frozen selection-file SHA-256: `0a540ac2c089d12ca176e904d1d7afa60ef7da7fc21bb99bc6add9633c34c02a`.
- Repeated-scene image hash prefixes: 168 `26bfdc80df9f`; 362 `55c5fbba47dd`; 479 `72d6ad4e3919`; 571 `8c92cb90c903`; 598 `93dcde35650d`. Full hashes and exact annotation coordinates remain in the evidence file.
