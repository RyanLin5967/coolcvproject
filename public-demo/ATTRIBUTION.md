# Public demo image attribution

All included dataset images are distributed under the source-declared
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
No endorsement by the original creators, Roboflow, or the dataset hosts is implied.
The exact image inventory and SHA-256 values are in `data/snapshot.json`.

## Chess Pieces

**Joseph Nelson and Brad Dwyer**, Chess Pieces, RF100 / Roboflow version 1.
[Dataset source](https://universe.roboflow.com/roboflow-100/chess-pieces-mjzgj/dataset/1).
[LibreYOLO mirror](https://huggingface.co/datasets/LibreYOLO/chess-pieces-mjzgj),
revision `17e0d3e7c76bea701ad623b0f7b13bec8859ff80`.
Source license and attribution recorded in the downloaded dataset metadata and
`research/roboflow/build_plan/DATA_PROTOCOL.md`.

CoverageCV converts source annotations to COCO and creates a two-pawn projection
and a separate all-13-class task. Training-source class withholding is deliberate;
it does not imply missing source labels. Exact duplicate boxes were removed in
the documented preparation. The shared validation images remain byte-identical.

## Construction Safety

**Anonymous**, original [Worker Safety project](https://universe.roboflow.com/computer-vision/worker-safety),
republished as [Roboflow100 Construction Safety, version 1](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/1).
[LibreYOLO mirror](https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb),
revision `342e545489a6b5f76d6c8225f1ef2629c5a4770a`.
The original creator credit and CC BY 4.0 declaration are preserved in the downloaded
`README.dataset.txt`; `data.yaml` also declares CC BY 4.0.

The upstream export auto-oriented and resized images to 640 × 640. CoverageCV
converts annotations, repairs grouped splits, and deliberately withholds class
families in training. Published validation labels have documented defects and
remain unchanged in the reported metrics. Included image bytes are unchanged.

## Display and selection

Each gallery contains six evenly spaced entries in the existing validation-image
order, including both endpoints. Selection does not use predictions or accuracy.
Chess galleries reuse the same six images. Browser overlays are model predictions
or reference boxes, not edits to image files. Metrics use the entire validation
split; gallery sampling does not change them. No test images are exported.
