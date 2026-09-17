# CV, RF-DETR, tracking, and evaluation research

Research/retrieval date: **2026-09-17**. Six Exa searches completed; raw results are in `raw/cv_models_{rfdetr,supervision,product,competitors,camera,calibration}.json`. Primary documentation, repository releases, issue reports, and papers were then checked directly. No credentials were printed or copied. The user subsequently removed the coding-time limit and prefers infrastructure/backend work while remaining open to CV/ML. Consequently, rankings below do not penalize coding ambition; compute, hardware, experimental access, and annotation requirements remain explicit.

## Main conclusion

The best opportunities in this lane sit **between a good detector and a dependable application**. Roboflow already has much more than a detector: six tracking algorithms, tracking benchmarks, an Optuna tuner, dynamic frame-rate support, mask-assisted tracking, sliced inference, multiple export runtimes, camera-calibration blocks, confidence filters, and event storage. Building one of those again would be a weak pitch.

Three concepts survive strongly: an event-level regression debugger, a serving/export failure minimizer, and a physical camera qualification system. A video tile scheduler has a credible but experimentally risky performance thesis. A calibrated decision-policy compiler is a reserve candidate because of competitive overlap and difficult statistical validation. These are **inferred useful projects**, not claims about Roboflow's internal roadmap or unpublicized shortcomings.

## Evidence ledger

All links retrieved 2026-09-17. `Observed` means the feature/document exists; it does not independently validate vendor performance claims. Issue reports are user reports unless a maintainer response establishes more.

| ID | Source and date | What it establishes | Confidence / implication |
|---|---|---|---|
| E1 | [RF-DETR current docs](https://rfdetr.roboflow.com/latest/), page updated 2026-08-12 | Detection, instance segmentation, and keypoint preview share an API. Nano–Large code/core models are Apache 2.0; XL/2XL detection uses the PML extension. Company-reported benchmarks emphasize accuracy/latency. | High on product surface; company claim on performance. Use open core sizes for an accessible project. |
| E2 | [RF-DETR paper](https://arxiv.org/abs/2511.09554), 2025 preprint / ICLR 2026 per docs | The research is architecture/search-oriented and evaluated on detection workloads. | High. A project should complement this research, rather than promise an unsubstantiated new SOTA model. |
| E3 | [RF-DETR issue #906](https://github.com/roboflow/rf-detr/issues/906), opened 2026-04-01, open when checked | A user reports missed moving metal rods despite motion-blur augmentation and asks about cropping/resizing. They also ask about RGB/BGR conversion. | High that report exists; low on general prevalence or root cause. Supports acquisition diagnostics and input-contract tooling, not a claim RF-DETR is defective. |
| E4 | [RF-DETR FAQ source](https://github.com/roboflow/rf-detr/blob/develop/docs/faq.md), current develop | Query count and postprocessing count are already documented; increasing query slots can add untrained embeddings. Export formats and evaluation APIs are documented. | High. Reject a “fix the 300-object cap” tool based solely on old issues. |
| E5 | [Trackers releases](https://github.com/roboflow/trackers/releases), v2.6.0 dated Aug 6, 2026 | Existing features include Optuna tuning against HOTA/MOTA/IDF1 (v2.4), C-BIoU and configurable IoUs (v2.5), McByte using SAM/Cutie and timestamp-aware updates (v2.6). | High. Generic tracker tuning, variable-FPS handling, and SAM-assisted association are duplicates. |
| E6 | [Supervision releases](https://github.com/roboflow/supervision/releases), 0.25.0 notes | LineZone already added `minimum_crossing_threshold` to reduce double counts from jitter. | High. Do not sell “add debounce to line counting” as a novel project. |
| E7 | [InferenceSlicer](https://supervision.roboflow.com/detection/tools/inference_slicer/) and [SAHI docs](https://obss.github.io/sahi/), current docs | Tiling, overlap, per-tile inference, and merged detections already exist. | High. Ordinary sliced inference is a tutorial, not a project gap. |
| E8 | [Vision Events workflow block source](https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/vision_events/v1.py), current main | Events can contain images, predictions, types, metadata, and operator feedback; cloud and local event-store modes exist. | High. An event dashboard alone duplicates product capabilities; replayable event correctness is a different proposed layer. |
| E9 | [Export-format audit issue #1024](https://github.com/roboflow/rf-detr/issues/1024), 2026; [export docs](https://rfdetr.roboflow.com/develop/learn/export/) | The maintainer audit records real conversion work, including GridSample conversion, preprocessing/mask-decoder parity fixes, and calibration needs. Current docs expose many export paths. | High on historical engineering concerns. Individual “remaining work” entries can become stale; verify current code before promising to fix one. |
| E10 | [Camera calibration guide](https://blog.roboflow.com/vision-ai-camera-calibration/), 2025-12-08 | Roboflow already explains OpenCV calibration and wiring a Camera Calibration block into Workflows. It discusses changes from vibration/repositioning. | High. A checkerboard wizard or homography wrapper is insufficient. |
| E11 | [USG customer case study](https://roboflow.com/case-studies/usg), undated | Company/customer account: local vision measures board dimensions and angles, can reroute/pause a line, and supports consistent quality standards across a manufacturing network exceeding 50 sites. | High on stated use case; vendor/customer claim, not independent ROI measurement. Strong real-user anchor for station qualification. |
| E12 | [Basler image-quality documentation](https://docs.baslerweb.com/optimizing-image-quality), revision released 2026-09-16 | Longer exposure can cause motion blur; higher gain amplifies noise; short exposure should limit moving-object displacement during exposure. | High. Physical image-acquisition tradeoffs must be measured, not represented by brightness sliders alone. |
| E13 | [Confidence threshold guide](https://blog.roboflow.com/how-to-set-a-confidence-threshold-for-object-detection/), 2026-09-03; [per-class confidence block](https://docs.roboflow.com/workflows/blocks/blocks/logic-and-branching/per-class-confidence-filter) | Roboflow already advises choosing thresholds from precision/recall and validating against annotated production images, and supports per-class filtering. | High. A threshold dashboard or configurable filter is redundant. |

## Candidate CV-01 — EventLab: explain and prevent wrong counts, dwell times, and alerts

**Verdict: strongest CV/backend hybrid.** A replay engine that answers, “The boxes look fine, so why did this application count 102 objects when 100 passed?”

**Real user and usefulness.** Roboflow solutions engineers and customers building counting, dwell-time, queue, or inspection applications. They care about correct events and event times, while detector mAP and tracking HOTA capture only parts of that outcome. E5–E8 establish that the necessary detector/tracker/event primitives already exist. Our inference is that a cross-layer event debugger would make those primitives more dependable; no claim is made that Roboflow currently lacks internal equivalents.

**Useful mechanism.** Treat the video pipeline as a replayable sequence of timestamped observations and state transitions. Record detector output, tracker associations, zone state, thresholds, and emitted events. A user marks sparse event truth (object crossed once at roughly this time), without initially labeling every frame. The system finds a minimal failing clip, compares alternative tracker and event configurations, and reports controlled interventions: “Replacing the missed detections fixes this event,” “Using the true identity fixes it,” or “The same trajectories still produce the wrong event because of zone logic.” These are intervention results, not a claim of uniquely identified causality. For clips with dense ground truth, oracle substitutions can separate upstream detection failures from association and event-state failures.

**How it differs from what exists.** Trackers already provides offline metrics and tuning. TrackEval already provides tracking metrics. Supervision already debounces crossings; Vision Events already stores events. The project survives only if its primary unit is a **business event and its minimal reproducer**, with sparse event supervision, state explanations, and reproducible regression cases. It should reuse the existing tracker tuner and evaluators where appropriate. “Trackers with a prettier dashboard” is rejected.

**Narrow first implementation.** RF-DETR → current `trackers` package → `sv.LineZone`; two event types: line crossing and time-in-zone. A local replay bundle contains video or frame references, normalized timestamps, cached detections, pinned config, event truth, and expected outputs. Add deliberate dropped-frame bursts, duplicated frames, timestamp offsets, and detection gaps. Persist deterministic trace IDs separately from tracker IDs, whose numeric values can change across versions. Export a compact regression fixture that runs without a GPU using cached detections.

**Full ambition.** A visual event-state debugger, counterexample shrinking over frames and detection records, event-specific optimization on sparse labels, automatic release comparisons, plus adapter blocks for Roboflow Workflows. Add deployment traces and camera identity later; keep the scientific difference between capture-time, inference-time, and delivery-time explicit.

**Benchmark.** Use [MOTChallenge](https://motchallenge.net/) training sequences or Roboflow's documented tracker datasets with scene-disjoint validation, plus self-recorded conveyor/doorway clips whose event truth is manually checked. Derive crossings from trusted tracks and manually inspect ambiguous boundary cases. Compare default settings, built-in HOTA-tuned settings, and event-optimized settings on held-out scenes. Measure event precision/recall, count absolute error, duplicate-event rate, dwell-time error, first-event delay, minimal-reproducer size, and replay determinism. Report HOTA alongside event results so an event gain cannot conceal identity deterioration. Success is lower held-out event error or faster verified diagnosis; no target improvement is promised in advance.

**Compelling demo.** A side-by-side video has nearly identical boxes and very different counts. Click one wrong count: the tool shrinks a long video to the responsible sequence, highlights an ID switch followed by a threshold interaction, applies the smallest justified change, and runs the regression suite to prove the fix did not break other scenes.

**External constraints.** Dense oracle attribution needs some track labels; sparse event labels alone cannot establish which layer was wrong. No model training is required for the first version. One ordinary GPU accelerates detection caching, but the replay engine itself is CPU work. Video redistribution needs dataset permission; ship manifests and downloader instructions where needed.

**Attack / kill criterion.** Event metrics can be only a thin Optuna objective, and handcrafted toy failures are easy to solve. Kill or merge this if it cannot produce useful minimized diagnoses on independent real clips, or if event tuning merely overfits one camera. Do not invent another tracker or claim new evaluation theory. Its moat is executable, cross-layer debugging and adoption quality.

## Candidate CV-02 — Vision Repro: find the smallest input that breaks model-serving equivalence

**Verdict: strong, especially for the user's infrastructure/backend preference; merge with any equivalent infrastructure-lane proposal.**

**Real user.** RF-DETR/Inference maintainers and engineers deploying a trained detector across Python, HTTP, ONNX, and an edge runtime. E3 supplies a concrete input-color confusion example; E9 supplies historical conversion/preprocessing parity issues.

**Useful mechanism.** Generate **contract-preserving** variants of an input and compare canonicalized output sets across paths. Examples: a PNG loaded by the documented file path versus an equivalent array; permutation of a batch; documented RGB conversion versus explicit conversion; inference before/after a supported serialization path. Instrument checkpoints at decoded pixels, normalized tensors, raw logits/boxes, and final detections. When a difference exceeds a documented tolerance, shrink the test image, input-format combination, model configuration, or exported subgraph while preserving the divergence. Produce a self-contained bug bundle with dependency versions, tensors, output matcher, and a one-command reproduction.

**Important scientific guardrail.** Cropping, brightness changes, and object insertion do not necessarily preserve a detector's predictions. They can expose robustness problems but are not automatically correctness bugs. Separate strict input/serving conformance tests from optional semantic stress tests. Use permutation-invariant detection matching; numerical tie order alone must not fail a case. Quantized runtimes need separately declared tolerances and task-level checks.

**Existing work / collision check.** [MetaOD](https://github.com/MetaOD/MetaOD) (ASE 2020) already applies metamorphic object insertion to detector testing. [GeMTest](https://github.com/tum-i4/gemtest) is a general metamorphic framework. [DeltaNN](https://www.dcs.gla.ac.uk/~josecr/pub/2023_icsme.pdf) studies computational-environment effects, and [Scalpel](https://arxiv.org/abs/2510.21451) tests automotive DL frameworks using generated model components. Consequently, claim no invention of metamorphic or differential testing. The differentiated deliverable is a Roboflow-native **pixel-to-event contract tracer and failure reducer** with maintainer-ready reproducers.

**First implementation.** Pin an Apache-licensed RF-DETR variant and support native PyTorch, ONNX Runtime CPU, and local Inference HTTP. Focus on detection before segmentation. Ten carefully justified properties, canonical tensor capture, bipartite box matching, and delta debugging. Run against historical versions with known, already-fixed regressions to validate discovery before looking for current failures. Add TensorRT/CoreML/TFLite only when real devices/runtimes are available.

**Full ambition.** CI farm accepting new model/adapter/export releases, cross-runtime corpus evolution, automatic bisect over release versions, output-stage fault localization, minimized graph witnesses, and a maintainer review queue. A single discovered and convincingly explained current bug is more valuable than a large wall of synthetic “failures.”

**Benchmark and demo.** Use historical regression cases plus explicit seeded mutations (wrong channel order, normalization, box rescaling, mask stride). Report detection rate by fault class, false alarms on supported paths, time to identify the first divergent stage, reproducer size reduction, and runtime overhead. Demo: a production-shaped portrait image fails only over HTTP or only after export; the tool reduces it and prints an exact tensor-stage mismatch, then a regression test passes after the real fix. Historical replay must be labeled historical; seeded bugs must be labeled seeded.

**External constraints.** Native/CPU paths need no training or special hardware. Genuine accelerator coverage requires actual target hardware and versions; emulation cannot establish device parity. Existing maintainers must ultimately judge intentional versus erroneous behavior. Avoid distributing restricted weights.

**Attack / kill criterion.** A parity benchmark alone is already common. The candidate dies without input reduction, stage localization, and low false-positive rates. It also dies if claims rest only on ordinary quantization differences or unverified metamorphic assumptions. Root should combine this with related export-validation ideas rather than counting them as distinct projects.

## Candidate CV-03 — CameraLab: qualify a vision station and recommend the smallest physical correction

**Verdict: strongest distinctive physical demo; technically useful to field deployment, less purely backend than CV-01/02.** This incorporates the customer lane's transferred station-qualification concept.

**Real user.** An engineer deploying a moving-part inspection or dimensional-measurement system. A model trained on good still images can fail on the actual line. The system should answer, “At what speed, illumination, exposure, and camera alignment is this station empirically reliable?” USG's board-angle/dimension application in E11 is an unusually direct anchor. E3 is a specific report of missed moving objects; E10 and E12 make clear that ordinary camera calibration already exists and optics cannot be ignored.

**Useful mechanism.** An experiment controller varies allowed exposure/gain settings and collects repeatable passes of a known object over measured speed and lighting conditions. It jointly measures detection recall, dimensional/angle error, capture blur, saturation, and latency. It fits an interpretable operating envelope, suggests a minimal change (shorter exposure, more light, resolution/crop adjustment, or recalibration), and marks untested combinations as unknown. A stationary reference target or fiducials detect whether the camera geometry remains inside the validated envelope. The report's validity is linked to camera/lens/settings/model hashes, not just “calibration succeeded.”

**Existing work / collision check.** Roboflow has calibration blocks and a guide; camera vendors already provide exposure controls. [DRL-AE](https://arxiv.org/abs/2404.01636) (CVPR 2024) learns exposure control, [JOCA](https://arxiv.org/abs/2512.06763) (2025 preprint) studies task-driven hardware/control optimization, and [ICCV 2023 autoexposure work](https://openaccess.thecvf.com/content/ICCV2023/papers/Tedla_Examining_Autoexposure_for_Challenging_Scenes_ICCV_2023_paper.pdf) provides a challenging exposure dataset. Do not present learned exposure control as new. The surviving proposition is a **reproducible task-acceptance experiment and portable qualification artifact** integrated with Roboflow, including invalidation when the physical station changes.

**First implementation.** One camera with verified manual exposure/gain controls, one fixed lens, adjustable light, and a repeatable moving target/turntable. Use a known planar part with measured dimensions. Start with standard calibration and classical contour measurement as a serious baseline, then RF-DETR detection/segmentation for the actual object decision. A small model may be fine-tuned if the chosen part is not represented by pretrained classes, but new model research is unnecessary. Export a deployment profile and a validity-monitor block; recommendations are reviewed before camera settings are changed.

**Full ambition.** Camera capability adapters, automated experimental design selecting the next most informative physical trial, cross-camera transfer with recalibration, measurement uncertainty propagation, and a fleet registry that detects outdated station qualifications after lens, firmware, model, or mounting changes. Do not silently control real industrial machinery.

**Benchmark.** Hold out entire capture sessions and combinations of speed/illumination, not neighboring frames. Compare default autoexposure, a simple physics-based short-exposure rule plus brightness control, manual expert settings, and the optimizer. Measure recall by speed/lighting, dimensional/angle error against a physical reference, uncertainty coverage, false pass/false fail rates of the qualification decision, time to qualify, and detection of deliberately introduced camera bumps. Report conditional performance only inside the tested domain. Distinguish predicted from physically measured lighting/speed limits.

**Compelling demo.** A part travels through the scene. Increasing speed makes a confident-looking measurement wrong. CameraLab identifies the validated limit, selects a better capture profile or reports that added lighting is needed, then physically reruns the experiment. A small camera bump immediately invalidates the qualification, and the report shows the resulting dimension error. This is much more persuasive than a synthetic blur slider.

**External constraints.** Requires controllable camera hardware, repeatable motion, reference dimensions/angles, and physical data collection. These cannot be replaced by more coding time. A replay-only mode helps development but cannot validate capture recommendations. Lighting measurements and camera controls vary by hardware. No manufacturing-site access is required for a credible tabletop proof, although commercial generalization eventually needs real stations.

**Attack / kill criterion.** Manual settings and a short-exposure heuristic may solve the demo equally well. The project survives if it provides reliable qualification/invalidation and repeatable evidence across conditions, even if learned control is unnecessary. Kill a version that reports only confidence, uses synthetic blur as its sole evidence, or repackages checkerboard calibration. Do not claim metrology-grade certainty from detector confidence.

## Candidate CV-04 — LookAgain: a deadline-aware small-object video scheduler that still searches for new arrivals

**Verdict: conditional survivor; best performance-oriented research option, with real risk that the simple baseline wins.**

**Real user.** A deployment engineer processing high-resolution cameras where small objects disappear after resizing, while running all image tiles on every frame is expensive. Current RF-DETR small-object/query-count questions and E7 establish demand and existing tools, not a proven universal performance gap.

**Useful mechanism.** Schedule fine-resolution tiles from a combination of uncertain tracks, observed motion, tile age, and coverage exploration. Maintain explicit limits on how long any region may go without inspection, so a scheduler cannot claim speed by repeatedly looking only where it already sees objects. Incorporate scene cuts, camera motion, dense-scene fallback, and admission control under a wall-clock budget. The output includes spatial freshness and recall/cost traces, not just boxes.

**Existing work / collision check.** SAHI and InferenceSlicer are strong baselines. [ROI-Gated SAHI](https://arxiv.org/abs/2608.23923), submitted 2026-08-25, explicitly studies content-based ROI gating: its own 128-image experiment finds static gating slower and less accurate on average, with only slight gain from adaptive routing; favorable large gains come from a tiny sparse-scene case study. Thus “detect coarsely, crop likely objects” is neither new nor reliably faster. The candidate must contribute temporal freshness/exploration and matched-quality, full-pipeline scheduling evidence. A maximum revisit time is a scheduling property, **not a guarantee that all objects will be detected**.

**First implementation.** RF-DETR small/nano, Supervision tile merge, current timestamp-aware trackers, a deterministic scheduler, and fixed-size inference microbatches. Start with one GPU and recorded video to separate algorithm quality from RTSP instability. Include round-robin exploration and always-full-tiling fallback. No new model training is necessary for the scheduling experiment, but a domain-matched detector may require fine-tuning on the selected dataset.

**Full ambition.** Shared budget allocation across multiple camera streams, learned tile-value predictors with explicit exploration, count-saturation diagnostics, queue-aware batching, and energy measurement on an actual edge device. The scheduling and observability/backend layer is substantial even if the predictor remains simple.

**Benchmark.** [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) contains image/video detection and tracking annotations. Use video scene-disjoint splits, plus a second high-resolution scenario and explicit object-entry/camera-motion/density-shift clips. Compare full-image inference, equal-budget uniform SAHI, full SAHI, coarse ROI gating, simple motion gating, and the proposed scheduler. Measure small-object recall/AP, new-object discovery delay, maximum unsampled-region age, dense-scene regressions, and p50/p95 end-to-end latency including proposer, crop, transfers, merge, and scheduling. Compare at matched recall and matched hardware; do not market a gain against an untuned tiling baseline.

**Compelling demo.** A heatmap shows where the system spends compute on a 4K scene. A tiny object enters a previously empty corner; exploration finds it, while a naive ROI gate misses it. As density rises, the scheduler changes strategy and visibly exposes the tradeoff between recall, freshness, and latency.

**External constraints.** Real hardware measurements and several scene types are mandatory; dynamic scenes can erase the benefit. GPU memory/decoding overhead can dominate. A larger model or simple resolution adjustment may beat the scheduler. Data usage terms must be checked before publishing footage.

**Attack / kill criterion.** Reject if a tuned uniform schedule matches the quality/cost frontier, if gains occur only on sparse cherry-picked clips, or if overhead is excluded. Do not spend months on a fancy scheduling algorithm before a small matched-quality experiment shows available headroom.

## Candidate CV-05 — DecisionPolicy: compile a labeled error budget into accept/rerun/review behavior

**Verdict: reserve / requires stronger differentiation; do not treat as equal to the top three.**

**Real user.** A solutions engineer choosing when an inspection result is actionable, when to rerun a larger model/crop, and when to request review. E13 establishes Roboflow's interest in production threshold selection, but much of the foundation already exists.

**Useful mechanism.** Given labeled calibration images and measured costs, choose among accepting the fast result, running a larger/cropped pass, or deferring. Optimize the policy against task-specific errors (for example a false pass) and compute/review costs; produce a versioned policy artifact with held-out error estimates and explicit unsupported slices. Compile the policy into existing Workflows branches rather than inventing another workflow editor. Empty predictions must be eligible for rerun/review; confidence calibration of existing boxes does not handle all missed objects.

**Existing work / collision check.** Per-class thresholds already ship. [PUNCC](https://deel-ai.github.io/puncc/theory_overview.html) already implements boxwise conformal detection, and its [YOLO discussion](https://github.com/deel-ai/puncc/discussions/63) explains that matching true positives does not address false negatives. [Conformal Object Detection](https://proceedings.mlr.press/v204/andeol23a/andeol23a.pdf) and [2026 probabilistic-detection research](https://arxiv.org/abs/2605.07549) further limit any novelty claim. “Conformal boxes for RF-DETR” is too thin. Statistical guarantees under exchangeability do not automatically transfer to a changed camera or factory.

**First implementation.** Two RF-DETR sizes, two inference resolutions, one inspect/pass/review task, separated tuning/calibration/test sets, and a small policy class that is auditable. Compare learned routing to per-class thresholds and always-running the larger model. Add uncertainty intervals on observed task error and explicit invalidation after a model/version/settings change. Do not issue a universal “safe” certificate.

**Benchmark / demo.** Hold out cameras or capture sessions. Measure risk versus automatic coverage, review rate, conditional false-pass rate including images with zero detections, compute cost, and calibration failure under shift. Demo changes the error-cost slider and shows the compiled flow plus actual held-out errors/cost. Improvement must survive against simple threshold/cascade baselines.

**External constraints.** Requires enough independently labeled examples, especially rare failures. Reliable low-error estimates can demand far more data than the UI suggests. No amount of coding fixes an absent failure distribution. Synthetic defects are useful diagnostics but cannot establish real-world failure rates.

**Attack / kill criterion.** Most of this can collapse into an Optuna search over thresholds and branches. Reject unless the project handles omitted objects, produces trustworthy held-out cost/risk curves, and makes materially better decisions than simple baselines. This is best absorbed into EventLab or CameraLab rather than promoted as a separate flagship.

## Explicitly rejected concepts

| Tempting concept | Why it fails this research bar |
|---|---|
| RF-DETR custom fine-tune with a clean detection demo | Official tutorials already cover it; it does not establish a reusable mechanism or company-specific usefulness. |
| General tracker benchmark / hyperparameter tuner | Trackers already has metrics, dataset workflows, comparisons, and an Optuna tuner. |
| Add timestamp-aware tracking for dropped frames | Trackers 2.6.0 already supports timestamp-based dynamic frame rate. |
| Stabilize tracking with SAM/Cutie masks | McByte already does this. A project would need a narrower demonstrable improvement beyond association. |
| Add line-crossing debounce | Supervision already has a crossing threshold; known past bugs are not current product gaps. |
| Plain small-object sliced inference | InferenceSlicer and SAHI already provide it. |
| Increase RF-DETR's 300 query slots / dense-count “fix” | Current FAQ covers it; extra slots may start untrained. A controller based only on output counts cannot reliably know how many objects were missed. |
| Basic camera calibration / perspective correction wizard | Existing Roboflow blocks and tutorial cover this. Physical qualification/invalidation is the worthwhile extension. |
| Cross-runtime model benchmark with a leaderboard | Useful evidence, but no novel useful mechanism unless it localizes/reduces failures or meaningfully selects deployment behavior. |
| Confidence calibration visualization alone | Existing libraries and per-class filters cover much of it; misses with no predicted box remain unaddressed. |

## Recommended handoff to synthesis

1. **EventLab** is the best standalone hybrid: practical backend replay systems, meaningful CV understanding, no expensive training prerequisite, and a demo that explains real application failures.
2. **Vision Repro** is a strong maintainer-facing infrastructure project; merge with export/serving candidates from other lanes and make failure minimization the distinguishing mechanism.
3. **CameraLab** is the most distinctive physical-world project; it has a real customer anchor and must be judged by measured station outcomes, not ML novelty.
4. **LookAgain** deserves a research-spike option. Its pitch becomes strong only after a rigorous baseline experiment demonstrates real headroom.
5. **DecisionPolicy** should be held in reserve or merged into another project. It should not crowd the user's final shortlist simply to increase idea count.

No performance improvement, private company pain point, research novelty, or hiring outcome has been established by this research alone. The sources justify plausible users and integration paths; each project's kill experiment is the next evidence needed.
