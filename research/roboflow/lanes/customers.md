# Customer and production-deployment research

Researched 2026-09-17. Six Exa searches; primary customer case studies, current official documentation, and customer-maintained source code checked. Later user steering removes the original 4–6 week constraint and prioritizes backend/infrastructure roles while retaining CV/ML ideas. Ambition is therefore a choice, not a rejection criterion by itself.

## What the evidence says

The strongest company-usefulness evidence is the **Forward Deployed Engineer job description**, rather than a speculative claim that Roboflow lacks a detector. The role takes an already validated customer proof of concept into production, deals with physical deployment constraints, and is explicitly expected to turn repeated field work into reusable artifacts. That makes a technically rigorous acceptance tool potentially useful to Roboflow itself and its customers. We have no evidence of Roboflow's private roadmap or whether it already has internal equivalents.

| Source | Observed facts / source's claims | Interpretation and confidence |
|---|---|---|
| [FDE job](https://jobs.ashbyhq.com/roboflow/444ee288-cb72-4751-b16d-67c27749e901), Exa date May 18, 2026 | Employer describes initial engagements of 4–12 weeks; data pipelines, local hardware, unreliable connectivity, lighting, camera calibration, drift, latency, handoff/runbooks, reusable deployment patterns. | High confidence about advertised work. This is direct evidence for an engineering artifact that reduces first-deployment uncertainty. Not proof of an unresolved internal defect. The ordinary browser view is JS-only; full text was obtained through Exa. |
| [USG](https://roboflow.com/case-studies/usg), undated | Customer testimonial describes detecting board dimensions/angles before jams, integrating with reroute/pause machinery, offline local inference, and consistent quality measurements across more than 50 manufacturing sites. | Strong task specificity. Benefits are vendor-published customer claims, not independently audited ROI. Supports physical outcome validation and camera measurement qualification. |
| [BNSF](https://roboflow.com/case-studies/bnsf), undated | Yard inventory and train-wheel inspections; quoted executive emphasizes scaling across an operating network without disruption. | High confidence about described use cases; no public technical incident data or architecture. Do not extrapolate a rail-safety project from this. Logistics inference state is a safer reusable research direction. |
| [FloVision](https://roboflow.com/case-studies/flovision), undated | Food yield, quality and operator-feedback applications; local/cloud choices; vendor reports 20 million pounds analyzed and one customer's trimming speed improvement. | Shows production outcomes matter beyond model accuracy. Do not repeat carbon-impact statements or infer financial savings from a home demonstration. |
| [Fletcher Sports](https://roboflow.com/case-studies/fletcher-sports), undated, includes Wimbledon 2024 image | Vendor says 56 cameras across 14 courts; over 20 FPS per stream and local operation without internet. | Supports multi-stream timing and operational resilience. Already a sophisticated deployed system: another player detector does not add much. |
| [PlayVision](https://roboflow.com/case-studies/playvision), undated | Court coordinates/events plus active-learning selection using missing balls, unexpected player counts and confidence anomalies. | Explicitly rejects generic sports active-learning or stats-dashboard proposals as differentiated work. |
| [Statsyuk](https://roboflow.com/case-studies/statsyuk), undated | Single-camera hockey pipeline with 14 specialized models; occlusion, player identity and downstream propagation of detection errors. | Supports pipeline-level evaluation, but a complete competitor would be excessive product scope without privileged data. |
| [Field Museum](https://roboflow.com/case-studies/field-museum), undated | Whole-drawer specimen digitization combines Roboflow models and LLM transcription through the open DrawerDissect project. | Especially valuable: actual public customer code and sample data permit a concrete contribution without enterprise access. |

All source pages retrieved on the research date. Case studies establish what customers describe doing; they do not establish that Roboflow currently needs our proposed implementation.

## Important overlap discoveries

1. **Calibration is already a feature.** [Camera Calibration block](https://docs.roboflow.com/workflows/blocks/blocks/transformations/camera-calibration), [Perspective Correction block](https://docs.roboflow.com/workflows/blocks/blocks/advanced-blocks/perspective-correction), and [calibration guide, Dec 8, 2025](https://blog.roboflow.com/vision-ai-camera-calibration/) cover lens correction and coordinate transforms. A checkerboard UI or homography wrapper is not a survivor.
2. **PLC integration is already a feature.** [Aug 3, 2026 PLC integration guide](https://blog.roboflow.com/computer-vision-plc-integration/) covers timing, duplicate rejection, heartbeats, acknowledgements, and records. Exa also retrieved PLC Relay and PLC Writer docs; the latter's old URL returned 404 when opened directly. Treat the live guide as the public source, and validate current block entitlement before implementation. Enterprise PLC access must not become an MVP dependency.
3. **Events and fleet management already exist.** [Vision Events](https://docs.roboflow.com/deploy/vision-events) stores important outcomes plus visual evidence. [Deployment Manager](https://docs.roboflow.com/deployment/self-hosted/enterprise/deployment-manager) covers provisioning, cameras, deployment and telemetry. [Model Monitoring docs](https://docs.roboflow.com/deployment/monitoring-and-analytics/model-monitoring) say it will be replaced by Vision Events. Another fleet dashboard/event store is weak.
4. **Simulators already exist.** [Open Industry Project](https://www.openindustryproject.org/) provides a browser/desktop industrial simulator with Structured Text. [EngineerHub conveyor timing simulator](https://myengineerhub.com/tools/conveyor-travel-time) appears in Exa; direct browser fetch failed. [A public smart-factory project](https://github.com/js-deepakgiduthuri/smart-factory-vision-public) already combines conveyor video, detector, virtual PLC and SCADA. Reuse simulation components; do not claim the first simulated factory.
5. **Sports tooling is substantially beyond ByteTrack demos.** [Roboflow sports](https://github.com/roboflow/sports) explicitly lists re-identification and calibration challenges. [Roboflow trackers](https://github.com/roboflow/trackers) includes several trackers, camera motion compensation, benchmarking and tuning. A benchmark-and-toggle UI is not novel enough.
6. **Inventory uncertainty is in existing positioning.** [Inventory cycle counting page](https://roboflow.com/ai/inventory-cycle-counting) explicitly describes WMS reconciliation, photo evidence, partial-view confidence, and requesting another capture when exact count is unsupported. This is marketing positioning, not proof of a generic API, but it still kills claims that Roboflow has overlooked uncertain counts.

## Candidate C1 — Conveyor acceptance lab

**Verdict: strong survivor; strongest fit for backend/infrastructure with an immediately legible physical demo.**

**One-line pitch:** A test bench that proves whether a vision workflow rejects the correct physical item under latency, disconnects and restarts—even when its detector accuracy looks excellent.

**Real user and need:** Roboflow FDEs, implementation engineers, and customers commissioning an inspection line. A model can correctly classify every observed frame while the integration acts on the wrong item because an old result arrives after the line advances. The FDE job and USG's action-integrated inspection establish relevance; existence of this exact unmet need inside Roboflow is an inference.

**Reusable artifact:** A headless acceptance runner, deterministic scenario format, simulator adapter, input/output adapters for local Roboflow Inference/Workflows, and an evidence report. A tabletop line is the demonstration fixture, not the entire product.

**Narrow MVP:** One lane, one inspection point, one reject station, one item family, and local inference. Use a discrete-event simulation or existing simulator for exact ground truth. Run the same detector trace through two integration policies: a reasonable frame-based baseline, then an inspection-ID/deadline-aware protocol. Keep PLC sequencing in the controller; the vision layer supplies evidence and decisions. Add a local protocol adapter or a simulated HTTP sink so a paid enterprise PLC block is optional.

**Test model:** Keep physical item identity, frame identity, observation timestamp, receipt timestamp, inference completion, decision expiry, PLC acknowledgement and physical outcome separate. Correlate those into one per-item timeline. Do not confuse a successful tag write with successful removal. Unknown actuation after a crash remains unresolved until reconciled; do not promise exactly-once physical actuation.

**Fault scenarios:** Delayed inference response; out-of-order frames; duplicated callbacks; dropped frame; frozen camera; wrong capture timestamp; process restart while an item is in flight; reconnected camera with reset frame counter; controller unavailable; lost acknowledgement; belt stop/change of speed; stale positive reject bit. Start with a small documented set and expand systematically. Expose deterministic seeds and retain failing traces.

**Useful assertions:** An expired result cannot act on the following item. An unseen item cannot count as inspected. Every input item ends in accepted, rejected, manual-review or unresolved state. A controller acknowledgement alone cannot satisfy a physical-outcome assertion. Restart cannot reuse an old inspection namespace. Abstention/manual review consumes a stated operational budget.

**Accessible data/hardware:** Generate simple marks/missing-component examples on self-owned objects, with real camera recordings as a second stage. A laptop can run the simulator and replay. For a physical version use a low-energy tabletop conveyor, USB camera, simple parts, and an independent outcome sensor or second view. Hardware cost requires a later bill of materials, not a fabricated price. Industrial-speed, washdown, EMI and PLC reliability are outside what this setup establishes.

**Ground-truth discipline:** In simulation, the simulator knows physical identity independently of the vision model. In a real rig, use an independent identifier/manifest or outcome sensor not exposed as the detector's shortcut. Do not use the same tracker under test to produce the ground-truth item mapping. Hold out some sequences and fault combinations from development.

**Measurement:** Correct-item rejection rate; accepted defective items; good-item false rejection; duplicate/misdirected actions; unresolved/manual-review rate; deadline miss distribution; detection-to-confirmed-outcome p50/p95/p99; recovery time; acceptance-run reproducibility. Report denominators and confidence intervals. Compare the same visual predictions, so gains demonstrate systems correctness rather than a changed detector. Later compare model versions on end-to-end outcomes too.

**Remarkable demo:** Two synchronized conveyors or split-screen replays share the same highly accurate detector. Turn a latency knob and inject a restart. The baseline rejects a good item while the faulty one escapes. The improved integration discards the expired decision and visibly accounts for the affected item according to its configured fallback. A click on either object opens the exact trace explaining the action. A single command regenerates the acceptance report.

**Adversarial attack:** This could be a generic digital twin, an overengineered fault simulator, or an implementation of already documented PLC practices. It survives only as an executable acceptance package with real item outcomes, reusable adapters, and a measured baseline. Open Industry Project and Roboflow's guide already cover much of the scaffolding. Do not claim certification, hard real-time guarantees, or industry savings from a toy line. If no plausible integration fails under the controlled suite, the project has not demonstrated its usefulness.

**Ambition ladder:** (1) deterministic offline acceptance runner; (2) real Workflow/inference integration plus virtual controller; (3) repeatable tabletop validation; (4) hardware-in-the-loop tests and customer-operable runbook; (5) multi-station/rework loops and minimally reproducing a failed deployment trace. Each level yields a useful artifact independently.

## Candidate C2 — CuratorPatch: human corrections that survive model reruns

**Verdict: survivor with a narrow, unusually concrete customer-code entry point. Less broad company impact than C1; excellent backend/data-lifecycle engineering.**

**One-line pitch:** Version and safely rebase human-approved records when a CV pipeline changes the identities, boundaries or ordering of the things it detects.

**Real user:** Field Museum/other DrawerDissect curators, then any Roboflow user converting densely packed images into durable records. The difficult operation is preserving a human correction when the upstream detector finds an extra object or a crop splits—not writing another OCR prompt.

**Current implementation evidence:** The [advanced functions documentation](https://github.com/EGPostema/DrawerDissect/blob/main/advanced_functions/README.md) uses appendable curation CSVs and approval gates. [crop_specimens.py](https://github.com/EGPostema/DrawerDissect/blob/main/functions/crop_specimens.py) orders detections spatially and names specimens by index. [scaffold_locations.py](https://github.com/EGPostema/DrawerDissect/blob/main/advanced_functions/scaffold_locations.py) skips existing specimen IDs, reports absent IDs, and offers a destructive full refresh with confirmation. Existing flags, pending exports and manual review are real features; do not pretend they are missing.

**Controlled research probe:** `raw/customers_identity_probe.py` creates a tiny red/green synthetic image, supplies old/new prediction arrays, and calls upstream crop, scaffold and GBIF export functions unchanged; only logging is stubbed. Adding the previously missed left object changes `spec_001` from green B to red A. The old `spec_001` approval is retained, only `spec_002` is added, and no orphan is reported. The actual GBIF export function then joins B's synthetic catalog number/location to A's current image path; B's new ID is pending. The supported `--rerun` path clears selected image outputs but leaves the top-level curation files. This establishes a **controlled synthetic failure mechanism**, not a production incident, real-model behavior, or prevalence estimate. Separate output folders emulate regenerated crops; the entire CLI and inference stack were not run. Results are in `raw/customers_identity_probe_result.json`.

**Immutable code evidence at commit `3823276df6b5afe3184af13b9701c3d17f440c96`:** [positional crop numbering](https://github.com/EGPostema/DrawerDissect/blob/3823276df6b5afe3184af13b9701c3d17f440c96/functions/crop_specimens.py#L113-L156); [skip previously curated IDs](https://github.com/EGPostema/DrawerDissect/blob/3823276df6b5afe3184af13b9701c3d17f440c96/advanced_functions/scaffold_locations.py#L92-L147); [rerun output clearing](https://github.com/EGPostema/DrawerDissect/blob/3823276df6b5afe3184af13b9701c3d17f440c96/process_images.py#L352-L390); [rerun call path](https://github.com/EGPostema/DrawerDissect/blob/3823276df6b5afe3184af13b9701c3d17f440c96/process_images.py#L581-L587); [GBIF identity joins and gates](https://github.com/EGPostema/DrawerDissect/blob/3823276df6b5afe3184af13b9701c3d17f440c96/advanced_functions/gbif_format.py#L140-L228). These paths check catalog number, approval and taxonomy; they do not compare source/crop hashes or old/new physical coordinates before using an existing approval. This is a bounded code finding, not a claim of absent safeguards throughout every museum workflow.

**MVP:** Import two immutable DrawerDissect result bundles plus a curated CSV; associate objects using source-image identity and coordinates; generate a three-way diff between prior machine output, human corrections and new machine output. Keep content-addressed source/crop provenance. Preserve a correction only when the underlying entity and relevant evidence remain matched. Route splits, merges, unmatched objects and conflicting changed fields to a visual review queue. Export current approved records in existing formats; do not replace DrawerDissect's digitization pipeline.

**Technical depth:** Versioned entity graph, durable edit log, field-level provenance, selective invalidation, transactional rebase/export, reproducible run manifests, and explicit conflict semantics. A useful API is `rebase(previous_run, human_edits, next_run)` returning carried, invalidated and ambiguous edits with evidence.

**Data/access:** Public [DrawerDissect sample images and models](https://github.com/EGPostema/DrawerDissect), plus self-made labeled specimen-like trays for known identity/correction ground truth. Check each dataset/model's license before redistributing; public access alone is not a license. A laptop and existing image files suffice. Human evaluation can begin with readable printed labels; domain-expert validation remains necessary for scientific metadata claims.

**Validation:** Seed added/missed objects, reorderings, crop shifts, splits/merges and changed OCR fields. Measure false correction transfers, correct edits retained, unnecessary re-review, stable-identity match precision, complete conflict detection, export reproducibility, and reviewer time against existing CSV review. Wrong transfers are more important than merely maximizing automatic carryover. A real handwritten-label benchmark is a later extension, not required to establish rebase correctness.

**Remarkable demo:** Approve a dense drawer's records, switch the detector to a version that reveals a previously missed insect, and show the IDs rearrange while approved facts stay attached to the correct physical specimens. Ambiguous pairs become side-by-side review cards; one click traces an exported field to its source pixels and human edit.

**Adversarial attack:** Generic data versioning tools already exist, and a full visual annotation UI would be redundant. This survives only if source-aligned entity rebasing materially reduces rework or prevents wrong transfers on a supported real pipeline. If upstream stable identity already solves the complete rerun flow, contribute a small patch/test instead and abandon the large platform. No maintainer contact or acceptance has occurred; practical demand remains unvalidated.

## Candidate C3 — Station operating-envelope qualification

**Verdict: survives as part of the CV-model lane's operating-envelope proposal; transferred there to avoid duplicate recommendations.**

Use USG-style board dimensional checks and the FDE's camera-calibration responsibility as the user/problem anchor. Measure acceptable operating regions across camera position, motion, exposure and lighting, including geometric measurement error and abstention after camera displacement. Export an evidence-backed station configuration rather than another homography/calibration tool. A ruler/printed target, controllable motion and an inexpensive camera are sufficient for initial validation. [Basler's image-quality documentation](https://docs.baslerweb.com/optimizing-image-quality) provides a primary-source anchor for exposure/motion tradeoffs. Main attack: ordinary calibration and adjustment guidance already exist; the unique artifact must be empirical acceptance coverage and automated acquisition improvement. The CV lane owns the full candidate and further alternative checks.

## Candidate C4 — Uncertainty-preserving inventory ledger

**Verdict: reject as a standalone final recommendation at present; could be a later C1 scenario pack.**

Proposed user: logistics FDE building BNSF-like yard inventory. Integrate Inference/trackers/Vision Events with a small event-sourced inventory service that preserves ambiguous counts through blind spots, missed observations, duplicated messages and restarts. MVP: two tabletop zones, identical boxes, known intake manifest, and a camera blackout; retain count ranges and request the most informative extra capture. Validate count error, interval coverage/tightness, false certainty and reconciliation workload. Demo: unplug a camera; the system preserves uncertainty instead of inventing disappearances.

Why it fails the attack: Roboflow already markets evidence-backed reconciliation and abstention for partial counts. Ordinary event sourcing adds limited novelty, while genuinely robust multi-camera inference introduces a large perception problem without realistic deployment data. Public evidence does not establish a specific remaining product gap. Keep the rigorous uncertainty scenario inside the acceptance lab instead of pitching an inventory platform.

## Candidate C5 — Sports identity repair workbench

**Verdict: reserve, not strong enough for the final survivor list without a sharper algorithmic result.**

Proposed user: sports analytics developers such as Statsyuk. Detect likely identity swaps from jersey/team/court constraints and propagate one human identity correction across a temporal graph; integrate the current Roboflow trackers and sports repositories. MVP: short, permitted sports clips with manually audited identity ground truth. Compare identity errors and reviewer interventions/time with tracker-only output and basic manual track merge. Demo: correct one mistaken identity and watch player statistics repair across a whole possession.

Why it currently fails: tracking and re-identification are well-populated fields, Roboflow's repositories explicitly target them, and a correction interface alone risks becoming a generic annotation product. Unknown demand and licensing of sports footage compound the problem. It would need a credible constrained-association algorithm, reproducible public benchmark and evidence of large reduction in review effort. No claim of unique invention is justified.

## Candidate C6 — Retail SKU/packaging migration assistant

**Verdict: reject.**

Proposed user: retail deployer adapting a model to new package designs. Match new packaging to old SKU classes, choose retraining samples and test changed/unchanged items. MVP could use self-photographed pantry goods plus [Unitail](https://unitedretail.github.io/) or [Retail-786k](https://arxiv.org/abs/2309.17164) after license/access checks; Roboflow dataset/inference integration; validation would require held-out package designs and false-merge rates. Demo: swap the package design and regenerate a small deployment update.

Why it fails: product matching datasets and methods already exist, the straightforward version is embeddings/OCR plus active learning, and proving business value needs an actual changing SKU catalog. It also overlaps the data lane. Packaging change alone is not enough differentiation.

## Candidate C7 — Food-yield optimization from video

**Verdict: reject the standalone vertical app; retain business-outcome framing.**

Proposed user: FloVision-like food processor; segment product/trim and choose threshold policies balancing yield, false rejection and throughput. A home version could use cut produce, a kitchen scale and permitted camera data; integrate RF-DETR/segmentation output into measurements, compare measured waste mass and time, and show an overtrim-reduction demo.

Why it fails: the credible product requires process-specific measurement, experts, conditions and customer economics that a home setup cannot establish. Inferring mass or profit from 2D pixels is weak without calibration. FloVision already sells this solution, so the result risks an inferior customer competitor rather than a reusable Roboflow engineering tool. The useful general lesson is to evaluate complete operational outcomes, which C1 does in a tractable setting.

## Final rank and handoff

1. **C1 Conveyor acceptance lab:** strongest overall in this lane; reusable systems artifact, hard failure modes, compelling causal demo, directly aligned with the advertised FDE workflow and backend target.
2. **C2 CuratorPatch:** strongest low-hardware alternative; concrete open customer code, reproduced synthetic data-lifecycle failure mechanism and meaningful lineage/backend work. Validate a complete supported CLI rerun with real sample predictions and user workflow before expanding scope.
3. **C3 Station qualification:** compelling CV/systems option, merged into the CV lane; do not count twice.

C4–C7 are retained as rejection audit, not ideas to pad the final shortlist. A broader project need not win by owning more components: C1 and C2 stand out through the quality of their correctness evidence.

## Research artifacts

Exa: `raw/customers_manufacturing.json`, `raw/customers_retail_sports.json`, `raw/customers_fde.json`, `raw/customers_calibration.json`, `raw/customers_plc.json`, `raw/customers_museum.json`.

Source inspection: `raw/customers_drawerdissect_advanced_readme.md`, `raw/customers_drawerdissect_locations.py`, `raw/customers_drawerdissect_barcodes.py`, `raw/customers_drawerdissect_crop_specimens.py`.

Controlled probe: `raw/customers_identity_probe.py`, with result `raw/customers_identity_probe_result.json`. Further pinned source snapshots: `raw/customers_drawerdissect_gbif.py`, `raw/customers_drawerdissect_emu.py`, `raw/customers_drawerdissect_process.py`, `raw/customers_drawerdissect_merge.py`. Read upstream snapshot retrieved on 2026-09-17; GitHub tree at inspection was `3823276df6b5afe3184af13b9701c3d17f440c96`. This was research only; no upstream files, services or museum records were changed.
