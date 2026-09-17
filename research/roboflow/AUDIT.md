# Research workflow and selection audit

Date: September 17, 2026. User requested broad research with subagents, Exa, many categories, and adversarial selection. Subsequent guidance: infrastructure/backend priority; retain CV/ML; do not optimize for coding duration. All final ideas follow that guidance.

## What was actually done

1. Located the Exa setup by inspecting `~/projects/dbresearch/scratchpad_exa.sh`. It references `~/.config/exa/keys`. Created a local helper that reads credentials internally and sends them only to the Exa API; it never prints or copies them into research files.
2. Ran seven independent research lanes, plus root-level company/competitor checks. Used Exa discovery followed by primary-source validation: company documentation, employer-authored live job descriptions, repositories, current issue/PR state, source code, papers and competitors' official materials.
3. Required each lane to propose named candidates and attack its own proposals against existing capabilities, integration access, actual users and measurable demonstrations.
4. Ran separate novelty and technical critics. They introduced additional competing work, rejected invalid assumptions, required strong baselines and mandated merges.
5. Updated the final cards after critique. Support replay, runtime parity and release infrastructure became one conformance project. Event-state diagnosis and streaming fault injection became one EventLab with two explicit execution modes. Capacity planning and runtime scheduling remain distinct: one recommends provision/configuration; the other enforces online policy.
6. Built a standalone comparison page and detailed report from one structured set of twelve cards. No proposed flagship project was implemented during research.

The raw log records **71 Exa queries, 447 result records and 383 unique result URLs**. These are discovery counts, not all-verified-source counts. [Search log](search_log.json), [counts and definitions](research_stats.json).

## Research coverage

| Lane | Named candidates examined | Main deliverable |
| --- | ---: | --- |
| Company strategy and internal needs | 6 | [company.md](lanes/company.md) |
| Inference, edge and runtime | 7 | [inference_edge.md](lanes/inference_edge.md) |
| Data platform and composition | 7 | [data.md](lanes/data.md) |
| CV, models and video | 5 | [cv_models.md](lanes/cv_models.md) |
| Workflows and developer experience | 6 | [workflows_devex.md](lanes/workflows_devex.md) |
| Customer applications | 7 | [customers.md](lanes/customers.md) |
| Infrastructure and security | 6 | [infra_security.md](lanes/infra_security.md) |
| **Total before consolidation** | **44** | Additional generic pitches appear in each lane's rejection list |

Independent critiques: [novelty](critiques/novelty.md), [technical validity](critiques/technical.md). These reviews are independent analytical passes over shared evidence; they are not independent customer interviews or external validation.

## Disposition of the 44 named lane candidates

Final IDs refer to [the twelve cards](REPORT.md#the-twelve-survivors). A merge is not a claim that the underlying idea was bad; it prevents counting components of the same system as separate choices.

| Lane / candidate | Disposition | Reason |
| --- | --- | --- |
| Company C1 incident-to-regression fixtures | Merge → 01 | Capture/reduction/adoption layer of the conformance engine |
| Company C2 visual reward falsification | Survive → 12 | Visual/temporal independent-truth specialization; research risk remains |
| Company C3 commissioning/handoff | Merge → 02, 08 | Event acceptance and physical camera qualification are the concrete mechanisms |
| Company C4 rubric-change shipment revalidation | Reserve; exclude | Real workflow/demand and impact-detection advantage unvalidated; related semantics can extend 03 |
| Company C5 support/docs chatbot | Reject | No evidence retrieval is the bottleneck; weak diagnostic validity |
| Company C6 model bakeoff service | Reject | Roboflow Playground and evaluation tooling already cover the generic concept |
| Inference E1 deployment-contract release lab | Merge → 01 | Same release/conformance kernel |
| Inference E2 semantic video chaos | Merge → 02 | EventLab fault mode |
| Inference E3 event-quality capacity planner | Survive → 09 | Distinct configuration/provisioning decision; must validate predictions |
| Inference E4 physical timing/compatibility rig | Merge → 02 | Hardware acceptance fixture, not a separate platform |
| Inference E5 cache-aware load isolation | Reject standalone; related to 07 | Preload/pinning and active async-loading work exist; runtime idea narrowed to dynamic fanout |
| Inference E6 offline dependency certifier | Reject standalone | Offline mode, resource discovery and preloading already exist; manifests support 01 |
| Inference E7 dynamic-workflow fair admission | Survive → 07 | Specific fanout/order/byte accounting mechanism; empirical performance gate |
| Data 1 coverage-aware composition | Survive → 03 | Concrete user/maintainer evidence and meaningful data/training semantics |
| Data 2 visual-data provenance/minimizer | Survive → 04 | Training-target corruption across transforms, localized and reduced |
| Data 3 fleet acquisition | Survive → 11 | Distinct delayed-feedback resource problem; learning benefit must be proven |
| Data 4 annotation-policy migration | Reserve; exclude | Crowded ontology/versioning area; possible 03 extension |
| Data 5 leakage repair | Reserve; exclude | Existing leaky-split and group-split tools; useful 03/04 module |
| Data 6 ensemble label dashboard | Reject | Native Workflow auto-labeling and adjacent quality tools already cover it |
| Data 7 Universe provenance/license score | Reject | Cannot infer ownership/permission or complete lineage from public overlap |
| CV 1 EventLab | Merge → 02 | Fixed-observation perception/event diagnosis mode |
| CV 2 serving/export minimizer | Merge → 01 | Pixel/model/runtime comparator and reducer within conformance |
| CV 3 CameraLab | Survive → 08 | Task-conditioned physical qualification and invalidation |
| CV 4 LookAgain tile scheduler | Survive → 10 | Exploration/freshness mechanism; needs matched-quality result |
| CV 5 decision-policy compiler | Reserve; exclude | Risk of thin threshold/conformal wrapper; possible 02/08 extension |
| Workflows D1 conformance generator | Survive → 01 | Strongest direct company fit |
| Workflows D2 lineage debugger | Survive → 06 | Distinct developer diagnosis choice; merge into 01 if no independent usability gain |
| Workflows D3 application release gate | Merge → 01, 02 | Application/event contracts, not a separate runner |
| Workflows D4 model integration contract forge | Merge → 01 | Integration family/adoption module |
| Workflows D5 dependency linker | Merge useful parts → 01 | Existing resource discovery; exact provenance useful but not a flagship |
| Workflows D6 tail-latency explorer | Merge useful parts → 07, 09 | Existing profiler baseline; diagnosis must justify an intervention/prediction |
| Customers C1 conveyor acceptance lab | Merge → 02 | Strong physical/event fixture for the fault laboratory |
| Customers C2 curator correction rebase | Survive → 05 | Concrete open code and reproduced synthetic failure mechanism |
| Customers C3 station qualification | Merge → 08 | Same physical operating-envelope mechanism |
| Customers C4 uncertain inventory ledger | Reject standalone | Existing positioning, weak differentiated mechanism; useful 02 scenario |
| Customers C5 sports identity repair | Reserve; exclude | Crowded tracking/re-ID and annotation tools; unproven algorithmic/user gain |
| Customers C6 SKU migration | Reject | Generic embeddings/OCR/active-learning combination and inaccessible realistic catalog needs |
| Customers C7 food-yield app | Reject | Customer already addresses it; home demonstration cannot establish process economics |
| Infra 1 release observatory | Merge → 01 | Same release/conformance kernel |
| Infra 2 camera-to-action fault lab | Merge → 02 | Same event/fault kernel |
| Infra 3 support capsules | Merge → 01, 02 | Reproduction/reduction feature, not another platform |
| Infra 4 event-cost optimizer | Merge → 09 | Constrained capacity/economics mode; ratio alone is a flawed objective |
| Infra 5 dataflow/egress checker | Reserve; exclude | Current controls exist; arbitrary code and incomplete taint semantics defeat broad assurances |
| Infra 6 secret-free integration environment | Merge → 01 | Explicit contribution path and useful adoption foundation; not enough standalone differentiation |

## The strongest adversarial findings

- **Roboflow already has substantial testing.** Main includes model, Workflow, HTTP and stream tests; some recent instrumentation and scheduling work is ahead of released builds. Final proposals extend coverage and diagnosis rather than assert testing is absent.
- **Model parity and reduction already exist elsewhere.** Polygraphy narrows the conformance opportunity to meaningful CV graph/API/event semantics and useful workflow/input reduction.
- **Video tooling is mature.** Trackers already offers metrics/tuning and timestamp support; DeepStream offers replay and scheduling controls. EventLab must add event truth, controlled interventions and useful counterexamples.
- **Basic data quality is crowded.** FiftyOne, Cleanlab, Datumaro, Encord and Roboflow cover many obvious ideas. Unknown-versus-absent label semantics and downstream human-edit identity are more distinctive integration problems.
- **Scheduling cannot evade overload.** It must report infeasible workloads, preserve state/side-effect order and include memory allocation before fanout. Simple fair queues, EDF and caps are serious baselines.
- **Physical validation is irreducible.** Camera exposure/lighting and actuation claims need physical ground truth. A simulator or confidence score cannot establish the real measurement error.
- **Eval testing is not new.** evalmut, Inspect/Harbor, multimodal reward-hacking work and hack-verifiable benchmarks substantially constrain novelty. Visual operators and real grader failures must carry idea 12.

The critiques link the exact prior art and detail all pass/kill gates. Main-report proposal text incorporates the corrections rather than leaving the original inflated claims in place.

## Controlled investigation: DrawerDissect

The customer lane used synthetic colored objects and prediction arrays, then invoked unchanged downloaded upstream crop, curation-scaffold and GBIF-export functions; only logging was stubbed. An added earlier detection reassigned the positional ID. Existing approved metadata persisted under that ID, and the export joined it to the new crop path. The technical critic independently reran the probe and obtained the same mechanism.

Source was pinned to commit `3823276df6b5afe3184af13b9701c3d17f440c96`; the supported rerun clearing path was inspected. Separate output directories emulated crop regeneration. The complete CLI, real model inference, museum data and any production deployment were not run or changed. This is a reproduced synthetic failure mechanism, not an observed customer incident or prevalence claim. [Probe source](raw/customers_identity_probe.py), [result](raw/customers_identity_probe_result.json), [pinned code and limitations](lanes/customers.md).

## Evidence and access limits

- Live employer job API data was captured; dates are the listing's publication fields, not proof descriptions stayed unchanged.
- Search excerpts sometimes lagged live docs. Current primary pages/source took precedence. Released packages, Docker images, current docs and `main` were not treated as interchangeable.
- Open issues are reports. Closed historical regressions are useful fixtures, not current missing features. Active PRs were explicitly considered.
- No private roadmap, revenue metrics, production traces, customer support history or internal platform credentials were available.
- No authenticated Workflows UI inspection, new training campaign or multi-device benchmark was performed.
- No external outreach, purchases, production changes, issue filing or PR submission occurred.
- Raw third-party text/code is retained as research evidence. It is not trusted instructions, and broad source-result counts are not used as a quality proxy.

## Separate user-requested configuration change

During research, the user requested persistent Full Access with no approval prompts. The root agent backed up `~/.codex/config.toml` and set the supported defaults `approval_policy = "never"` and `sandbox_mode = "danger-full-access"`. `codex --strict-config doctor --summary` successfully loaded those settings and reported unrestricted filesystem/network with approval Never. Its unrelated state/connectivity checks were not all healthy, so this was not represented as an overall clean doctor result. The running conversation continued to carry its earlier managed restriction; the configuration change was not claimed to retroactively alter that policy. No state database or authentication files were modified.
