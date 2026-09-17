# Infrastructure, reliability, security, and support research

Research date: 2026-09-17. Seven Exa searches completed; raw results are under `raw/infra_*.json`. Live official documentation and source were used to challenge the search results. Rankings below concern usefulness and distinctiveness, not a coding-time cutoff. External dependencies are listed separately from engineering work.

## What the evidence actually supports

Roboflow is explicitly hiring for high-availability inference, SLOs, incident response, cost optimization, observability, customer security integrations, and infrastructure automation. Its employer-written job page also says building with Roboflow and contributing to its open source are ways applicants can stand out. These are unusually direct signals for an infrastructure project, rather than an inferred need from a generic company description. [E1]

The highest-confidence opportunities are **qualification and diagnosis across the entire deployed vision application**. Generic monitoring, profiling, model preloading, offline caching, and fleet administration are already product capabilities. The research does not establish what internal release tooling or roadmap Roboflow has beyond public evidence.

### Evidence ledger

All URLs retrieved on 2026-09-17. Undated documents are current retrieved documentation, not independently verified release claims.

| ID | Source, date/type | Precise finding and confidence |
| --- | --- | --- |
| E1 | [Infrastructure Engineer, employer-written YC job](https://www.ycombinator.com/companies/roboflow/jobs/MEleCqH-infrastructure-engineer), undated | Explicit responsibilities include inference reliability, cost, SLOs/SLAs, observability, on-call, security and customer onboarding. High confidence about stated hiring priorities; not evidence of a specific incident. |
| E2 | [Support request requirements](https://docs.roboflow.com/platform/support/getting-help-faster-what-to-include-in-a-support-request), official docs | Requests exact server image, hardware/JetPack, model IDs, concurrency, image dimensions, environment overrides, timing and memory traces. Describes troubleshooting performance and Workflow errors. High confidence of a support-reproducibility need; no quantified ticket volume. |
| E3 | [Workflow caching guide](https://blog.roboflow.com/workflow-caching-in-self-hosted-roboflow-inference/), 2026-06-18, company engineering blog | Server image, Workflow definition and model weights update independently. Definition caching can explain apparently ignored changes. High confidence; this is documented behavior, not necessarily a defect. |
| E4 | [Workflow profiling](https://docs.roboflow.com/workflows/developer-guide/developer-guide/profiling), official docs | Profiling already emits Chrome traces via `ENABLE_WORKFLOWS_PROFILING` and SDK `enable_profiling=True`; serverless profiling is not supported. High confidence of product overlap. |
| E5 | [Inference telemetry](https://docs.roboflow.com/deployment/self-hosted/inference-server/configuration/telemetry), official docs | Prometheus `/metrics` and Docker statistics already exist. High confidence; a new metrics dashboard alone is weak differentiation. |
| E6 | [Production readiness checklist](https://docs.roboflow.com/deployment/production-checklist), official docs | Documents retries, cold starts, rate limits, autoscaling replica choices and persistent production deployments. High confidence that basic production guidance already exists. |
| E7 | [Security configuration migration](https://docs.roboflow.com/deployment/self-hosted/inference-server/configuration/security-migration.md), source checked by document 2026-09-11 | Distinguishes merged source from unverified published versions; runtime hardening PR #2952 is described as pending. This distinction is crucial for any release/security project. High confidence in the document's stated scope. |
| E8 | [Securing self-hosted Inference](https://docs.roboflow.com/deployment/self-hosted/inference-server/configuration/security), official docs | Authentication, local Custom Python, model access, image fetching, webhook destinations, networking and TLS have explicit controls. Current docs describe CLI loopback publishing, with Jetson/tunnel exceptions; consult E7 for release status. High confidence in documented controls, not installed deployments. |
| E9 | [Secure Gateway](https://docs.roboflow.com/deployment/self-hosted/enterprise/secure-gateway), official docs | Existing controlled egress proxy caches model weights, containers and API responses. Basic gateway/cache/air-gap packaging ideas duplicate this direction. High confidence of overlap. |
| E10 | [Video configuration](https://docs.roboflow.com/deployment/self-hosted/inference-server/configuration/video-configuration), official docs | Managed pipeline process cap and RAM allocation guard are documented; process cap is not a per-user rate quota or hard GPU memory cap. Some media validation is pending. High confidence with release caveat. |
| E11 | [Inference issue #2448](https://github.com/roboflow/inference/issues/2448), opened 2026-06-11, public report | Reports severe multi-model Workflow latency amplification from synchronous model loads, eviction and shared execution resources despite idle accelerators. Page is open. Medium confidence as a reported incident; not reproduced here or established as current. |
| E12 | [Inference issue #685](https://github.com/roboflow/inference/issues/685), opened 2024-09-27, public report | Reports inability to terminate a pipeline after initial source connection failure. Useful historical fault scenario. Current runtime behavior remains unverified, regardless of issue state. |
| E13 | [Inference HTTP implementation](https://github.com/roboflow/inference/blob/b77b7a08cb1e484742d9eaf5b48e48b534081289/inference/core/interfaces/http/http_api.py) and [cache implementation](https://github.com/roboflow/inference/blob/b77b7a08cb1e484742d9eaf5b48e48b534081289/inference/core/managers/decorators/fixed_size_cache.py), source inspected | Current main contains `PRELOAD_MODELS`, `PINNED_MODELS`, `pin_model`, startup loading and readiness machinery, plus cache pressure handling. High confidence these mechanisms exist in source. Do not propose basic warmup/pinning as missing because an older issue requested it. |
| E14 | [CONTRIBUTING.md](https://github.com/roboflow/inference/blob/b77b7a08cb1e484742d9eaf5b48e48b534081289/CONTRIBUTING.md), source inspected | Explicitly suggests making server integration tests run without Roboflow API keys. Notes some tests/runners need private resources. High confidence of a stated contribution avenue; precise remaining test coverage requires implementation audit. |
| E15 | [Block testing](https://docs.roboflow.com/workflows/developer-guide/developer-guide/testing), official docs | Already recommends practical integration cases, assertions on predictions, and consistency with external implementations. New testing projects must improve automation, corpus or deployment realism, not claim testing is absent. |
| E16 | [Workflow versioning](https://docs.roboflow.com/workflows/developer-guide/developer-guide/versioning), official docs | Engine compatibility ranges and versioned blocks already exist. Runtime qualification must test behavior inside these contracts rather than invent versioning. High confidence. |
| E17 | [Deploy a Workflow](https://docs.roboflow.com/workflows/deploy/deploy-a-workflow), official docs | SDK can send a full `specification`; locally executable definitions can avoid API-key-gated blocks. Direct Python execution and local HTTP are public seams. High confidence that a meaningful public demonstration need not require private cloud credentials. |
| E18 | [Test a Workflow](https://docs.roboflow.com/workflows/build/test-a-workflow), official docs | Browser test UI already previews results and caches unchanged block results; sinks can be explicitly enabled. High confidence of overlap with generic preview/debugger ideas. |
| E19 | [NVIDIA Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), competitor/tool | Cross-backend comparisons, ONNX subgraph manipulation and inference debugging already exist. High confidence; generic numerical parity is not a distinctive project. |
| E20 | [Triton Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/model-analyzer-README.html), competitor/tool | Searches configurations and reports performance/resource tradeoffs for single, multiple, ensemble and BLS models. High confidence; generic batching/throughput optimization is crowded. |
| E21 | [Toxiproxy](https://github.com/Shopify/toxiproxy), existing infrastructure | Fault injection for network conditions already exists. A vision-specific fault lab should reuse it and contribute domain assertions, not build another proxy. High confidence. |
| E22 | [WireMock](https://wiremock.org/docs/), existing infrastructure | API stubbing, stateful behavior, faults and record/playback already exist. A cloud test environment must contribute Roboflow contracts and actual integration coverage. High confidence. |
| E23 | [Open Policy Agent](https://www.openpolicyagent.org/docs), existing infrastructure | General structured-data policy evaluation already exists. Security-project differentiation must be vision data lineage and evidence, not a policy-language wrapper. High confidence. |

### Important research corrections

- Exa returned an older security-page excerpt implying broadly permissive listening. Live docs describe newer loopback defaults and explicitly distinguish source changes from released artifacts. Therefore no statement that all current installations expose an unauthenticated service is justified.
- E11's requested warmup/pinning functionality partly exists in inspected source. The report is useful for workload design, not a license to claim an unfixed vulnerability or missing feature.
- Main-branch code, current documentation, latest published Python package, and Docker image tags are four different evidence objects. A useful release tool should record each separately.
- There are already many mocks and Workflow stub plugins in the repository tree. A secret-free integration project should extend them, and measure new real integration coverage rather than claim there are no mocks.
- The downloaded HTTP, cache and contributing files were verified by Git blob hash against source commit `b77b7a08cb1e484742d9eaf5b48e48b534081289`; E13–E14 use immutable links.

## Candidate 1 — Vision Release Observatory

**Verdict: strongest infrastructure candidate; merge with any broader release-certification idea from other lanes.**

**User and job:** Inference maintainers and deployment engineers deciding whether a server/SDK/runtime upgrade is safe. The tool qualifies the actual deployed Workflow against a versioned contract, then gives the smallest observable counterexample when the contract breaks.

**Why Roboflow:** Public versioning promises, cache independence, platform breadth, current migration ambiguity, and existing integration-test guidance create a concrete fit [E3, E7, E14–E17]. It does not require assuming Roboflow has no CI.

**Integration:** Local Inference containers and `InferenceHTTPClient.run_workflow(specification=...)`; direct `ExecutionEngine` for block-level isolation; Workflow JSON, server/SDK versions, image digest, model artifact hash, backend and decoder metadata form the run manifest. Start with public/local models and known Workflow blocks. Real hardware targets register as optional workers.

**Distinctive core:** Test decoded images → resize/crop/coordinate handling → model → postprocessing → event output, including schema and security/migration contracts. Tensor comparison is an optional Polygraphy adapter, not the product. Compare boxes using class-aware matching and explicit tolerance, masks geometrically, and event sequences by timestamp windows and task semantics. Preserve exact artifact identity. Treat nondeterminism statistically and explicitly distinguish intentional model changes from runtime changes.

**Narrow first proof:** Two pinned released CPU images, one candidate source build, three Workflows (detection, crop/classify/remap, tracker/count), a small public labeled video corpus, and a set of known geometry/serialization/timeout regression fixtures. Output a release matrix and self-contained failing case. Add CUDA/Jetson/decoder matrices once hardware is available.

**Compelling demo:** Two deployments show almost identical bounding boxes, but an upgrade changes a crop-coordinate convention or drops a short crossing event. Click the failed contract to see the exact frames, responsible step, environment difference and replay command. Include both a historical reproduced regression and disclosed seeded faults; do not present seeded faults as Roboflow defects.

**Validation:** Failure detection on held-out seeded mutations, false alarms on repeat runs, reproducibility on a clean machine, number of real supported targets, localization accuracy, and regression corpus acceptance by maintainers. Avoid arbitrary speedup promises before measurement.

**Adversarial attack:** “Polygraphy plus screenshots”; noisy floating-point differences; duplicate CI; green matrix hiding untested platforms. **Survival condition:** full application semantics, explicit untested cells, content-bound artifacts, empirical tolerances, useful historical cases and straightforward upstream fixture adoption. Otherwise reject.

**Engineering vs external constraints:** Engineering includes runners, event oracles, artifacts and minimization. Genuine external constraints are access to different GPUs/Jetsons and representative labeled event footage. No private Roboflow infrastructure or Enterprise license is needed for the local core.

## Candidate 2 — Camera-to-Action Fault Lab

**Verdict: strong, visually memorable reliability project; distinct from release parity.**

**User and job:** A customer integration engineer needs to know what a vision application does when a camera disconnects, a sink slows down, a model cold-loads or the process restarts. An HTTP health check cannot answer whether the inventory count remained correct.

**Evidence:** Production retry/cold-start guidance, source lifecycle reports, performance reports and documented pipeline limits support this class of scenario [E6, E10–E12]. They do not prove the particular current server fails any proposed test.

**Integration:** Real local `InferencePipeline`/managed pipeline API, RTSP from a local replay server, Toxiproxy on TCP boundaries, local HTTP sink and Prometheus/profiling capture. Use public `workflow_specification` and `on_prediction`/SDK result interfaces. All industrial outputs terminate in a simulator.

**Distinctive core:** A timeline with ground-truth events and fault schedule; assertions for frame age, recovery time, bounded queues, duplicate external actions and missed events. Most load tools count requests. This evaluates whether camera evidence turned into the correct timely action. Use existing fault tools [E21].

**Narrow first proof:** A recorded conveyor-like sequence of numbered objects crossing a line; one detector/tracker/count Workflow; reconnect, delayed webhook response, restart and burst-density scenarios. A reference sink records stable event IDs, demonstrating when deduplication is possible and when uncertainty must be surfaced. Never promise universal exactly-once semantics.

**Demo:** Pull the virtual network cable while boxes keep arriving, then recover it. The frame overlay, event ledger and memory/freshness graph explain whether the system silently lost ten items, duplicated two actions, or recovered according to its contract. Compare baseline integration with an explicit bounded-buffer/retry/deduplication example.

**Validation:** Event recall and duplicate count, p95 frame age, time to recovery, process cleanup, memory ceiling, and repeated-run variability. Include successful fault tolerance to avoid a benchmark designed only to fail.

**Adversarial attack:** “Toxiproxy wrapper”; synthetic scenes too easy; cannot infer correct behavior after missing camera data. **Survival condition:** domain-specific temporal/event oracle, honest unknown intervals, real inference and a reusable scenario format. If only ping latency changes, reject.

**Engineering vs external constraints:** Core runs locally with generated/owned video and CPU inference. Real camera/codec/hardware coverage and realistic line-event labels are external expansions; no customer access is needed for the first convincing result.

## Candidate 3 — Replay-Verified Support Capsules

**Verdict: strongest direct support-engineering utility; serious privacy and validity constraints make it technically interesting.**

**User and job:** Someone experiencing a Workflow failure needs to send a maintainer an executable, minimal reproducer instead of ten rounds of version and configuration questions. E2 directly describes the missing diagnostic inputs.

**Integration:** Wrap a local SDK run or attach to an explicitly selected local container. Collect allowlisted configuration, Docker digest, package/backend metadata, Workflow definition hash, trace, clocked request pattern and failure predicate. A DAG-aware reducer removes unneeded steps, frames and parameter variations while re-running the predicate. E3's three independent dependencies become explicit provenance fields.

**Distinctive core:** A capsule is called reproducible only after it reproduces in a clean environment. The minimizer understands Workflow selectors, batches, parent coordinates and video state; generic traceback collection does not. It can create a support-request draft following Roboflow's own categories without sending anything.

**Narrow first proof:** Support failures involving configuration, schema, HTTP timeouts and synthetic frame inputs. Start with deterministic exceptions; add probabilistic predicates for timing failures only with a specified repeat criterion. Replace media with a synthetic fixture only if the original failure still reproduces.

**Demo:** Start with a 15-step pipeline and a long sample clip. The tool reduces it to a 3-step, 8-frame reproducer, launches a fresh container, confirms the same assertion, and produces a precise environment diff. These numbers are proposed demo targets, not observed results.

**Validation:** Reproduction rate on unseen machines, reduction in retained steps/media, preserved failure identity, correct classification of non-reproducible cases, and secret scanning on a planted-secret corpus. Have an independent reviewer reconstruct the issue using only the capsule.

**Adversarial attack:** Redaction can erase the failure; model outputs and embeddings may still contain private information; timing is not deterministic; dumping environment variables leaks keys. **Survival condition:** allowlist collection, local review before export, no automatic upload, explicit restricted/unshareable output when privacy-preserving reduction fails, and a reproducibility acceptance gate. A cosmetic log zipper is rejected.

**Alternatives:** Existing support checklist, browser previews/block cache, general diagnostics, WireMock playback and Polygraphy debugging [E2, E18, E19, E22]. The gap is a valid whole-Workflow reproducer, not general replay as an invention.

**Engineering vs external constraints:** No private infrastructure is needed to prove the tool on public/local workloads. Obtaining real support cases or adoption feedback is an external dependency for measuring actual ticket-time improvement; do not fabricate that metric.

## Candidate 4 — Cost per Correct, Timely Event Planner

**Verdict: survives only with labeled event workloads and measured recommendations; more ambitious than a pricing calculator.**

**User and job:** A deployment engineer choosing a model, frame sampling, crop strategy, batch/concurrency settings and target hardware wants the cheapest configuration that still catches the events that matter before a deadline.

**Evidence and integration:** Profiling, public SDK controls, dedicated replica settings and multi-model cache interactions provide real knobs [E4, E6, E11, E17]. Run a user-supplied local Workflow under recorded arrival traces and labeled video events; collect inference traces, memory, cold/warm behavior and external action timing. Pricing is an explicit user-supplied schedule with date and units. Hosted evaluation is optional and metered, not necessary for the project.

**Distinctive core:** Optimize cost subject to event recall, false-action rate and deadline constraints over the entire Workflow, including input-dependent crop fanout and cold model working sets. Show uncertainty and Pareto tradeoffs; decline recommendations outside measured regimes. Triton Model Analyzer already handles model and ensemble configuration optimization [E20], so plain throughput/RAM sweeps are rejected.

**Narrow first proof:** One crop/classify/count task with annotated event intervals; three configuration families; one CPU and one GPU target; smooth vs bursty traffic. Show that the highest FPS option can lose fast rare events under temporal sampling, while a measured alternative satisfies the task constraint at lower cost.

**Demo and validation:** Interactive same-footage comparison: adjust a target event deadline and watch the measured feasible configurations change. Validate chosen recommendations against a held-out exhaustive sweep, report workload assumptions, end-to-end p95, event recall, false-action rate and cost sensitivity. Never infer dollar savings from synthetic service time alone.

**Adversarial attack:** “Triton tuner with a new axis”; event labels expensive; measurements do not transfer; cost improvements achieved by discarding hard cases. **Survival condition:** held-out event correctness, total workload coverage, actual target runs and uncertainty. This is not worthwhile if the only result is a GPU benchmark leaderboard.

**Engineering vs external constraints:** Optimization and tracing are engineering work. Representative event annotations, at least two target configurations and any paid cloud test budget are external. No internal Roboflow bills are available or required; company-wide savings cannot be claimed.

## Candidate 5 — Workflow Data-Flow and Egress Contracts

**Verdict: conditional security survivor; narrower than a claim of universal privacy certification.**

**User and job:** A customer security engineer reviewing a Workflow wants to know whether raw frames, cropped faces, OCR text or only aggregate counts can reach each external destination. A secure transport proxy does not describe which upstream image branch feeds a webhook.

**Evidence:** Workflows can execute custom code and make external calls; official hardening separates image fetch, webhook destination, model and transport controls [E8]. Secure Gateway already provides controlled Roboflow egress [E9], so another proxy is unnecessary.

**Integration and differentiator:** Parse public Workflow definitions and a versioned supported-block manifest; propagate data provenance across known image/crop/transform/output steps. Compile a policy such as “this sink receives counts only” into a graph assertion and a local runtime test using synthetic marker pixels/text and sink capture. OPA can evaluate the structured policy [E23]; the contribution is vision-specific provenance and a concrete counterexample.

**Narrow first proof:** Support a small enumerated set of image, crop, redaction, OCR and HTTP sink blocks. Flag a raw-image branch that bypasses the intended redaction step. Unknown custom Python and unsupported blocks produce `unverified`, never a green check. Isolated test networking enforces the declared destinations during the observed run.

**Demo and validation:** An understandable graph colors the precise raw-image route to a simulated external VLM endpoint. Rewire the branch; rerun the synthetic marker test; export evidence tied to the Workflow hash. Validate recall/false alarms on deliberately miswired graphs and check that unknown blocks cannot silently pass.

**Adversarial attack:** Static graph reachability is easy; content may survive transformations; arbitrary code defeats taint tracking; redaction detection can miss sensitive pixels. **Survival condition:** useful block coverage, conservative unknowns and executable evidence. This certifies a narrowly stated routing contract, not legal compliance, anonymization or absence of all leaks. Do not advertise “HIPAA-certified Workflows.”

**Engineering vs external constraints:** The core can run entirely against synthetic data and local sink emulators. Real customer policy fit and production deployment authority are external. The project is weaker than Candidates 1–3 unless a security-oriented role is the target.

## Candidate 6 — Secret-Free Inference Integration Environment

**Verdict: high adoption plausibility, lower standalone spectacle; strong component of Candidate 1.**

**User and job:** An outside contributor should be able to fork Inference and exercise real HTTP → Workflow → model → output integration without access to company secrets or internal runners. CONTRIBUTING explicitly identifies this avenue [E14].

**Integration:** Existing `tests/inference/integration_tests`, Workflow integration suites and stub plugins; local API/model-registry fixtures, tiny redistributable model artifacts, real server process, real preprocessing/runtime/postprocessing, and hermetic network rules. Reuse existing fixtures and WireMock-like tools [E22]; don't invent a whole mock cloud platform.

**Distinctive core:** Replace only inaccessible cloud boundaries while retaining the inference implementation under test. Version the cloud-boundary contracts and provide deterministic failure modes for authorization, corrupt downloads, timeouts, stale Workflow definitions and model cache transitions. The harness distinguishes “mocked boundary passed” from “live cloud parity checked.”

**Narrow first proof:** Select an existing integration-test slice that currently requires keys, document its baseline skipped/failing state, make that slice execute in a public fork with no secrets, and prove that seeded server regressions still fail. Do not simply replace model predictions with canned responses.

**Demo and validation:** A fresh public fork runs real model/Workflow tests offline after downloading permitted dependencies and fixtures. Report additional meaningful integration cases executed, failure sensitivity, runtime, and any lost coverage. A small optional live contract check can validate boundary drift using an authorized ordinary account; it is not required for routine contributor CI.

**Adversarial attack:** “Just mocks”; fake cloud silently diverges; maintainers already solved it; tiny model misses real backend behavior. **Survival condition:** audit current tests, upstream-compatible architecture, mutation sensitivity and explicit coverage boundaries. Reject as a separate flagship if the actual delta is only several patched fixtures; incorporate that work into Release Observatory instead.

**Engineering vs external constraints:** Public core has no private credential requirement. Maintainer feedback/acceptance and optional live contract checks are external, while hardware-specific coverage requires real devices. Engineering ambition can scale, but mock fidelity does not replace those facts.

## Ideas rejected after overlap and usefulness checks

| Idea | Why rejected or folded into a survivor |
| --- | --- |
| Inference metrics dashboard | Prometheus, profiling and deployment/model monitoring already exist. Only interesting as an output of a better diagnostic tool. |
| Generic workflow profiler | E4 already provides block timing and Chrome traces. |
| Model preloader / pinning CLI | Current main already has preloading and pinning machinery [E13]. A concrete concurrency fix could be excellent, but requires a fresh reproducer and current code audit before planning it as an open gap. |
| Generic air-gap cache/gateway | Secure Gateway and offline features already occupy the product space [E9]. Disconnect rehearsal can be a fault-lab scenario. |
| Security checklist or Docker hardening template | Detailed current and pending hardening docs already exist [E7–E10]; low distinctiveness unless tests expose a specific version mismatch. |
| Pure ONNX/TensorRT output diff | Polygraphy exists [E19]. Whole application geometry/events are the differentiated layer. |
| GPU capacity / batching tuner | Triton Model Analyzer already supports rich workloads [E20]. Candidate 4 survives only through event correctness and complete workload economics. |
| LLM support chatbot | Weak diagnostic validity and scarce proprietary support data. Reproducible evidence is a more defensible contribution. |
| Private cloud cost optimizer | No access to Roboflow's production bills, traces or deployment authority; a simulated “savings” report would not establish usefulness. |

## Ranking and recommendation to synthesis

1. **Vision Release Observatory:** widest strategic fit, substantial technical depth, strong upstream artifact path. Needs careful semantic oracles and honest hardware coverage.
2. **Camera-to-Action Fault Lab:** clearest impressive demo; connects infrastructure failures to real CV outcomes. Particularly compelling when tied to release qualification without making one giant first milestone.
3. **Replay-Verified Support Capsules:** strongest support-specific primary-source grounding and immediate developer utility. Preserve privacy and exact failure identity.
4. **Cost per Correct, Timely Event Planner:** potentially valuable and interesting; requires real labels and target measurements to avoid benchmark theater.
5. **Workflow Data-Flow and Egress Contracts:** useful security-focused option with clearly bounded claims; more uncertain adoption and novelty.
6. **Secret-Free Integration Environment:** unusually explicit requested contribution, but strongest as the adoption foundation of #1 rather than a separate grand product.

The most persuasive combined project is **a public qualification bench that demonstrates a meaningful event failure, emits a minimized replay capsule, and can run its core tests in an ordinary fork**. Keep the implementation modules separable. Do not add cost optimization and privacy analysis merely to inflate scope; they are alternate projects with different users and validation needs.
