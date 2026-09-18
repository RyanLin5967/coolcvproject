# Current presentation checkpoint — 2026-09-17

The user asked for a polished, distinctive merge-focused demo, a GitHub push, Cloudflare subdomain steps and a demo video. Those presentation changes are now implemented. Default page is an interactive missing-label/coverage explanation; benchmarks curate one matched comparison per dataset; predictions use the same final recipe checkpoints. Generic training/import/history/provider tools are secondary local workbench controls. Public static export is `public-demo/`, served locally at http://127.0.0.1:8767; full workbench remains http://127.0.0.1:8765. Source is `src/coveragecv/workbench/static`; rebuild copied assets with `python3 scripts/build_public_demo.py`. See docs/DEPLOYMENT.md for exact Cloudflare Pages settings (root public-demo, command exit 0, output .).

No new cloud jobs, training or model result changes. Imported final saved presentation jobs: pawns dbee2b4299204c35, all-pieces fb48928676e54b30; construction ecd38e0e6be34958. Public snapshot checks 9 checkpoint/evaluation contracts and uses 12 licensed image files across18 evenly spaced gallery slots. Benchmarks display matched seed cohorts: pawns76.25/78.82/78.94; all13Large72.78/73.68/74.97; construction46.34/48.14/52.76. Single-run galleries explicitly differ from seed averages. A full-label reference is not frontier/SOTA.

Fresh SSE history replay and canvas replacement were fixed. 162 tests pass (14.79s), Ruff/JS syntax pass, and actual desktop1440/mobile390 browser QA passes for BOTH local and standalone public app. Evidence artifacts/qa/presentation/evidence.json. Presentation commit 6179205 was pushed successfully to authorized PUBLIC origin https://github.com/RyanLin5967/coolcvproject.git. The final media checkpoint preserves the opening shot and trims recording shutdown padding; playback/captions/close-to-pause browser checks passed. Check git status/log/remote for the latest final commit. Cloudflare deployment itself still requires connecting the repo in the user's dashboard; do not claim coverage.ryanlin.dev is already live. Credentials stay outside Git. There is no active experiment to resume.

The older completed ML checkpoint follows for historical context; its server PID is obsolete.

---

# Execution status — 2026-09-17, 21:47 EDT

Read this file before resuming. User authorizes autonomous work and explicitly approved important remaining experiments; checkpoint means save/report, not abandon active work. Full access, approval never. Do not resume the ancient restricted research agents listed in AGENTS.md. Credentials stay outside Git, artifacts and output.

## Current work

All authorized training and inference experiments are complete. No active experiment remains. The primary acquisition comparison, independent metric rescoring, fixed tiling interaction, full tests and desktop/mobile UI checks passed. The final LOCAL interaction applied the EXACT existing CPU384px/stride256 tile/NMS.5/edge2 policy and ORIGINAL partial-TRAIN class-size rule<=96px to all3newcheckpoints. Declaration frozen before secondary scores; no new thresholds, label edits, GPU training or parameter search. Results artifacts/acquisition-tiled/construction/20260917/results.json; all3semantic/perclassAPaudits passed. Guided51.6204168610 AP50:95/92.2047344949AP50, random49.3253654598, complete53.7706935561. Zero-reviewwithsamegate49.8820500802. Guided+1.73837vszero,+2.29505vsrandom; fivepasses~4.9timesCPUinference. Original single-pass primary comparison remains unchanged. Best construction score is51.62, NOT95AP.

Workbench http://127.0.0.1:8765, serverPID62013/session39374, log artifacts/workbench_server_latest.log. New acquisition panel verifies source/checkpoint/evaluation contracts and shows real completed scores. All owned cloud apps were verified at zero containers in artifacts/acquisition/cloud_status.json; deployed apps use min_containers=0. Do not stop/restart cloud workloads gratuitously or resubmit completed calls.

## New completed acquisition results

Primary metric complete-validation COCO AP50:95 times100, all120images/717boxes/5classes, CPU512, unchanged reference. One seed20260917. Same6000total updates; guided/random/zero start from original aware4000, complete control starts from original complete4000. Added2000 updates,batch4,augmented LR5e-5/encoder5e-6cosine,restartedoptimizer,no intermediatevalidation.

| Arm | Review units | Training boxes | AP50:95 | AP50 | No-helmet AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero review, reused aware_standard | 0 | 2119 | 48.0026116692 | 85.7482122217 | 20.0247 |
| Random | 150 | 2317 | 47.3432212702 | 81.0997501790 | 15.1395 |
| Guided | 150 | 2384 | 49.8485836091 | 90.9374466852 | 26.6056 |
| Complete standard | control | 6380 | 52.4241494091 | 92.2507970538 | 34.1330 |

Guided minus random +2.5053623389 points; guided minus zero +1.84597193998. Random regresses0.6594. Useful one-seed result, NOT95AP and not evidence of universal improvement. These extra-label studies are separate from the original same-label-budget coverage experiment.

Both150image-class query lists(30/class) were frozen before published TRAIN oracle access. Guidedmaxconfidence recovers265boxes; random198. No-helmet33→84 vs33→38. All existing labels retained; only requested pair coverage changes; all121validation payloads byte-identical. Guided140exhaustive/10absent versusrandom86/64. Review units are not human-hours; extra boxes can cost more to annotate. Not actual new human annotation or adjudication. Independent plan/view/parent/control audit passed, artifact artifacts/acquisition/independent_audit.json. Acquired-label census51unique nohelmetboxes showed no obvious severe class error but9boxes from5frames ofonecamera sequence; see docs/ACQUISITION_LABEL_AUDIT.md. Known construction reference defects remain unchanged.

Cloud calls (completed; reconnect only): guided fc-01M2S28RRS9VFYYD1K8TE0TSNX, random fc-01M2S28RWAHFS0Y31Q53GBNZHP, complete_standard fc-01M2S28RRJ7FKYRF70E5T5YRPT. Checkpoint SHA guided a1bcc7c0ea7864a1c57e7732b038689b8c185093d723c106f2bb8defb9506572; random75cf1710cd7664a75ec834780eca5195b272ee3f4aa460d9aed6f383e890d310; complete1ee494473f5df5e94dd2870db40c7a2b119f6ea9dfac140d989ba956f513d0da. Client scripts/run_acquisition.py saves reservations+callIDs, fails closed on orphan reservation, collects ALL checkpoints before CPU scoring, verifies protocol/view/parent/weights. Source snapshot artifacts/modal/acquisition-source/snapshot.json. Primarycomparison artifacts/acquisition/construction/20260917/comparison.json. All4savedpredictionmetrics reproduced exactly by scripts/audit_acquisition.py, receipt artifacts/acquisition/rescoring_audit.json; not a second inference run.

## Existing strongest results and prior negatives

Original study55/64cloudruns complete,9interrupted; acquisition/crop/refiner follow-ups are separate, not additional original seeds. Pawn original3seed aware76.347 vsnaive63.789 (+12.558); augmentation512 aware78.8218±.4234 vscomplete78.9359,AP50/recall100. All13chess augmentationaware72.9108±.1073(n3),complete74.1557(n2); larger704/EMA one-seed aware73.6825 vsnaive72.7762/complete74.9654, modest. Chess tasks sharecamera/domain.

Construction originalseed17 naive46.3388/aware48.1388/complete52.7558. Original nine-model testplanUNEXECUTED; constructiontestlabelsunevaluated. Crop continuationnegative: aware47.2780vsordinary48.0026; notpromoted. Priorfixedsizegate bestordinaryaware49.88205AP/89.23498AP50, fivepasses, one seed. Humanboxrefinernegative: aware72.9954→72.5534. Teacher variants andfixed3seedWBF also notpromoted. All results/sourcehashes/limitations in docs/RESULTS.json and docs/DECISIONS.md. No annotations/classes/referenceboxes removed to improve scores.

## Budget and account state

Hard original userbudget0cash. User reports spendlimit0,12.91creditsremaining/25.41workspaceusage/no cardcharge and repeatedly authorizes use. EarlierAPI32.68metered/2.68billed after30credits conflicted with dashboard; no confirmed cardpayment and no guaranteed cashoutcome. Originalapps stopped19:59 beforeboundedrestart. Current locked reservation ceilings: capacity3×1.10=3.30,refiner2×.85=1.70,crops4×1.10=4.40,acquisition3×1.10=3.30; total12.70<=reported12.91. These are commitments, not actualspend. All authorized reservations now consumed; no additional cloud calls/retries. Policy ~/.config/coveragecv/cloud_lock.json0600, separate flock ledgers. Exact Decimal comparison avoids binaryfloat cap edge. Keys ~/.config/coveragecv/credentials.json and ~/.modal.toml, never print/commit. Do not create extra accounts forcredits.

## Product, integration, validation

Compiler4coverage states/ontology/provenance, immutable views; RFDETR1.10.1 selective criterion fullcoverage loss+gradientparity, positive/box/aux/encoder preservation. DurableSQLiteWAL jobqueue/isolatedworkers/restart/cancel, safeZIP, localpolicyeditor/provenance/predictions/research. CPU/MPSandNano/Large selectors. Added accountfree CLI plan-reviews and simulate-reviews; published-label simulation namedexplicitly. NativeSDK metadataexport loads actualLargecheckpoint with exacttensorparity.

Roboflow real20epochhostedbaseline finished, observedupload201train451boxes/58valid241exact, customawareversion3 FINISHED/served with correctedreservedrowlayout;3image28detectionhostedparityminIoU.93556. Hostedbaselineusedoriginalchesstest; don'tclaimalltestdata globallyuntouched. Customcriterion trainingislocal/Modal; ordinaryRoboflowhostedtrainingdoesnotconsumethecoveragesidecar.

Currentfullsuite154passed19.18s artifacts/acquisition/full_tests.log; Ruffsrc/tests/scripts,JSsyntax,diffchecks pass. Real2stepCPUacquisitiontrainingpassed. ActualcompletedUIchecks1440/390pass artifacts/qa/acquisition-evidence/evidence.json, noerrors/overflow/mutations. Priorfullappimport/train/restart/cancel andprovider checks passed. Priorcompletedcheckpoint44ebacf; final acquisition checkpoint is being committed with this record.

## Completion and resume

All requested work for this iteration is complete. Final source/docs/portable results are being committed after the staged credential scan and manifest refresh. No experiment is queued and no further cloud call should be launched: the conservative remaining-credit reservation is exhausted. The live app remains available. A future turn can iterate from these actual results without rerunning completed work or claiming95AP. See docs/DECISIONS.md for decisions, failures and limitations, docs/RESULTS.json for portable metrics, and RESTART_MANIFEST.json for source integrity.
