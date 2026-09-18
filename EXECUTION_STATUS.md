# Execution status — 2026-09-17, 20:57 EDT

Continue autonomous work. User explicitly says checkpoint means save/report, not stop. Latest user is angry about modest accuracy gains and demands much better numbers; do not promise or manufacture 95. Full access, approval never. Credentials remain outside repo. Old research agents listed in AGENTS.md must not be resumed.

## Immediate active work

- FOUR new construction object-crop continuation calls launched; client `scripts/run_object_crops.py`, session38110, log `artifacts/object-crops/client.log`. Three GPU containers maximum. The fourth queues. Frozen deployment `coveragecv-object-crops`. Do not resubmit: client reconnects saved call IDs and reservations.
- Cases/IDs: aware_standard fc-01M2S00ZXGWX1J25RB4C0KTJ53; aware_object_crops fc-01M2S00ZXM1W6WJ3E5RA1FBC6A; naive_object_crops fc-01M2S00ZYTT7MSMBFM249B1NEM; complete_reference_object_crops fc-01M2S00ZXD4YR6NZW42J4WE76K.
- Parent checkpoints are original construction seed20260917, 4000updates, Nano512. Each case adds2000updates/batch4/LR5e-5 encoder5e-6 cosine, no intermediate validation, final fixedstep only. Standard aware is primary equal-budget control. Crop-aware/naive/complete replay SAME partial-observed-only immutable8000sampleplan:4000fullframe+4000class-balanced anchorcrops, sequential no model RNG dependence. All labels retained/clipped by upstream transforms; IDs/coverage preserved. JPEG draft decode explicitly disabled. Plan digest b76b88d8792054be7dc0f7d7e081d7d264dc5f76ecbe83799a6dac3c503c47da.
- Crop protocols and source snapshot: artifacts/object-crops/{protocol,plan,plan_audit,smoke-evidence}.json, artifacts/modal/object-crops-source/. Seven focused tests+REAL2update CPU aware training passed. Deployment finished71.9sec, GPU training logs show actual startup. Function train timeout1250/hard1500,max3,min0,retries0,noVolumes. Return checkpoints FIRST then localCPU evaluation, so scoring errors don't lose training. Client has exclusive flock/reconnect/integrity bindings. Results artifacts/object-crops/construction/20260917/<case>; comparison.json only after all4.
- Fresh /root/tiled_inference agent independently implementing/running local MPS fullframe+four384px tiles on originalconstruction3arms. Fixed stride256, floor.001,NMS.5, internaledge2px; same-device fullframe and NMS-only controls, geometry tests, no GT in tiling. Owns tiled.py/evaluate_tiled.py/test_tiled.py. MPS free; root cloud work. Wait for actual results, no more inference parameter sweeps.
- /root/refiner_evidence_ui now adding construction crop evidence panel/API/tests. Owns API/app.js/tests/test_workbench.py. Prior refiner completednegative panel and desktop/mobile actualbrowser checks passed. Ask agent before editing samefiles; restart API when agentready.
- Main webserver PID82677, session64131, http://127.0.0.1:8765. Latest refiner panel loaded. Next API change needsrestart. Logs artifacts/workbench_server.log.

## Current actual scores

Metric primary complete-validation COCO AP50:95×100. Three-seed SD is sampleSD, notCI.
- Pawns original3seeds naive63.789 aware76.347 complete78.041 (+12.558paired); continued augmentation512/4000total aware78.8218±.4234, naive76.2511±1.5489, complete78.9359±.2103. AwareAP50/recall100; strictAP90~23.6. Do not call AP50=100 strictAP95.
- All13chess original naive58.226 aware69.642 complete72.537. Augmentation512 aware72.9108±.1073 n3; naive69.0324 n3; complete74.1557 n2 (third interrupted). Genericbishop0trainpositives remains included. Same camera/domain as pawns.
- Large704/4decoder/EMA fresh2000seed17 all3 completed: naive72.7761593 aware73.6824745 complete74.9653603. Aware+0.6871 vs same-seed Nano512; modest, not breakthrough. Unequal cross-recipe updates (Large2k/Nano4k), separate pilot. Independent rescoring all3 exact, artifacts/capacity/rescoring_audit.json.55/64originalrunscomplete;9remaininterrupted.
- Construction initialseed17naive46.3387973 aware48.1388477 complete52.7557506. Onlythese3+seed18naive45.365complete. Other5originalrunsinterrupted. Nine-model heldout testplan UNEXECUTED, testlabelsunevaluated. Freshaudit nohelmet33observedboxes19/995trainimages; validmedian22x26pxat512; classAP19.09vs33.95complete explains64.37%ofgap; smallAP10.71vslarge59.25. Validationboxlossplateauswhiletrainimproves. Motivatesnewcroppilot.
- Teacher augmentation/cautiousboxweight and fixed3seed WBF ensembles did not improve aware results; do not promote. See docs/DECISIONS.

## Refiner completed, negative — not promoted

- New human-only ResNet18 spatial crop refiner trained1000updates on994observedhumanallchessboxes; fixed224crop/context1.5,4x4features,frozenBN,25%identityproposals; no pseudo/hidden/trainreference labels. Same trainedrefiner all3seed17Nano512controls, classes/scores/order preserved, threshold.05/top100.
- First cloud attempt trained then evaluation crashed on zero-area detector boxes and lost temporary checkpoint. Archived attempts/1. Fixedzeroareapreservation and separated collection from scoring.
- Retry fc-01M2RZ8E6Q0FZKZVSTF5SRC8XK trained98.45seconds, saved checkpoint SHA d201b55a8a1f498d8afcfed9bd0348c9ef06680c21600a66570ea5824ceff364. MPS evaluation failed because7x7->4x4adaptivepoolunsupported; checkpoint durable, CPU rerun no new cloud. Client nowdefaultsCPU.
- ALL3scored: naive69.2098→69.3292 (+.1194), aware72.9954→72.5534 (-.4420), complete74.0649→73.2851 (-.7798). NOTPROMOTED. artifacts/refinement/all-pieces/20260917/partial-human/{comparison,receipt,independent_audit}.json. Audit checksidentity/class/score/order; COCO scorer regenerates geometry-derivedarea fields.
- Refiner stage explicitly separate from original64 detector runs.23unitgeometry tests+real2stepCPU/save/reloadpassed. InferencecostrecordedCPUtiminginclcheckpoint/crops/excludingdetector/scorer.

## Budget and credentials

Hard userrequest $0cash. User reportedModalspendlimit0, later12.91creditsremaining/25.41workspaceusage/no visible cardcharge and repeatedlyauthorizeduse. EarlierAPI reported32.68metered/2.68billedafter30credit; accountdashboardconflict unresolved. No actualcardpaymentverified; no zero-cashguarantee. All4apps stopped19:59 thenboundeduser-authorizedrestart.
- ~/.config/coveragecv/cloud_lock.json0600 allows capacity/refinement/object-crops ONLY. Atomic fcntl reservation ledgers track commitments independentoflaggingAPI. Capacity3×1.10=3.30; refiner2attempts×.85=1.70; crop4×1.10=4.40. Cumulativepostrestartmax9.40 againstreported12.91; actualspendnotclaimed. No automatic retries/newadditionalcalls. Otheroriginalappsremainstopped.
- Keys ~/.config/coveragecv/credentials.json;Modalprofileidident5967~/.modal.toml. Neverprint/commit. Do not createaccountsforcredits.

## Product, verified integration, tests, docs

Compiler4coveragestates, immutablehashviews, upstreamloss+gradientparity, learnerexcludeshidden/testlabels, perclass ontology. DurableSQLiteWALqueue/isolatedworkers/restart/cancelidentity, safeZIPimport/policyeditor/diffs/gallery/predictions. CPU/MPSavailable andNano/Large selectors, Research3tasks/pairedseeds/APbyIoU/ensembles/refinernegative. Actualbrowserrestartsurvival/cancel passed.
Roboflow real20epochNanohostedtrainingFINISHED coveragecv-chess-hosted/1 (differentbaselineusesoriginaltestsplit). Observeduploadversion2 exact201train451boxes/58valid241. Customcoverage-awareversion3 FINISHED/served, correctedreservedrowlayout; hosted3image28detectionsminsameclassIoU.93556 verified. Nativeexportsourceimmutable. PublicSDK metadatafix model_name nowwritten, export-native legacycopy addsmetadataonly; actualLargeaware publicSDKexacttensorparity artifacts/exports/large-704-aware/sdk_parity.json.

- Last codecommit a0131e2. Since then largeuncommittedrefiner/nativeSDK/croppipeline/UI/docsupdates. Needcheckpointcommit after currentagentsfinish andsecretscan; do NOTstopaftercommit.
- Latest full suite **97passed in20.30sec**, artifacts/upgrade_tests_v6.log, before tiled/newcropUI tests. ExistingLargeCPU2step,MPS20stepspassed. Refiner23 andobjectcrop7 included. Ruffcoreclean; rootcaught2minorruffissues inagentin-progresstiledfiles,agentnotifiedtofix.
- README/docsARCHITECTURE/DECISIONS/RESULTS updatedfor55Large+negativeRefiner by evidence_docs. RootaddedconstructionsectiontoDECISIONS andaccepted2000stepupdate toNEXT_EXPERIMENT_AUDIT. RESULTSaccountingstill5.00 untilnewcroppilotfinalupdate; cloudincident9.40current. Finalportableupdate/manifestpending.
- docs/METRIC_AUDIT: independentlyrescored52initialruns allAPexact; Large481nonclasstensorspreservedfromofficialweights,EMAcorrect; residualcorrelation.93-.97 andnoglobalcoordinatebias. Sharedbias/annotationvariabilityplausiblebutunproven.

## Next actions

1. Monitor/collect4cropcalls withexistingclient (no new calls); checkpointsreturnbeforelocalCPUscoring. Shareactualscorespromptly, includingregressions. Standardvsawarecropprimary andcompletecontrol allvisible.
2. Collect localtiledagent3armresults/paritytests, addreportedcost. No silentinferenceprotocolreplacement.
3. FinishUIactualbrowserchecksafterAPIrestart; docs/RESULTS snapshotaccuratecounts+newpilotseparatefromoriginal64.
4. Appropriatefinaltests/ruff, configured-keysecret-scan, regenerateRESTART_MANIFEST andcommitdurablecheckpoint. Keepworkingpaststatus/checkpoint; user explicitly saysdo notpause.
