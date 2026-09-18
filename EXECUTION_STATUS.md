# Execution status — 2026-09-17, 20:08 EDT

Continue autonomous work; user says checkpoint means save/report, never stop. Full access, approval never. Read docs/DECISIONS.md for implementation difficulties. Old research agents must not be resumed.

## Immediate state

- Local app http://127.0.0.1:8765, PID81665. Needs restart for latest research/ensemble/budget API changes. Latest browser research check passed1440/390px across3tasks, no errors/overflow.
- 56 tests passed in artifacts/upgrade_tests_v4.log. Latest MPS/kill-switch/UI incident changes need checks. Large704/EMA realCPU2-update/save/reload smoke passed. RealMPS20updates passed (46sec training; ~84sec including save).
- All upgrades remain uncommitted as of this write. Commit checkpoint after secret scan/tests. Do NOT treat saving/committing as stopping.
- Local Large aware2000-update run is ACTIVE in session19584, artifacts/local_capacity.log, outputs artifacts/local-capacity/all-pieces/20260917/aware_large_704_ema. AppleM5Pro MPS48GB. Initial init artifacts/large-smoke/init/initialization.pt. No cloud cost.

## Billing discrepancy and bounded restart

User requires $0cash. Credentials ~/.config/coveragecv/credentials.json0600 and ~/.modal.toml profileidident5967; no secrets in repo/output.

API initially showed19.72 metered/0billed, then32.68metered/2.68billed after30credits. All4ownedModalapps were stopped19:59EDT and verified0containers. We disclosed the discrepancy. User then explicitly said dashboard has12.91credits remaining and25.41workspaceusage and directed us to USE those. Workspace identity idident5967 confirmed. Environment budget API unavailable on thisplan. Documentation calls billed_cost invoiced, but no actual payment/cardcharge was verified; dashboard/API conflict unresolved. Never assert a confirmed cash charge or zero cash until reconciled.

User-authorized boundedrestart: THREE Large704/EMA controls only, maximum conservative reservation3.30USD (<3.50cap). Persistent ~/.config/coveragecv/cloud_lock.json records permission, remaining12.91asUSERREPORTED, and allowed_app coveragecv-capacity/3runs. Do not resurrect other compute automatically. Keep reservation accounting; do not rely only on lagging metered usage.

capacity app ap-BGuIFBIOzbHes6SyIXX4TH currently being REDEPLOYED via module, log artifacts/modal_capacity_redeploy.log. Frozen source artifacts/modal/capacity-source/coveragecv, requirementscloud-requirements.txt, large official130MBweights. Original3failedcalls archived under each artifacts/capacity/all-pieces/20260917/<method>/attempts/1. Launch scripts/run_capacity.py only after redeploysuccess; preserve oldattempts. DecoratorL40S,max3,timeout1500,CPU4,RAMmax24GiB,min0,scaledown2,noVolumes,retries0. train2000steps/1250sec cap, EMA,704px. Training source frozen BEFORE subsequent MPS/scorer/localwork edits.

Otherapps coveragecv-improvement ap-EaDb1d0vAAvn40oqQuU4j8, coveragecv-ablation ap-5Xn3Xmns7nr3Ho1oQi5l73, coveragecv-benchmark ap-622r2hQTPM66elE9AtyiwQ remainSTOPPED. Collector and construction holdout waiter were terminated. scripts/collect_experiments.py would wait until64/64, now12interrupted, so update it before reuse.

## Results

52/64cloudrunscompleted;12stoppedforbudget (3capacitynowapprovedrestart). artifacts/research_summary.json verifiedhashes/seeds/budgets; artifacts/cloud_incident.json listsstoppedpaths. Keepactualseedcounts.

Three-seed pawns: initial naive63.789±2.074, aware76.347±.704, complete78.041±.533 AP50:95. Pairedgain12.558pts. Augmented512 after4000totalupdates: naive76.251±1.549, aware78.822±.423, complete78.936±.210. Longer384aware75.355; teacher77.921. Teacher0.1boxweightseed17=78.705 < matchedaugmentation78.903, so notpromoted. Classonly/stuffinterrupted.

All13chess initial3seeds: naive58.226 aware69.642 complete72.537. Augmentationaware3seeds72.911±.107; naive3seedsnowcomplete/readlatestsummary; complete2seeds74.156thirdinterrupted. Teacher3seeds72.752. SAMEboarddomainaspawns, genericbishop0trainpositives.

Independentconstruction:995train/120valid/91test,5classes,2119visible4261hiddenboxes. Final3arms×3seeds4000planned, onlyfirstseed3arms+seed18naivecompleted. Firstseednaive46.339 aware48.139 complete52.756. Other5interrupted. Nine-run test plan NOTexecuted; heldoutlabelsremainunevaluated. Donotquietlypresentselected4asfinishedstudy.

WBF inferenceensemble (ensemble-boxes1.0.9) fixed3seeds/IoU.55/skip.001 equalweight testedexistingpredictions withoutGTfusion. Pawnsaware78.686,complete78.781,naive76.952; all13aware72.476. Noawareimprovement: donotpromote. scripts/evaluate_ensembles.py finishavailable3seedcontrols, noGPUcost. New common scorer exactly matchesexistingmetricdict, artifacts/scorer_parity.json.

## Product and provider evidence

Compiler4coveragestates + hashverifiedimmutableviews, arbitraryontology, stockloss/gradientparity, durableSQLitequeue/isolatedworkers/restart/cancel, ZIPimport,policyeditor,diffs,images,predictions. Researchdashboard3tasks,actualn/SD/perclass,APbyIoUdiagnostic,ensembletable,stoppedcohortnotice. Coreinitial18cloudresultsadoptedSQLite. Latest researchAPI/JS not all restarted. CLI Largevia initialize --variant large and train --recipe large_fresh --device mps works. App currentlyCPU/Nano; optionalMPS/Largeselector is notyetimplemented.

Roboflow actual20epochNano hostedbaselineFINISHED, coveragecv-chess-hosted/1. Correctedcustomaware2000modelversion3 FINISHED, class/boxparity3validationimages28detectionsminIoU.93556. Originalv1classlayoutfailed; explicitrowpermutationfixedallmain/encoderheads. Exactobserveddatasetroundtripv2passedVOCorigin+1. Evidence artifacts/provider/model/semantic-corrected/{deployment,semantic_parity}.json. PlatformAPIpathfixed; UI stilldescribesoldv1prominently, improve.

## Next work

1. Finishboundedcapacityrestart and collect; continueMPSwhilecloudstartup, don'tclaimscorebeforefinished. Newcloudreservationsmuststay<=3.50 unlessuserbudgetreconciled/updated.
2. Finish trackedcheckpointcommit plusupdateREADME/docs withcompleteactualresults andbillingdiscrepancy (olderdocscurrentlyclaimallcredits!). Maintain decisionsrecord asuserasked.
3. Finish browser/serverlatestAPI,tests/ruff/node; no unnecessary repeated costlytests.
4. EvaluateLargeagainstsame-seedNano andmatchedLargecontrols, no cherry-picking. Iffails, report. Userwantslargeaccuracygain butitcannotbeguaranteed.
5. Roboflowremainingcreditsuserreportsplentiful; no additionalRFtrainingcurrentlyrunning. Ordinaryhostedtrainingdoesnotconsumeourcoveragecriterion.
