# Assignment 2 — ordered execution plan and agent checkpoint

## 0. Purpose, authority, and current state

- Project: Astana IT University Computer Vision; hand gesture recognition (`peace`, `stop`, `thumbs_up`). Instructor: Nurym Kaiyrkhan.
- Authoritative requirements: Assignment 2 text supplied by the user in this conversation. This document translates them into work; proposed targets below are not additional instructor requirements.
- Deadline: **7 November 2026, 23:59, Asia/Almaty**. Practice-lesson defense is also required; its date is unknown and must be confirmed by the student.
- Plan created: 4 November 2026, Asia/Almaty. Version: 1.1 (execution-readiness review).
- Workspace: `/home/nurik/Projects/cv-assignment`.
- Current authorization: user explicitly authorized execution on 2026-11-04. Continue ordinary implementation autonomously within that scope.
- Source of truth during execution: this workspace file. Update it after every task, failure, review, or interrupted run. Reconcile any exported copy against this file before resuming.
- Execution outcome: **Presentation v2 verified locally; student narration and external handoff pending.** Current state is this dashboard and section 16; earlier dated checkpoints are historical.

### Resume checkpoint

| Field | Current value |
|---|---|
| Current task | GitHub source repo published privately; student narration and final handoff pending |
| Next action after execution authorization | Record deliverables/A2_Demo_Transcript.md at its timestamps, add audio to the silent 04:10 master, review playback; then authorized publication/submission. |
| Last completed work | Refined responsive app, seven-page report/charts, presentation screens, silent 04:10 video, timestamped transcript, audio helper and refreshed verified source export |
| Implementation tasks passed | 13 of 15: P01-P13 PASS. P14 student audio/video link pending; P15 reviewer access/submission/defense pending |
| Active training/generation jobs | None. Local app preview server on 127.0.0.1:8765; no recording or training active. |
| Dataset version / split version | a1_limited_v2 FROZEN, 344 train / 58 val / 51 test |
| Selected SOTA / weights revision | ConvNeXtTiny, Keras 3.15.1; official ImageNet weights SHA256 in pilot_convnext.json |
| Runtime / compute | Python 3.12.15, TensorFlow CPU 2.21.0; host RTX 3050 4 GiB/camera visible; elevated network/HTTP verified |
| Final experiment run IDs | B0_limited_v1, B1_limited_v1, S1_limited_v1, A0_limited_v1 — completed and reviewed |
| App / repository / video / submission links | App http://127.0.0.1:8765; repo https://github.com/lastnurik/cv-assignment (private); silent video/transcript in deliverables. Final video link/submission receipt absent. |
| Open decisions | Student voice recording; repository/video destination and publication authorization; defense time. Exact frontend-design/anti-slop skill install path not visible. |
| Next reviewer question | Does the student narration align with the 04:10 transcript, and can the instructor access the final repository/video links? |

## 1. Assignment requirements and evidence map

| ID | Requirement | Planned evidence | Completion owner |
|---|---|---|---|
| R01 | Continue A1 topic and dataset; cleaning/expansion allowed | Original inventory, transformation manifest, dataset card | P01, P04 |
| R02 | Integrate at least one eligible SOTA component and compare against A1 | Method decision with sources, executable integration, experiment matrix | P03, P06–P08 |
| R03 | Small interactive web/mobile app: upload/camera → prediction → explanation | Tested local web app, screenshots and demonstration | P09–P11 |
| R04 | Accuracy/F1 and suitable task metrics; training curves | Per-image predictions, metric tables, confusion matrices, history files | P06–P08 |
| R05 | At least one controlled ablation | Matched runs differing in one declared factor | P07–P08 |
| R06 | Explainability visualization | Class-specific Grad-CAM or other verified explanation and limitations | P09 |
| R07 | Average inference latency and device/browser environment | Warm/cold timing protocol, recorded measurements, app display | P08, P10–P11 |
| R08 | Hand-gesture robustness by environment | Real evaluation samples with environment tags and group counts | P04, P08 |
| R09 | Ethics, consent, and synthetic disclosure | Dataset consent/provenance notes; labels and notice in app/report/video | P04, P07, P10, P13 |
| R10 | Heavy generators may run offline | Reproducible offline generation script/notebook, metadata and displayed outputs | P03, P07, P10 |
| R11 | Report PDF, 6–8 pages | Rendered and visually checked final PDF | P13 |
| R12 | One-page Neurohive-style method explanation, diagram, relevant Neurohive citation | Method section with primary-source cross-checks | P03, P13 |
| R13 | Repository and reproducibility | Repo link, pinned environment, run guide, obtainable weights and inputs | P02, P12 |
| R14 | Demo video, 3–5 minutes | Playable video link, duration check, matching final build | P14 |
| R15 | Moodle submission and practice defense | Submission receipt/status, working links, defense notes | P15 |

Scope decisions: web app preferred over mobile; a local run guide is an explicitly permitted delivery option. Implement upload and camera if feasible; file upload remains the reliable fallback when browser camera permissions fail. mAP and IoU/Dice are not required for this classification task unless a detection/segmentation claim is added. Optional physics enrichment and a second SOTA method are out of scope until mandatory gates pass. A1 notebook/report repair is separate backlog: A2 requires its own report and repo, not explicitly a notebook, though a runnable experiment notebook is useful if training is remote.

## 2. Known issues — do not lose these across sessions

All entries below remain OPEN until evidence closes them. At first execution, add status, resolution evidence and closure date to each issue. I10 is A1-only backlog, not an A2 completion blocker; do not close it merely to make the A2 dashboard green.

| ID | Finding and evidence | Risk | Required resolution / blocking gate |
|---|---|---|---|
| I01 | `collect_data.py` draws class text before `cv2.imwrite`; inspected samples contain it | Label leakage; model may read labels | Preserve originals, clean consistently, visually verify; blocks P05–P08 final runs |
| I02 | Sampled photos show one person and largely one setting | Poor generalization; related frames across splits | Collect independent sessions/environments; group split; limit claims if unavailable; P04/P08 |
| I03 | Collection counter starts at zero each run | Existing data overwritten | Append-safe or unique session filenames; regression check; before any new collection |
| I04 | `demo.py` passes OpenCV BGR into RGB-trained model | Training/inference mismatch | Shared preprocessing with explicit input color contract; P05/P10 |
| I05 | Demo applies softmax to already-softmax output | Incorrect displayed confidence | Use output probabilities directly; P05/P10 |
| I06 | MobileNet curve is pre-fine-tuning; CM is produced after fine-tuning in current code | Ambiguous experiment attribution | Separate stage/run IDs and saved histories/metrics; P06/P08 |
| I07 | Precision/recall/F1 printed, not persisted | Results cannot be traced or audited | Save structured predictions, metrics and history; P06/P08 |
| I08 | No dependency manifest; available Python lacked TF, OpenCV, NumPy, sklearn during audit | Cannot reproduce/load/train | Isolated tested runtime and pinned versions; P02 |
| I09 | Saved models identify Keras `3.16.0.dev2026101004` | Compatibility not verified | Test loading and inference; do not assume compatibility; P02 |
| I10 | A1 notebook and 3–5 page report absent from this folder | A1 deliverables incomplete locally | Track separately; do not silently mark done or let this replace A2 deliverables |
| I11 | Validation-only evaluation with random image split | Similar-session leakage; no independent generalization estimate | Freeze group-aware train/val/test protocol before experiments; P04 |
| I12 | `git rev-parse` failed in this workspace | Repo deliverable not yet satisfied | Establish writable repository/export path without altering protected metadata; P12 |

Audit facts: 151 peace + 163 stop + 150 thumbs_up = 464 JPEGs, all 640×480; image decoding passed; no byte-identical duplicates. This does not rule out near duplicates. Three `.keras` archives and four plot PNGs exist. Python syntax passed. Runtime inference and training were NOT tested. Existing CMs imply 70/92 = 76.1% baseline and 89/92 = 96.7% MobileNet accuracy; these are historical, leakage-affected results, not fresh verified benchmarks.

## 3. Dependency table and work order

Status vocabulary: `TODO`, `IN_PROGRESS`, `BLOCKED`, `IN_REVIEW`, `PASS`, `STALE`. A task is PASS only with artifacts and successful checks. `STALE` means prior evidence was invalidated by an upstream change.

| Order | Task | Hard dependencies | Can start early / parallel work within one agent | Gate / initial status |
|---|---|---|---|---|
| P01 | Preserve A1 and record project inventory | Execution authorization | None | G01 / PASS |
| P02 | Establish runtime and compute | P01 | P03 source research | G02 / PASS |
| P03 | Select feasible, eligible SOTA and freeze experiment design | P01; P02 for executable feasibility | UI wireframe and report outline | G03 / PASS |
| P04 | Fix collection, clean data, consent, freeze splits | P01; P02 for tooling | Independent of generator availability | G04 / PASS |
| P05 | Implement shared preprocessing and training/evaluation contracts | P02, P04 | Generic inference interface can be drafted earlier | G05 / PASS |
| P06 | Reproduce clean A1 baselines | P03, P05 | P07 preparation after P03/P04/P05 | G06 / PASS |
| P07 | Integrate SOTA; train improved model and ablation | P03, P04, P05 | Does not need P06 training to finish | G07 / PASS |
| P08 | Freeze runs and evaluate all models fairly | P06, P07 | Metric code may be prepared earlier using fixtures | G08 / PASS |
| P09 | Implement and verify explainability | P05 + a trained model from P06/P07 | Prototype on interim model; final check needs final selected weights | G09 / PASS |
| P10 | Integrate interactive app and disclosures | P03, P05, P09; P07 for SOTA artifacts | UI skeleton early; final integration needs chosen weights | G10 / PASS |
| P11 | End-to-end QA and performance validation | P08, P09, P10 | None | G11 / PASS |
| P12 | Reproducible repo and delivery package | P11 | Dependency docs and run guide drafted from P02 onward | G12 / PASS (private owner access; reviewer access pending P15) |
| P13 | Final 6–8 page report | P08, P09, P11; P12 local reproducibility evidence | Outline and method references start after P03 | G13 / PASS |
| P14 | Record 3–5 minute video and prepare defense | P11, P13; P12 local reproducibility evidence | Script outline early | G14 / BLOCKED |
| P15 | Final audit and submission handoff | P12, P13, P14 | None | G15 / BLOCKED |

Critical path: preservation → usable runtime/data/SOTA feasibility → consistent preprocessing → matched experiments → evaluation → final app QA → reproducibility/report → video → submission. Do not delay P04 while polishing UI. Do not claim SOTA integration because a dropdown or sample video exists.

## 4. Detailed execution steps and AI review checks

### P01 — Preserve and establish evidence

- [x] P01.1 Inventory source, images, weights, plots with sizes and SHA-256; save `artifacts/audit/a1_inventory.json`.
- [x] P01.2 Preserve original files unchanged. Use separate versioned output directories; make an independent backup if storage permits. Do not rerun current `train.py` into existing `models/` or `results/`.
- [x] P01.3 Record original class order, model metadata, collection/training settings and historical-result caveats.
- [x] P01.4 Inspect applicable local instructions; establish proposed project structure and evidence paths.
- [x] **G01 review:** original inventory exists; hashes are reproducible; output paths cannot overwrite A1 artifacts. Record evidence and mark P01 PASS.

### P02 — Runtime and compute feasibility

- [x] P02.1 Inspect Python, RAM/disk, CPU/GPU availability and permitted remote runtime. Record actual hardware, not assumed Colab availability.
- [x] P02.2 Create isolated environment; resolve compatible classification dependencies. Record versions and install commands. Keep generator dependencies separate if they conflict.
- [x] P02.3 Load each saved A1 model in a compatible runtime and run one valid RGB batch; record shape, finite probabilities and class mapping. If incompatible, record failure and reproduce from architecture; do not claim original weights were evaluated.
- [x] P02.4 Run a tiny disposable training smoke test to estimate classification cost; never use smoke-test metrics in the report.
- [x] P02.5 Identify accessible generator compute, quotas, download size and budget. Do not provision paid resources without user authorization.
- [x] **G02 review:** dependency imports and classification smoke tests have logs; runtime reproducible; available compute documented. Original-weight compatibility may resolve either by successful loads or documented incompatibility plus a successful reconstruction smoke test; the latter is not an evaluation of original weights. Generator feasibility is owned by G03 and does not prevent classification tooling/data work. Unavailable hardware is an explicit blocker for dependent work, not an excuse to fabricate runs.

### P03 — SOTA decision and experiment contract

- [x] P03.1 [Alternative branch] Review ConvNeXt primary paper/repository, relevant Neurohive citation, implementation/weights/source hashes and licensing scope; see feasible_alternative.md.
- [x] P03.2 [N/A after method selection] Step-Video standard deployment infeasible on 4 GiB GPU; selected ConvNeXt alternative instead. No video generation claimed.
- [x] P03.3 Run one small genuine pilot on accessible compute; record command, model revision, seed, time, memory, output and visual judgment. A downloaded unrelated demo is not a project generation run.
- [x] P03.4 If infeasible, evaluate authentic ArtAug or another modern method allowed by the assignment. Ordinary gamma correction, blur, or CLAHE must not be renamed ArtAug. An alternative needs primary evidence of modern-method relevance and a valid assignment/citation mapping; record any instructor clarification needed rather than inventing approval.
- [x] P03.5 Timebox feasibility to approximately two focused hours plus a bounded pilot job. When blocked, continue data/UI-independent tasks and present concrete options; do not silently remove R02.
- [x] P03.6 Freeze method, why it helps, app wireframe, primary metric (macro-F1), split policy, checkpoint-selection rule, compute budget, seed(s), ablation factor and synthetic sampling ratio BEFORE final runs.
- [x] **G03 review:** `artifacts/method/decision.md` has one selected method, source links, pilot evidence, resource plan and experiment matrix. No unsupported SOTA claim. If no method passes, P03 is BLOCKED.

### P04 — Repair data collection, clean data and freeze splits

- [x] P04.1 Save clean camera frame before drawing overlays; draw UI only on a display copy. Fix append/unique session naming and check write success.
- [x] P04.2 Verify collection twice in a temporary test directory: no overwrites; output has no class text; existing images unchanged.
- [x] P04.3 Choose deterministic cleanup for original text-contaminated images. Prefer a fixed crop removing the complete overlay region if it preserves useful hand content; inspect full-resolution samples before choosing coordinates. Do not use class-dependent crops or inpainting that could create new shortcuts.
- [x] P04.4 Audit all cleaned images for decode failures and overlays; inspect contact sheets of all samples, investigate ambiguous/empty/occluded gestures. Quarantine with reasons instead of silently relabeling. Record retained per-class counts; if below A1's stated range, collect replacements.
- [ ] P04.5 [DEFERRED by explicit user scope; not required to run the accepted limited protocol.] Collect new clean sessions across lighting/background conditions. Proposed target: three named conditions with at least 20 independent examples per class per condition where practical; this is a project target, not a grading minimum. More independent sessions are preferable to rapid adjacent frames.
- [x] P04.6 Obtain consent for identifiable participants; avoid/background-crop bystanders or exclude samples without permission. Record consent availability without committing private identity documents. Do not claim consent based on assumption.
- [x] P04.7 Write dataset manifest: sample ID, original hash, derived path/hash, label, real/synthetic, person/session/clip ID where known, environment, transformation, provenance/consent status, split. Mark unknown metadata unknown.
- [x] P04.8 Split by session/person/near-duplicate group, not adjacent frame. Prefer independent newly captured sessions for real test data. Use train/validation for tuning; lock test labels/data from model selection. A 70/15/15 target is optional; group separation and class representation take precedence.
- [x] P04.9 Keep all derivatives of one original and all frames from one synthetic clip together. Synthetic data may enter training only; keep separate generated stress demos clearly distinguished from real evaluation.
- [x] P04.10 Freeze manifest hash and split version. If independent sessions are unavailable, record the limited protocol and block claims of unseen-session/general-person performance.
- [x] **G04 review:** no readable label overlays; no known duplicate/group overlap; every class represented in evaluation; counts and exclusions reconcile; real test data is real; consent/provenance unresolved items excluded or explicitly block release. Human input required for collection/consent must remain visibly BLOCKED.

### P05 — Shared preprocessing and experiment infrastructure

- [x] P05.1 Centralize fixed crop, resize, color conversion and model-specific normalization. Define RGB/BGR input contract; apply crop exactly once. Keep normalization inside existing models where already embedded.
- [x] P05.2 Fix inference confidence: no second softmax. Persist class order with weights; never infer order from an unrelated hard-coded list.
- [x] P05.3 Use versioned manifests for loading; training-only augmentation; deterministic validation/test order. Pair labels and predictions in one pass or use stable sample IDs.
- [x] P05.4 Save config, seed, package/device info, dataset/split hash, checkpoint-selection rule, histories and per-sample predictions under `runs/<run_id>/`.
- [x] P05.5 Configure training to save the best validation checkpoint separately from the last checkpoint; distinguish frozen and fine-tuned phases. Review fine-tuning normalization-layer behavior against framework documentation.
- [x] **G05 review:** same image via file and simulated camera paths yields equivalent model input/probabilities; RGB color fixture passes; crop not doubled; probability shape/range/sum valid; no test augmentation; labels aligned; saved/reloaded model agrees within stated numeric tolerance.

### P06 — Clean A1 comparisons

- [x] P06.1 Train B0: original simple CNN architecture on clean real training data using frozen protocol.
- [x] P06.2 Train B1: original MobileNetV2 transfer/fine-tuning approach on the same real data. Save frozen-stage and fine-tuning histories separately.
- [x] P06.3 Preserve old A1 models/results as historical only. Clearly distinguish historical reported A1 from clean re-trained A1 comparisons.
- [x] P06.4 Record parameter count, settings, total training time, chosen epoch and checkpoint hash. Select on validation only.
- [x] **G06 review:** both runs complete with full histories/configs; clean dataset/split hashes match; checkpoints reload; historical contaminated accuracy is not used as a directly comparable improvement claim.

### P07 — ConvNeXt integration and controlled pretraining ablation (selected branch)

- [x] P07.1 Freeze ConvNeXtTiny spatial backbone, pooled Dense3 head and raw RGB preprocessing.
- [x] P07.2 Load authentic official ImageNet weights; record checksum, source revision and genuine pilot outputs.
- [x] P07.3 Verify finite input-dependent features, save/reload parity and real Grad-CAM compatibility.
- [x] P07.4 Record zero synthetic training share and actual method disclosure.
- [x] P07.5 Train S1 with frozen ImageNet ConvNeXt and A0 with frozen random ConvNeXt on identical real train IDs.
- [x] P07.6 Verify identical initial head weights, seed, optimizer, epochs, steps and augmentation policy; only backbone initialization differs.
- [x] P07.7 Freeze the non-generative contract before execution; retain original A1 comparisons B0/B1 separately.
- [x] **G07 review:** actual runs/history/checkpoints verified; no test tuning; controls and mixed outcomes recorded.

### P08 — Final evaluation and robustness/latency results

- [x] P08.1 Freeze final checkpoints and configs using validation only, then evaluate B0, B1 and S1 on exactly the same real held-out sample IDs.
- [x] P08.2 Save accuracy, macro-F1, per-class precision/recall/F1/support, raw confusion matrices, probabilities and predictions. Record zero-division policy.
- [x] P08.3 Report results by environment with support counts, not just one aggregate score. Show failures, minority-class weaknesses and limitations of a small correlated dataset.
- [x] P08.4 Plot both training stages with correctly labeled epochs; derive comparison and ablation tables from saved files, not manual transcription.
- [x] P08.5 Benchmark batch-size-1 inference on stated device: cold start separately; proposed 10 warmups plus at least 100 timed calls with synchronization/materialization where needed. Record mean, median, p95 and sample count. Keep explanation and end-to-end app latency separate from classifier time.
- [x] P08.6 [Single-seed branch; variation not measured.] Where time permits, repeat matched runs with three seeds and report variation. Otherwise disclose single-seed results. Confidence intervals, if used, should respect session grouping rather than pretend adjacent frames are independent.
- [x] **G08 review:** CM totals equal prediction count; recomputed metrics match tables; ablation comparison is controlled; all evaluated samples are identical across methods; no unsupported gain/generalization claims. If test data influences a fix/tuning decision, label subsequent results exploratory or create a fresh held-out set.

### P09 — Explainability

- [x] P09.1 Implement Grad-CAM on the actual classifier, with explicit target class and correct convolution layer/gradient path. If model structure prevents reliable gradients, refactor/export equivalently or choose another assignment-accepted explanation and document why.
- [x] P09.2 Project heatmap onto the correctly cropped image; if shown on the original, account for crop offset. Display target class and method name.
- [x] P09.3 Verify non-None, finite gradients and non-degenerate maps. Handle zero activation gracefully; never generate decorative/random heatmaps.
- [x] P09.4 Save representative correct and incorrect examples across classes/environments. Inspect attention to hands versus background and removed-overlay boundary; explain that heatmaps are diagnostic, not proof of causal reasoning.
- [x] **G09 review:** map changes with input/target class where expected; selected model/checkpoint identified; geometry correct; app and report examples derived from real computations. Repeat checks if weights/preprocessing change.

### P10 — Interactive app

- [x] P10.1 Implement local web app with file upload and camera input, image preview, class probabilities and visual explanation; use a shared inference service rather than duplicate preprocessing.
- [x] P10.2 Show active model, average inference time, backend device and browser/client environment. Distinguish server hardware from user's camera device; do not invent device details.
- [x] P10.3 [Non-generative branch] Display actual ConvNeXt/pretraining comparison and zero synthetic training images; no generated-media panel required or claimed.
- [x] P10.4 Provide camera denied/unavailable fallback, invalid/oversized-file errors, model-loading state, inference failure handling, clear/reset and resource cleanup.
- [x] P10.5 State supported classes and closed-set limitation: an unrelated image can still get a confident class. Do not imply a validated no-hand detector unless implemented/tested. Any confidence threshold must be chosen on validation.
- [x] P10.6 Add synthetic disclosure, consent notice for people, and accurate image-retention policy. Default to transient processing; do not log camera images by accident.
- [x] **G10 review:** upload → real prediction → real explanation works; camera exercised where available (otherwise visibly unverified); probabilities/class mapping correct; generated media clearly labeled; no placeholder outputs presented as model results.

### P11 — Integrated QA

- [x] P11.1 Test each gesture, difficult lighting/backgrounds, no hand/unrelated input, invalid upload and denied camera. Record expected versus observed behavior; do not relabel failures as successes.
- [x] P11.2 Compare CLI and app outputs on the same fixed fixtures; check model selection and crop/color handling across all routes.
- [x] P11.3 Measure app latency and repeated requests; verify no repeated model reload, stale explanation, request cross-contamination or unbounded resource use.
- [x] P11.4 Perform a fresh startup on the intended defense machine. Record browser/OS/runtime and camera availability.
- [x] **G11 review:** all mandatory paths verified, exceptions documented, no leakage or inference regression, displayed measurements honest. Hardware-only checks require actual evidence, not static inspection.

### P12 — Repository and reproducibility

- [x] P12.1 Package modular code, pinned dependencies, configs, data-cleaning/splitting/generation/training/evaluation scripts and run guide. Add remote notebook only where needed.
- [x] P12.2 Include commands for setup, model acquisition, inference/app, training, evaluation and reproducing figures; weights have stable paths and hashes.
- [x] P12.3 Resolve writable git checkout/export location; do not delete or modify protected metadata to force initialization. Prepare repository contents and publication handoff; external upload must follow user authorization and consent scope.
- [x] P12.4 Exclude credentials, environments, temporary outputs and unconsented images; document licensed model weights and dataset access. Do not assume all original camera images are safe to publish.
- [x] P12.5 Test from a fresh environment/export: download/access permitted weights, start app, run sample prediction and regenerate metrics from saved predictions.
- [ ] **G12 review:** record two results: G12-local (fresh-environment reproducibility) and G12-publication (accessible repository URL). P13/P14 can proceed after G12-local passes while publication remains blocked. P12 overall remains incomplete until both pass. A local run guide satisfies app delivery but does not replace the repo-link requirement.

### P13 — Report, 6–8 pages

- [x] P13.1 Build an approximately 7-page report: (1) objective/A1 context; (2) dataset cleanup, split and synthetic share; (3) one-page method explanation with simple diagram and sources; (4) experiment setup and curves; (5) results/ablation/environment/latency; (6) explanation/failures/ethics; (7) limitations, reproducibility and references. Adjust to keep total PDF 6–8 pages including references unless instructor specifies otherwise.
- [x] P13.2 Cite relevant Neurohive article as required; cross-check technical claims against primary paper/repository. Distinguish generative quality from downstream classification performance.
- [x] P13.3 Explain embedded-label cleanup, corrected baseline, sampling/split limits, consent, synthetic disclosure, bias, misuse and storage policy honestly.
- [x] P13.4 Pull all numbers/figures from frozen evidence. Include negative or mixed findings; no fabricated gains, runtime measurements or consent claims.
- [x] P13.5 Render PDF, inspect every page for readability/clipping, verify captions/links/citations, check 6–8 page count and matching final model/version.
- [x] **G13 review:** all report-specific R01–R13 content is present and verified; unresolved mandatory content blocks PASS (a documented scientific limitation is not a missing result); report matches evidence; page count and visual QA pass.

### P14 — Demo video and defense

- [x] P14.1 Prepare approximately 4-minute script: 0:00–0:30 topic/A1 and cleanup; 0:30–1:45 upload/camera predictions and explanation; 1:45–2:30 ConvNeXt and real failure example with disclosure; 2:30–3:20 comparison/ablation/latency; 3:20–4:00 limitations/ethics/reproducibility.
- [ ] P14.2 Record actual final app with legible results and audible explanation. Show offline generation as offline; no simulated live processing. Include a failure/limitation if practical.
- [ ] P14.3 Check duration 3–5 minutes, playback, sound, captions if used, link access and consistency with report/repo.
- [ ] P14.4 Prepare defense answers: why method qualifies, what changed from A1, label leakage, group splitting, ablation fairness, Grad-CAM interpretation, latency protocol and ethics. Student confirms practice-lesson defense time.
- [ ] **G14 review:** playable accessible video with correct duration, real execution, clear synthetic labeling and final-build identity; recording/upload tasks requiring user actions remain open until confirmed.

### P15 — Final audit and submission

- [ ] P15.1 Audit R01–R15 against evidence; scan this plan for OPEN/BLOCKED/STALE items and resolve or disclose them.
- [ ] P15.2 Freeze release version/hash; verify PDF, repository, app link or local guide, weights and video links from the reviewer perspective.
- [ ] P15.3 Prepare exact Moodle submission checklist and files/URLs. Submit only with explicit authorization/access; otherwise provide user handoff and do not mark submitted.
- [ ] P15.4 Record actual submission time (Asia/Almaty), receipt/confirmation and remaining defense obligation. Target handoff by 7 November 20:00, leaving buffer before 23:59.
- [ ] **G15 review:** record G15-ready (complete bundle and working links) and G15-submitted (actual submission confirmation) separately. Record defense as pending/completed independently; submission does not establish defense completion. Deadline or defense completion is never inferred from artifact creation.

## 5. Execution loop and checkpoint rules

1. Read this file, applicable instructions and the latest log. Verify actual filesystem and any running jobs; do not repeat expensive work merely because conversation context was lost.
2. Choose the earliest TODO task whose hard dependencies PASS. A permitted early draft is not a completed gate. Mark current task/subtask IN_PROGRESS before work.
3. Execute a bounded unit. Keep originals safe and outputs versioned. Capture commands, exit codes, paths, versions and relevant logs without secrets.
4. Run the listed review checks. Inspect actual results; a tool returning without an error is not sufficient evidence of model/data correctness.
5. Mark a subtask checked only after its acceptance record passes review. Once all required subtasks pass, mark the stage IN_REVIEW, run its gate, and record PASS only when all mandatory gate criteria pass. Same-agent self-review is allowed; do not claim independent review. Do not spawn agents unless the user or applicable instructions authorize delegation.
6. If a check fails, record failure, likely cause, next corrective action and affected downstream tasks. Try a reasoned correction; after two substantially identical failures, reassess instead of looping blindly.
7. If human data/consent/compute/access is missing, mark BLOCKED with exact unblock condition. Continue independent eligible work. Do not invent evidence or treat elapsed time as approval.
8. Update resume checkpoint, issue register, task status table, artifact index and append-only log before stopping or yielding. One task's PASS does not close unrelated issues.
9. For long training/generation, record job/session ID, command, runtime location, start time, expected output, last observed epoch and resume checkpoint. Poll existing job before launching another; record interruption/failure and partial outputs.
10. On upstream changes, mark affected dependent gates STALE and invalidate their completion checkboxes. Dataset/split changes invalidate all final comparisons; preprocessing/weights changes invalidate inference, explanations, latency and report/video evidence. UI-only edits require appropriate UI checks, not needless retraining.
11. Never report completion with mandatory BLOCKED/STALE items. State implemented, verified, unverified and user-required work separately.

### Per-task checkpoint record (copy for every execution unit)

```yaml
task_id: P01
subtask_id: P01.1
status: IN_PROGRESS
started_at_almaty: null
finished_at_almaty: null
owner: primary_agent
dependencies_verified: []
inputs_and_hashes: []
commands_or_actions: []
outputs_and_hashes: []
checks:
  - name: null
    result: NOT_RUN
    evidence_path: null
issues_closed: []
issues_opened: []
reviewer: self_review
review_findings: []
job_id_and_location: null
failure_or_blocker: null
next_action: null
upstream_changes_invalidating_downstream: []
```

### Run registry (fill after jobs exist)

| Run ID | Role | Data/split hash | Method/config/seed | Weights hash | History/metrics path | Status |
|---|---|---|---|---|---|---|
| B0_limited_v1 | B0 | a1_limited_v2; hash in final_evidence_index.json | configs/B0.convnext_v1.json; seed 123 | d78fed9e9e52ba3b8a72d94c78fc1bbdf7dc4520a75bd4e5b3068781bfad3a10 | runs/B0_limited_v1/ | VERIFIED |
| B1_limited_v1 | B1 | a1_limited_v2; hash in final_evidence_index.json | configs/B1.convnext_v1.json; seed 123 | f38422e9cf7144a70b2330f7a7a13c93bddb3d7e2d7071f727a077389d606e9a | runs/B1_limited_v1/ | VERIFIED |
| S1_limited_v1 | S1 | a1_limited_v2; hash in final_evidence_index.json | configs/S1.convnext_v1.json; seed 123 | f177f4a9aa4db75fffec1b01b322cd05de0231e58a5d2ee2e30b57d07f3f2eee | runs/S1_limited_v1/ | VERIFIED |
| A0_limited_v1 | A0 | a1_limited_v2; hash in final_evidence_index.json | configs/A0.convnext_v1.json; seed 123 | 6f4aa7252f523650a02f47374be0b9946483dbd3e16349213893c27daf5dd30a | runs/A0_limited_v1/ | VERIFIED |

### Evidence index (planned paths; existence is not implied)

| Evidence | Proposed location |
|---|---|
| A1 immutable inventory and audit | `artifacts/audit/` |
| Method choice, sources and pilot | `artifacts/method/` |
| Dataset versions/manifests and card | `data/manifests/`, `docs/dataset_card.md` |
| Cleaned derived real data / synthetic data | `data/clean/`, `data/synthetic/` |
| Versioned experiments | `runs/<run_id>/` |
| Metrics, figures, explanations and QA | `artifacts/evaluation/`, `artifacts/explainability/`, `artifacts/qa/` |
| Final report, demo script and submission checklist | `deliverables/` |
| Setup and run instructions | `README.md`, dependency manifest |

### Append-only execution log

| Date (Almaty) | Task | Event | Evidence / next action |
|---|---|---|---|
| 2026-11-04 | PLAN | Plan created from user-supplied A2 requirements and prior audit. Git availability checked: not a usable repository here. Method references located. | No implementation or final experiment has run; next P01.1 after execution authorization. |

## 6. Deadline schedule and contingency

These are target windows, not promises or completed milestones.

| Date (Asia/Almaty) | Target | Must not defer |
|---|---|---|
| Nov 4 | P01–P05; method pilot; collection/cleanup/split; initial UI outline | SOTA feasibility, overwrite fix, leakage cleanup, compute access |
| Nov 5 | P06–P09; baseline/improved/ablation runs; explanations | Controlled comparisons and real held-out data |
| Nov 6 | P08–P13; app QA, fresh-run guide, report draft | Metrics traceability and final inference correctness |
| Nov 7 | Final report QA, P14–P15; handoff target 20:00 | 3–5 min video, working repo/run guide, submission before 23:59 |

If behind: cut second SOTA, visual polish, live generation and optional enrichment first. Keep one genuine method, fair comparison, one ablation, explanation, honest latency, ethics and required deliverables. If compute prevents the selected method, resolve P03 early using a verified feasible alternative; do not present conventional augmentation as SOTA. If fresh collection is unavailable, document narrower evaluation and limits rather than manufacture diversity. Student actions likely needed: real collection/consent, accessible compute if absent, practice defense, and final account-specific submission.

## 7. Sources and unresolved method decision

The assignment permits another modern SOTA, but requires a relevant Neurohive citation in the report. Do not assume the earlier assistant's use of “approved” means instructor approval of a particular implementation. Listed methods are expressly suggested; alternatives require a defensible mapping and clarification if citation compliance is ambiguous.

Sources located on 2026-11-04; recheck exact versions before implementation:

- Step-Video-T2V primary repository: https://github.com/stepfun-ai/Step-Video-T2V
- Step-Video-T2V primary paper: https://arxiv.org/abs/2502.10248
- Step-Video-T2V model card: https://huggingface.co/stepfun-ai/stepvideo-t2v
- Required-style Neurohive reference: https://neurohive.io/en/state-of-the-art/step-video-t2v-text-to-video-open-source-model-achieves-16x-video-compression-breakthrough/
- ArtAug Neurohive reference: https://neurohive.io/en/state-of-the-art/artaug-a-novel-synthesis-understanding-framework-for-image-generation-enhancement/
- ArtAug paper linked by that article (read before choosing): https://arxiv.org/abs/2412.12888
- ArtAug implementation repository linked by article (verify exact implementation before using): https://github.com/modelscope/DiffSynth-Studio

Planning conclusion: generator availability and hand correctness are unresolved feasibility risks. This plan intentionally requires a genuine pilot before committing final experiments; neither listed method is assumed to be a simple drop-in image filter. Primary sources control technical claims if a secondary article is imprecise.


## 8. Execution-readiness amendments (v1.1)

These rules refine the preceding task descriptions. They are mandatory for the agent loop and resolve ambiguous completion criteria; they do not claim that the implementation has passed.

### 8.1 Start/resume instruction for the executing agent

> Read this entire plan and the latest checkpoint. Carry out the earliest eligible task after the user has requested execution. Do not ask permission again for ordinary authorized implementation steps. Before acting, write its acceptance record. Implement, run the recorded checks, inspect evidence, and review against both its task criteria and applicable assignment requirements. Fix failures before marking it complete. Update this file and evidence records after each bounded unit. Continue to the next eligible task without requesting confirmation at every gate. Stop only for a real blocker, missing authorization for an external action, or a user instruction to stop; continue independent authorized work when possible. Never substitute a plan, stub, mocked prediction or unchecked assertion for completed work.

This is suitable for GPT Astra or another capable coding agent. No model can guarantee completion without actual compute, data, permissions and external access. This file is a manual execution protocol, not a background scheduler: it does not launch an autonomous service by itself.

### 8.2 Subtask acceptance records and state transitions

Before starting EACH numbered subtask, create `artifacts/reviews/<subtask_id>.md` with these fields. Instantiate concrete paths and expected values; do not leave `TBD` in a completed record.

| Field | Required content |
|---|---|
| Identity | Subtask ID, plan version, current state, implementing agent |
| Requirement mapping | Relevant Rxx and Ixx IDs; upstream PASS records |
| Preconditions | Actual input paths/hashes; runtime; branch of conditional task |
| Deliverable | Exact expected files or observed behavior |
| Acceptance criteria | Observable assertions covering EVERY clause in the numbered subtask |
| Verification procedure | Exact command or manual procedure, fixtures, expected output and numeric tolerance if relevant |
| Actual evidence | Exit code, log/result paths, artifact hashes, observation and timestamp |
| Review | Read outputs and compare against criteria; list defects and unresolved uncertainties |
| Decision | PASS/FAIL/BLOCKED, reason, reviewer identity (self-review if same agent) |
| Continuation | Next eligible subtask; invalidated downstream evidence if any |

Rules:

- Valid transitions: TODO → IN_PROGRESS → IN_REVIEW → PASS; IN_REVIEW → IN_PROGRESS on correctable failure; any active state → BLOCKED on a concrete external dependency; PASS → STALE on relevant input changes. BLOCKED/STALE → IN_PROGRESS only after recording the resolved prerequisite.
- A checkbox means reviewed PASS, not “code written.” No stage may pass if a mandatory subtask or review check is unexecuted, failed, blocked or stale.
- Conditional tasks can be `N/A` only inside a chosen method branch, with an explicit reason, requirement mapping and reviewer acceptance. N/A cannot remove an assignment requirement. Update the main checklist annotation so it is not mistaken for executed work.
- Drafting independent downstream work is allowed; it must be labeled DRAFT/UNVERIFIED and cannot clear a gate whose dependencies have not passed.
- Expected outcomes must be declared before running a check. Never weaken a threshold retrospectively merely to obtain PASS; amend the design with a reason and invalidate affected evidence.
- Keep one authoritative stage status in section 3 and the current task in section 0. Gate records live in `artifacts/reviews/Gxx.md`; subtask records link to them. On disagreement, mark state unresolved and reconcile against evidence rather than choosing the more favorable status.
- Use atomic file replacement when updating this checkpoint. Record work in progress before long jobs so interruption does not lose the command/job identity.

### 8.3 Concrete gate checks to supplement section 4

The values below are proposed project verification defaults, not instructor scoring thresholds. Freeze any hardware-specific changes before final experiments.

| Gate | Required operational checks and evidence |
|---|---|
| G01 | Recompute original dataset/weights/plots hashes and compare with inventory: zero unexpected differences. Archive source snapshots before editing scripts; source edits are allowed afterward and tracked. Do not interpret “preserve originals” as banning fixes to working code. |
| G02 | One classifier batch produces shape `(N,3)`, all finite probabilities, each in [0,1] and row sums within 1e-5 of 1. Tiny training run writes/reloads a checkpoint. Capture exact interpreter and package versions; record generator compute unknown separately. |
| G03 | Decision file specifies exact method/weights revision, executable pilot output and license; all experiment fields below populated. Only browsing papers does not pass this gate. |
| G04 | Manifest IDs unique; paths decode; labels valid; hashes/counts reconcile; train/val/test nonempty with all classes; group intersections and exact-duplicate intersections empty. Visual-review index covers all retained images and rejected cases. Log near-duplicate method/threshold and human resolution of cross-split candidates. |
| G05 | Test a known RGB color fixture through file and camera adapters. Compare identical decoded pixel inputs (avoid lossy re-encoding); float32 outputs agree at atol=1e-5, rtol=1e-4 on same runtime, or document predeclared precision-specific tolerance. Two collection sessions preserve old file hashes. Reloaded model predictions pass the same tolerance. |
| G06 | B0 and B1 history files contain every completed epoch and phase; final checkpoint exists/reloads; no NaN/Inf losses; selected epoch follows the frozen validation rule. Failed or interrupted runs cannot be summarized as completed. |
| G07 | Synthetic samples trace to genuine generation logs and accepted review IDs. Matched ablation configs differ only in the declared factor and necessary sampling settings; list every difference. Reject missing per-class synthetic coverage or amend experiment scope before final runs. |
| G08 | Recompute metrics independently from saved labels/predictions: absolute numeric discrepancy ≤1e-6 before rounding. CM total equals test sample count; per-class and per-environment support sum correctly. Save timing repetitions and protocol, not just an average. |
| G09 | Save at least one inspected map per class and one failure example when failures exist. No fabricated failure if none exists. Confirm finite gradients, matching image geometry, and nontrivial maps on multiple valid samples; document zero-map cases. |
| G10 | At least one real uploaded example per class completes prediction/explanation. Camera works on target machine or is explicitly pending; inaccessible camera cannot be reported tested. Error paths yield an actionable UI message, not a traceback or fake prediction. |
| G11 | Scripted fixtures, manual browser checks and observed timing logs refer to the same release/model hash. Repeated requests preserve image-to-result correspondence. No arbitrary minimum accuracy/FPS is invented; observed performance and limitations are reported. |
| G12-local | Fresh exported checkout/environment follows documented commands to start app and evaluate a fixture, without hidden absolute paths or undocumented files. Record commands and outputs. |
| G12-publication | Repository link and required artifact links are accessible to the intended reviewer; no secrets/private consent records included. |
| G13 | PDF has 6–8 pages; every page visually inspected; all quantitative claims trace to run/evaluation artifacts; method citation and diagram present. No placeholder or unresolved mandatory result remains. |
| G14 | Video duration 180–300 seconds; playable video/audio, legible app/results and accessible link verified. Report/video/app model and metrics agree. |
| G15-ready | Checklist contains report PDF, app link or local guide, repository URL and video URL; each checked from reviewer perspective; unresolved A2 blockers zero. |
| G15-submitted | Actual receipt/confirmation recorded; defense stays separate and pending until student confirmation. |

### 8.4 Decision fields that must be frozen before final training

P03/P04/P05 together must write `artifacts/method/experiment_contract.md`. These are intentionally unresolved now; selecting them requires the runtime/data pilot. Final training must not start with missing values.

- Selected SOTA name, exact source/weights revision, genuine pilot evidence, license, compute location and method-specific acceptance criteria.
- App framework and startup command, inference location (server/device), camera mode (continuous or capture), browser requirement, local delivery versus hosting, model loading/caching strategy and file retention behavior. Select the framework during implementation using available capabilities and current documentation; do not prescribe unverified framework APIs here.
- Cleanup crop coordinates as normalized fractions or a documented resolution-specific rule; orientation/EXIF handling; camera mirror convention; input RGB order; resize interpolation; normalization location; original/processed-image coordinate mapping. UI previews and inference must use the same orientation.
- Dataset/split IDs and hashes, session grouping provenance, environments, seed(s), exclusion rules, test isolation and the explicitly permitted scope of performance claims.
- B0/B1/S1 architecture and initialization; frozen/fine-tune phases, epochs or step budget, batch size, optimizer/schedule, augmentations and stopping criterion. Save shared initial weights for B1/S1 or document deterministic recreation so initialization is not an uncontrolled difference.
- Validation checkpoint-selection metric, tie-break and selected app model policy. Choose deployed model on validation and practical constraints, never on held-out test accuracy. If SOTA underperforms, report it and keep its integrated demonstration accessible; a claimed upgrade need not fabricate an improvement.
- Synthetic prompts, number of clips, extraction stride/maximum frames per clip, acceptance criteria, per-class counts, mixture/sampling ratio, generation seeds and compute budget. Count both unique images and training exposure; repeated generated frames are not independent examples.
- Final metric formulas/averaging, zero-division policy, supported classes, latency protocol, abstention policy if implemented, precision/tolerances and all planned comparisons.

### 8.5 Prevent dataset deadlocks and unfair claims

Original session metadata is not established. Do not assign independent session IDs from filename ranges without evidence. Conservatively group correlated original captures; require genuinely separate labeled sessions for independent validation/test claims. If this makes three-way grouping impossible, P04 remains blocked for final experiments until new data arrives or an explicitly recorded limited evaluation protocol is agreed. Merely writing a limitation does not make group overlap pass G04. Data cleaning, inference tooling and app drafts can continue meanwhile.

For novel environments, report per-environment sample and session counts. Separating by session does not establish person-independent performance when the same person appears throughout. Where backgrounds include identifiable bystanders, permission or exclusion must be resolved before releasing those images; private use and public distribution must be distinguished in the dataset card.

If using a non-generative SOTA alternative, rewrite P07 and P10.3 plus their gate checks to match the real pipeline before execution; synthetic share may legitimately be 0%. Keep a meaningful controlled ablation, environment evaluation and the explanation requirement. Do not force fake generated assets to satisfy an inapplicable branch.

### 8.6 Final completion is four separate states

| State | Required evidence |
|---|---|
| DEVELOPED_AND_VERIFIED | P01–P11 passed; scientific limitations disclosed; no missing mandatory app/experiment checks |
| DELIVERABLES_READY | G12-local, G12-publication, G13, G14 and G15-ready passed |
| SUBMITTED | G15-submitted passed with actual confirmation |
| DEFENDED | Student confirms completion during practice lesson |

A1-only backlog I10 may remain open without blocking A2. For other issues distinguish FIXED, ACCEPTED_LIMITATION and BLOCKED; acceptance of a limitation must identify affected claims and cannot waive a required deliverable, hidden label leakage, invalid metrics or missing SOTA execution.

### 8.7 Plan-review result

Review date: 2026-11-04. Reviewer: primary agent, self-review; no independent agent review claimed.

Changes from v1.0: added P03 dependency to baseline training; separated classification runtime readiness from generator feasibility; allowed reports/video to proceed after local reproducibility while repo publication waits; clarified original preservation versus working-code changes; added per-subtask evidence contracts, observable gate assertions, branching and state-transition rules, experiment decision fields, unknown-session handling and separate readiness/submission/defense states.

Plan status: **READY TO GUIDE IMPLEMENTATION WITH EXPLICIT DECISION GATES**. This is not a claim that runtime feasibility, final method selection, fresh data, camera testing, publication or submission have been completed. Historical plan-review note: execution was subsequently authorized; see current checkpoint and section 9.


## 9. Earlier execution checkpoint — superseded by section 10

User execution authorization received. Work progressed through all currently runnable independent units. **Only P01 is a fully passed stage; no final A2 model, metrics, explanation, report, video or submission is claimed.**

### Verified results

- P01: 474 originals archived and hashed; preserved data/models/plots still match. Working scripts changed only after source snapshots.
- P02.1: runtime inventory completed. Python 3.12.15 isolated data environment recovered from compatible cached NumPy/Pillow wheels. Full dependency installation failed with a recorded DNS error.
- P04.1–P04.4 and P04.7: clean capture storage and append-safe sessions implemented/tested; 464 crop derivatives validated and all six contact sheets reviewed; 11 stop samples without a gesture excluded, leaving 453 candidates; manifest records unknown provenance/consent honestly.
- 15 unit/contract tests passed. Coverage includes overwrite prevention, clean storage, BGR/RGB and crop consistency, direct probability handling, leakage rejection, metric arithmetic/alignment, input errors, no fabricated result when runtime is missing, and training preflight rejection.
- JavaScript and Python syntax checks passed. These are not browser, camera or TensorFlow verification.

### Prepared but not model-verified

- Shared inference and Grad-CAM implementation; local upload/camera UI and HTTP handlers.
- Baseline/MobileNet training implementation with guarded manifests/contracts, versioned outputs, best/last checkpoints, phase histories and no automatic original overwrite. TensorFlow execution is unverified.
- Metric comparison tool; report outline; video script outline; run guide. No invented experiment results or filled-in performance tables.

### Issue disposition

| Issue | Current state | Evidence / remaining condition |
|---|---|---|
| I01 | FIXED in derived candidates; original archive intentionally unchanged | top-20% crop; full visual review; originals never used automatically for final training |
| I02 | BLOCKED | new independent environments/sessions/persons unavailable |
| I03 | FIXED in storage path | two-session/exclusive-write regression test; real camera capture still pending |
| I04 | IMPLEMENTED, model verification blocked | BGR/RGB adapter tests pass; no actual model parity test |
| I05 | IMPLEMENTED, model verification blocked | direct-probability fixture passes; real model load pending |
| I06 | IMPLEMENTED, training verification blocked | per-phase versioned histories/checkpoints in new training code |
| I07 | IMPLEMENTED, experiment evidence blocked | structured histories/configs and metric tool; no final predictions yet |
| I08 | BLOCKED | data-only packages available; full runtime download failed |
| I09 | BLOCKED | no TensorFlow/Keras model-load evidence |
| I10 | OPEN A1-only backlog | not an A2 gate blocker |
| I11 | BLOCKED | one conservative original group; no independent frozen split |
| I12 | PARTIAL | local git snapshot at artifacts/repository; no external repo URL |

### Actual blockers and exact unblocking actions

1. **Runtime/network:** package downloads fail DNS; localhost socket creation returns PermissionError. Run installation and smoke tests from an online local terminal or provide an accessible ML runtime. See README.md. No approval escalation was attempted; environment restrictions remain in force.
2. **Data:** collect genuinely independent sessions with all three gestures and environment metadata, and confirm applicable participant/bystander permission or exclude/redact affected images. Do not invent sessions or set consent flags without confirmation. Current training validator intentionally fails.
3. **SOTA:** provide suitable accessible compute/weights for a genuine pilot, or investigate a feasible modern alternative once package/model access works. No method has passed G03.
4. **Later external steps:** real camera/browser QA, repository/video publication, Moodle submission and student defense remain pending after upstream work.

### Resume procedure

- Read this checkpoint and review files; verify `artifacts/audit/preservation_check.json` and `artifacts/qa/contract_tests.log` against current inputs.
- Restore P02 runtime and execute `.venv/bin/python -m scripts.runtime_smoke`; log failures truthfully, then freeze compatible dependency versions.
- Complete P03 pilot and P04 collection/consent/group split independently; update reviewed manifests and contract. Continue at P05 model consistency checks.
- Do not run final training using `configs/run.example.json`: it deliberately references an unfrozen contract and must fail.
- No active training or server job remains. Do not restart expensive work from a claim that it previously completed: it has not.

### Execution log additions

| Date (Almaty) | Tasks | Result | Evidence |
|---|---|---|---|
| 2026-11-04 | P01 | PASS | 474-file inventory and preservation check |
| 2026-11-04 | P02/P03 | BLOCKED | package DNS failure, no working GPU/camera; no SOTA pilot |
| 2026-11-04 | P04 | PARTIAL / gate BLOCKED | cleanup/review passed; 453 retained candidates; sessions/consent/split pending |
| 2026-11-04 | P05/P08/P09/P10 | PREPARATORY WORK / gates BLOCKED | 15 passing contract tests; no ML/browser execution; socket startup denied |
| 2026-11-04 | P12–P15 | DRAFTS / gates BLOCKED | run guide/report/video outlines; no final delivery claims |

Packaging checkpoint: `artifacts/repository/` is an initialized local git snapshot, not a published repository. Its commit is recorded in `artifacts/qa/repository_snapshot.json`. Exported-source contract tests pass (15 tests); this is not a fresh full-ML runtime test. `deliverables/A2_WORK_CHECKPOINT.zip` contains source, plan, manifests and text evidence; it deliberately excludes images, weights, environments and credentials. Apply it alongside the preserved original project inputs.

Backup status: persistent upload attempted after packaging and failed with DNS before saving. Local workspace files and local git snapshot remain authoritative; the previously saved external plan copy is stale. Do not assume a saved remote execution checkpoint exists.

### Resume execution started — 2026-11-04

P02.2 IN_PROGRESS: authorized elevated `uv pip install --python .venv/bin/python -r requirements-runtime.in` succeeded. Sandbox DNS/socket restrictions persist, but do not describe host network access. Preservation recheck: all 474 archived originals match; data/model/result originals unchanged. Data integrity passes on 464 records; 15 contract tests pass. Evidence: `artifacts/audit/resume_environment.json`, `artifacts/qa/resume_contract_tests.log`. Runtime smoke is running; no final training authorized by the still-unfrozen experiment contract.

## 10. Current execution checkpoint — 2026-11-04 resume

**P01 and P02 PASS (2/15). P03/P04 blocked; remaining final-stage gates blocked.** No active job or server remains. No final model, dataset split, final metric, report, video, publication or submission produced.

| Unit | Reviewed result | Evidence |
|---|---|---|
| P01 recheck | PASS: 474 archive hashes; original data/models/results unchanged | artifacts/audit/resume_environment.json |
| P02.2–P02.5 / G02 | PASS: resolved runtime, three model loads, tiny checkpoint reload, baseline training smoke, compute inventory | artifacts/reviews/G02.md; requirements-runtime.lock.txt; artifacts/qa/runtime_smoke.json |
| P03 feasibility | BLOCKED: standard Step-Video exceeds local compute; alternative citation/pilot/contract unresolved | artifacts/method/decision.md |
| P04 validation | Integrity PASS (464); training readiness correctly FAILS (915 errors: consent/splits/groups unresolved) | artifacts/qa/resume_training_readiness.log |
| P05/P09/P10 early diagnostics | Historical model only: actual probabilities, RGB/file parity, finite target/input-dependent Grad-CAM, repeat consistency PASS; final gates remain BLOCKED | artifacts/qa/diagnostic_smoke.json |
| HTTP early QA | Actual loopback routes PASS: legacy predictions 200, missing final model 503, invalid input 400; no browser/camera claim | artifacts/qa/http_smoke.json |
| Regression/preflight | 15 tests PASS; unfrozen example contract correctly rejected before training | artifacts/qa/resume_contract_tests.log; artifacts/qa/resume_preflight.log |

Issue updates: I08 and I09 CLOSED for this CPU runtime. I04/I05 verified with real historical-model diagnostics; final A2 parity remains dependent on final weights. I02/I11 remain BLOCKED on independent data/consent/group splits. I06/I07 remain pending full experiments. I12 remains PARTIAL, publication unperformed. All other dispositions unchanged.

Host checks corrected the old sandbox diagnosis: 4 GiB RTX 3050 and /dev/video0,/dev/video1 exist; host network and loopback work after approval. TensorFlow is CPU-only. Camera presence is not capture or consent evidence. Sources and resource limitations are recorded in the method decision. The alternative-method question requests compliance clarification only; section 7 already allows other modern methods.

Resume: do not reinstall a working runtime or rerun final training preflight until its inputs change. Resolve selected method/pilot and independent consented sessions, freeze P03/P04 contracts, then continue P05 and final experiments. Do not fabricate consent or split original adjacent frames into independent groups. Older repository/ZIP exports are STALE; workspace files are authoritative. This resume changed scripts and evidence, not original images/weights.

Protocol deviation: detailed P02 subtask records were written retrospectively from actual outputs; the main plan recorded runtime work in progress. No independent reviewer or pre-existing acceptance record is claimed. Self-review inspected all output assertions before marking G02 PASS.

## 11. Active continuation — user-authorized limited dataset protocol

2026-11-04: user confirms teammate agrees to project use and explicitly authorizes continuing with current data, expansion deferred. P04.5 new capture is DEFERRED under this scope; do not claim varied-environment R08 satisfied. P04.8 IN_PROGRESS: versioned within-participant/setting near-duplicate-component split; session IDs remain unknown. Consent is participant-specific; visible bystanders require confirmation or exclusion/redaction. Public release not authorized.

State reconciliation: all files in resume_evidence_index.json match; all 474 archive hashes match; no running training/server/install job found. P03.4 feasible alternative research/pilot running in research-skill agent; no final experiments launched. See P04.8-limited-v1.md for predeclared acceptance.

Execution log 2026-11-04 continuation: P04.8 structural checks PASS after fixing a cross-class-component allocation assumption (first run exit 1 with no output; corrected run exit 0). Full recomputed graph matches existing threshold-4 graph; deterministic rebuild matches. Teammate consent recorded. G04 still BLOCKED on bystanders; no threshold lowered, no synthetic session IDs. EfficientNetB0 feasibility pilot PASS; newer method research active. Original resume_evidence_index is historical after these intentional source/state changes; new continuation index will identify current files.

## 12. Authoritative current checkpoint — final experiments

2026-11-04: user confirms all background classmates mutually consented to project use. P04.6 resolved; privacy candidate shelved, original 453-image cleaned dataset retained. Data frozen as a1_limited_v2. G03/G04/G05 self-review PASS on recorded evidence; 5/15 stages pass. See current gate records; earlier blocked narratives are historical.

Authorized scope amendment: P04.5 independent collection deferred, not claimed performed. P03/P07 non-generative ConvNeXt branch replaces video-generation steps, prompts, generated stress panel and synthetic-mixture requirements with real pretrained representation integration and A0/S1 frozen-backbone pretraining ablation. No synthetic image claim. App must show selected method and zero synthetic share. One-setting evaluation limits disclosed; no person/session/environment generalization claim. P07's matched architecture/initial head/budget controls apply to A0/S1, not B1/S1.

P06/P07 IN_PROGRESS; pre-execution acceptance artifacts/reviews/P06-runs.md and P07-runs.md. Four run IDs B0_limited_v1, B1_limited_v1, S1_limited_v1, A0_limited_v1. Runner logs/state will identify active process. Do not duplicate jobs or inspect test predictions during selection. Expected next: verify histories/reloads and controlled ablation; select app on validation; then evaluate fixed test once and build final artifacts.

## 13. Required scope and current evidence — 2026-11-04

This matrix is the scope agreed with the user; optional expansion/generation/privacy-crop work is excluded from the release. Original instructor text permits another modern SOTA and upload OR camera. Hardware-camera capture is unverified; it does not replace or invalidate the required working upload route.

| Requirement | Implementation | Current verified status / evidence |
|---|---|---|
| Same A1 topic and dataset | Three hand gestures; existing dataset cleaned | PASS: a1_limited_v2, 453 accepted; frozen split 344/58/51 |
| One modern SOTA component | ConvNeXt, compared with A1 CNN and MobileNetV2 approaches | PASS: four completed runs, primary/Neurohive citations, G03/G06/G07 |
| Interactive app | Upload/camera → prediction → Grad-CAM | PASS: actual Chrome upload/Grad-CAM QA; physical camera capture unverified (optional) |
| Metrics and training curves | Accuracy, macro-F1, per-class results, curves and measured latency | PASS: artifacts/evaluation; G08 |
| Controlled ablation | Same frozen ConvNeXt architecture/head training, with versus without ImageNet pretraining | PASS: S1/A0 identical initial head and matched settings; selection.json |
| Ethics and synthetic disclosure | Recorded consent, limitations, zero synthetic training images | PASS for private project use; public distribution not authorized; one-person/one-setting limits |
| Deliverables | Web app, 6–8-page report, 3–5-minute video and repository | Local app, 7-page PDF, 4:21 video and source/private packages produced; human audio review and external handoff pending |

Current verified experiment results: B0 accuracy 47.06%, macro-F1 .3670; B1 86.27%, .8050; S1 78.43%, .7300; A0 41.18%, .4108. S1 remains deployed selection based on validation (100% vs B1 89.66%), not test results. The SOTA method improves over random features and simple CNN but does not beat MobileNetV2 on this test; report this honestly. Selected model CPU mean classifier latency 395.36 ms (100 calls, excluding explanation). Final report numbers must be read from JSON, not copied from this rounded summary.

G06/G07/G08/G09 review PASS. All training/evaluation jobs finished; local demo server and Chrome QA may still run (check processes before resuming). No final deliverable/publication/submission claim.

## 14. Delivery checkpoint — local app/report/package verified

P10/G10 and P11/G11 PASS using actual upload browser flow (physical camera remains unverified, optional under instructor upload/camera wording). P12 G12-local PASS after clean exported checkout/environment test; G12-publication pending. P13/G13 PASS: 7-page report rendered/reviewed. Local source repository and ZIP, private asset ZIP exist; exact commit/hashes in artifacts/qa/release_package.json. P14 actual app recording with disclosed generated narration is in progress; expected about 4m19s. Publication/submission/defense not performed.

Required external authorization only after final video verification: repository/media account and audience, publish approval, then submission access/authorization if requested. User project-use consent is resolved for all pictured classmates; do not re-ask. No new data collection, model alternatives or optional polish is needed.

## 15. Authoritative final local checkpoint — 2026-11-04

12/15 full gates PASS. All feasible local deliverables produced. P12-local PASS, publication pending. P14 duration 261.24 seconds, full ffmpeg decode exit 0, visual review PASS; AAC signal verified, human listening UNVERIFIED because agent audio input is unavailable. P14 overall stays incomplete pending listening/link access. Defense answers and submission checklist prepared in deliverables/FINAL_HANDOFF.md. P15 submission/defense unperformed. No active training, recording or app server.

Integrity: 474 preserved backups rehashed; source/private ZIP CRC checks PASS; report hash matches reviewed seven-page PDF. Current hashes: artifacts/qa/final_evidence_index.json; archive checks: artifacts/qa/final_integrity.json. Source export includes preparation/splitting provenance; README check-only command on an existing run ID removed because overwrite protection correctly rejects it. No model/data inputs changed, so experiment evidence remains valid.

Exact resume action: obtain student's listening result and destination/audience plus explicit publication approval (user-required boundary), verify instructor link access, then submit only if separately authorized. Existing dataset/consent is sufficient for the agreed limited evaluation; no additional pictures required.

Export metric recheck: first invocation exited 127 because source-only export has no .venv; corrected explicit runtime invocation exited 0. Evidence: artifacts/qa/export_metric_recheck.json.

## 16. Authoritative presentation v2 checkpoint

User requested refinement with local frontend-design/anti-slop skills, refreshed UI/presentation/deliverables, a re-recorded video and timestamped Markdown transcript for their own voice. Humanizer and PDF skills applied. Exact frontend-design/anti-slop packages could not be located in the visible npx registry/directories; user asked for install command/source. No claim of using missing skill contents.

Completed and reviewed: compact responsive web interface; correct camera preview lifecycle; consistent report/chart/video styling; seven-page PDF; actual browser recording edited to a silent 04:10 master; eight-scene Markdown voiceover transcript; presentation HTML screens; verified audio-merge helper; synchronized source repository/archive. Review: artifacts/reviews/refinement_v2.md. Evidence: artifacts/qa/refinement_v2/, current report/video QA records, fresh_checkout.json and release_package.json.

All model, dataset, experiment config and numerical result evidence remains unchanged. v1 presentation files/drafts and generated speech recording are archived in artifacts/releases/v1. Current deliverables intentionally contain no generated narration. Full G14 remains pending until student audio and final link verification; P12 publication and P15 submission/defense remain unperformed. Local preview server remains on port 8765; no training/recording job active.

Exact resume: user records deliverables/A2_Demo_Transcript.md, aligns voice at timestamps with deliverables/A2_Demo_Silent.mp4, optionally runs scripts/add_demo_audio.py, checks playback, then chooses/authorizes repository/video sharing. Obtain explicit authorization for publication/submission as previously instructed. Additional pictures and retraining are unnecessary.

## 17. Defense PPTX and report confirmation

User explicitly requested a PPTX presentation. The supplied deliverables require a 6-8 page report; a slide deck is not explicitly listed, so it is an additional user-authorized defense aid. Existing seven-page report verified against its reviewed hash. Created deliverables/A2_Defense_Presentation.pptx: nine slides, source-backed speaker notes, native editable tables and training charts with embedded XLSX data. Actual final PPTX reopened/rendered in LibreOffice; all nine slides visually inspected; structural/layout checks PASS. Evidence artifacts/qa/defense_pptx/evidence.json and artifacts/reviews/defense_pptx.md. Scientific artifacts unchanged.

12/15 required stage gates remain PASS. This extra deck does not substitute for the video or count as a completed defense. Resume: student records/adds the timestamped narration, reviews the final video, then authorized publication/submission. No additional user action is needed to obtain the report or PPTX.

## 18. GitHub publication checkpoint

User explicitly authorized repository initialization, GitHub creation and push. SSH authenticates as lastnurik. Restored official GitHub CLI 2.103.3 with verified release checksum, using the pre-existing keyring login. Initialized the project root Git repository on main. Created private https://github.com/lastnurik/cv-assignment and set origin to git@github.com:lastnurik/cv-assignment.git. Initial push succeeded. Raw images, weights, report/PPTX/video and private archives stay local. Source, experiment histories/metrics, manifests and reproducibility scripts are included.

G12 publication is now complete for authenticated owner access. Instructor access remains a P15 handoff check. G12-local reproducibility already passed. Required stages: 13/15 PASS (P01-P13); P14 requires student narration/final video access; P15 requires accessible reviewer links, authorized submission and actual defense. Exact resume: record/add narration, review playback, share the final video and grant the instructor repository access, then submit when authorized. No submission or defense is claimed.

Final remote/local commit verification is recorded in artifacts/qa/github_publication.json. The source ZIP and export are synchronized after this checkpoint. A previously empty .git directory initially bypassed the setup check; normal git init resolved it without replacing history. No credentials, SSH keys or authentication files were committed.
