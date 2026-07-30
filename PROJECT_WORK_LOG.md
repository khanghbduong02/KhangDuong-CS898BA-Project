# Project Work Log

> **Purpose:** Persistent technical record for the 3D-print failure detector. Update this file after every code change, dataset generation, training run, evaluation, validation finding, or experimental decision.
>
> **Separate from the AI accountability log:** This file records engineering work and experiment evidence. `AI_Log.md` is not edited automatically.
>
> **Last updated:** 2026-07-30 — repository documentation and regression tests were moved into dedicated `docs/` and `tests/` folders without changing runtime pipeline modules or experimental results.

## 1. Current Decision

- **Repository organization:** runtime modules and grading-facing Markdown documents remain at the repository root to preserve stable commands and easy navigation. Presentation and reference assets live under `docs/`; test modules live under `tests/`; generated data, checkpoints, archives, recordings, and editor artifacts remain ignored. See [README.md](README.md#repository-structure).
- **Clean RGB reference:** `runs/yolo26/baseline_seed42_e50/best.pt`, trained on `processed-data/baseline` with normalized labels and seed `42`.
- **Provisional relative preprocessing result:** `runs/yolo26/clahe_seed42_e50/best.pt` has the highest observed validation mAP50 (`0.0674`) and mAP50-95 (`0.0274`) among the three clean preprocessing variants. Because validation shares exact image content with training, it is not an independent generalization selection.
- **Selection caveat:** CLAHE's improvement over RGB is very small and comes with lower warping AP50 (`0.0058` versus `0.0137`). The three-way comparison remains a controlled relative comparison under the same split, but it is not evidence of a statistically decisive or leakage-free preprocessing improvement.
- **CLAHE+Canny is rejected:** it underperformed RGB and CLAHE across aggregate mAP, precision, recall, and all per-class AP values.
- **Test-split report is contaminated:** the CLAHE checkpoint was evaluated once, but a later exact-hash audit found 58 image hashes shared between train and test. Treat its metrics as descriptive diagnostics, not an independent held-out generalization result; do not tune from them.
- **Group-disjoint CV result:** fixed custom focal-plus-weight training reached mAP50 $0.0615 \pm 0.0064$, mAP50-95 $0.0180 \pm 0.0019$, precision $0.0940 \pm 0.0045$, and recall $0.0538 \pm 0.0065$ at threshold `0.25` across three folds.
- **Faster R-CNN status:** the initial 960-pixel three-fold Faster R-CNN run is invalid as a benchmark because it predates an RPN positive-supervision repair. Its mAP50 $0.0075 \pm 0.0079$ is retained only as a diagnostic of the defect. The repaired replacement scratch baseline is valid: mAP50 $0.0935 \pm 0.0522$, mAP50-95 $0.0284 \pm 0.0203$, precision $0.1441 \pm 0.0157$, and recall $0.1445 \pm 0.0369$. The one permitted ImageNet-initialized treatment reached $0.0665 \pm 0.0391$, $0.0187 \pm 0.0114$, $0.1087 \pm 0.0189$, and $0.2546 \pm 0.0650$, respectively. It increased recall but reduced both AP metrics and precision, so it is rejected as an aggregate-AP improvement. The completed P2 plus plateau/early-stopping treatment reached mAP50 $0.0395 \pm 0.0401$, mAP50-95 $0.0110 \pm 0.0120$, precision $0.1871 \pm 0.0082$, and recall $0.0340 \pm 0.0324$; it lost $0.0540$ mAP50, $0.0174$ mAP50-95, and $0.1105$ recall versus the repaired scratch baseline despite a $0.0430$ precision increase. Retain the repaired run as the valid scratch two-stage reference; retain one-to-many YOLO26 + NMS as the selected custom configuration.
- **Strict-scratch mAP50-selection outcome:** the completed study below supersedes the prior validation-loss-selected reports as the current local-model results. The historical repaired Faster R-CNN and one-to-many YOLO26 runs remain valid controls for the paired comparison.
- **Current selected scratch configurations:** retain fixed-LR, one-to-many YOLO26 with mAP50 checkpoint selection as the selected local one-stage configuration. The completed photometric policy was essentially neutral on aggregate YOLO26 mAP50 but changed class trade-offs, so it does not replace the raw baseline. For Faster R-CNN, retain the completed warmup-plus-cosine result as the current mAP-oriented local two-stage configuration because it improves both aggregate AP metrics and fixed-threshold precision, while disclosing that its mAP50 increase is modest and selection/reporting reused the same validation folds. EMA and photometric augmentation are rejected as overall replacements for both systems.
- **Experiment boundary:** do not reopen completed scratch YOLO26 or scratch Faster R-CNN DFL, loss, sampler, class-weight, offline minority-augmentation, online photometric augmentation, threshold, duration, resolution, anchor, postprocessing, P2, plateau-controller, combined architecture/controller, horizontal-flip, cosine-schedule, or EMA sweeps. The ImageNet-transfer, P2/controller, strict-scratch checkpoint-selection, horizontal-flip diagnostic, schedule-only, EMA, and photometric studies are complete. Do not use the contaminated candidate test split or add another variation on this export. The one-to-many branch correction is a completed no-retraining inference selection, not a new training sweep.
- **Do not continue** image-level balanced sampling, global `cls_gain` tuning, or positive-class weight-power tuning; each was inferior overall.
- **Exact 20% augmentation conclusion:** it improved warping AP but reduced aggregate mAP and eliminated stringing detections at the standard threshold. It is retained as a class-specific trade-off, not the overall model.
- **Candidate five-class neutral-baseline conclusion:** epoch 25 was the validation-loss minimum, but only spaghetti had true positives at thresholds `0.25` and `0.10`. Focal loss subsequently broke this collapse; do not repeat neutral-loss, sampler, threshold, or longer-epoch sweeps.
- **Candidate best custom result:** `focal_gamma=2` plus `--class-positive-weight-power 0.25` improved mAP50 to `0.0871`, mAP50-95 to `0.0303`, and produced true positives for all five classes at threshold `0.25`. This completes the final permitted published-split loss diagnostic; do not use the candidate test split or run more loss/sampling/threshold/epoch sweeps.
- **Custom-model arm:** the local YOLO26-style architecture remains randomly initialized. The separate pretrained YOLO11n reference below was authorized solely as a practical diagnostic baseline, not as a direct architecture-only comparison or a replacement for the custom arm.
- **Pretrained practical-system result:** the official COCO-pretrained Ultralytics YOLO26n detector is the strongest same-dataset practical result to date. It is separate from the local scratch YOLO26-style implementation and should be presented as an official pretrained-system reference, not as evidence that the local model reached the same accuracy.
- **Strict-scratch mAP50 checkpoint-selection result: completed and selected.** Both local trainers use `--checkpoint-selection map50` to decode their normal inference path after each epoch and select `best.pt` by project validation mAP50; validation loss remains the only plateau/early-stopping signal. The pre-registered protocol changed no dataset, labels, grouped folds, preprocessing, architecture, initialization, loss, optimizer, augmentation, inference, or candidate-test policy: both models remain random-initialized, P2 is disabled, the plateau controller is disabled, and each study runs for at most 50 epochs with seed `42`. The local YOLO26 run `runs/yolo26/candidate_cv3_imgsz960_n_map50_selection_focal_g2_posweight_p025_seed42_e50` selected epochs `48` / `47` / `49` and reached mAP50 $0.1586 \pm 0.0104$, mAP50-95 $0.0512 \pm 0.0043$, precision $0.1106 \pm 0.0060$, and recall $0.1306 \pm 0.0067$. Relative to the matched one-to-many validation-loss baseline, this is $+0.0316$ mAP50, $+0.0092$ mAP50-95, $-0.0047$ precision, and $+0.0147$ recall. The local Faster R-CNN run `runs/faster_rcnn/candidate_cv3_imgsz960_s_map50_selection_posweight_p025_seed42_e50` selected epochs `50` / `47` / `35` and reached mAP50 $0.3191 \pm 0.0164$, mAP50-95 $0.1145 \pm 0.0099$, precision $0.1777 \pm 0.0329$, and recall $0.1271 \pm 0.0201$. Relative to its repaired validation-loss baseline, this is $+0.2256$ mAP50, $+0.0861$ mAP50-95, $+0.0336$ precision, and $-0.0174$ recall. The large Faster R-CNN improvement comes from later mAP-selected checkpoints with much stronger Layer cracking, Over extrusion, Stringing, and Warping ranking; Spaghetti AP remains low. These are controlled selection-aligned validation results, not independent held-out generalization estimates, because each fold's validation data both selects its epoch and reports the metric. No candidate public-test data was used.
- **User-authorized horizontal-flip test-time augmentation: completed and rejected for both models.** `inference_tta.py` supplies a shared horizontal-flip/coordinate-unflip/merge path for both local evaluators. `--tta-hflip` runs a frozen checkpoint on each original validation image and its horizontal flip, maps flip predictions back to original pixel coordinates, then applies the existing class-aware NMS thresholds and detection limit. It is off by default, preserves trained weights and source data, forbids YOLO legacy-top-k mode, and writes distinct `_tta_hflip_` outputs so standard artifacts cannot be overwritten. Synthetic coordinate restoration, duplicate suppression, wrapper dry runs, evaluator smoke coverage, and normal model regressions passed. The fixed local YOLO26 mAP-selected checkpoints were evaluated once: mAP50 fell from $0.1586 \pm 0.0104$ to $0.1409 \pm 0.0089$, mAP50-95 from $0.0512 \pm 0.0043$ to $0.0475 \pm 0.0044$, and precision from $0.1106 \pm 0.0060$ to $0.0828 \pm 0.0046$, despite recall rising from $0.1306 \pm 0.0067$ to $0.1741 \pm 0.0093$; AP50 fell for Layer cracking, Over extrusion, Stringing, and Warping. The matching Faster R-CNN run on the other PC likewise lost aggregate ranked detection: mAP50 $0.3191 \pm 0.0164 \rightarrow 0.3075 \pm 0.0186$, mAP50-95 $0.1145 \pm 0.0099 \rightarrow 0.1100 \pm 0.0104$, and precision $0.1777 \pm 0.0329 \rightarrow 0.1335 \pm 0.0221$, despite recall rising $0.1271 \pm 0.0201 \rightarrow 0.1711 \pm 0.0286$. Therefore horizontal-flip TTA is rejected as an aggregate-AP improvement for both local models. These evaluations used only existing mAP-selected checkpoints and grouped `valid` folds; no retraining or candidate-test data was used.
- **User-authorized warmup-plus-cosine schedule: completed; rejected for YOLO26 and retained cautiously for Faster R-CNN.** `training_control.py` provides a deterministic epoch-based scheduler shared by both trainers. Historical constant learning rate remains the default. The treatment used three linear warmup epochs from $0.1\times$ to $1.0\times$ the base rate, then cosine decay to $0.02\times$ at epoch 50; it was independent of validation metrics, ran with plateau/early stopping disabled, and preserved mAP50 checkpoint selection. Formula/compatibility tests, model regressions, runner dry runs, and direct CUDA smokes passed. The YOLO26 run selected epochs `38` / `34` / `49` but was decisively worse: mAP50 $0.1586 \pm 0.0104 \rightarrow 0.0947 \pm 0.0051$, mAP50-95 $0.0512 \pm 0.0043 \rightarrow 0.0299 \pm 0.0029$, precision $0.1106 \pm 0.0060 \rightarrow 0.0874 \pm 0.0269$, and recall $0.1306 \pm 0.0067 \rightarrow 0.0798 \pm 0.0080$; every mean class AP50 declined. Reject cosine scheduling for YOLO26 and retain fixed LR. The Faster R-CNN run selected epochs `32` / `46` / `35` and produced a modest aggregate AP improvement: mAP50 $0.3191 \pm 0.0164 \rightarrow 0.3258 \pm 0.0275$, mAP50-95 $0.1145 \pm 0.0099 \rightarrow 0.1239 \pm 0.0111$, and precision $0.1777 \pm 0.0329 \rightarrow 0.2960 \pm 0.0525$, while recall fell $0.1271 \pm 0.0201 \rightarrow 0.1146 \pm 0.0069$. mAP50 increased in two of three folds, mAP50-95 increased in all three, and class AP50 rose for Spaghetti, Over extrusion, and Stringing but fell for Layer cracking and Warping. Retain this schedule as the current FRCNN mAP-oriented configuration, but report the $+0.0067$ mAP50 change as modest rather than decisive. Both studies used only the existing grouped `valid` folds, random initialization, 960-pixel input, P2 disabled, the same loss/weighting/NMS/seed, and no candidate test data.
- **User-authorized model EMA: implementation and pre-run protocol (superseded by the completed result below).** `model_ema.py` maintains a shadow copy of every local model state tensor. It exponentially averages floating parameters and buffers—including BatchNorm running mean/variance—while copying non-floating counters such as `num_batches_tracked` from the current raw model. A supplied `--ema-decay` is interpreted at epoch scale and converted to an equivalent per-optimizer-step factor using each fold's number of training batches; this makes the same EMA time horizon comparable across YOLO26 batch `8` and Faster R-CNN batch `2`. The pre-registered full CV fixes `--ema-decay 0.9`, which provides about a ten-epoch smoothing horizon and leaves only $0.9^{50} \approx 0.0052$ of the initialization contribution by epoch 50. EMA updates only after a successful optimizer step. Raw-model validation loss remains the sole plateau/early-stopping signal. With `--checkpoint-selection map50`, EMA weights produce validation predictions and select `best.pt`; `last.pt` retains raw model state plus EMA metadata, while selected `best.pt` stores EMA weights directly and explicitly records EMA as its weight source. This avoids doubling the large Faster R-CNN last-checkpoint write in trainers that intentionally do not support resume. Default `--ema-decay 0` preserves historical raw-weight behavior. Unit coverage verifies averaging, epoch normalization, BatchNorm-buffer handling, temporary raw-state restoration, decay validation, and legacy no-EMA checkpoint matching. Existing model/controller tests, complete two-model CUDA train-and-evaluate smoke runs, evaluator restoration of EMA-selected checkpoints, and both EMA CV runner dry runs passed. The pre-registered full CV changed only EMA behavior: YOLO26 retained fixed LR and its mAP50-selected 960 protocol; Faster R-CNN retained its cautiously selected warmup-plus-cosine mAP50-selected 960 protocol. Both retained random initialization, P2 disabled, existing grouped folds/losses/class weights/NMS/seed/50-epoch cap, and no candidate-test use.
- **User-authorized model EMA: completed and rejected for both models.** This result supersedes the preceding prepared-study description. The fixed study used `--ema-decay 0.9`; every selected checkpoint reported `checkpoint_weights=ema`, so the intended EMA state—not raw weights—was evaluated. YOLO26 EMA selected epoch `50` in all three folds but collapsed ranked detection: mAP50 $0.1586 \pm 0.0104 \rightarrow 0.0507 \pm 0.0154$, mAP50-95 $0.0512 \pm 0.0043 \rightarrow 0.0151 \pm 0.0068$, and recall $0.1306 \pm 0.0067 \rightarrow 0.0097 \pm 0.0047$; precision rose $0.1106 \pm 0.0060 \rightarrow 0.1967 \pm 0.0411$ only because detections nearly disappeared. Faster R-CNN EMA selected epochs `47` / `45` / `49` and lost the project AP targets relative to the selected cosine configuration: mAP50 $0.3258 \pm 0.0275 \rightarrow 0.3166 \pm 0.0322$ and mAP50-95 $0.1239 \pm 0.0111 \rightarrow 0.1190 \pm 0.0170$, despite precision $0.2960 \pm 0.0525 \rightarrow 0.3181 \pm 0.0341$ and recall $0.1146 \pm 0.0069 \rightarrow 0.1200 \pm 0.0136$. The implementation had already passed unit, CUDA smoke, evaluator-restoration, and runner checks, so this is negative experimental evidence rather than a pipeline failure. Do not use EMA checkpoints for either architecture. Both studies retained random initialization, P2 disabled, the existing grouped folds/losses/class weights/NMS/seed/50-epoch cap, and no candidate-test use.
- **User-authorized online photometric augmentation: implementation and pre-run protocol (superseded by the completed result below).** `online_augmentation.py` supplies the same in-memory `photometric` policy to both training datasets only. It applies brightness, per-channel contrast, and gamma factors independently sampled from $[0.9, 1.1]$, plus Gaussian noise with standard deviation `0.01`, after resize and normalization. It has no flip, crop, translation, scaling, rotation, blur, Mosaic, MixUp, class balancing, or other geometry/class-frequency effect, so detection boxes and labels remain unchanged. The source files remain read-only and validation datasets always use `none`. This is intentionally distinct from the rejected offline, minority-targeted augmentation treatment: it creates no images or labels and does not alter class counts. The default remains `--online-augmentation none`; K-fold settings/checkpoints/evaluation JSON record the policy, and legacy no-policy checkpoints map to `none`. Deterministic transform, range, source-tensor preservation, YOLO target invariance, Faster R-CNN target invariance, full regression, CUDA smoke, and runner dry-run checks passed before the one pre-registered CV per architecture began. The fixed study preserved raw fixed-LR YOLO26 and raw cosine-scheduled Faster R-CNN, `--ema-decay 0`, all existing grouped folds/losses/weights/NMS/seed/50-epoch caps, and no candidate test use.
- **User-authorized online photometric augmentation: completed and not adopted as an overall replacement.** The same in-memory policy trained both local models while leaving source files, labels, boxes, validation images, class counts, model initialization, losses, inference, and candidate-test policy unchanged. Every evaluated checkpoint reported `training_online_augmentation=photometric` and `checkpoint_weights=raw`. YOLO26 selected epochs `45` / `50` / `49`: aggregate mAP50 changed only $0.1586 \pm 0.0104 \rightarrow 0.1588 \pm 0.0103$ and mAP50-95 improved $0.0512 \pm 0.0043 \rightarrow 0.0535 \pm 0.0054$, while precision and recall rose slightly. However, the paired mAP50 result was mixed across folds and mean AP50 traded a large Layer cracking gain ($0.0678 \rightarrow 0.1256$) for lower Warping ($0.3596 \rightarrow 0.3023$) and Stringing ($0.0535 \rightarrow 0.0482$), so raw fixed-LR YOLO26 remains selected rather than claiming a robust overall improvement. Faster R-CNN selected epochs `25` / `44` / `38` and regressed: mAP50 $0.3258 \pm 0.0275 \rightarrow 0.2999 \pm 0.0234$, mAP50-95 $0.1239 \pm 0.0111 \rightarrow 0.1134 \pm 0.0092$, and precision $0.2960 \pm 0.0525 \rightarrow 0.2808 \pm 0.0947$, with only a negligible recall increase. Its Stringing and Warping AP50 values also fell. Reject photometric augmentation as an overall Faster R-CNN improvement and do not launch a magnitude/policy sweep. These are completed validation-only experiments, not independent test estimates.
- **Final demonstration support:** `demo_custom_yolo26.py` creates annotated batch-inference images plus a JSON prediction summary from the selected local custom YOLO26 checkpoint. It is explicitly a visual demonstration tool rather than a new metric-selection path; the documented workflow uses grouped fold-1 validation images and continues to exclude the public candidate test split. White boxes are validation annotations and colored boxes are custom YOLO26 predictions. A four-image CUDA smoke demonstration passed with the selected raw local fold-1 checkpoint (epoch `48`), saving annotated JPEG outputs plus `demo_predictions.json`; it included one output with a high-confidence Over extrusion prediction covering the main annotated defect and several honest diffuse-Spaghetti examples with imperfect localization. `demo_custom_detector_comparison.py` now produces the final qualitative architecture grid in the fixed order **Ground Truth | Custom YOLO26 | Custom Faster R-CNN** for the same grouped-validation source image. It restores only the selected raw scratch checkpoints (YOLO26 one-to-many/class-aware NMS and the scheduled Faster R-CNN/per-class NMS), writes one JPEG row per image plus `comparison_predictions.json`, and deliberately does not calculate another metric or make an independent-test claim. Use this comparison grid for the architecture-comparison slide, and use the single-model script when a YOLO26-only batch-processing view is useful. `demo_ultralytics.py` remains only for a separately labeled pretrained practical-reference demonstration and must not be substituted for the local-model video.
- **Scratch-only comparison-grid guard:** each grid has a visible qualitative-only title, and the script rejects EMA checkpoint weights plus Faster R-CNN checkpoints whose metadata reports ImageNet/backbone transfer instead of `backbone_weights=none` and random initialization. A one-image CUDA technical smoke passed using the selected local raw YOLO26 fold-1 checkpoint and an older locally available raw scratch Faster R-CNN fold-1 checkpoint; this only validated image loading, inference, coordinate restoration, headers, and JPEG/JSON output. It is not final qualitative evidence because the selected scheduled Faster R-CNN run folder is on the other PC and must be co-located with the selected YOLO26 folder before creating the presentation grids.
- **Faster R-CNN qualitative-demo diagnosis and correction:** the original alphabetical fold-1 eight-image grid contained `20` ground-truth Spaghetti boxes and no other ground-truth class. This is an unrepresentative qualitative slice for the mAP-selected Faster R-CNN: its saved fold-1 metrics report Spaghetti AP50 `0.0399` and recall `0.0834`, versus Layer cracking `0.2766`, Over extrusion `0.4273`, Stringing `0.2191`, and Warping `0.6092`; its macro mAP50 is `0.3144`. The custom YOLO26 fold-1 Spaghetti AP50 is similarly low (`0.0384`), so blank/weak two-stage Spaghetti panels are expected model behavior rather than a loader, coordinate, checkpoint, or mixed-precision defect. A direct full-precision versus CUDA-autocast check on the first two images retained the same displayed Faster R-CNN counts (`1` then `0` at threshold `0.25`), while the comparison script now runs Faster R-CNN in float32 to match `eval_faster_rcnn.py` exactly. The new optional `--image-selection ground_truth_coverage` rule selects source images from strict labels only until every available class is covered, then fills remaining slots alphabetically; it never examines predictions and is recorded in the JSON artifact. Use it with four images for a reproducible class-coverage qualitative slide, while retaining at least one visible Spaghetti/other failure and reporting the complete metrics separately. This is a presentation-sampling correction, not a new performance result, threshold change, or model-selection change.
- **Candidate group-disjoint CV:** all three folds used the identical fixed focal-plus-weight configuration and detected all five classes at threshold `0.25`. The resulting metric variability is modest overall, but class-specific performance remains low and uneven.
- **CV-fold choice:** three folds were intentional. Five folds would reduce each validation fold from about 39 to 23 warping boxes and from about 27 to 16 warping-containing groups, producing substantially noisier minority metrics while roughly doubling total 50-epoch CV training exposure.

### External PASCAL VOC 2007 benchmark preparation (2026-07-20)

To distinguish custom-detector behavior beyond the 3D-print corpus, the official PASCAL VOC 2007 archives were downloaded into ignored `candidate-data/pascal-voc-2007/` and converted by the tracked `prepare_voc2007.py` script into ignored `processed-candidate-data/pascal-voc-2007/official/`. This is a separate project-metric benchmark, not an official VOC leaderboard result: the project uses its own AP implementation and excludes VOC objects marked `difficult=1` because the metric code has no difficult-object ignore semantics.

| Check | Result |
| --- | --- |
| Official source archives | Downloaded from the official Oxford VOC URLs: train/validation archive SHA-256 `7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508`; test archive SHA-256 `6836888e2e01dca84577a849d339fa4f73e1e4f135d312430c4856b5609b4892`. |
| Taxonomy | All 20 official VOC classes retain canonical IDs `0`--`19` from `aeroplane` through `tvmonitor`. |
| Official split integrity | `2,501` train, `2,510` validation, and `4,952` test images; source IDs are disjoint across all three splits. |
| Strict conversion | One-to-one image/label pairing passed for all `9,963` images; all labels are five-field normalized YOLO boxes. All images were hardlinked; no pixels were resized, re-encoded, or augmented during conversion. |
| Difficult/invalid handling | `5,998` difficult objects excluded consistently; `0` invalid boxes excluded. Emitted boxes: `6,301` train, `6,307` validation, `12,032` test. |
| Existing model tests | `test.py` and `test_faster_rcnn.py` passed after conversion. |
| Capacity-matched resource smoke | YOLO26 `l`: `27,082,448` parameters, batch `8`, 640 pixels, finite forward/loss/backward, 5.00 GiB peak allocated CUDA memory. Faster R-CNN `s`: `28,872,040` parameters, batch `8`, 640 pixels, finite forward/loss/backward, 1.47 GiB peak. |

The frozen scratch benchmark protocol is: official VOC train for training, official val for checkpoint selection/evaluation during development, and official test evaluated once only after both training runs complete with no configuration changes. Both models use 640-pixel stretch resizing, seed `42`, 100 epochs, AdamW (`lr=1e-4`, `weight_decay=5e-4`), batch size `8`, no focal loss, no class weighting, no sampler, no augmentation, and project metrics. The capacity-matched custom pair is YOLO26 `l` versus Faster R-CNN `s`; both are randomly initialized. YOLO26 uses direct regression (`reg_max=1`) and standard class-aware NMS (`0.001` candidate score, `0.70` IoU, 300 detections); Faster R-CNN uses equivalent per-class NMS settings. VOC uses official train/validation/test folders rather than `fold_<n>` directories, so the one-fold `train_*` / `eval_*` primitives—not the generic K-fold runners—must be used. The exact direct commands are documented in `README.md`.

- **Frozen scratch YOLO26 `l` validation baseline: completed.** The 100-epoch run selected epoch 14 by validation loss (`10.0969`); the final epoch had lower training loss but validation loss `14.1346`, confirming that `best.pt` correctly avoids the later overfit trajectory. At confidence `0.25` on official VOC `val` only, the project metrics were mAP50 `0.1450`, mAP50-95 `0.0567`, precision `0.6276`, and recall `0.0772`. Highest AP50 classes were Car `0.3231`, Horse `0.2917`, Motorbike `0.2889`, and Person `0.2731`. This is a frozen from-scratch project-metric result, not an official VOC score or a pretrained-model comparison.
- **Frozen scratch Faster R-CNN `s` validation baseline: completed.** At confidence `0.25` on official VOC `val` only, project metrics were mAP50 `0.0723`, mAP50-95 `0.0278`, precision `0.1974`, and recall `0.1327`. Its strongest AP50 classes were Car `0.2479`, Motorbike `0.1865`, Person `0.1220`, and TV Monitor `0.1218`. Relative to frozen YOLO26 `l`, Faster R-CNN has lower AP/precision but higher operating recall (`0.1327` versus `0.0772`). This is a valid frozen from-scratch project-metric result, not an official VOC score or a pretrained-model comparison.
- **Final one-time VOC test results: completed.** Under unchanged evaluation settings (confidence `0.25`, candidate score `0.001`, NMS IoU `0.70`, maximum 300 detections), YOLO26 `l` achieved mAP50 `0.1318`, mAP50-95 `0.0527`, precision `0.5958`, and recall `0.0824` on official VOC `test`; Faster R-CNN `s` achieved mAP50 `0.0726`, mAP50-95 `0.0292`, precision `0.2094`, and recall `0.1514`. Both results are close to their respective validation outcomes: YOLO26 changed by `-0.0132` mAP50 and `-0.0040` mAP50-95, while Faster R-CNN changed by `+0.0003` and `+0.0014`. YOLO26 retains the stronger AP/precision result; Faster R-CNN retains higher operating recall. Strongest test AP50 classes were Horse `0.3230`, Car `0.3212`, Person `0.2846`, and Motorbike `0.2472` for YOLO26; Car `0.2742`, Motorbike `0.1450`, TV Monitor `0.1439`, and Person `0.1378` for Faster R-CNN.
- **VOC benchmark closure:** no more VOC training, checkpoint selection, threshold changes, NMS changes, or test evaluations are permitted. These are project-metric external results only: VOC difficult objects were excluded and the project AP implementation differs from the official VOC leaderboard protocol. The benchmark demonstrates that both scratch custom systems learn nontrivial external detection signal, while YOLO26 provides stronger ranked detection and Faster R-CNN trades AP/precision for recall.

### Training-system diagnostics and practical transfer study (2026-07-21--22)

The low scratch VOC results are expected from the current training systems: only 2,501 official train images supervise randomly initialized 27--29M parameter detectors, with no pretrained features, no online detection augmentation, no learning-rate warmup/schedule, 640-pixel stretch resize, and no stride-4/P2 small-object level. The models demonstrate nontrivial learning, but they are not comparable to mature pretrained systems trained on much broader corpora.

- **YOLO26 inference-branch correction: completed without retraining.** The local architecture trains a full-gradient one-to-many branch and a detached-feature one-to-one branch, but historical inference decoded only the one-to-one branch. Identical class-aware NMS diagnostics showed one-to-many is stronger on VOC 2007 `val` (mAP50 `0.1657` / mAP50-95 `0.0683` / recall `0.2191` versus one-to-one `0.1452` / `0.0569` / `0.0774`) and 3D-print candidate fold 1 (`0.1293` / `0.0453` / `0.0888` versus `0.1107` / `0.0326` / `0.0412`). The complete three-fold candidate validation confirms the correction: one-to-many + NMS gives mAP50 `0.1270 ± 0.0210`, mAP50-95 `0.0420 ± 0.0081`, precision `0.1153 ± 0.0207`, and recall `0.1159 ± 0.0246`, versus historical one-to-one + NMS `0.1127 ± 0.0232`, `0.0343 ± 0.0076`, `0.1621 ± 0.0497`, and `0.0588 ± 0.0172`. Every fold improved both AP measures; mean AP50 also rose for every class, including Layer cracking `0.0511 -> 0.0738`, Over extrusion `0.1890 -> 0.1997`, Stringing `0.0259 -> 0.0310`, and Warping `0.2636 -> 0.2928`. The lower fixed-threshold precision is the expected recall/false-positive trade-off, but ranked AP improves. `eval_yolo26.py` and its K-fold wrapper now default to one-to-many when using class-aware NMS; explicitly select `--inference-branch one2one` to reproduce historical branch results, and use `--postprocess legacy_topk` to reproduce the historical top-k-only decoder. No VOC test re-evaluation is permitted after its closed one-to-one benchmark protocol.
- **Pretrained official YOLO26n practical reference: completed.** The local `yolo26n.pt` checkpoint was verified as an 80-class COCO-pretrained Ultralytics detection model, then fine-tuned through the same generic Ultralytics workflow used for YOLO11n: unchanged five-class grouped folds, 50 maximum epochs, 640-pixel input, batch `8`, seed `42`, `optimizer=auto`, patience `100`, standard Ultralytics augmentation/NMS, and project-metric validation at confidence `0.25`. The aggregate validation result was mAP50 `0.4592 ± 0.0214`, mAP50-95 `0.1942 ± 0.0087`, precision `0.4075 ± 0.0325`, and recall `0.2105 ± 0.0192`; no candidate public-test data was used. Against the matched YOLO11n practical reference (`0.4799 ± 0.0262` mAP50, `0.1979 ± 0.0076` mAP50-95, precision `0.3766 ± 0.0133`, recall `0.2459 ± 0.0137`), YOLO26n is lower by `0.0207` mAP50 and `0.0037` mAP50-95, higher by `0.0309` precision, and lower by `0.0354` recall. It reaches 95.7% of YOLO11n mAP50 and 98.1% of its mAP50-95, so it nearly matches the reference on the same fine-tuning data. This is the strongest practical same-dataset detector result, but not a local scratch-model improvement or a causal architecture-only comparison: both official systems retain COCO pretraining plus the Ultralytics training, augmentation, and inference stack.
- **Pre-registered practical Faster R-CNN transfer study:** `models/faster_rcnn.py` supports architecture-compatible ImageNet initialization through torchvision: local `s` maps to ResNet-18 and local `m` maps to ResNet-34; local `l` is intentionally rejected because its wider channel shapes are incompatible. Mapping tests, actual ResNet-18 ImageNet weight download, optimizer parameter groups, trainer dry run, generic runner dry run, and all existing model tests passed. The single permitted candidate treatment changed backbone initialization from random to ImageNet and applied a fixed `0.1` backbone learning-rate multiplier; FPN, RPN, heads, data folds, 960 input, batch 2, 50 epochs, seed 42, AdamW base LR `1e-4`, weight decay `5e-4`, positive-weight power `0.25`, no sampler, no augmentation, NMS, and project metrics remained frozen. This is a practical transfer-learning comparison, not a scratch architecture-only comparison with YOLO26 or YOLO11n.
- **ImageNet-transfer result: completed and rejected as an aggregate-AP improvement.** The distinct run root `runs/faster_rcnn/candidate_cv3_imgsz960_s_imagenet_backbone_lr0p1_posweight_p025_seed42_e50` trained all three folds and selected validation-loss checkpoints at epoch/loss `3` / `1.3066`, `1` / `1.2431`, and `2` / `1.2390`. The one permitted aggregate validation evaluation used the existing `best.pt` files only, the group-disjoint `valid` folds, confidence `0.25`, per-class NMS candidate score `0.001`, NMS IoU `0.70`, and maximum 300 detections; no candidate test data was used. It produced mAP50 `0.0665 ± 0.0391`, mAP50-95 `0.0187 ± 0.0114`, precision `0.1087 ± 0.0189`, and recall `0.2546 ± 0.0650`. Mean AP50 was Spaghetti `0.0910`, Layer cracking `0.0113`, Over extrusion `0.0961`, Stringing `0.0749`, and Warping `0.0592`. Relative to the identical repaired scratch Faster R-CNN protocol, this is `-0.0270` mAP50, `-0.0097` mAP50-95, and `-0.0354` precision, with `+0.1101` recall. It also remains below selected one-to-many YOLO26 + NMS (`0.1270` / `0.0420` mAP50 / mAP50-95), while the practical pretrained YOLO11n reference is much stronger (`0.4799` / `0.1979` mAP50 / mAP50-95) but differs in COCO pretraining, detector/loss implementation, augmentation, preprocessing, optimizer, and NMS. Therefore no causal architecture-only comparison to YOLO11n is claimed, no second transfer variant is authorized, and the scratch Faster R-CNN baseline remains the valid two-stage reference.
- **User-authorized same-dataset P2 and training-control implementation:** The user requested that both local architectures pursue the practical YOLO11 reference on the same existing dataset, without changing source data, folds, labels, or the candidate test boundary. `training_control.py` supplies both trainers with validation-loss `ReduceLROnPlateau` and early stopping. The defaults are reduction patience `8`, factor `0.5`, cooldown `2`, minimum LR `1e-6`, and early-stopping patience `18`, enforcing $18 \ge 8 \times 2 + 2$. An actual LR reduction resets the stopping counter so the lower rate receives a fresh optimization window. Controller state, learning rates, completion state, and stop reason are saved in checkpoints; K-fold runners recognize a matching early-stopped fold as complete. `--use-p2` adds a stride-4 P2 detector branch to YOLO26 and a P2--P6 FPN/RPN path with 16-pixel base anchors plus stride-aware RoI assignment to Faster R-CNN. Historical P3--P5/P3--P6 checkpoints remain the default and restore with `use_p2=false`. Regression coverage passed for the patience formula, LR reduction/reset, disabled mode, early-stopped K-fold completion, historical YOLO26/Faster R-CNN smoke tests, the new YOLO26 P2 head plus loss/backpropagation path, the Faster R-CNN P2 FPN/anchor/RPN path, and neutral YOLO class weights on intentionally reduced smoke subsets that omit a class.
- **YOLO26 P2 plus training-control result: completed and rejected as an aggregate-AP improvement.** The run root `runs/yolo26/candidate_cv3_imgsz960_n_p2_plateau_es_seed42_e150` used the unchanged five-class grouped folds, 960-pixel input, `n` scale, focal gamma `2`, positive-weight power `0.25`, one-to-many class-aware NMS (`0.001` / `0.70` / `300`), seed `42`, and validation-only evaluation. Its folds reached the minimum LR and stopped validly at epochs `120`, `122`, and `122`; their selected `best.pt` validation-loss checkpoints were epochs `38`, `40`, and `50`. No candidate public-test data was used. Aggregate validation results were mAP50 `0.1067 \pm 0.0271`, mAP50-95 `0.0345 \pm 0.0100`, precision `0.0987 \pm 0.0067`, and recall `0.1009 \pm 0.0259`. Relative to the selected historical 960 direct-regression, one-to-many, P3--P5 baseline (`0.1270 \pm 0.0210` / `0.0420 \pm 0.0081` / `0.1153 \pm 0.0207` / `0.1159 \pm 0.0246`), the combined treatment changes the aggregate metrics by `-0.0203` mAP50, `-0.0075` mAP50-95, `-0.0166` precision, and `-0.0150` recall. Mean AP50 also declined for every class: Spaghetti `0.0379 -> 0.0351`, Layer cracking `0.0738 -> 0.0487`, Over extrusion `0.1997 -> 0.1885`, Stringing `0.0310 -> 0.0200`, and Warping `0.2928 -> 0.2415`. This is a valid negative result for the combined P2-plus-controller treatment; it does not isolate P2 from the new training-control policy. Retain the historical 960 one-to-many P3--P5 model as the selected YOLO26 configuration, do not launch another YOLO26 sweep on this export, and retain the P2/controller code as tested experimental capability. The paired P2 Faster R-CNN study remains in progress under its distinct run root.
- **YOLO26 P2 plus training-control result: completed and rejected as an aggregate-AP improvement.** The run root `runs/yolo26/candidate_cv3_imgsz960_n_p2_plateau_es_seed42_e150` used the unchanged five-class grouped folds, 960-pixel input, `n` scale, focal gamma `2`, positive-weight power `0.25`, one-to-many class-aware NMS (`0.001` / `0.70` / `300`), seed `42`, and validation-only evaluation. Its folds reached the minimum LR and stopped validly at epochs `120`, `122`, and `122`; their selected `best.pt` validation-loss checkpoints were epochs `38`, `40`, and `50`. No candidate public-test data was used. Aggregate validation results were mAP50 `0.1067 \pm 0.0271`, mAP50-95 `0.0345 \pm 0.0100`, precision `0.0987 \pm 0.0067`, and recall `0.1009 \pm 0.0259`. Relative to the selected historical 960 direct-regression, one-to-many, P3--P5 baseline (`0.1270 \pm 0.0210` / `0.0420 \pm 0.0081` / `0.1153 \pm 0.0207` / `0.1159 \pm 0.0246`), the combined treatment changes the aggregate metrics by `-0.0203` mAP50, `-0.0075` mAP50-95, `-0.0166` precision, and `-0.0150` recall. Mean AP50 also declined for every class: Spaghetti `0.0379 -> 0.0351`, Layer cracking `0.0738 -> 0.0487`, Over extrusion `0.1997 -> 0.1885`, Stringing `0.0310 -> 0.0200`, and Warping `0.2928 -> 0.2415`. This is a valid negative result for the combined P2-plus-controller treatment; it does not isolate P2 from the new training-control policy. Retain the historical 960 one-to-many P3--P5 model as the selected YOLO26 configuration and retain the P2/controller code as tested experimental capability.
- **Faster R-CNN P2 plus training-control result: completed and rejected as an aggregate-AP improvement.** The run root `runs/faster_rcnn/candidate_cv3_imgsz960_s_p2_plateau_es_seed42_e150` used the unchanged grouped folds, 960-pixel input, random `s` backbone, positive-weight power `0.25`, per-class NMS (`0.001` / `0.70` / `300`), seed `42`, and validation-only evaluation. The controller marked every fold complete after valid early stopping at epochs `99`, `95`, and `98`; the selected validation-loss checkpoints were epochs `6`, `2`, and `5`. It produced mAP50 `0.0395 \pm 0.0401`, mAP50-95 `0.0110 \pm 0.0120`, precision `0.1871 \pm 0.0082`, and recall `0.0340 \pm 0.0324`. Compared with the repaired P3--P6 scratch baseline, aggregate AP and recall collapsed, and all class mean AP50 values fell: Spaghetti `0.0578 -> 0.0299`, Layer cracking `0.0854 -> 0.0414`, Over extrusion `0.1118 -> 0.0586`, Stringing `0.0447 -> 0.0087`, and Warping `0.1681 -> 0.0590`. This is a valid negative result for the combined P2-plus-controller treatment; it does not isolate P2 from the controller. Keep the repaired P3--P6 scratch model as the selected Faster R-CNN configuration, do not launch another Faster R-CNN sweep on this export, and do not use the candidate public test split.

### Faster R-CNN workflow modernization (2026-07-19)

The previously untracked local Faster R-CNN files were reviewed against the current project protocol and updated before any comparison experiment. This is a fully local, randomly initialized ResNet-style/FPN/RPN/RoI-Align detector; it does not use a pretrained backbone or high-level torchvision detector model.

- **Correctness and reproducibility:** fixed the FPN proposal-level assignment to the canonical P3--P6 mapping; added seed control, DataLoader worker seeding, strict five-field YOLO label validation, `data.yaml` taxonomy validation, safe empty-sample losses, and validation-loss computation in `eval()` mode so validation batches cannot update BatchNorm statistics.
- **Training/evaluation parity:** added optional tempered foreground class weights and class-balanced sampling (both disabled by default), preserved Faster R-CNN's internal background index while keeping external labels zero-indexed, saved class counts/weights and inference metadata in checkpoints, restored legacy checkpoint compatibility, and added configurable per-class NMS using the current candidate score `$0.001$, IoU `$0.70$, and at most 300 detections defaults.
- **Reusable CV tooling:** added generic sequential `run_faster_rcnn_kfold_cv.py` and `eval_faster_rcnn_kfold_cv.py` scripts. They discover/validate standard `fold_<n>` layouts, reject incompatible partial or completed checkpoints, emit machine-readable per-fold metrics, and aggregate overall/per-class mean and sample SD.
- **Validation completed:** syntax/type checks passed; a synthetic forward/loss/backward/inference test passed; the five-class grouped-fold data loader/weighting test passed; both CV tools passed dry runs; and a one-epoch CUDA smoke pass at 128 pixels on a deliberately tiny fold-1 subset created and restored a checkpoint successfully. The tiny subset had only Spaghetti labels and obtained zero metric values, so it is **not** an experiment, model result, or comparison with YOLO26/YOLO11n. Separate fold-1 resource smokes passed with the `s` model at batch size 2 and tempered class-weight power `0.25`: 0.58 GiB peak allocated CUDA memory at 640 pixels and 0.77 GiB at 960 pixels. Therefore 960/batch-2 is available for a separately scoped matched-resolution baseline. The local Faster R-CNN `s` model has 28.86M parameters versus YOLO26 `n` at 2.66M, so any future comparison must disclose this capacity mismatch.
- **Pre-fix same-images diagnostic: failed:** the intentionally leaked 72-image, three-class subset (`debug-data/overfit-balanced`) produced mAP50 `0.0486`, mAP50-95 `0.0313`, and recall `0.0458` after 100 epochs; Stringing and Warping recall were zero. Therefore the declining training/validation loss was not sufficient proof of detector learning. A pre-fix proposal diagnostic showed that, although the RoI head classified and localized exact GT boxes perfectly, the RPN recalled only `9/153` targets (`0.0588`) at IoU 0.50 under its 300-proposal inference path. Raw anchors covered `125/153` (`0.8170`) targets at IoU 0.50, so the primary failure was RPN supervision/ranking rather than anchor capacity, NMS, or the RoI head.
- **RPN repair and overfit validation:** the from-scratch RPN now uses foreground IoU `0.50` rather than `0.70`, forces up to four highest-IoU anchors positive per target, and repeats scarce positive anchors when forming the fixed 256-anchor RPN loss sample. This affects training supervision only; inference receives no GT boxes or labels. The assignment is isolated in `assign_rpn_targets()` and covered by `test_faster_rcnn.py`. Repeating the exact same intentionally leaked 72-image diagnostic with no other configuration change gave mAP50 `0.9881`, mAP50-95 `0.8109`, precision `0.8032`, and recall `0.9869`; class AP50 was `0.9643` / `1.0000` / `1.0000` for Spaghetti / Stringing / Warping. The corrected RPN recalled `151/153` boxes (`0.9869`) at IoU 0.50 and `146/153` (`0.9542`) at IoU 0.75. This is a successful implementation validation only, not a generalization result.
- **Replacement 960-pixel, three-fold Faster R-CNN baseline: completed:** using run root `runs/faster_rcnn/candidate_cv3_imgsz960_s_rpn_stable_posweight_p025_seed42_e50`, the repaired model selected validation-loss checkpoints at epochs `7` / `7` / `5`. Aggregate validation metrics at confidence `0.25` were mAP50 `0.0935 ± 0.0522`, mAP50-95 `0.0284 ± 0.0203`, precision `0.1441 ± 0.0157`, and recall `0.1445 ± 0.0369`. Per-class mean AP50 was Spaghetti `0.0578`, Layer cracking `0.0854`, Over extrusion `0.1118`, Stringing `0.0447`, and Warping `0.1681`. All classes obtain nonzero ranked AP, but minority-class operating recall is highly unstable: fold 3 produces no Layer cracking, Over extrusion, Stringing, or Warping detections at the fixed `0.25` threshold. This is therefore a valid scratch two-stage reference, not a replacement for YOLO26 or a deployment result.
- **Faster R-CNN decision:** do not run optimizer, loss, sampler, class-weight, threshold, epoch, resolution, anchor, or architecture sweeps on this export. The one permitted repaired rebaseline is complete. For the final report, compare it with direct YOLO26 + NMS as a scratch one-stage/two-stage systems comparison, disclose the Faster R-CNN `s`/YOLO26 `n` capacity mismatch, and do not treat either scratch model as an architecture-only comparison to pretrained YOLO11n.

## 2. Dataset Facts and Label Audit

### Source split sizes

| Split | Images |
| --- | ---: |
| Train | 2,696 |
| Validation | 1,524 |
| Test | 329 |

Classes are `spaghetti` (0), `stringing` (1), and `warping` (2).

### Raw-label issue discovered and fixed

Raw Roboflow labels mix standard five-field YOLO bounding boxes with polygon rows. The detector requires exactly:

```text
class_id x_center y_center width height
```

The preprocessing pipeline now preserves valid five-field rows and converts every polygon to its enclosing normalized bounding box.

| Split | Five-field boxes | Polygon rows converted | Strict boxes written |
| --- | ---: | ---: | ---: |
| Train | 5,211 | 693 | 5,904 |
| Validation | 3,144 | 276 | 3,420 |
| Test | 539 | 116 | 655 |

### Normalized box distribution

| Split | Spaghetti | Stringing | Warping |
| --- | ---: | ---: | ---: |
| Train | 4,120 (69.8%) | 1,404 (23.8%) | 380 (6.4%) |
| Validation | 1,354 (39.6%) | 1,449 (42.4%) | 617 (18.0%) |
| Test | 539 (82.3%) | 94 (14.4%) | 22 (3.4%) |

The train, validation, and test class distributions differ substantially. Select models with per-class validation metrics, not aggregate mAP alone.

### Post-hoc duplicate-content and annotation-disagreement audit

The baseline processed images were hashed with SHA-256. This revealed exact duplicate pixel content both within and across the original Roboflow splits. Normalizing polygon rows corrected label *format*, but it did not resolve duplicate images that carry different annotations.

| Audit result | Count | Interpretation |
| --- | ---: | --- |
| Train image files / unique exact hashes | 2,696 / 2,334 | 362 duplicate file entries occur within training. |
| Validation image files / unique exact hashes | 1,524 / 1,489 | 35 duplicate file entries occur within validation. |
| Test image files / unique exact hashes | 329 / 325 | 4 duplicate file entries occur within test. |
| Exact hashes shared between train and validation | 230 | Validation is not independent of training. |
| Exact hashes shared between train and test | 58 | The existing test report is not an independent held-out result. |
| Exact hashes shared between validation and test | 43 | Validation and test also overlap. |

For the shared exact-image hashes, annotations frequently disagree. At IoU 0.50, train–validation duplicate pairs matched only 329 of 878 train boxes and 1,007 validation boxes. Most disagreements retain the same class but use different box geometry; a smaller number use different class sets.

The combined train-plus-validation development pool has 4,220 image files but only **3,589 unique exact-image hashes**. It contains 631 duplicate file entries and 366 duplicate-content groups requiring annotation adjudication before automatic folding.

**Consequence:** existing validation/test metrics are useful diagnostics of the current exported split, but they are not clean independent generalization estimates. Do not blindly concatenate train and validation labels or apply image-level random K-fold splitting.

## 3. Implemented Engineering Changes

### Data pipeline

- [preprocess_dataset.py](preprocess_dataset.py) clears each generated split before rebuilding it, preventing stale images and labels.
- Polygon labels are normalized to strict five-field detection labels, with numeric/range validation and clipping safeguards.
- Train-only augmentation uses realistic transforms: horizontal flip, brightness, contrast, gamma, Gaussian noise, blur, translation, and scale. Vertical flips and 180° rotations were removed.
- Generated samples use parameter signatures and image hashes to prevent exact duplicates.
- Python and NumPy augmentation random sources are both seeded by `--minority-aug-seed`.
- **2026-07-14 correction:** minority-target accounting now tracks actual generated object boxes, not generated image count. This makes `--minority-target-ratio` match its documented object-count meaning.

### Training

- [train_yolo26.py](train_yolo26.py) rejects any processed label row that does not contain exactly five fields.
- Added CUDA AMP, AdamW, gradient clipping, checkpointing, strict label validation, and seed control.
- `--seed` defaults to `42` and seeds model initialization, DataLoader ordering, and weighted sampling.
- Added image-level class-balanced sampling with `--balanced-sampling-power`; it was tested and is currently rejected as an overall strategy.
- Added positive-only classification weighting with `--class-positive-weight-power`; it was tested and is currently rejected as an overall strategy.
- Positive class weights are saved in every new checkpoint.

### Evaluation and metrics

- [detection_metrics.py](detection_metrics.py) computes mAP50, mAP50-95, per-class AP, precision/recall, and a background-aware confusion matrix.
- Fixed AP bookkeeping so each true-positive match remains associated with its original prediction confidence before global AP sorting.
- [eval_yolo26.py](eval_yolo26.py) restores the model scale, loss settings, assignment settings, and positive class weights saved in a checkpoint unless a CLI option explicitly overrides them.
- This fixed misleading loss reporting when evaluating a checkpoint trained with a non-default `cls_gain`.
- `--conf-thresh` affects precision, recall, and the confusion matrix; it does **not** affect mAP.

### Sanity-check tool

- [make_overfit_subset.py](make_overfit_subset.py) builds a deliberately small, balanced dataset copied into both train and validation.
- The model successfully overfit that set, including the rare warping class. This showed the main limitation is full-data generalization/class imbalance rather than a fundamentally broken model pipeline.
- Its high validation metrics are **intentional memorization evidence**, not a benchmark result: every validation image is also a training image and must never be compared with the held-out full-dataset validation or test metrics.

## 4. Validation Checks Performed

| Check | Result |
| --- | --- |
| Strict processed-label parsing | Passed; malformed rows are rejected. |
| AP score/match alignment toy test | Passed. |
| Overfit sanity test | Passed: mAP50 `0.8424`, mAP50-95 `0.6065`; all three classes learned. Train and validation intentionally contain the same samples, so this verifies pipeline fit/memorization rather than generalization. |
| Seed reproducibility for model/sampler | Passed. |
| Neutral positive-class weights | Exactly reproduce the original BCE loss. |
| Positive-only BCE behavior | Passed; only positive terms receive class weights. |
| Checkpoint positive-weight restoration | Passed for new and legacy checkpoints. |
| Seeded NumPy noise augmentation | Passed. |
| Object-count augmentation target test | Passed with a multi-object synthetic image. |
| Augmentation split isolation | Passed: validation and test labels are byte-identical to clean data; only train images are added. |
| Exact-target augmentation integrity | Passed: strict labels, 824 warping training boxes, 310 added train images, and unchanged validation/test labels. |
| Clean preprocessing-variant parity | Passed: baseline, CLAHE, and CLAHE+Canny have equal split sizes and byte-identical labels. |
| Exact image-hash and annotation audit | Found leakage and label disagreement: 230 train–validation, 58 train–test, and 43 validation–test shared exact hashes; future evaluation must be group-disjoint. |
| Custom focal-loss mathematics | Passed: `focal_gamma=0` is exactly the prior BCE; `focal_gamma=2` suppresses easy correct anchors more than uncertain anchors; negative gamma is rejected. |
| Custom focal checkpoint restoration | Passed: evaluator restores saved focal gamma while legacy checkpoints default to `0`. |
| Five-class focal training smoke test | Passed: one CUDA epoch completed with `focal_gamma=2`; evaluator restored the setting and decoded five classes. |

## 5. Controlled Experiment Record

All results in this section are validation-split diagnostics. A later exact-hash audit found train–validation leakage, so they are not independent generalization estimates. The test-split diagnostic appears only in Section 7 and was not used for tuning.

### Earlier clean-label runs without a fixed seed

| Run | Change from clean baseline | mAP50 | mAP50-95 | Precision @ 0.25 | Recall @ 0.25 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `baseline_cleanlabels_lr5e-5_e50` | Clean labels, learning rate `5e-5` | 0.0767 | 0.0319 | 0.3899 | 0.0906 | Useful diagnostic; not strictly reproducible because no fixed seed. |
| `baseline_cleanlabels_balanced_e50` | Full inverse-frequency image sampling | 0.0517 | 0.0179 | 0.3178 | 0.0439 | Rejected: overall quality declined, despite nonzero warping recall. |
| `baseline_cleanlabels_balanced_p05_e50` | Tempered image sampling, power `0.5` | 0.0465 | 0.0165 | 0.3140 | 0.0269 | Rejected. |
| `baseline_cleanlabels_cls1_e50` | Global `cls_gain=1.0` | 0.0565 | 0.0207 | 0.4926 | 0.0392 | Rejected: fewer detections, especially stringing. |

### Reproducible seed-42 reference and ablations

| Run | Data / change | Selected epoch | mAP50 | mAP50-95 | Precision @ 0.25 | Recall @ 0.25 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_seed42_e50` | Clean normalized RGB baseline | 36 | 0.0670 | 0.0260 | **0.4103** | 0.0702 | Clean RGB reference; retains the strongest clean-data warping AP signal. |
| `clahe_seed42_e50` | Clean CLAHE preprocessing | 33 | **0.0674** | **0.0274** | 0.4012 | **0.0784** | Selected by validation aggregate mAP for one held-out test report; improvements are small and warping AP declined. |
| `clahe_canny_seed42_e50` | Clean CLAHE+Canny preprocessing | 26 | 0.0445 | 0.0176 | 0.3636 | 0.0433 | Rejected: worst clean preprocessing variant across aggregate metrics. |
| `baseline_seed42_posweight_p05_e50` | Positive-only inverse-frequency class weighting, power `0.5` | 27 | 0.0518 | 0.0190 | 0.3282 | 0.0251 | Rejected overall; warping became detectable but false positives increased. |
| `baseline_seed42_aug20_e50` | Offline warping augmentation, intended 20% target but actually 24.2% | 27 | 0.0603 | 0.0215 | **0.4896** | 0.0345 | Exploratory result only; target accounting was incorrect. |
| `baseline_seed42_aug20_exact_e50` | Exact 20% warping-box target, 824 total warping boxes | 27 | 0.0595 | 0.0224 | **0.5202** | 0.0377 | Rejected overall; strongest warping AP so far, but aggregate mAP and stringing recall declined. |

### Per-class comparison at confidence threshold 0.25

| Run | Spaghetti AP50 / recall | Stringing AP50 / recall | Warping AP50 / recall |
| --- | --- | --- | --- |
| Seed-42 clean reference | 0.1484 / 0.1388 | 0.0390 / 0.0359 | 0.0137 / 0.0000 |
| Clean CLAHE | **0.1601 / 0.1610** | 0.0363 / 0.0345 | 0.0058 / 0.0000 |
| Clean CLAHE+Canny | 0.1017 / 0.1093 | 0.0270 / 0.0000 | 0.0048 / 0.0000 |
| Positive weighting, power 0.5 | 0.1108 / 0.0369 | 0.0330 / 0.0131 | 0.0117 / 0.0276 |
| Exploratory augmentation | 0.1300 / 0.0783 | 0.0328 / 0.0000 | **0.0180 / 0.0194** |
| Exact 20% augmentation | 0.1239 / 0.0894 | 0.0312 / 0.0000 | **0.0234 / 0.0130** |

### Low-confidence behavior at threshold 0.10

| Run | Overall precision / recall | Warping precision / recall | Interpretation |
| --- | --- | --- | --- |
| Seed-42 clean reference | 0.2211 / 0.1173 | 1.0000 / 0.0032 | Only two correct, low-confidence warping detections. |
| Clean CLAHE | 0.1941 / **0.1301** | 0.0000 / 0.0000 | Higher aggregate recall, but lower precision and no warping detections. |
| Clean CLAHE+Canny | 0.1753 / 0.0848 | 0.0000 / 0.0000 | Rejected: lower aggregate precision and recall, with no warping detections. |
| Positive weighting, power 0.5 | 0.1829 / 0.0968 | 0.0636 / 0.0519 | More warping recall, but many false positives. |
| Exploratory augmentation | 0.2552 / 0.0860 | 0.1062 / 0.0389 | Warping signal improved, but overall/stringing recall declined. |
| Exact 20% augmentation | 0.2479 / 0.0863 | 0.1959 / 0.0308 | Better warping precision than other imbalance methods, but stringing recall remained very low. |

## 6. Offline-Augmentation Accounting Issue

The first offline augmentation run used:

```text
--minority-target-ratio 0.20
```

The majority training count was 4,120 boxes, so the intended warping target was:

$$
0.20 \times 4120 = 824
$$

However, the old generator created 444 **images** and each contained between 1 and 8 warping boxes. The resulting data had:

| Quantity | Value |
| --- | ---: |
| Original warping boxes | 380 |
| Intended target | 824 |
| Actual resulting warping boxes | 999 |
| Overshoot | 175 |
| Actual warping-to-majority ratio | 24.2% |
| Added training images | 444 |
| Mean warping boxes per added image | 1.394 |

The existing `processed-data-aug20` data and `baseline_seed42_aug20_e50` checkpoint are valid exploratory artifacts, but they must not be described as an exact 20% augmentation experiment.

The corrected implementation now:

1. counts actual class boxes in each accepted augmentation;
2. updates deficits using those counts;
3. rejects a transformed sample if it loses the target class entirely;
4. prints the target and final per-class box counts.

### Exact-target rerun: completed

The corrected exact-target dataset and run are valid:

| Quantity | Value |
| --- | ---: |
| Exact target and final warping boxes | 824 / 824 |
| Added train images | 310 |
| Train images after augmentation | 3,006 |
| Selected checkpoint epoch | 27 |
| mAP50 / mAP50-95 | 0.0595 / 0.0224 |
| Warping AP50 / recall @ 0.25 | 0.0234 / 0.0130 |
| Stringing recall @ 0.25 | 0.0000 |

The run increased warping AP50 from `0.0137` to `0.0234`, but the reference clean RGB baseline remains the overall validation choice because its aggregate mAP and recall are higher and it retains stringing detections.

## 7. Clean Preprocessing Comparison and Contaminated Test-Split Diagnostic

All three clean preprocessing variants used the same normalized labels, `seed=42`, learning rate, gains, batch size, epochs, and model scale. The only intended change was image preprocessing.

| Metric at confidence 0.25 | Clean RGB | Clean CLAHE | Clean CLAHE+Canny |
| --- | ---: | ---: | ---: |
| Selected epoch / validation loss | 36 / 12.7404 | 33 / **12.7356** | 26 / 13.5461 |
| mAP50 | 0.0670 | **0.0674** | 0.0445 |
| mAP50-95 | 0.0260 | **0.0274** | 0.0176 |
| Precision | **0.4103** | 0.4012 | 0.3636 |
| Recall | 0.0702 | **0.0784** | 0.0433 |
| Spaghetti AP50 | 0.1484 | **0.1601** | 0.1017 |
| Stringing AP50 | **0.0390** | 0.0363 | 0.0270 |
| Warping AP50 | **0.0137** | 0.0058 | 0.0048 |

### Relative preprocessing comparison

Within the same exported validation split, clean CLAHE leads on mAP50 and mAP50-95. The margin over RGB is small; RGB retains a stronger weak warping signal; and CLAHE+Canny is harmful. Because 230 exact image hashes overlap train and validation, this is a controlled **relative** preprocessing comparison, not a leakage-free generalization selection.

### Post-selection test-split diagnostic: completed

The CLAHE checkpoint was selected before the test command was run and was evaluated on `processed-data/clahe/test` (329 images) without subsequent tuning. A later exact-hash audit found 58 hashes shared with training and 43 hashes shared with validation, so this is **not** an independent held-out test result.

Executed reporting commands:

```text
python eval_yolo26.py --checkpoint runs/yolo26/clahe_seed42_e50/best.pt --data-root processed-data/clahe --split test --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.25
python eval_yolo26.py --checkpoint runs/yolo26/clahe_seed42_e50/best.pt --data-root processed-data/clahe --split test --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.10
```

#### Aggregate test metrics

| Metric | Threshold 0.25 | Threshold 0.10 |
| --- | ---: | ---: |
| Loss / classification loss / box loss | 11.0359 / 5.8058 / 5.2301 | 11.0359 / 5.8058 / 5.2301 |
| mAP50 | 0.1401 | 0.1401 |
| mAP50-95 | 0.0466 | 0.0466 |
| Precision | 0.4966 | 0.2714 |
| Recall | 0.1115 | 0.1985 |

#### Per-class test metrics

| Class | Ground-truth boxes | AP50 | AP50-95 | Precision / recall at 0.25 | Precision / recall at 0.10 |
| --- | ---: | ---: | ---: | --- | --- |
| Spaghetti | 539 | 0.1617 | 0.0763 | 0.4892 / 0.1262 | 0.2796 / 0.2189 |
| Stringing | 94 | 0.0802 | 0.0323 | 0.6250 / 0.0532 | 0.2105 / 0.1277 |
| Warping | 22 | 0.1783 | 0.0311 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

#### Interpretation

- The test mAP values are higher than validation values, but this is descriptive only. The test split has a different class composition, just 22 warping boxes, and exact image leakage from training/validation, so its per-class estimates have high uncertainty and cannot support an independent generalization claim.
- Warping AP50 is `0.1783` even though no warping detection survives either reporting threshold. This is expected: AP is threshold-independent and evaluates the complete confidence-ranked prediction list, while the precision/recall summaries retain only predictions at or above `--conf-thresh`. The result therefore indicates some low-confidence warping ranking signal below `0.10`, not usable warping detections at the chosen operating thresholds.
- Lowering the reporting threshold from `0.25` to `0.10` increased recall from `0.1115` to `0.1985`, but reduced precision from `0.4966` to `0.2714`.
- The final report should state that CLAHE had the highest observed mAP in the current exported validation split, RGB retained a stronger warping signal, CLAHE+Canny was harmful, and the split-level numbers are limited by duplicate-content leakage and annotation disagreement.

### Protocol closure

Do **not** evaluate more checkpoints on this test split, change a threshold, or tune a model based on these results. Future research must create a deduplicated, annotation-adjudicated, group-disjoint development protocol before using cross-validation or a newly held-out evaluation set.

## 8. Diagnosis: Why Metrics Remain Low After Label Repair and Augmentation

### Confirmed data and split limitations

| Evidence | Train | Validation | Test | Implication |
| --- | ---: | ---: | ---: | --- |
| Warping boxes | 380 (6.4%) | 617 (18.0%) | 22 (3.4%) | The rare class is underrepresented during training but overrepresented in validation. |
| Images containing warping | 268 | 353 | 17 | The custom model has limited distinct warping scenes from which to learn. |
| Median normalized warping-box area | 0.0255 | 0.0085 | 0.0354 | Validation warping objects are about one-third the median training area; test warping objects are larger again. |
| Clean RGB warping AP50 | — | 0.0137 | — | The baseline has only a weak rare-class signal before augmentation. |
| Exact 20% augmentation warping AP50 | — | 0.0234 | — | Duplicated/transformed warping images help the rare class somewhat, but do not solve generalization and reduce aggregate performance. |

The class proportions and warping object-size distributions differ markedly across splits. This explains why the model struggles especially on validation warping examples and why the selected model's test mAP is not directly comparable to validation mAP. The test set has comparatively few, generally larger warping objects.

### Duplicate content and annotation disagreement are an additional root cause

- The same exact pixels appear repeatedly with different boxes. In the combined train-plus-validation pool, 366 duplicate-content groups require annotation adjudication; 358 retain the same class set but disagree on box geometry, and 8 disagree on the class set itself.
- At IoU 0.50, exact train–validation duplicate pairs agree on only 329 matches across 878 train boxes and 1,007 validation boxes. This is materially more than harmless coordinate rounding.
- Therefore the fixed five-field format is necessary but not sufficient: annotation *consistency* remains unresolved. This adds noisy/contradictory supervision during training and invalidates naive independent validation/test claims.

### What label repair and augmentation did—and did not—solve

- Label normalization fixed invalid row format and polygon-to-box geometry. It did **not** reconcile duplicate images with inconsistent boxes, nor add new defect appearances, camera conditions, printer backgrounds, or small-warping examples.
- The exact-target augmentation increased training warping boxes from `380` to `824`, and raised warping validation AP50 from `0.0137` to `0.0234`. However, it reused transformations of the same limited source images and reduced mAP50 from `0.0670` to `0.0595` while stringing recall at threshold `0.25` became zero.
- Therefore, the current augmentation is evidence of a class-specific trade-off, not a full replacement for more diverse real training data.

### What can and cannot be concluded about the architecture

- The custom YOLO26-style model is **not proven broken**. The deliberate overfit test learned all three classes, so the data loader, labels, forward pass, loss, optimizer, and decoder can fit a small balanced dataset.
- The model is nevertheless an unpretrained, local educational implementation trained from scratch. Its `n` scale contains about 2.66 million trainable parameters, and the training objective uses BCE classification loss. It has not been benchmarked against a standard pretrained detector on the same cleaned train/validation split.
- The custom training code currently chooses `best.pt` by validation loss rather than validation mAP. This can select a checkpoint that is not the mAP-best one.
- Consequently, the evidence supports **both** a major data/split problem and a plausible custom-model/training-recipe contribution. It does not justify the claim that architecture alone caused the low metrics.

### Dataset strategy decision

**Recommendation: prioritize a better curated dataset, or a major cleanup of the current source, before further model tuning.** The current export remains useful as a diagnostic proof-of-concept and data-quality case study, but it is not a sound basis for final generalization claims because of duplicate content, annotation disagreement, severe class imbalance, and group/split shift.

If project requirements permit, use a new dataset as the final experimental corpus and retain the current dataset only to document the audit and preliminary pipeline work. A suitable replacement or supplement must provide:

1. **Source-level provenance:** a source image, printer/session, or video identifier so related frames can remain together during splitting.
2. **Consistent annotations:** one annotation policy per defect, with reviewed boxes or masks and no conflicting labels for duplicate content.
3. **Adequate unique rare-class coverage:** hundreds of distinct warping scenes/instances across printers, materials, lighting, viewpoints, and defect sizes—not many transformed copies of the same scenes.
4. **Consistent class taxonomy:** labels that can map unambiguously to spaghetti, stringing, and warping. Do not combine datasets with incompatible definitions without a documented relabeling policy.
5. **A group-disjoint split created before training:** no exact or near-duplicate source groups across train, validation, and final test partitions.
6. **Enough normal/negative images:** background and normal-print examples must represent deployment conditions, not only defect images.
7. **Appropriate licensing and documentation:** record source, license, class definitions, annotation method, and split construction.

Do not choose another dataset merely because it reports more image files. Audit its exact hashes, labels, source grouping, class balance, and annotation quality before adoption.

### Hugging Face candidate shortlist (2026-07-16)

The Hugging Face main dataset pages were intermittently unavailable from the development environment, but repository `README.md`, `dataset.yaml`, split summaries, dataset-viewer metadata, and DOI records were retrieved directly. The Platform Cam source was subsequently cloned and audited locally; no shortlisted dataset has been adopted.

| Rank | Dataset | Why it is promising | Verified metadata | Major unknowns before adoption |
| --- | --- | --- | --- | --- |
| 1, audited and rejected | `DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Plattform_Cam` | A documented standard-YOLO, whole-platform camera corpus from a named institutional source. It was the closest Hugging Face candidate for a **different nine-class platform-defect study**. | 2,510 / 301 / 334 images and 10,542 / 1,324 / 1,428 boxes in train / validation / test; 9 classes: Nozzle, Object, Purge Line, Spaghetti, Stringing, Unterextrusion, Warping, poor first layer, Double Print; DOI `10.57967/hf/7897`; CC BY-SA 4.0 stated in the README. | The completed local audit confirmed zero Warping labels, only 29 Stringing labels, two invalid rows, exact cross-split duplicates, and severe provenance-group leakage. It cannot be the current project's primary corpus. |
| 2, supplementary-view candidate | `DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Nozzle_Cam` | Larger documented YOLO detection corpus from the same institutional project; useful only as a potential nozzle-view supplementary domain. | 4,438 / 567 / 536 images and 15,172 / 1,912 / 1,888 boxes in train / validation / test; same 9-class taxonomy; DOI `10.57967/hf/7891`; CC BY-SA 4.0 stated in the README. | The provided summary lists 670 Stringing boxes but only 2 Warping boxes. Its nozzle-near camera domain and class imbalance make it unsuitable as a direct replacement or blind merge. |
| Supplement only | `ppak10/Additive-Manufacturing-Benchmark` with configuration `fdm_3d_printing_defect` | Contains 1,912 FDM defect images and explicit `label` / `label_id` fields. | Image-classification-style configuration, one train split, roughly 6.27 GB download. | It has no verified detection boxes or validation/test split, so it is not a direct replacement for this detector. |

Rejected from the search:

- `NilsHagenBeyer/3D_printing_errors`: Hugging Face dataset viewer reports an empty dataset.
- `g4ndh1/Additive-Manufacturing-Digital-Twin`: 3,300 image-classification samples of object types such as Boat, Gears, Phil, StressTest, and Vase, not printing defect detections.
- `Javiai/failures-3D-print`: standard object-detection structure but only 73 images, unknown license, one train split, and classes `Error`, `Extrusor`, `Part`, and `Spagheti`; it does not cover the required defect taxonomy.

**Updated shortlist decision:** no currently discovered Hugging Face dataset is a direct high-quality replacement for the five-class objective, because none supplies sufficient verified Warping coverage. The completed Platform Cam audit also rejects it as a stand-alone nine-class main corpus. The Nozzle Cam source remains unaudited locally and is unsuitable as a direct replacement on its published count of only two Warping boxes; it must not be merged blindly. Any future candidate must be acquired into a separate ignored directory, audited for pairing, taxonomy, exact and meaningful near duplicates, provenance groups, minority coverage, and annotation consistency, then split group-disjointly before any training.

### Platform Cam local acceptance audit: completed and rejected (2026-07-16)

**Acquisition:** the untouched source was cloned with Git LFS into `candidate-data/hf-errors-additive-manufacturing-platform-cam/` at pinned commit `ca192db90b68cd5df4f8a0adcd7dbb9c1b8699fa`. The LFS fetch contained 3,145 image objects (about 5.19 GiB). No source image or label was normalized, repaired, or used for training.

| Audit | Result | Consequence |
| --- | --- | --- |
| Image/label pairing | Exact one-to-one pairing: 2,510 train, 301 validation, and 334 test images and labels. | The raw directory structure is complete. |
| Image decoding | All 3,145 PNG files decoded successfully with OpenCV; every image is 1,920 × 1,080. | No corrupt or unreadable image was found. |
| Label rows | 13,294 raw box rows exactly reproduce the supplied split and class summaries. Every non-empty row has five numeric fields and a class ID in `0..8`, except two train Object rows with normalized vertical centres `1.010076...` and `1.048070...`. | Repair or exclude those two rows before any use; valid Object count falls from 3,138 to 3,136. |
| Boundary review | 22 otherwise numeric YOLO boxes extend past an image boundary when centre and size are combined. | These rows meet the README's individual-coordinate rule, but require a documented clipping/review policy for any processed copy. |
| Defect coverage | Warping has **0** boxes. Stringing has 29, poor-first-layer has 21, and Double Print has 49 boxes across all splits. | It fails the current five-class objective immediately and lacks viable coverage for several classes in a nine-class study. |
| Exact duplicate audit | 3,145 image files reduce to 3,000 unique SHA-256 values: 145 duplicate pairs, including 55 groups across official splits (22 train--validation, 32 train--test, 1 validation--test). Every cross-split duplicate pair has the same parsed annotation signature. | The duplicated labels are consistent, but the public validation and test partitions are not independent. |
| Filename provenance proxy | All 3,145 names match `<prefix>__<8-hex-token>-frame_<integer>`. Grouping by the repeated prefix before `__` yields 78 conservative provenance candidates; 77 span official splits and contain 3,123 records. | The exported split is not group-disjoint under the only repeated provenance-like filename component. The prefix is not a verified session ID, so source metadata is still needed for a defensible final grouping. |
| Minority provenance proxy | The 78 prefix groups contain only 2 Stringing, 11 Unterextrusion, 5 poor-first-layer, and 4 Double Print groups; Warping has none. | Even treating the prefix as a conservative group key, three folds cannot represent Stringing in every validation fold. |
| Perceptual screening | Exhaustive 64-bit DCT pHash comparison at Hamming distance ≤ 5 found 718,455 candidate pairs, 251,526 crossing official splits, and joined the static-camera corpus into 13 transitive components. | This threshold is over-permissive for this fixed platform view and cannot be used alone as a source/near-duplicate grouping rule. Exact hashes and real source/session provenance remain necessary. |
| Visual spot check | Raw stratified samples were inspected for Spaghetti, Stringing, Unterextrusion, poor first layer, and Double Print; visible printer/defect content is consistent with the broad platform-view taxonomy. Warping has no sample because it has no labels. | A small spot check does not overcome the missing Warping class, rare-class scarcity, leakage, or need for full annotation review. |

The class totals are Nozzle 3,051, Object 3,138 raw / 3,136 geometrically valid, Purge Line 3,634, Spaghetti 3,094, Stringing 29, Unterextrusion 278, Warping 0, poor first layer 21, and Double Print 49. All 3,145 label files are non-empty, so the export also provides no fully unannotated normal-image files.

**Acceptance decision: REJECT as the primary dataset.** Do not normalize it into a training copy, create CV folds, run `train_yolo26.py`, use its published validation/test scores, or merge it with Nozzle Cam. Its lack of Warping makes it unusable for the existing five-class scope; its rare nine-class defects are too sparse for stable group-disjoint evaluation; and the supplied split has proven exact and provenance-proxy leakage. It can only be reconsidered for a newly scoped platform-view study after obtaining verified source/session metadata, repairing the two invalid rows under a documented policy, manually reviewing annotations, adding substantial independent rare-class data (especially Warping), and rebuilding all splits from scratch.

#### User-authorized one-class Spaghetti architecture control: prepared (2026-07-16)

The project requirement then changed from finding a replacement five-class failure dataset to determining whether the local YOLO26-style implementation can learn **any** reasonably large, visually coherent print-failure detection task. This does **not** reverse the rejection above: Platform Cam remains rejected as the primary multi-defect corpus. It is used only as a clearly scoped one-class `Spaghetti` control, with no claim about Warping, the nine-class taxonomy, or real-world multi-printer generalization.

New builder [prepare_platform_spaghetti_cv.py](prepare_platform_spaghetti_cv.py) was added specifically for this control. It combines all official Platform Cam folders as development data, validates every source row, projects source class `3` (`Spaghetti`) to derived class `0`, clips seven target boxes that cross an image edge, excludes the two invalid non-target Object rows from the derived labels, removes exact duplicate images only after verifying their projected labels agree, and writes hard-linked one-class YOLO folds. It deliberately does **not** use the previous distance-5 pHash rule because that rule is over-permissive for this static platform camera. Instead, it keeps the filename prefix before `__` and every exact SHA-256 connection together as a conservative provenance proxy.

Executed build:

```text
python prepare_platform_spaghetti_cv.py --input-root candidate-data/hf-errors-additive-manufacturing-platform-cam --output-root cv-data/hf-platform-spaghetti-1class --folds 3 --seed 42 --attempts 100 --overwrite
```

| Build property | Result |
| --- | --- |
| Raw source records / exact-unique records | 3,145 / 3,000; 145 redundant exact copies removed. |
| Derived labels | 3,040 valid Spaghetti boxes in 1,382 positive images; 1,618 empty-label hard-negative images. |
| Grouping | 74 exact/provenance-proxy groups, including 41 groups with at least one Spaghetti box. This is conservative but not verified session provenance. |
| Fold 1 validation | 1,001 images; 467 positive / 534 negative; 1,027 Spaghetti boxes. |
| Fold 2 validation | 999 images; 466 positive / 533 negative; 1,003 Spaghetti boxes. |
| Fold 3 validation | 1,000 images; 449 positive / 551 negative; 1,010 Spaghetti boxes. |
| Materialization and validation | 9,000 hardlinks; each fold has one-to-one image/label pairing and strict one-class five-field YOLO rows. A `YoloDetectionDataset` loader smoke test on fold 1 returned valid `(4, 3, 640, 640)` batches for train and validation. |

The architecture-control protocol is intentionally neutral: randomly initialized local YOLO26 `n` scale, `--num-classes 1`, seed `42`, a fixed 31-epoch budget, image size `640`, batch size `8`, learning rate `5e-5`, `cls_gain=0.5`, no focal loss, no positive weighting, no sampler, no preprocessing variant, no augmentation, and no pretrained/hosted weights. All three folds use this unchanged configuration and report mean ± sample standard deviation. The result can demonstrate whether the implementation learns this one-class, platform-view task; it cannot certify the model for the original rare multi-defect setting.

**Fold 1 interim result:** the first fixed-configuration run was manually stopped after completing epoch 31 of the planned maximum 50 epochs, so it is an **interim**, not final cross-fold result. The saved best checkpoint is epoch 31 (training loss `3.7212`, validation loss `8.1248`). Its independent-within-proxy-group validation result at threshold `0.25` is mAP50 `0.2449`, mAP50-95 `0.0763`, precision `0.6109`, and recall `0.1850` on 1,027 Spaghetti boxes. The confusion matrix has 190 matched detections, 837 false negatives, and 121 background false positives. This is substantially above the prior five-class grouped-CV mAP50, confirming that the local implementation learns meaningful signal on a larger one-class control. It does not prove a final 50-epoch optimum, a multi-defect solution, or external generalization. Do not use this interim fold to tune settings.

##### Reusable K-fold automation: refactored

The initial Platform-specific wrappers were replaced rather than merged into the one-fold trainer. [train_yolo26.py](train_yolo26.py) remains the reusable single-fold training primitive; it now reads `nc` and class names from any dataset's `data.yaml` and rejects a conflicting explicit `--num-classes`. [eval_yolo26.py](eval_yolo26.py) uses the same shared parser and can emit machine-readable metrics with `--metrics-output`.

[run_yolo26_kfold_cv.py](run_yolo26_kfold_cv.py) and [eval_yolo26_kfold_cv.py](eval_yolo26_kfold_cv.py) are dataset-agnostic orchestrators for any standard layout containing `fold_1/`, `fold_2/`, and so on, where each fold has `data.yaml`, `train/images`, `train/labels`, `valid/images`, and `valid/labels`. They auto-discover folds when `--folds` is omitted, infer the class count and taxonomy from `data.yaml`, run folds sequentially, write a reproducible plan, reject mismatched or partial checkpoints instead of silently reusing them, and aggregate both overall and per-class mean/sample-SD metrics.

[run_ultralytics_kfold_cv.py](run_ultralytics_kfold_cv.py) and [eval_ultralytics_kfold_cv.py](eval_ultralytics_kfold_cv.py) provide the corresponding generic pretrained Ultralytics reference workflow. They are intentionally separate from the custom runner because pretrained weights, augmentation, model API, and NMS differ; merging those backends would obscure the experimental protocol.

The new generic custom runner/evaluator was dry-run validated against both existing fold layouts: the one-class Platform Spaghetti control (`nc=1`) and the five-class Roboflow candidate (`nc=5`). Its evaluator was also run against the existing fold-1 control checkpoint and reproduced the prior metrics exactly. The generic Ultralytics runner/evaluator dry runs validated the same one-class folds and local `yolo11n.pt` path. The currently validated Ultralytics dependency (`8.4.89`) is now explicitly pinned in [requirements.txt](requirements.txt). No new model training was launched by this refactor.

For any future custom YOLO26 study, use the pattern:

```text
python run_yolo26_kfold_cv.py --data-root <cv-data-root> --run-root <new-run-root> --epochs <fixed-budget> [fixed training options]
python eval_yolo26_kfold_cv.py --data-root <cv-data-root> --run-root <same-run-root> --conf-thresh 0.25
```

For the completed one-class control, the equivalent historical command would be:

```text
python run_yolo26_kfold_cv.py --data-root cv-data/hf-platform-spaghetti-1class --run-root runs/yolo26/platform_spaghetti_cv3_neutral_e31 --epochs 31 --lr 5e-5
python eval_yolo26_kfold_cv.py --data-root cv-data/hf-platform-spaghetti-1class --run-root runs/yolo26/platform_spaghetti_cv3_neutral_e31 --conf-thresh 0.25
```

The completed run is recognized as configuration-matching and skipped; do not rerun it.

##### Three-fold one-class architecture-control result: completed

The fresh sequential run and aggregate evaluation completed under `runs/yolo26/platform_spaghetti_cv3_neutral_e31/`. Every fold trained through epoch 31; checkpoints are selected strictly by minimum validation loss, which occurred at epoch 31 for fold 1 and epoch 29 for folds 2 and 3. The latter is normal checkpoint selection rather than an incomplete training run.

| Metric | Fold 1 | Fold 2 | Fold 3 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Selected checkpoint epoch | 31 | 29 | 29 | 29.7 ± 1.2 |
| Validation loss | 8.1248 | 8.8885 | 9.2494 | 8.7543 ± 0.5742 |
| mAP50 | 0.2449 | 0.1831 | 0.2167 | **0.2149 ± 0.0309** |
| mAP50-95 | 0.0763 | 0.0440 | 0.0671 | **0.0625 ± 0.0166** |
| Precision at threshold `0.25` | 0.6109 | 0.4904 | 0.5034 | **0.5349 ± 0.0662** |
| Recall at threshold `0.25` | 0.1850 | 0.1525 | 0.2178 | **0.1851 ± 0.0326** |
| Validation Spaghetti boxes | 1,027 | 1,003 | 1,010 | 3,040 total across folds |

At threshold `0.25`, the three folds respectively produced 190 / 153 / 220 matched Spaghetti boxes, 837 / 850 / 790 false negatives, and 121 / 159 / 217 background false positives. The important outcome is that no fold collapsed: the randomly initialized implementation learns a repeatable Spaghetti detection signal on all independent-within-proxy-group folds. The moderate cross-fold variation supports that conclusion.

Two post-training diagnostics explain why this is still a weak control result. First, the validation boxes are not all large: at the training size of 640, the 10th-percentile box width is only 21.7--25.6 pixels and the 10th-percentile height is 29.7--37.9 pixels; the smallest-scale detector feature map is stride 8, so these examples occupy only a few feature-map cells. A size-stratified IoU-0.50 analysis at threshold `0.25` confirms this is material: across all folds, recall is `0.0343` for 379 COCO-small boxes (area below $32^2$ pixels), `0.1853` for 2,002 medium boxes, and `0.2716` for 659 large boxes. By short side, recall is `0.0675` below 32 pixels, `0.1470` from 32 to below 64 pixels, and `0.3199` at 64 pixels or larger. Second, a predeclared diagnostic re-evaluation at threshold `0.10` raised mean recall only from `0.1851 ± 0.0326` to `0.2535 ± 0.0419`, while mean precision fell from `0.5349 ± 0.0662` to `0.3987 ± 0.0733`. This recovers 208 additional correct boxes across all folds but adds 710 background false positives; it is therefore evidence of limited low-confidence signal, not a deployment threshold or a cure for the low recall. mAP is unchanged because it is threshold-independent.

**Architecture-control decision:** this is evidence that the local YOLO26-style pipeline is functioning and can learn nontrivial visual features; it is **not** evidence that it is already a strong detector. A mean recall of `0.1851` means roughly 81.5% of annotated boxes are missed at the fixed `0.25` operating threshold, and mAP50 `0.2149` remains far below a deployment-quality result. The remaining limits plausibly include the unpretrained small custom architecture, fixed 640-square resizing of 16:9 images, stride-8 resolution for small/diffuse targets, loss/checkpoint selection by validation loss rather than mAP, and annotation/domain variation across the 74 provenance-proxy groups. The original five-class study additionally suffers from severe rare-class scarcity, annotation inconsistency, and split contamination, so its much lower result cannot be attributed to one cause.

##### Pretrained YOLO11n practical reference: completed (2026-07-18)

The predeclared practical reference was run after the custom control completed. It used the **same three frozen one-class folds**, a maximum of 31 epochs, image size 640, batch size 8, seed 42, and the project metric implementation at threshold `0.25`. The reference used local `yolo11n.pt` (SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`), COCO pretraining, Ultralytics `optimizer=auto`, standard Ultralytics augmentation/preprocessing, and Ultralytics NMS. It is therefore a practical reference—not an architecture-only or single-variable comparison.

| Metric | Fold 1 | Fold 2 | Fold 3 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| mAP50 | 0.4595 | 0.4403 | 0.5098 | **0.4699 ± 0.0359** |
| mAP50-95 | 0.2158 | 0.1721 | 0.2162 | **0.2014 ± 0.0254** |
| Precision at threshold `0.25` | 0.4974 | 0.4513 | 0.4941 | **0.4809 ± 0.0257** |
| Recall at threshold `0.25` | 0.4684 | 0.4756 | 0.5347 | **0.4929 ± 0.0364** |

| Cross-fold metric | Custom YOLO26 from scratch | Pretrained YOLO11n reference | Reference minus custom |
| --- | ---: | ---: | ---: |
| mAP50 | 0.2149 ± 0.0309 | 0.4699 ± 0.0359 | **+0.2550** (2.19×) |
| mAP50-95 | 0.0625 ± 0.0166 | 0.2014 ± 0.0254 | **+0.1389** (3.22×) |
| Precision at 0.25 | **0.5349 ± 0.0662** | 0.4809 ± 0.0257 | -0.0540 |
| Recall at 0.25 | 0.1851 ± 0.0326 | **0.4929 ± 0.0364** | **+0.3077** (2.66×) |

At the shared operating threshold, the reference matches 1,498 of 3,040 ground-truth boxes versus 563 for custom YOLO26: 935 additional correct detections and 935 fewer false negatives, but 1,122 additional false positives. This explains its modestly lower precision and much stronger recall.

**Reference decision:** the one-class data and folds can support materially stronger detection, so the largest practical bottleneck is the custom system—not an inherently unlearnable Platform Spaghetti task. The result does **not** isolate a single cause: the reference differs simultaneously in pretrained visual features, model/head/loss implementation, optimizer and schedule, online augmentation/preprocessing, and NMS decoding. It rules out neither the remaining data limitations nor the need for new multi-failure data; it does rule out treating the custom model's low control score as evidence that no model can learn the task.

**Transition decision:** this plan was executed. The five-class pretrained reference completed and substantially outperformed every custom class, so the single pre-registered 960-resolution custom ablation below was run without changing any loss, sampler, class-weight, augmentation, threshold, seed, fold, or duration setting.

##### Pretrained YOLO11n practical reference on five-class grouped folds: completed (2026-07-18)

The required multi-class diagnostic completed on the same 823-group, three-fold candidate data used by the custom CV. The reference used local `yolo11n.pt` (the same SHA-256 recorded for the one-class reference), COCO pretraining, 50 maximum epochs, image size 640, batch size 8, seed 42, Ultralytics `optimizer=auto`, standard Ultralytics augmentation/preprocessing, and Ultralytics NMS. Metrics below use the project metric implementation after NMS at threshold `0.25`. As with the one-class comparison, this is a practical baseline rather than an architecture-only comparison.

| Metric | Fold 1 | Fold 2 | Fold 3 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| mAP50 | 0.5102 | 0.4642 | 0.4654 | **0.4799 ± 0.0262** |
| mAP50-95 | 0.2035 | 0.2011 | 0.1893 | **0.1979 ± 0.0076** |
| Precision at threshold `0.25` | 0.3832 | 0.3612 | 0.3853 | **0.3766 ± 0.0133** |
| Recall at threshold `0.25` | 0.2370 | 0.2390 | 0.2617 | **0.2459 ± 0.0137** |

| Class | Reference AP50 mean ± SD | Reference recall mean ± SD | Custom AP50 mean ± SD | Custom recall mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| Spaghetti | 0.1507 ± 0.0144 | 0.1900 ± 0.0166 | 0.0195 ± 0.0019 | 0.0314 ± 0.0068 |
| Layer cracking | 0.5062 ± 0.0795 | 0.5474 ± 0.1114 | 0.0372 ± 0.0211 | 0.1368 ± 0.0737 |
| Over extrusion | 0.7704 ± 0.0309 | 0.7561 ± 0.0520 | 0.1323 ± 0.0274 | 0.2862 ± 0.0836 |
| Stringing | 0.4062 ± 0.0451 | 0.4886 ± 0.0570 | 0.0612 ± 0.0421 | 0.0913 ± 0.0440 |
| Warping | 0.5661 ± 0.0396 | 0.7607 ± 0.0534 | 0.0572 ± 0.0035 | 0.3504 ± 0.0148 |

Overall, the pretrained reference improves mAP50 by `+0.4184` (7.80×), mAP50-95 by `+0.1799` (11.00×), precision by `+0.2826` (4.01×), and recall by `+0.1921` (4.57×) relative to the fixed custom focal-plus-weight CV. It improves AP50 and operating recall for every class, including the sparse Warping and Stringing classes. The public candidate data still has limited independent source coverage and the rare-class estimates remain uncertain, but it is demonstrably capable of supporting much stronger multi-class detection than the custom result.

**Multi-class reference decision:** data quality and rarity remain serious final-study limitations, but they are not the immediate reason for the custom detector's near-collapse. The main practical bottleneck is the custom YOLO26 training/inference system. The reference changes several variables at once—pretraining, model/head/loss, optimizer/schedule, augmentation/preprocessing, and NMS—so it does not identify one isolated code defect and must not be described as a pure architecture comparison.

##### 960-resolution grouped-CV ablation: completed (2026-07-18)

The single pre-registered custom follow-up changed **only** `--imgsz` from 640 to 960. The candidate export is already 640 × 640, so the larger input does **not** create new raw visual detail or introduce aspect-ratio distortion; it resamples the source and increases the detector grid density from $80\times80$ to $120\times120$ at stride 8. This was motivated by the source-box audit: Layer cracking has a median short side of only 30.5 pixels at 640 (10th percentile 17.2), while Over extrusion, Stringing, and Warping also have small lower-tail objects.

A CUDA forward/loss/backward readiness test passed at image size 960, batch size 8, five classes, focal gamma 2, and positive-weight power 0.25, with peak allocated CUDA memory 2.70 GiB on the RTX 4070 Laptop GPU. Therefore batch size and all other fixed custom settings remained unchanged. The full three-fold run used the following command:

```text
python run_yolo26_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/yolo26/candidate_cv3_imgsz960_focal_g2_posweight_p025_seed42_e50 --epochs 50 --batch-size 8 --imgsz 960 --workers 0 --lr 5e-5 --weight-decay 5e-4 --seed 42 --device cuda --scale n --box-gain 7.5 --cls-gain 0.5 --reg-gain 1.5 --one2many-topk 10 --one2one-topk 1 --focal-gamma 2.0 --class-positive-weight-power 0.25 --balanced-sampling-power 1.0

python eval_yolo26_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/yolo26/candidate_cv3_imgsz960_focal_g2_posweight_p025_seed42_e50 --imgsz 960 --batch-size 8 --workers 0 --device cuda --conf-thresh 0.25
```

| Metric | 640 fixed custom CV | 960 resolution-only CV | 960 minus 640 |
| --- | ---: | ---: | ---: |
| Selected checkpoint epoch | 41.3 ± 1.2 | 43.3 ± 6.1 | — |
| Validation loss | 6.6858 ± 0.0331 | **6.5474 ± 0.0269** | -0.1384 |
| mAP50 | 0.0615 ± 0.0064 | **0.0782 ± 0.0161** | **+0.0167** |
| mAP50-95 | 0.0180 ± 0.0019 | **0.0259 ± 0.0065** | **+0.0079** |
| Precision at threshold `0.25` | 0.0940 ± 0.0045 | **0.1062 ± 0.0312** | +0.0122 |
| Recall at threshold `0.25` | 0.0538 ± 0.0065 | **0.0604 ± 0.0186** | +0.0066 |

The aggregate mAP result improved in every paired fold: mAP50 differences were `+0.0083`, `+0.0101`, and `+0.0315`; mAP50-95 differences were `+0.0052`, `+0.0053`, and `+0.0134`. With only three folds this is not a formal significance test, but it is coherent directional evidence that grid resolution helps aggregate custom localization/ranking.

| Class | 640 AP50 mean ± SD | 960 AP50 mean ± SD | Resolution effect |
| --- | ---: | ---: | --- |
| Spaghetti | 0.0195 ± 0.0019 | 0.0225 ± 0.0029 | Essentially unchanged / slight aggregate gain. |
| Layer cracking | 0.0372 ± 0.0211 | 0.0430 ± 0.0390 | Small, highly uncertain AP change; recall declined. |
| Over extrusion | 0.1323 ± 0.0274 | **0.1497 ± 0.0340** | Moderate improvement; recall increased from 0.2862 to 0.3874. |
| Stringing | **0.0612 ± 0.0421** | 0.0242 ± 0.0238 | Material regression; recall declined from 0.0913 to 0.0776. |
| Warping | 0.0572 ± 0.0035 | **0.1514 ± 0.0362** | Largest gain; recall increased from 0.3504 to 0.5726. |

Fold 3 selected its final epoch 50 checkpoint, while folds 1 and 2 selected epochs 38 and 42. The 50-epoch cap remains fixed for this comparison; do not extend only fold 3 or run a duration sweep after observing this result.

**Resolution decision:** retain 960 as the current **aggregate-mAP / Warping-and-Over-extrusion-oriented custom candidate**, because aggregate mAP50 and mAP50-95 improved in all folds. Do not describe it as a universal multi-class improvement: Stringing regressed and Layer cracking remains unstable. Do not test 1024, 1280, a changed batch size, or a changed duration; that would convert this single controlled result into an unbounded search.

##### Fixed-spec class-aware NMS diagnostic: completed without retraining (2026-07-19)

The legacy custom evaluator selected a global top-$k$ set from decoded predictions but did not suppress overlapping same-class boxes. A new [class_aware_nms()](models/yolo26_torch.py) decoder now operates on all raw decoded one-to-one candidates **before** legacy top-$k$ truncation. It uses `torchvision.ops.batched_nms`, retains scores at least `0.001` for AP, applies class-aware IoU suppression at `0.70`, and keeps at most 300 detections per image. [tests/test_yolo26.py](tests/test_yolo26.py) now verifies that duplicate same-class boxes are suppressed while an overlapping different-class detection remains. `torchvision` is explicitly listed in [requirements.txt](requirements.txt).

This was a fixed decoder diagnostic on already trained checkpoints—not a new training run, confidence sweep, or NMS-IoU sweep. The custom evaluator and generic K-fold evaluator now default to `class_aware_nms`; pass `--postprocess legacy_topk` only to reproduce the historical reports.

| Aggregate metric at threshold `0.25` | 640 legacy top-$k$ | 640 class-aware NMS | 960 legacy top-$k$ | 960 class-aware NMS |
| --- | ---: | ---: | ---: | ---: |
| mAP50 | 0.0615 ± 0.0064 | 0.0740 ± 0.0042 | 0.0782 ± 0.0161 | **0.1127 ± 0.0232** |
| mAP50-95 | 0.0180 ± 0.0019 | 0.0205 ± 0.0018 | 0.0259 ± 0.0065 | **0.0343 ± 0.0076** |
| Precision | 0.0940 ± 0.0045 | 0.1346 ± 0.0093 | 0.1062 ± 0.0312 | **0.1621 ± 0.0497** |
| Recall | 0.0538 ± 0.0065 | 0.0518 ± 0.0068 | 0.0604 ± 0.0186 | **0.0588 ± 0.0172** |

For the 960 checkpoints, NMS improves mAP50 by `+0.0345`, mAP50-95 by `+0.0084`, and precision by `+0.0558`, while reducing recall only `-0.0016`. Every 960 fold improved both aggregate mAP measures. It also preserves the resolution conclusion: 960 + NMS exceeds 640 + NMS by `+0.0387` mAP50 and `+0.0139` mAP50-95.

| Class at 960 | Legacy AP50 | NMS AP50 | Legacy recall | NMS recall |
| --- | ---: | ---: | ---: | ---: |
| Spaghetti | 0.0225 | 0.0338 | 0.0309 | 0.0291 |
| Layer cracking | 0.0430 | 0.0511 | 0.1123 | 0.1123 |
| Over extrusion | 0.1497 | 0.1890 | 0.3874 | 0.3874 |
| Stringing | 0.0242 | 0.0259 | 0.0776 | 0.0776 |
| Warping | 0.1514 | **0.2636** | 0.5726 | 0.5726 |

**NMS decision:** adopt class-aware NMS at IoU `0.70`, candidate score `0.001`, and maximum 300 detections as the standard custom inference/evaluation decoder. The gains show duplicate-box postprocessing was a real custom inference defect. It does not close the large gap to pretrained YOLO11n, and it does not solve Stringing or the sparse-data limitations, but it is a valid no-retraining improvement.

**Historical one-to-one result:** the table above remains the valid historical one-to-one inference comparison. At 960 + class-aware NMS it yielded mAP50 `0.1127 ± 0.0232`, mAP50-95 `0.0343 ± 0.0076`, precision `0.1621 ± 0.0497`, and recall `0.0588 ± 0.0172` at `0.25`.

##### One-to-many inference-branch correction: completed without retraining (2026-07-21)

The completed branch evaluation in the training-system section showed that the full-gradient one-to-many head is consistently stronger than the historically decoded detached-feature one-to-one head. Applying the same class-aware NMS settings to the already frozen 960 checkpoints improves every fold's mAP50 and mAP50-95. The selected no-retraining custom inference configuration is now direct regression (`reg_max=1`), 960 input, focal gamma 2, positive-weight power .25, **one-to-many** branch, candidate score .001, NMS IoU .70, and at most 300 detections:

| Aggregate metric at threshold `0.25` | Historical one-to-one + NMS | Selected one-to-many + NMS | Difference |
| --- | ---: | ---: | ---: |
| mAP50 | 0.1127 ± 0.0232 | **0.1270 ± 0.0210** | +0.0143 |
| mAP50-95 | 0.0343 ± 0.0076 | **0.0420 ± 0.0081** | +0.0077 |
| Precision | **0.1621 ± 0.0497** | 0.1153 ± 0.0207 | -0.0467 |
| Recall | 0.0588 ± 0.0172 | **0.1159 ± 0.0246** | +0.0570 |

All class AP50 means improve with one-to-many inference: Spaghetti `0.0379`, Layer cracking `0.0738`, Over extrusion `0.1997`, Stringing `0.0310`, and Warping `0.2928`. This is an inference-selection correction, not a retraining or threshold sweep. One-to-one remains available through `--inference-branch one2one` for historical reproduction; `--postprocess legacy_topk` automatically selects the historical one-to-one path.

##### Distributional box regression (`reg_max=16`): implemented, validated, and rejected by three-fold CV (2026-07-19)

The custom architecture milestone was implemented and tested. The previous head used `reg_max=1`, which directly regressed four box distances. The configurable DFL path uses `reg_max=16`: each left/top/right/bottom distance is represented by 16 logits, softmax-normalized, and decoded by its expected discrete distance. It was intended to improve localization quality, so mAP50-95 was the primary experiment outcome.

Implementation changes include:

1. [models/yolo26_torch.py](models/yolo26_torch.py) now provides `DistributionIntegral`, configurable detection-head channels $4\times\texttt{reg\_max}$, and DFL decoding before box construction;
2. [train_yolo26.py](train_yolo26.py) now accepts `--reg-max`, decodes DFL logits for task assignment, and uses Ultralytics `BboxLoss(reg_max)` to enable the DFL loss path when `reg_max > 1`;
3. [eval_yolo26.py](eval_yolo26.py) restores `reg_max` from new checkpoints while treating missing legacy metadata as `reg_max=1`, preserving all existing checkpoint compatibility;
4. [run_yolo26_kfold_cv.py](run_yolo26_kfold_cv.py) records and verifies `reg_max` across every fold; and
5. [tests/test_yolo26.py](tests/test_yolo26.py) verifies DFL integral decoding, DFL-head output shape, finite decoded predictions, and class-aware NMS compatibility.

Validation completed before the DFL training run:

| Check | Result |
| --- | --- |
| Legacy 640/960 checkpoint restoration | Passed: pre-DFL checkpoint with no saved `reg_max` restores as `reg_max=1` and loads successfully. |
| New DFL checkpoint restoration | Passed: a `reg_max=16` model state, including `detect.dfl.project`, restores successfully. |
| DFL architecture smoke test | Passed: five-class raw head shape at 640 is `(1, 69, 8400)`, where $69=5+4\times16$; decoded output remains `(1, 9, 8400)` and postprocessed output remains `(1, 300, 6)`. |
| Five-class CUDA forward/loss/backward smoke test | Passed at 960, batch size 8, focal gamma 2, positive-weight power 0.25: finite total loss `17.3030`, including finite DFL regression loss `9.2143`, with peak allocated CUDA memory 2.93 GiB. |
| Generic K-fold command validation | Passed: the runner generates the intended `--reg-max 16` command for all three candidate folds. |

The pre-registered one-variable experiment changed only `--reg-max` from 1 to 16; it kept the candidate data, group-disjoint folds, 960 input, batch size 8, focal gamma 2, positive-weight power 0.25, no sampler, no augmentation, seed 42, 50-epoch budget, and class-aware NMS fixed. The completed command was:

```text
python run_yolo26_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/yolo26/candidate_cv3_imgsz960_dfl_r16_focal_g2_posweight_p025_seed42_e50 --epochs 50 --batch-size 8 --imgsz 960 --workers 0 --lr 5e-5 --weight-decay 5e-4 --seed 42 --device cuda --scale n --box-gain 7.5 --cls-gain 0.5 --reg-gain 1.5 --reg-max 16 --one2many-topk 10 --one2one-topk 1 --focal-gamma 2.0 --class-positive-weight-power 0.25 --balanced-sampling-power 1.0

python eval_yolo26_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/yolo26/candidate_cv3_imgsz960_dfl_r16_focal_g2_posweight_p025_seed42_e50 --imgsz 960 --batch-size 8 --workers 0 --device cuda --conf-thresh 0.25
```

| Metric at threshold `0.25` | Historical one-to-one 960 + NMS direct regression | Historical one-to-one 960 + NMS DFL (`reg_max=16`) | DFL minus direct |
| --- | ---: | ---: | ---: |
| Selected checkpoint epoch | 38 / 42 / 50 | 17 / 25 / 27 | Earlier minimum validation loss in every DFL fold. |
| mAP50 | **0.1127 ± 0.0232** | 0.0802 ± 0.0258 | -0.0325 |
| mAP50-95 | **0.0343 ± 0.0076** | 0.0258 ± 0.0093 | -0.0086 |
| Precision | **0.1621 ± 0.0497** | 0.1083 ± 0.0898 | -0.0538 |
| Recall | **0.0588 ± 0.0172** | 0.0351 ± 0.0097 | -0.0237 |

The DFL run is a **valid negative result** within the historical one-to-one inference protocol: architecture smoke tests, DFL loss/backpropagation, checkpoint restoration, and evaluation all passed, but the fixed three-fold experiment degraded both aggregate AP measures and recall. It did not solve localization under the frozen 50-epoch, fixed-gain protocol. Its selected validation-loss epochs were earlier than the direct-regression run, but extending epochs, changing DFL gain, or reopening DFL evaluation under a new inference selection would be a new study and is not authorized by this controlled protocol.

Only Layer cracking AP50 rose slightly (`0.0511` to `0.0576`), with high uncertainty; all other classes declined in AP50 relative to 960 + NMS. In particular, Over extrusion fell from `0.1890` to `0.1131`, Stringing from `0.0259` to `0.0153`, and Warping from `0.2636` to `0.1885`. Therefore **reject `reg_max=16` as the current custom configuration**. Preserve the implementation as a documented, tested experimental capability, but do not use its checkpoints as the model baseline and do not tune DFL hyperparameters on this candidate export.

In parallel, the data-first work should start now: use [group_manifest.csv](cv-data/roboflow-3d-print-fail-v1/group_manifest.csv) with `fold=1` to avoid repeated rows, then review every group containing Layer cracking, Stringing, or Warping for box/class consistency. Prioritize the rare groups before collecting new printer sessions and normal/hard-negative examples. Do not use the old public test partition for decisions.

##### Minority-group visual review package: added (not a data change)

[review_grouped_annotations.py](review_grouped_annotations.py) creates an ignored local HTML review package from any compatible `group_manifest.csv`. It draws normalized CV labels onto the materialized fold images, creates one page per group, and writes editable `review_groups.csv` and `review_images.csv` decision sheets. It never modifies raw images, labels, folds, checkpoints, or manifest files.

For the current candidate, use fold 1 and classes `1 3 4`; this selects 179 of 823 unique groups and 272 images, covering the 57 Layer-cracking, 59 Stringing, and 80 Warping groups. Build the package with:

```text
python review_grouped_annotations.py --manifest cv-data/roboflow-3d-print-fail-v1/group_manifest.csv --fold 1 --class-ids 1 3 4 --output-dir review-data/candidate-minority-fold1 --overwrite
```

Open `review-data/candidate-minority-fold1/index.html`, inspect every image within a selected group, and fill `review_groups.csv` with `keep`, `correct`, or `exclude` plus notes. The ignored `review-data/` output is intentionally local only; corrections must first be documented and then applied to a separately versioned curated dataset rather than the downloaded source archive.

**Review-policy clarification:** for diffuse defects—especially Spaghetti and Stringing—multiple bounding-box geometries can be semantically valid. A reviewer must **not** mark an annotation `correct` merely because they would draw a tighter, looser, or differently partitioned box. For the current audit, mark `keep` if the defect class is visually present and the existing box(es) provide a defensible coarse region for that defect. Mark `correct` only for a wrong/absent class, a box that targets an unrelated region or plainly misses the defect, or an instance-count violation after a written policy is agreed. Mark `exclude` when the image/group cannot support a consistent semantic decision. Record `diffuse_geometry` in the review notes where multiple boundary choices remain valid.

This is a task-definition limitation, not a reason to rewrite labels to one reviewer's arbitrary box preference. It partly limits the absolute interpretation of strict IoU metrics such as mAP50-95; however, it does not invalidate the relative 640/960/NMS or pretrained-reference comparisons because those comparisons use the same frozen labels and folds. Before creating a curated label version, define one documented policy per defect: whether separated but causally related filament clusters are one instance or multiple instances, what visible region a box must cover, and when a diffuse defect should instead be labeled at image/event level. If the operational goal is simply to raise a defect alarm, a future multi-label image-level classification or segmentation study may be more appropriate than requiring pixel-tight object boxes for Spaghetti/Stringing.

##### Multi-label image-classification scope assessment: recommended after label-policy work (not started)

**Recommendation:** if the intended project question is “which failure types are visible in this image?” rather than “where is each individual failure instance?”, make **multi-label image-level classification** the primary final task and retain the current detector as a secondary localization/diagnostic study. This is a scope change, not a quick metric substitution. It is particularly appropriate for diffuse Spaghetti/Stringing because it avoids forcing a single arbitrary bounding-box boundary. It must be **multi-label**, not multi-class/softmax: one image can contain multiple failure types.

The existing fold-1 labels confirm multi-label behavior: 95 of 2,281 training images and 48 of 1,140 validation images contain more than one class. Image-level positive counts in fold 1 are `2,116 / 66 / 83 / 52 / 59` for Spaghetti / Layer cracking / Over extrusion / Stringing / Warping in training, and `1,057 / 33 / 42 / 27 / 29` in validation. However, all 3,421 images in this grouped development pool have at least one existing box label. Thus it contains no verified all-negative/normal images, and a missing box class is a reliable image-level negative only after the label audit confirms that annotation is exhaustive for that class.

Do **not** train a final classifier from the current box-derived labels yet. Before implementation:

1. finalize the semantic policy for every defect, including diffuse-instance rules and an explicit “normal/no listed defect” definition;
2. complete the group review, adding `image_label_missing`, `uncertain`, or `diffuse_geometry` notes where appropriate;
3. collect genuinely independent normal/hard-negative printer-session groups and additional minority groups; never create fake normal images by deleting labels;
4. create image-level multi-hot targets from the curated labels while retaining the existing source-/near-duplicate-group-disjoint folds; and
5. evaluate with per-class and macro average precision plus per-class recall/precision. Do not use ordinary accuracy, and select any per-class operating thresholds only within the cross-validation protocol.

The initial classifier should use five independent sigmoid outputs with BCE/focal-style loss, not one softmax output. After the curated image-level protocol exists, compare a custom YOLO26-backbone classifier with a documented pretrained classification reference on the same folds. Detection can still be used for compact/localizable defects or as an optional visual explanation, but it should not be the sole final metric for diffuse failure alerts.

### Candidate review: Roboflow `3d print fail` Dataset v1

Candidate URL: <https://universe.roboflow.com/mikes-workspace-oebho/3d-print-fail-7ipuj/dataset/1>

**Acquisition status (2026-07-14):** the authenticated archive `3d print fail.v1i.yolo26.zip` was downloaded and extracted to `candidate-data/roboflow-3d-print-fail-v1/`. The source archive and extracted candidate directory are ignored by Git. The extracted layout is valid: `data.yaml`, `train/`, `valid/`, and `test/` exist with 2,951 / 301 / 169 image-label pairs respectively.

Publicly reported properties at review time:

| Property | Reported value | Assessment |
| --- | --- | --- |
| Task | Object detection | Compatible with the current detection pipeline. |
| License | CC BY 4.0 | Suitable for academic use if attribution is retained. |
| Images | 3,421 | Exact-hash audit found 3,421 unique byte-level images, but near-duplicate screening still found cross-split leakage risk. |
| Split | 2,951 train / 301 validation / 169 test | The public split must not be trusted until duplicate/group auditing is complete. |
| Export preprocessing | Auto-orient and stretch resize to 640 × 640 | Compatible, but native image resolution is unavailable in the export. |
| Export augmentation | None reported | Positive: any new augmentation can be applied only inside training folds. |
| Classes | 5: a long-named spaghetti class, Layer-Cracking, Over Extrusion, String, Warping | Candidate study scope can expand to all five classes if the labels pass audit. |
| Public description/provenance | No dataset description published | Public page does not provide enough source/session provenance for group splitting. |

#### Downloaded-export audit

| Audit | Result | Consequence |
| --- | --- | --- |
| Image/label pairing | 2,951 / 301 / 169 matching image-label pairs | Archive structure is usable. |
| Raw label format | 7,516 / 844 / 469 five-field rows plus 41 / 9 / 10 polygon rows in train / valid / test | The strict local loader cannot consume the source directly; normalize polygons into a separate processed candidate dataset before any training. |
| Exact byte-level duplicates | None within or across candidate splits | Better than the current export at the exact-hash level. |
| Source-stem overlap | 96 train–valid, 56 train–test, and 33 valid–test shared stems | Split independence remains uncertain. |
| Perceptual near-duplicate screening (pHash Hamming distance ≤ 5) | 74 train–valid, 27 train–test, and 11 valid–test pairs among matching source stems | Treat matching source stems and near-duplicate clusters as indivisible groups; do not trust the published split. |
| Cross-dataset exact overlap | 8 exact image hashes shared with the current dataset | Do not merge the datasets without a source-group audit. |
| Five-class visual spot check | Representative samples match spaghetti, layer cracking, over extrusion, stringing, and warping | Mapping is plausible but requires a larger manual review before formal adoption. |
| Training coverage | class 0: 6,791 boxes / 2,778 images; class 1: 208 / 69; class 2: 320 / 88; class 3: 158 / 59; class 4 warping: 80 / 58 | The candidate is much weaker than the current source for unique warping coverage and remains severely imbalanced. |

**Decision:** the archive download succeeded, but the candidate does **not** currently pass the replacement-dataset acceptance gate. It is cleaner than the current export for exact hashes, but it has likely near-duplicate split leakage and only 80 training warping boxes versus 380 in the current export. Do not cite its public model metrics, use its published split, or replace the current study with it yet.

**Five-class architecture compatibility test (2026-07-15):** the candidate source was normalized without modifying the downloaded archive. A baseline-only processed copy was created at `processed-candidate-data/roboflow-3d-print-fail-v1/baseline/` using the canonical names `spaghetti`, `layer_cracking`, `over_extrusion`, `stringing`, and `warping`. It contains strict five-field labels after converting 41 train, 9 validation, and 10 test polygon rows to enclosing boxes. Both raw candidate data and processed candidate outputs are ignored by Git.

| Step | Result |
| --- | --- |
| Five-class forward/loss/backward test | Passed on CUDA using a batch containing every class ID `0` through `4`. `one_to_many` and `one_to_one` output shapes were both `(5, 9, 8400)`, where $9 = 4 + 5$ channels. |
| Optimizer-step test | Passed with finite total loss `22.1807`; gradients were clipped and one AdamW step completed. |
| Full training-loop smoke test | Passed: one complete epoch over 2,951 training and 301 validation images using `--num-classes 5`, `seed=42`, and neutral class weighting. Train loss `22.6053`; validation loss `21.3396`. |
| Five-class evaluator test | Passed: evaluator read all five canonical class names from `data.yaml`, decoded predictions, printed five per-class metric rows, and produced a six-by-six background-aware confusion matrix. |

Executed command:

```text
python train_yolo26.py --data-root processed-candidate-data/roboflow-3d-print-fail-v1/baseline --device cuda --epochs 1 --batch-size 8 --imgsz 640 --workers 0 --lr 5e-5 --cls-gain 0.5 --seed 42 --num-classes 5 --save-dir runs/yolo26/candidate_fiveclass_smoke_e1
```

The one-epoch validation evaluation produced mAP50 `0.0008`, mAP50-95 `0.0001`, precision `0.0000`, and recall `0.0000` at threshold `0.25`. These values are expected after one epoch from random initialization and are **not** a candidate-dataset benchmark or a conclusion about five-class learnability.

**Conclusion:** the local YOLO26-style architecture, loss, checkpointing, and evaluator support five classes and normalized polygon labels. Do not treat any result from the candidate's published split as a reliable benchmark until its near-duplicate groups, source grouping, label review, and severe five-class imbalance have been addressed.

#### Published-split five-class direct baseline: completed

On 2026-07-15, the user requested a direct full training run before cross-validation. One 50-epoch five-class baseline was run on the normalized candidate's published train/validation split with the same base hyperparameters used for the prior three-class seed-42 baselines.

- **Purpose:** compare end-to-end architecture behavior on the old and new exports.
- **Not valid for:** cross-dataset mAP comparison, model selection, final generalization claims, or test-split reporting.
- **Controls:** `seed=42`, `lr=5e-5`, batch size `8`, image size `640`, neutral class weighting, no image-level balanced sampling, and no offline augmentation.
- **Evaluation:** evaluate the candidate validation split only at thresholds `0.25` and `0.10`; do not evaluate its published test split.

```text
python train_yolo26.py --data-root processed-candidate-data/roboflow-3d-print-fail-v1/baseline --device cuda --epochs 50 --batch-size 8 --imgsz 640 --workers 0 --lr 5e-5 --cls-gain 0.5 --seed 42 --num-classes 5 --save-dir runs/yolo26/candidate_fiveclass_seed42_e50
```

##### Result

The selected checkpoint was epoch 25, with validation loss `12.5835` and training loss `9.3741`. Training loss continued to fall after this point while validation loss rose, so extending this run would increase overfitting rather than solve minority detection.

| Candidate training class | Boxes | Share of train boxes | Majority-to-class ratio |
| --- | ---: | ---: | ---: |
| Spaghetti | 6,791 | 89.86% | 1.0× |
| Layer cracking | 208 | 2.75% | 32.6× |
| Over extrusion | 320 | 4.23% | 21.2× |
| Stringing | 158 | 2.09% | 43.0× |
| Warping | 80 | 1.06% | 84.9× |

| Validation result | Threshold 0.25 | Threshold 0.10 |
| --- | ---: | ---: |
| mAP50 | 0.0370 | 0.0370 |
| mAP50-95 | 0.0092 | 0.0092 |
| Precision | 0.4000 | 0.1667 |
| Recall | 0.0281 | 0.0785 |
| Predicted classes with true positives | Spaghetti only | Spaghetti only |

Per-class AP50 was spaghetti `0.0609`, layer cracking `0.0009`, over extrusion `0.0120`, stringing `0.0154`, and warping `0.0957`. No non-spaghetti class produced a true positive at either reporting threshold. The nonzero minority AP values indicate only low-confidence ranking signal below `0.10`, not usable detections.

**Decision:** this direct neutral-loss run is a valid architecture/export compatibility diagnostic, but it confirms that the candidate is not a better practical training corpus in its current form. Do not repeat neutral-loss, sampler, threshold, or longer-epoch sweeps. The focal-loss and focal-plus-weight results below complete the allowed custom-loss diagnostics. The primary issue remains the candidate's five-class imbalance, especially 80 warping boxes, compounded by likely near-duplicate split leakage.

#### Custom focal-loss experiment: completed

The website’s reported model is tagged `yolov11n`, while the local baseline above used the project's own randomly initialized YOLO26-style detector. The website does not publish enough training detail to reproduce its weights, pretraining status, augmentation, optimizer schedule, or split protocol. Its `67.5%` mAP@50 must therefore not be treated as a reproducible target from an unknown recipe.

This run remained entirely custom: no `yolo11n.pt`, no hosted weights, and no transfer learning. It changed **one variable** relative to the neutral direct baseline: the custom classification loss used focal modulation with `focal_gamma=2`, reducing the loss contribution from easy spaghetti/background anchors so uncertain minority errors matter relatively more.

```text
python train_yolo26.py --data-root processed-candidate-data/roboflow-3d-print-fail-v1/baseline --device cuda --epochs 50 --batch-size 8 --imgsz 640 --workers 0 --lr 5e-5 --cls-gain 0.5 --focal-gamma 2.0 --seed 42 --num-classes 5 --save-dir runs/yolo26/candidate_fiveclass_focal_g2_seed42_e50
```

##### Focal-only result

The selected checkpoint was epoch 35 with validation loss `6.3801` and training loss `4.7372`. This improves on the neutral-loss checkpoint selected at epoch 25, but continuing beyond epoch 35 again increased validation loss while training loss fell.

| Metric | Neutral BCE baseline | Custom focal loss, gamma 2 | Change |
| --- | ---: | ---: | ---: |
| mAP50 | 0.0370 | 0.0776 | +0.0406 |
| mAP50-95 | 0.0092 | 0.0287 | +0.0195 |
| Precision at 0.25 | 0.4000 | 0.1176 | -0.2824 |
| Recall at 0.25 | 0.0281 | 0.0657 | +0.0376 |
| Recall at 0.10 | 0.0785 | 0.3962 | +0.3177 |

At threshold `0.25`, focal loss produced genuine true positives for every class: spaghetti `38`, layer cracking `1`, over extrusion `14`, stringing `1`, and warping `2`.

| Class | Neutral BCE AP50 | Focal AP50 |
| --- | ---: | ---: |
| Spaghetti | 0.0609 | 0.0351 |
| Layer cracking | 0.0009 | 0.0249 |
| Over extrusion | 0.0120 | 0.0783 |
| Stringing | 0.0154 | 0.0485 |
| Warping | 0.0957 | 0.2012 |

This proves the custom focal loss is working in the intended direction: it no longer collapses to spaghetti only. The trade-off is severe false positives: at threshold `0.10`, the confusion matrix contains 17,567 unmatched background predictions, so `0.10` is diagnostic only and not a usable operating threshold.

The result remains an architecture/data diagnostic, not a leakage-free benchmark. Warping has only 17 validation boxes, and the candidate split has likely near-duplicate leakage.

##### Focal plus modest positive weighting: completed final loss diagnostic

The permitted follow-up retained `focal_gamma=2` and added `--class-positive-weight-power 0.25`. Its normalized positive BCE weights were `[0.876, 2.095, 1.881, 2.244, 2.660]` for spaghetti, layer cracking, over extrusion, stringing, and warping respectively.

```text
python train_yolo26.py --data-root processed-candidate-data/roboflow-3d-print-fail-v1/baseline --device cuda --epochs 50 --batch-size 8 --imgsz 640 --workers 0 --lr 5e-5 --cls-gain 0.5 --focal-gamma 2.0 --class-positive-weight-power 0.25 --seed 42 --num-classes 5 --save-dir runs/yolo26/candidate_fiveclass_focal_g2_posweight_p025_seed42_e50
```

The selected checkpoint was epoch 38 with validation loss `6.3384` and training loss `4.5937`. Later training did not improve validation loss.

| Metric | Neutral BCE | Focal only | Focal + positive weighting | Best custom result |
| --- | ---: | ---: | ---: | --- |
| mAP50 | 0.0370 | 0.0776 | **0.0871** | Focal + weighting |
| mAP50-95 | 0.0092 | 0.0287 | **0.0303** | Focal + weighting |
| Precision at 0.25 | **0.4000** | 0.1176 | 0.1282 | Neutral baseline is more conservative, but misses minorities. |
| Recall at 0.25 | 0.0281 | 0.0657 | **0.0774** | Focal + weighting |
| Recall at 0.10 | 0.0785 | 0.3962 | **0.4256** | Diagnostic only; precision is unusably low. |

At threshold `0.25`, focal-plus-weighting produced the following per-class results:

| Class | AP50 | Precision | Recall | Confusion-matrix diagonal count |
| --- | ---: | ---: | ---: | ---: |
| Spaghetti | 0.0367 | 0.1520 | 0.0381 | 26 |
| Layer cracking | 0.0784 | 0.0977 | 0.2708 | 11 |
| Over extrusion | 0.1096 | 0.1224 | 0.2169 | 17 |
| Stringing | 0.0716 | 0.1429 | 0.1364 | 3 |
| Warping | 0.1391 | 0.1395 | 0.3529 | 4 |

Relative to focal-only training, modest weighting improved AP50 for layer cracking, over extrusion, and stringing, and substantially improved warping recall from `0.1176` to `0.3529`. Warping AP50 fell from `0.2012` to `0.1391`, so the weighting trade-off is greater operating recall rather than better confidence ranking for warping.

The class-specific precision/recall matcher requires a matching class label. The confusion matrix instead matches boxes globally by IoU and records off-diagonal class confusions, so its diagonal count can differ from the class-specific true-positive total when an incorrect-class prediction claims the same ground-truth box first.

At threshold `0.10`, recall rose to `0.4256`, but precision fell to `0.0180` with 19,745 unmatched background predictions. Therefore `0.10` is diagnostic only and not a usable operating threshold.

**Final published-split decision:** focal loss plus modest positive weighting is the best custom result on this exported candidate validation split. Do not tune another focal gamma, class-weight power, sampler, augmentation, threshold, training duration, or candidate test split. The next meaningful performance gain requires a cleaned group-disjoint development protocol and more real minority examples, especially warping.

##### Hosted YOLOv11n result: comparison limitation

The Roboflow page for this candidate reports a hosted model tagged `yolov11n` with mAP@50 `67.5%`, precision `70.6%`, and recall `67.0%`. This score is not a fair architecture-versus-architecture comparison with the custom local model because:

1. The local detector is a randomly initialized, educational YOLO26-style implementation with a custom dual-branch loss and decoder; it is not an official pretrained YOLO26 checkpoint.
2. The public page does not disclose the hosted model's initialization, training epochs, optimizer and schedule, augmentations, confidence/NMS settings, data revision, or exact evaluation protocol. It is unknown whether the hosted model used pretrained weights or additional Roboflow training defaults.
3. The candidate's published splits have demonstrated source-stem and perceptual near-duplicate overlap. If the hosted model used those splits, its reported metrics may be optimistic due to related content appearing across train and evaluation partitions.
4. “Newer” does not mean a local randomly initialized model must outperform a mature smaller model. Performance depends on the exact implementation, training recipe, initialization, data, labels, and evaluation protocol—not only the version name.

The current focal-plus-weight result demonstrates that the custom architecture can learn all five classes when its loss is made imbalance-aware. It does not establish that the custom architecture is inferior to YOLOv11n; a valid comparison would require both models to use the same cleaned, group-disjoint folds and a documented common training protocol.

##### Why the hosted Roboflow score is not directly comparable

The public candidate page reports a model tagged `yolov11n` with mAP@50 `67.5%`, precision `70.6%`, and recall `67.0%`. The page does **not** publish the trained weights, initialization/pretraining status, epoch count, optimizer schedule, online augmentation policy, loss configuration, NMS settings, or exact metric protocol. It also provides no model description beyond the model tag.

The local result uses a different system: a randomly initialized, local YOLO26-style educational implementation with a custom dual-branch head, custom assignment/loss path, and no external or hosted weights. Therefore the two reported scores are not an apples-to-apples architecture comparison.

The candidate export also has measured near-duplicate risk across its published split: 74 perceptually near train–validation pairs and 27 train–test pairs among matching source stems. These overlaps can make a published validation metric materially easier than a group-disjoint evaluation. They do not prove the hosted score is wrong, but they prevent treating `67.5%` as independent real-world performance or a reproducible target from the public information alone.

The custom focal-plus-weight result is still meaningful: it shows that a custom loss improvement can move the randomly initialized local detector from spaghetti-only detections to true positives for all five classes. To make a credible claim of exceeding the hosted model, both systems would need evaluation on the same cleaned, source-group-disjoint split with the same metric implementation.

**Five-class scope decision:** if the candidate passes the acceptance audit, detect all five classes rather than discarding `Layer-Cracking` and `Over Extrusion`. Preserve the source class order in the downloaded `data.yaml`, then create documented canonical display names such as `spaghetti`, `layer_cracking`, `over_extrusion`, `stringing`, and `warping` only after visual verification of each source label. The current local model supports an arbitrary class count: training must use `--num-classes 5`, while evaluation obtains `nc` and class names from the candidate dataset's `data.yaml`.

This is a **new five-class study**, not a direct continuation of the prior three-class experiment. Do not compare the five-class mAP directly with the current three-class mAP, because mAP averages a different set of classes. Retain the current work as the three-class proof-of-concept and data-quality audit.

Before adoption, download the unaugmented YOLO export to a separate candidate directory and complete this acceptance audit:

1. Complete a broader stratified visual review of every class and of source-stem/near-duplicate groups. The five-class mapping currently has only a preliminary spot check.
2. Preserve all five audited classes in any candidate study. Do not silently drop any annotated defect class or treat it as background.
3. Normalize the 60 candidate polygon rows into strict five-field boxes in a **separate** processed candidate directory; never overwrite the downloaded source.
4. Build source-stem and perceptual-near-duplicate groups before any new fold allocation. The exported 86/9/5 split is not acceptable for a final evaluation.
5. Count unique image groups, boxes, images per class, and object-size distributions after grouping and annotation review.
6. Manually adjudicate any group with incompatible annotations, then create group-disjoint multi-label folds.
7. Accept it as a replacement only if the cleaned groups provide enough unique examples across all five classes—especially materially more warping scenes than the current source—or supplement it with new real warping data.
8. Only after acceptance, build fresh five-class baselines using the cleaned candidate development set, then compare RGB, CLAHE, and CLAHE+Canny within the same group-disjoint folds.

#### Candidate group-disjoint three-fold build: completed

The initial `prepare_grouped_cv.py` run exposed an output-directory creation bug: the builder attempted to write `fold_1/data.yaml` before creating `fold_1/`, and would subsequently have lacked the fold label directory. Both directory creation steps were corrected and covered by an output-directory regression test.

The following completed command rebuilt the folds with `--overwrite`:

```text
python prepare_grouped_cv.py --input-root candidate-data/roboflow-3d-print-fail-v1 --output-root cv-data/roboflow-3d-print-fail-v1 --splits train valid test --folds 3 --seed 42 --attempts 50 --phash-distance 5 --num-classes 5 --class-names spaghetti layer_cracking over_extrusion stringing warping --overwrite
```

| Build property | Result |
| --- | --- |
| Source image records included | 3,421 across the public train, validation, and test folders; the public test partition is now part of the development pool, not a final test set. |
| Normalized annotations | 8,829 original boxes plus 60 polygon rows converted to strict five-field boxes. |
| Grouping | 823 groups created from 2,338 source-stem unions and 260 perceptual-hash unions; exact-hash unions were zero. |
| Materialization | 10,263 hardlinks, avoiding three physical copies of the image data. |
| Fold 1 | 2,281 train / 1,140 validation images; validation boxes `[2602, 95, 153, 73, 39]`. |
| Fold 2 | 2,279 train / 1,142 validation images; validation boxes `[2600, 95, 155, 73, 39]`. |
| Fold 3 | 2,282 train / 1,139 validation images; validation boxes `[2602, 95, 156, 73, 39]`. |

Post-build integrity checks passed: every generated label has exactly five valid fields, every fold has matching image/label counts, each fold's groups appear in only one partition, and all five classes are represented in every validation fold.

##### Next CV training protocol

Use the fixed best custom-loss configuration in each fold: `focal_gamma=2`, `class_positive_weight_power=0.25`, seed `42`, no sampler, no offline augmentation, and no external/pretrained weights. Train and evaluate folds sequentially; do not choose hyperparameters from individual folds.

Fold 1 was run with:

```text
python train_yolo26.py --data-root cv-data/roboflow-3d-print-fail-v1/fold_1 --device cuda --epochs 50 --batch-size 8 --imgsz 640 --workers 0 --lr 5e-5 --cls-gain 0.5 --focal-gamma 2.0 --class-positive-weight-power 0.25 --seed 42 --num-classes 5 --save-dir runs/yolo26/candidate_cv3_fold1_focal_g2_posweight_p025_seed42_e50
```

Fold 1 selected epoch 42 by validation loss (`6.7220`; train loss `4.5607`).

| Fold 1 metric | Threshold 0.25 | Threshold 0.10 |
| --- | ---: | ---: |
| mAP50 | 0.0661 | 0.0661 |
| mAP50-95 | 0.0176 | 0.0176 |
| Precision | 0.0934 | 0.0162 |
| Recall | 0.0479 | 0.3565 |

At threshold `0.25`, all five classes received genuine class-specific detections:

| Class | AP50 | Precision | Recall |
| --- | ---: | ---: | ---: |
| Spaghetti | 0.0173 | 0.0766 | 0.0277 |
| Layer cracking | 0.0581 | 0.1192 | 0.1895 |
| Over extrusion | 0.1022 | 0.1449 | 0.1961 |
| Stringing | 0.0994 | 0.2045 | 0.1233 |
| Warping | 0.0534 | 0.0726 | 0.3333 |

The group-disjoint fold is more difficult than the published-split diagnostic (`mAP50=0.0871`), as expected after grouping related content. It is the more meaningful estimate. At threshold `0.10`, precision is unusably low due to 63,981 unmatched background predictions; use that threshold only to inspect recall behavior.

For reference, fold 1 was evaluated at the standard and diagnostic thresholds:

```text
python eval_yolo26.py --checkpoint runs/yolo26/candidate_cv3_fold1_focal_g2_posweight_p025_seed42_e50/best.pt --data-root cv-data/roboflow-3d-print-fail-v1/fold_1 --split valid --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.25
python eval_yolo26.py --checkpoint runs/yolo26/candidate_cv3_fold1_focal_g2_posweight_p025_seed42_e50/best.pt --data-root cv-data/roboflow-3d-print-fail-v1/fold_1 --split valid --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.10
```

Run folds 2 and 3 with the exact same fixed configuration. Fold 2:

```text
python train_yolo26.py --data-root cv-data/roboflow-3d-print-fail-v1/fold_2 --device cuda --epochs 50 --batch-size 8 --imgsz 640 --workers 0 --lr 5e-5 --cls-gain 0.5 --focal-gamma 2.0 --class-positive-weight-power 0.25 --seed 42 --num-classes 5 --save-dir runs/yolo26/candidate_cv3_fold2_focal_g2_posweight_p025_seed42_e50
python eval_yolo26.py --checkpoint runs/yolo26/candidate_cv3_fold2_focal_g2_posweight_p025_seed42_e50/best.pt --data-root cv-data/roboflow-3d-print-fail-v1/fold_2 --split valid --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.25
python eval_yolo26.py --checkpoint runs/yolo26/candidate_cv3_fold2_focal_g2_posweight_p025_seed42_e50/best.pt --data-root cv-data/roboflow-3d-print-fail-v1/fold_2 --split valid --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.10
```

Fold 3:

```text
python train_yolo26.py --data-root cv-data/roboflow-3d-print-fail-v1/fold_3 --device cuda --epochs 50 --batch-size 8 --imgsz 640 --workers 0 --lr 5e-5 --cls-gain 0.5 --focal-gamma 2.0 --class-positive-weight-power 0.25 --seed 42 --num-classes 5 --save-dir runs/yolo26/candidate_cv3_fold3_focal_g2_posweight_p025_seed42_e50
python eval_yolo26.py --checkpoint runs/yolo26/candidate_cv3_fold3_focal_g2_posweight_p025_seed42_e50/best.pt --data-root cv-data/roboflow-3d-print-fail-v1/fold_3 --split valid --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.25
python eval_yolo26.py --checkpoint runs/yolo26/candidate_cv3_fold3_focal_g2_posweight_p025_seed42_e50/best.pt --data-root cv-data/roboflow-3d-print-fail-v1/fold_3 --split valid --device cuda --batch-size 8 --imgsz 640 --workers 0 --conf-thresh 0.10
```

Folds 2 and 3 completed with the same unchanged configuration. Selected checkpoints were fold 2 epoch 40 (validation loss `6.6789`; training loss `4.8330`) and fold 3 epoch 42 (validation loss `6.6564`; training loss `4.6449`).

##### Three-fold group-disjoint CV result

All summary values below are the arithmetic mean and sample standard deviation across the three fixed-configuration folds. They are the primary custom-model result for this candidate export.

| Metric | Fold 1 | Fold 2 | Fold 3 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Selected epoch | 42 | 40 | 42 | 41.3 ± 1.2 |
| Validation loss | 6.7220 | 6.6789 | 6.6564 | 6.6858 ± 0.0331 |
| mAP50 | 0.0661 | 0.0542 | 0.0643 | **0.0615 ± 0.0064** |
| mAP50-95 | 0.0176 | 0.0163 | 0.0200 | **0.0180 ± 0.0019** |
| Precision at 0.25 | 0.0934 | 0.0898 | 0.0987 | **0.0940 ± 0.0045** |
| Recall at 0.25 | 0.0479 | 0.0608 | 0.0526 | **0.0538 ± 0.0065** |
| Precision at 0.10 | 0.0162 | 0.0200 | 0.0169 | 0.0177 ± 0.0020 |
| Recall at 0.10 | 0.3565 | 0.3504 | 0.3494 | 0.3521 ± 0.0038 |

Per-class metrics at the standard threshold `0.25`:

| Class | AP50 mean ± SD | AP50-95 mean ± SD | Precision mean ± SD | Recall mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| Spaghetti | 0.0195 ± 0.0019 | 0.0047 ± 0.0005 | 0.0889 ± 0.0166 | 0.0314 ± 0.0068 |
| Layer cracking | 0.0372 ± 0.0211 | 0.0097 ± 0.0045 | 0.0852 ± 0.0353 | 0.1368 ± 0.0737 |
| Over extrusion | **0.1323 ± 0.0274** | **0.0421 ± 0.0085** | 0.1320 ± 0.0112 | 0.2862 ± 0.0836 |
| Stringing | 0.0612 ± 0.0421 | 0.0186 ± 0.0113 | **0.1475 ± 0.0500** | 0.0913 ± 0.0440 |
| Warping | 0.0572 ± 0.0035 | 0.0147 ± 0.0011 | 0.0735 ± 0.0131 | **0.3504 ± 0.0148** |

All five classes had genuine class-specific detections in every fold at threshold `0.25`. Over extrusion had the strongest average AP; warping had the most stable minority recall but low precision. Threshold `0.10` yields high recall but unusably low precision because of many background false positives, so it is diagnostic only and must not be presented as a deployment operating point.

The CV result is more credible than the earlier published-split diagnostic because source-stem and perceptual near-duplicate groups are held within one fold. It is not fully unbiased: the focal-plus-weight configuration was selected using earlier public-split experiments from the same overall dataset, and source provenance/annotations have not yet been manually adjudicated. Therefore report this as a **group-disjoint CV stability estimate**, not a final external generalization claim.

**CV decision:** no more focal-gamma, class-weight, sampler, offline-augmentation, threshold, training-duration, or candidate-test sweeps. The next meaningful improvement requires data work: manually review grouped annotations, add real minority scenes—especially warping—and reserve a new external group-disjoint test set after all model choices are frozen.

##### Why three folds rather than five or more

The grouped candidate development pool has 823 groups. The minority group coverage, not the total image count, set the fold count:

| Class | Groups containing class | Total boxes | Approx. validation support with 3 folds | Approx. validation support with 5 folds |
| --- | ---: | ---: | ---: | ---: |
| Spaghetti | 645 | 7,804 | 2,601 boxes / 215 groups | 1,561 boxes / 129 groups |
| Layer cracking | 57 | 285 | 95 boxes / 19 groups | 57 boxes / 11 groups |
| Over extrusion | 70 | 464 | 155 boxes / 23 groups | 93 boxes / 14 groups |
| Stringing | 59 | 219 | 73 boxes / 20 groups | 44 boxes / 12 groups |
| Warping | 80 | 117 | 39 boxes / 27 groups | 23 boxes / 16 groups |

Five folds would provide each model with about 80% rather than 67% of the data for training, but it would make every minority validation estimate much less stable. For warping, one correct or incorrect detection would change recall by roughly $1/23 \approx 4.3\%$ in five-fold validation compared with $1/39 \approx 2.6\%$ in three-fold validation. Five 50-epoch models also require roughly twice the total image-epochs of three folds.

Three folds are therefore the better compromise for the current sparse grouped data. Move to five folds only after data collection produces roughly 150--200 independent source groups for each minority class, so each validation fold has enough independent examples to estimate class performance more stably.

##### Data-first improvement plan

1. **Freeze the current custom model.** Use focal gamma `2` plus positive class-weight power `0.25` as the baseline. Do not change its loss, sampler, threshold, training duration, or augmentation on the current candidate export.
2. **Review annotations group by group.** Use `cv-data/roboflow-3d-print-fail-v1/group_manifest.csv`, filter to `fold=1` to avoid repeated manifest rows, and inspect every group containing layer cracking, over extrusion, stringing, or warping. Correct inconsistent boxes/classes or exclude ambiguous groups. The current `source_image` and `group_id` fields identify the images to review.
3. **Collect genuinely new data, not transformed copies.** Prioritize real printer sessions and viewpoints for warping, stringing, and layer cracking. Retain a source/session identifier with every image. Add normal-print and hard-negative images with empty label files so the model can learn what should not be detected; this directly targets low-precision background false positives.
4. **Use a fixed annotation policy.** Define one box or mask convention for each defect, annotate every visible instance, and have a second reviewer check a stratified sample. Polygon-to-box conversion fixes format but cannot resolve inconsistent semantic annotations.
5. **Create splits before augmentation.** Build source/session- and near-duplicate-group-disjoint train, validation, and test partitions first. Apply any augmentation only to each training partition after the split; never augment first and then split.
6. **Reserve a new external test set.** After the model configuration is frozen and the new data are grouped, reserve about 15--20% of independent source groups as an external test set. Do not use it while selecting the configuration or confidence threshold.
7. **Re-run the fixed custom baseline.** First reproduce the same focal-plus-weight configuration on the cleaned three-fold development data. Only then consider a single training-only, box-aware online-augmentation ablation using conservative domain-valid transforms. Do not use `tf.keras.Sequential`; implement it in the PyTorch/OpenCV path so image geometry and boxes remain synchronized.

The practical initial collection target is not perfectly equal class counts. It is at least a few hundred independently sourced images/groups per minority defect, with particular emphasis on increasing warping from the current 80 source groups to at least 150--200 independent groups. More diverse real examples and clean negative images are more valuable than repeating existing files or running more hyperparameter sweeps.

### Required cleanup and cross-validation protocol

Combining the current training and validation splits is appropriate **only after** duplicate handling. It provides 4,220 image records, 3,589 unique exact-image hashes, and 997 warping boxes before annotation adjudication.

1. **Freeze the current exported test split.** It is already contaminated by overlap and must not be used again for evaluation, tuning, or model selection.
2. **Create a duplicate report from exact image hashes.** Keep one copy automatically only when every duplicate has exactly the same annotation. For the 366 disagreement groups, manually adjudicate a canonical annotation or exclude the ambiguous group; do not automatically merge all boxes.
3. **Create source groups before folds.** Use exact-image hash as the minimum group. If images originate from videos, printer sessions, or near-duplicate sequences, group all related frames together as well. The Roboflow filenames alone are not reliable group identifiers.
4. **Use group-disjoint, multi-label stratified K-fold CV.** A custom fold allocator should balance images and box counts for all three classes plus warping size bins. Do not use naive image-level random `KFold`, because it can leak duplicate content across folds.
5. **Start with 3 folds.** After cleanup, 3-fold CV gives each validation fold roughly one-third of the development data and avoids the cost/noise of five full 50-epoch trainings. Use 5 folds only if group counts and compute budget support stable class/size balance.
6. **Augment only the training partition inside each fold.** Never generate an augmented pool once and then split it, because transformed copies of an image can leak into a fold's validation partition.
7. **Report mean and standard deviation across folds.** Select future changes from cross-validation only. A final model may then train on all cleaned development data, but it needs a new externally sourced or newly reserved group-disjoint test set for a valid final generalization claim.

### Architecture-control experiment after cleanup

The installed environment contains the Ultralytics Python API, but pretrained `yolo11n.pt` weights are not cached and would need to be obtained before use. Run a standard pretrained YOLO control under the same cleaned group-disjoint folds:

- If pretrained YOLO substantially outperforms the local model, the local architecture/training recipe is the main practical bottleneck.
- If pretrained YOLO is also weak, data scarcity, annotation consistency, split shift, and defect representation are the dominant bottlenecks.

Only after this control should future custom-model ablations test focal loss or hard-negative mining, higher resolution for small warping objects, and validation-mAP checkpoint selection. Consider segmentation later because enclosing boxes include substantial background for diffuse stringing/warping defects.

## 9. Current Constraints and Known Limitations

- `best.pt` is selected by validation loss, not validation mAP. Loss is useful for training selection but does not necessarily choose the highest-mAP checkpoint.
- The detector is a local educational YOLO26-style PyTorch implementation, not an official installed YOLO26 model.
- The original exported splits contain exact duplicate images and annotation disagreements. Existing split metrics are diagnostic results, not leakage-free generalization estimates.
- The clean preprocessing comparison is complete under the same exported split. CLAHE+Canny is rejected as a relative result, while CLAHE's small observed aggregate advantage must be rechecked under corrected group-disjoint folds.
- Faster R-CNN has a completed repaired group-disjoint scratch baseline. Its single ImageNet-initialized transfer treatment increased fixed-threshold recall but reduced aggregate AP and precision, so the scratch baseline remains the valid two-stage reference rather than a selected replacement for custom YOLO26 or pretrained YOLO11n.
- The current test split overlaps training/validation and must never be reused for model selection or tuning. A future final test set must be newly reserved and group-disjoint.

## 10. Files Changed During This Work

- [preprocess_dataset.py](preprocess_dataset.py): normalization, safe augmentation, reproducibility, duplicate prevention, and corrected object-count targeting.
- [train_yolo26.py](train_yolo26.py): strict label checking, reproducible seeds, sampling options, and positive-only class weighting.
- [eval_yolo26.py](eval_yolo26.py): checkpoint-setting and class-weight restoration.
- [detection_metrics.py](detection_metrics.py): corrected AP match/score alignment and expanded reporting.
- [make_overfit_subset.py](make_overfit_subset.py): new deliberate-overfit subset builder.
- [prepare_grouped_cv.py](prepare_grouped_cv.py): normalized source-group/perceptual-duplicate-aware candidate CV fold builder.
- [prepare_platform_spaghetti_cv.py](prepare_platform_spaghetti_cv.py): exact-deduplicated, provenance-proxy-grouped one-class Spaghetti control fold builder.
- [prepare_voc2007.py](prepare_voc2007.py): official VOC 2007 downloader/converter with strict labels, difficult-object exclusion, split validation, and conversion audit metadata.
- [yolo_dataset_config.py](yolo_dataset_config.py): shared `data.yaml` class-metadata parser for training, evaluation, and CV orchestration.
- [cv_utils.py](cv_utils.py): shared standard fold-layout validation, fold discovery, path templating, and metric-summary utilities.
- [run_yolo26_kfold_cv.py](run_yolo26_kfold_cv.py): generic sequential custom-YOLO26 K-fold trainer.
- [eval_yolo26_kfold_cv.py](eval_yolo26_kfold_cv.py): generic custom-YOLO26 K-fold evaluator and overall/per-class aggregate reporter.
- [run_ultralytics_kfold_cv.py](run_ultralytics_kfold_cv.py): generic sequential pretrained-Ultralytics K-fold reference trainer.
- [eval_ultralytics_kfold_cv.py](eval_ultralytics_kfold_cv.py): generic pretrained-Ultralytics K-fold evaluator using project metrics after NMS.
- [models/faster_rcnn.py](models/faster_rcnn.py): local two-stage Faster R-CNN with ResNet-style FPN, RPN, RoI Align, class-agnostic box regression, and configurable per-class NMS.
- [training_control.py](training_control.py): shared validation-loss plateau scheduler and formula-validated early-stopping policy for both custom trainers.
- [tests/test_faster_rcnn.py](tests/test_faster_rcnn.py): Faster R-CNN FPN/RPN target-assignment and model-behavior regression tests.
- [tests/test_training_control.py](tests/test_training_control.py): scheduler, early-stopping, and early-stopped K-fold completion regression tests.
- [train_faster_rcnn.py](train_faster_rcnn.py) and [eval_faster_rcnn.py](eval_faster_rcnn.py): strict-label, reproducible one-fold Faster R-CNN training/evaluation with checkpoint metadata and JSON metrics output.
- [run_faster_rcnn_kfold_cv.py](run_faster_rcnn_kfold_cv.py) and [eval_faster_rcnn_kfold_cv.py](eval_faster_rcnn_kfold_cv.py): generic sequential Faster R-CNN K-fold trainer/evaluator with aggregate reporting.
- [requirements.txt](requirements.txt): explicitly pins the installed Ultralytics version required by the custom loss imports and generic pretrained reference scripts.
- [README.md](README.md): links to this work log and distinguishes preliminary metrics from future group-disjoint evaluation.
- [PROJECT_WORK_LOG.md](PROJECT_WORK_LOG.md): this persistent engineering and experiment record.

## 11. Update Checklist

After each future action, append a dated entry or update the relevant table with:

1. exact command and key options;
2. dataset root and checkpoint directory;
3. any code/data change and validation performed;
4. selected epoch and validation loss;
5. mAP50, mAP50-95, precision, recall, and per-class metrics at the chosen threshold;
6. the decision: keep, reject, or treat as exploratory;
7. the next single controlled variable to change.
