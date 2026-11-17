# Post-Submission Practical Improvement Plan

## Purpose

Continue the detector work as a separate practical-performance study. The submitted strict-scratch results remain frozen and must not be replaced, relabeled, or compared as though they used the same protocol.

Current submitted grouped-CV development baselines at 960 pixels:

| System | mAP50 |
| --- | ---: |
| Custom YOLO26, selected raw scratch configuration | 0.1586 |
| Custom Faster R-CNN, selected raw scheduled scratch configuration | 0.3258 |
| Official pretrained Ultralytics YOLO26n reference (Phase 3, 960 px) | 0.4611 |
| Official pretrained Ultralytics YOLO11n reference (Phase 3, 960 px) | 0.4735 |

The `640` px official references (`0.4592` / `0.4799`) that predated Phase 3 are retained only for the completed resolution comparison; they are no longer the practical target.

## Requirements In Execution Order

Reaching a credible Ultralytics-level practical system requires all three items below. They are separate from, and do not change, the submitted strict-scratch study.

The dataset, labels, grouped folds, and public-test boundary are frozen for this study. The active work is therefore the Ultralytics-style training stack first, followed by detector pretraining; data improvement is explicitly deferred.

| Execution order | Requirement | Why It Is Needed | Plan Phases | Status |
| --- | --- | --- | --- | --- |
| 1. First code implementation | Mature training stack | Official Ultralytics systems combine correct image geometry, strong train-only augmentation, tuned optimization, losses, normalization, and postprocessing. | Phases 1-3 | **Phases 1-3 complete:** letterbox adopted, augmentation components scored, and the geometry-matched official-reference benchmark finished (YOLO11n `0.4735 ± 0.0326` at `960` px). Phase 4 is unblocked. |
| 2. After the transform foundation | Detector pretraining | Large-scale detection pretraining provides transferable visual features and localization priors that the frozen 3D-print dataset cannot supply from scratch. | Phase 4 | Begin only after letterbox and evaluation parity are validated; use the existing grouped folds as development data only. |
| 3. Deferred beyond this study | More independent, consistent data and a protected holdout | Sparse minority examples, ambiguous boxes, and repeatedly reused folds limit real-world generalization and final-claim credibility. | Deferred | Do not alter, clean, collect, or reserve data in this same-dataset performance study. |

This study may improve same-fold development performance, but it cannot turn the reused grouped folds into an independent generalization claim. A future data-improvement study remains necessary for that purpose and must use a separate protocol.

## Rules For The New Study

- Use a distinct `post_submission` run root and record every setting, checkpoint, dataset manifest, and result.
- Freeze the existing `cv-data/roboflow-3d-print-fail-v1/` dataset, labels, grouped folds, class taxonomy, preprocessing inputs, and public candidate test boundary. Do not clean labels, collect data, create a holdout, or regenerate folds in this study.
- Treat the existing grouped folds as development data only. Do not use the contaminated candidate public test split for tuning, model selection, or final claims.
- Clearly label pretrained weights, Ultralytics components, and external data. This is transfer learning and practical engineering, not strict scratch learning.
- Keep the original submitted checkpoints and metrics unchanged as historical baselines.

## Deferred Data And Holdout Governance

Data work is not active in the frozen-dataset practical study. The following utilities and local artifacts were created before the scope was frozen; retain them for a separate future data-governance study, but do not use them to alter the current data, labels, folds, or reported comparisons.

### Completed automated groundwork

- [x] Added [prepare_post_submission_data_inventory.py](prepare_post_submission_data_inventory.py), which reads one canonical grouped-CV fold, validates every manifest row against its materialized strict label, and creates ignored local governance artifacts without changing data.
- [x] Generated the canonical development inventory from fold 1: `3,421` image records in `823` source/perceptual groups. Existing group coverage is Spaghetti `645`, Layer Cracking `57`, Over Extrusion `70`, Stringing `59`, and Warping `80` groups.
- [x] Created `source_group_inventory.csv`, `development_image_registry.csv`, and `holdout_intake_template.csv` under ignored `review-data/post_submission_data_inventory/`.
- [x] Added [screen_post_submission_holdout.py](screen_post_submission_holdout.py), which blocks a future holdout candidate when it overlaps the development registry by exact SHA-256 hash, source stem, or perceptual hash at the grouping threshold. It also requires documented new-provider, physical-printer/job, capture-session, and reviewer-confirmed provenance.
- [x] Ran the empty-intake smoke screen. It correctly reported zero candidates; no new independent source group has been collected or approved yet.

### Deferred human data work

- [ ] Review the rarity-prioritized annotation queue in a separate data-governance study.
- [ ] Create a versioned corrected-data derivative only after review decisions are complete and reproducible.
- [ ] Collect genuinely new source groups with printer/job and capture-session provenance.
- [ ] Fill the holdout intake template, screen every candidate against the development registry, and freeze a final holdout under a separate protocol.

### Requirement 1 Commands

```bash
# Rebuild the canonical inventory and blank holdout-intake template.
python prepare_post_submission_data_inventory.py --manifest cv-data/roboflow-3d-print-fail-v1/group_manifest.csv --fold 1 --output-dir review-data/post_submission_data_inventory --overwrite

# Generate a visual review package for the rare and ambiguous classes.
python review_grouped_annotations.py --manifest cv-data/roboflow-3d-print-fail-v1/group_manifest.csv --fold 1 --class-ids 1 3 4 --output-dir review-data/post_submission_minority_review --overwrite

# Screen a populated future holdout intake against the immutable development registry.
python screen_post_submission_holdout.py --development-registry review-data/post_submission_data_inventory/development_image_registry.csv --holdout-intake path/to/populated_holdout_intake.csv --output-dir review-data/post_submission_holdout_screen --overwrite
```

## First Active Implementation: Shared Letterbox And Transform Foundation

**Completed 2026-08-24.** Both custom datasets historically stretched each image directly to a square. The new shared [image_geometry.py](image_geometry.py) module now provides a reversible `stretch|letterbox` contract that:

1. letterboxes images to a fixed canvas while preserving aspect ratio;
2. maps boxes and predictions between source, letterboxed, and model coordinates;
3. exposes deterministic train and validation behavior;
4. is used by both YOLO26 and Faster R-CNN training, evaluation, and demo code; and
5. has synthetic box-round-trip and image/label alignment tests.

Historical `stretch` remains the default. New post-submission checkpoints record `resize_mode`, while evaluators and custom demos restore that mode from checkpoint metadata unless explicitly overridden. K-fold runners include it in their frozen settings and output distinct metric filenames when an evaluator override is supplied.

**Validation completed:** synthetic stretch compatibility, label/detection round trips, both dataset target contracts, demo restoration, runner forwarding, K-fold dry runs, and one-epoch CUDA smokes passed for both architectures. The tiny YOLO26 smoke used `22` train / `11` validation images; the tiny Faster R-CNN smoke used `4` train / `2` validation images. Their zero AP values are geometry-smoke outcomes, not reportable performance results. Both evaluators restored `resize_mode=letterbox` from their new checkpoints without a CLI override.

### Grouped-CV Letterbox Baseline Results (2026-08-24)

Both architectures completed full three-fold group-disjoint cross-validation under `runs/yolo26/post_submission_letterbox` and `runs/faster_rcnn/post_submission_letterbox` at 960 px with `resize_mode=letterbox`, leaving all other selected hyperparameters, loss weights, seeds (`42`), and fold data unchanged.

#### Aggregate Performance: Historical Stretch vs Post-Submission Letterbox

| Detector | Image Geometry | Selected Epochs | mAP50 | mAP50-95 | Precision @ 0.25 | Recall @ 0.25 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Custom YOLO26 (scale `n`) | Historical Stretch | 48, 47, 49 | 0.1586 ± 0.0104 | 0.0512 ± 0.0043 | 0.1106 ± 0.0060 | 0.1306 ± 0.0067 | Historical submitted scratch baseline |
| Custom YOLO26 (scale `n`) | Post-submission Letterbox | 49, 45, 45 | **0.1985 ± 0.0063** | **0.0691 ± 0.0053** | **0.1137 ± 0.0073** | **0.1453 ± 0.0139** | **Adopted (+0.0399 mAP50 / +25.2% relative gain)** |
| Custom Faster R-CNN (scale `s`) | Historical Stretch | 32, 46, 35 | 0.3258 ± 0.0275 | **0.1239 ± 0.0111** | 0.2960 ± 0.0525 | 0.1146 ± 0.0069 | Historical submitted scratch baseline |
| Custom Faster R-CNN (scale `s`) | Post-submission Letterbox | 28, 42, 50 | **0.3280 ± 0.0282** | 0.1198 ± 0.0097 | **0.2985 ± 0.1086** | **0.1167 ± 0.0183** | **Adopted (Parity preserved / standardized geometry)** |

#### Per-Fold Validation Metrics

- **Custom YOLO26 (`post_submission_letterbox`):**
  - Fold 1 (epoch 49): mAP50 `0.2004`, mAP50-95 `0.0739`, Precision `0.1195`, Recall `0.1296`
  - Fold 2 (epoch 45): mAP50 `0.2036`, mAP50-95 `0.0699`, Precision `0.1161`, Recall `0.1560`
  - Fold 3 (epoch 45): mAP50 `0.1914`, mAP50-95 `0.0634`, Precision `0.1055`, Recall `0.1504`
  - Mean ± SD: mAP50 `0.1985 ± 0.0063`, mAP50-95 `0.0691 ± 0.0053`, Precision `0.1137 ± 0.0073`, Recall `0.1453 ± 0.0139`, Loss `6.5549 ± 0.0618`.
- **Custom Faster R-CNN (`post_submission_letterbox`):**
  - Fold 1 (epoch 28): mAP50 `0.3203`, mAP50-95 `0.1183`, Precision `0.1737`, Recall `0.1367`
  - Fold 2 (epoch 42): mAP50 `0.3592`, mAP50-95 `0.1302`, Precision `0.3498`, Recall `0.1124`
  - Fold 3 (epoch 50): mAP50 `0.3043`, mAP50-95 `0.1110`, Precision `0.3719`, Recall `0.1008`
  - Mean ± SD: mAP50 `0.3280 ± 0.0282`, mAP50-95 `0.1198 ± 0.0097`, Precision `0.2985 ± 0.1086`, Recall `0.1167 ± 0.0183`.

#### Per-Class AP50 and Recall at Threshold 0.25

| Class | Ground Truth | YOLO26 Letterbox AP50 | YOLO26 Letterbox Recall | Faster R-CNN Letterbox AP50 | Faster R-CNN Letterbox Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spaghetti (id 0) | 7,804 | 0.0336 ± 0.0030 | 0.1075 ± 0.0118 | 0.0434 ± 0.0049 | 0.0689 ± 0.0168 |
| Layer cracking (id 1) | 285 | 0.1510 ± 0.0272 | 0.2526 ± 0.0482 | 0.2694 ± 0.0506 | 0.3018 ± 0.0955 |
| Over extrusion (id 2) | 464 | 0.3432 ± 0.0354 | 0.5813 ± 0.0853 | 0.5533 ± 0.0920 | 0.6141 ± 0.0281 |
| Stringing (id 3) | 219 | 0.0884 ± 0.0258 | 0.1918 ± 0.0362 | 0.2352 ± 0.0462 | 0.2740 ± 0.0898 |
| Warping (id 4) | 117 | 0.3763 ± 0.0367 | 0.5897 ± 0.0769 | 0.5384 ± 0.0796 | 0.5812 ± 0.0783 |

**Adoption Decision:** Letterboxing is formally adopted as the geometry standard for all practical track runs. It provides an unambiguous aggregate mAP gain (+25.2%) and consistent per-fold improvements for Custom YOLO26, while maintaining Faster R-CNN performance parity and matching official Ultralytics letterbox geometry.

## Phase 3: Ultralytics Parity Benchmark

**Gate:** establish the practical-reference target on the frozen grouped folds under *matching fold inputs, image geometry, and project evaluation code*.

The official references that predated Phase 3 were fine-tuned at `640` px, while every post-submission custom run uses `960` px letterbox. The `0.4799` (YOLO11n) and `0.4592` (YOLO26n) mAP50 values therefore compared a different input geometry against the custom `0.2320` (YOLO26 `letterbox + affine`) and `0.4429` (Faster R-CNN `letterbox + hflip + affine`) results. Phase 3 closed exactly that gap: the same official fine-tuning workflow, same frozen folds, same evaluation code, now at the adopted custom geometry. Nothing about the custom models, folds, labels, or public-test boundary changed, and the benchmark is now complete.

### Protocol

| Item | Value |
| --- | --- |
| Data | `cv-data/roboflow-3d-print-fail-v1`, frozen; group-disjoint folds `1 2 3`, unchanged |
| Weights | official COCO-pretrained `yolo11n.pt` and `yolo26n.pt`; SHA-256 recorded in each `training_plan.json` |
| Geometry | `960` px native Ultralytics letterbox, matching the adopted custom standard |
| Epochs / patience | `50` maximum, patience `100` (no early stop) |
| Batch / workers / seed | `8` / `0` / `42`, `deterministic=True` |
| Optimizer | Ultralytics `auto` |
| Augmentation | standard Ultralytics train-only stack (mosaic, HSV, horizontal flip, scale/translate, erasing) |
| Checkpoint selection | Ultralytics fitness `0.1·mAP50 + 0.9·mAP50-95` on its own validation pass |
| Evaluation | `eval_ultralytics_kfold_cv.py` using project `detection_metrics.py` after class-aware NMS; `conf 0.25`, pre-NMS floor `0.001`, NMS IoU `0.70`, `max_det 300` |
| Outputs | `runs/ultralytics/post_submission_parity_yolo11n_imgsz960/`, `runs/ultralytics/post_submission_parity_yolo26n_imgsz960/` |
| Public test | not used; development folds only |

### Commands

One resumable entry point from the repository root (completed folds are skipped via `baseline_complete.json`, so an interrupted run can simply be restarted):

```bat
run_phase3_parity_benchmark.bat
```

Equivalent individual steps, if a single model needs to be re-run or a different device selector is required:

```bash
# YOLO11n reference: train, then evaluate
python run_ultralytics_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/ultralytics/post_submission_parity_yolo11n_imgsz960 --weights yolo11n.pt --folds 1 2 3 --epochs 50 --imgsz 960 --batch-size 8 --workers 0 --seed 42 --device 0 --optimizer auto --patience 100
python eval_ultralytics_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/ultralytics/post_submission_parity_yolo11n_imgsz960 --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device 0 --conf-thresh 0.25

# YOLO26n reference: train, then evaluate
python run_ultralytics_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/ultralytics/post_submission_parity_yolo26n_imgsz960 --weights yolo26n.pt --folds 1 2 3 --epochs 50 --imgsz 960 --batch-size 8 --workers 0 --seed 42 --device 0 --optimizer auto --patience 100
python eval_ultralytics_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/ultralytics/post_submission_parity_yolo26n_imgsz960 --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device 0 --conf-thresh 0.25
```

The frozen folds, both `.pt` weights, and `runs/` are Git-ignored, so the run device needs the same checkout plus `cv-data/roboflow-3d-print-fail-v1/` and `yolo11n.pt` / `yolo26n.pt` present. Expect roughly one hour per fold per model at `960` px with batch `8` on a single mid-range GPU; the whole benchmark is several hours, so run it unattended.

### Decision Rule

The higher official mAP50 becomes the Phase 3 target. Carry it, together with its per-class AP50, into Phase 4 as the number the pretrained custom architectures are measured against. Record per-class AP50 for all five classes rather than relying on aggregate mAP50 alone, because the two official systems already differ sharply in precision/recall balance on the minority classes.

### Results (completed 2026-11-05 on the remote GPU host)

| System | Geometry | imgsz | mAP50 | mAP50-95 | Precision @ 0.25 | Recall @ 0.25 | Per-fold mAP50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Ultralytics YOLO11n (fitness-selected) | letterbox | 960 | **0.4735 ± 0.0326** | 0.1924 ± 0.0180 | 0.3793 ± 0.0292 | 0.2393 ± 0.0190 | 0.4878 / 0.4362 / 0.4964 | completed |
| Ultralytics YOLO26n (fitness-selected) | letterbox | 960 | 0.4611 ± 0.0186 | 0.1923 ± 0.0067 | 0.3942 ± 0.0400 | 0.2172 ± 0.0122 | 0.4579 / 0.4810 / 0.4443 | completed |
| YOLO11n earlier reference | letterbox | 640 | 0.4799 ± 0.0262 | 0.1979 ± 0.0076 | 0.3766 ± 0.0133 | 0.2459 ± 0.0137 | 0.5102 / 0.4642 / 0.4654 | resolution control |
| YOLO26n earlier reference | letterbox | 640 | 0.4592 ± 0.0214 | 0.1942 ± 0.0087 | 0.4075 ± 0.0325 | 0.2105 ± 0.0192 | see its run root | resolution control |

**Resolution was not the confound.** Re-running the identical official workflow at `960` px changed almost nothing: YOLO11n moved by `-0.0064` mAP50 and `-0.0055` mAP50-95, YOLO26n by `+0.0019` and `-0.0019`. Every one of those deltas is well inside one fold sample SD (`± 0.0186` to `± 0.0326`), and precision/recall moved `± 0.0133` at most. This closes the geometry objection by measurement rather than assumption: on this dataset official fine-tuning is flat from `640` to `960` px, so the `640`-derived comparison was already fair in practice and the `960` custom results were not flattered by the mismatch.

**Phase 3 target (decision rule applied).** YOLO11n at `960` px, mAP50 `0.4735 ± 0.0326` with mAP50-95 `0.1924 ± 0.0180`, becomes the Phase 4 target. YOLO26n trails by `0.0124` mAP50 but leads on precision (`0.3942` vs `0.3793`), matches it on mAP50-95 (`0.1923` vs `0.1924`), and wins fold 2 outright (`0.4810` vs `0.4362`), so the gap between the two official systems is smaller than the fold spread and they should be treated as statistically indistinguishable. Operationally, the practical ceiling on this dataset is **≈ `0.47` mAP50 / ≈ `0.19` mAP50-95** — not `0.55`, and not a data limit that the custom stack cannot reach.

**Where the custom systems now stand at the matched geometry.**

| Custom system (`960` px) | mAP50 | mAP50-95 | Precision @ 0.25 | Recall @ 0.25 | vs YOLO11n @ 960 |
| --- | ---: | ---: | ---: | ---: | --- |
| Faster R-CNN `letterbox + hflip + affine` | 0.4429 ± 0.0084 | 0.1713 ± 0.0080 | 0.1849 ± 0.0286 | 0.2384 ± 0.0210 | −0.0306 mAP50 (93.5% of target) |
| YOLO26 `letterbox + affine` | 0.2320 ± 0.0113 | 0.0891 ± 0.0075 | 0.1500 ± 0.0137 | 0.1390 ± 0.0444 | −0.2415 mAP50 (49.0% of target) |

The Faster R-CNN result is the practical headline. Its operating recall at `0.25` matches the official target to `0.0009` (`0.2384` vs `0.2393`) and its fold 2 is a tie (`0.4332` vs `0.4362`), yet its precision is `0.1944` lower (`0.1849` vs `0.3793`). Ranked AP with equal operating recall therefore points at the same defect: the custom model emits roughly twice the false positives and its confidence scores are far less well calibrated. Localization tightness is not the bottleneck — mAP50-95 retention is `38.7%` of mAP50 for the custom model versus `40.6%` for the official one. The residual `0.0306` mAP50 gap is a score-ranking and hard-negative problem concentrated in two classes, not a generic capacity deficit; the per-class breakdown below localises it.

### Per-class AP50 (decision rule, completed 2026-11-05)

Both official references were also resolved per class from their `cv_evaluation_valid_project_metrics_conf_0p25.json` files. Custom `960` px values are the adopted selected runs (`Faster R-CNN letterbox + hflip + affine`, `YOLO26 letterbox + affine`).

| Class | GT boxes | YOLO11n @ 960 | YOLO26n @ 960 | Custom Faster R-CNN | Custom YOLO26 | Faster R-CNN − YOLO11n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spaghetti | 7804 | 0.1458 ± 0.0215 | 0.1289 ± 0.0228 | 0.0819 | 0.0576 | **−0.0639** |
| Layer cracking | 285 | 0.4619 ± 0.0551 | 0.4909 ± 0.0474 | 0.3770 | 0.1856 | **−0.0849** |
| Over extrusion | 464 | 0.7273 ± 0.0555 | 0.7201 ± 0.0242 | 0.6951 | 0.3751 | −0.0322 |
| Stringing | 219 | 0.4170 ± 0.0887 | 0.4312 ± 0.0946 | 0.3896 | 0.1636 | −0.0274 |
| Warping | 117 | 0.6153 ± 0.0274 | 0.5343 ± 0.0503 | 0.6710 | 0.3780 | **+0.0557** |

The per-class view sharpens the diagnosis considerably, and it also corrects one reading of the aggregate:

- **The aggregate mAP50 is an unweighted class mean.** Spaghetti carries `7804` of `8889` ground-truth boxes (`87.8%`) yet only one fifth of the score weight. A mean AP50 of `0.4429` therefore means four classes are performing well and one is failing badly, not a uniformly mediocre detector.
- **The deficit is concentrated in two classes.** Spaghetti (`−0.0639`) and Layer cracking (`−0.0849`) account for `0.1488` of the `0.1527` total shortfall — `97%` of the gap. Over extrusion and Stringing are within `0.033` of the official reference, i.e. effectively at parity.
- **Custom Faster R-CNN already beats the official target on Warping** (`0.6710` vs `0.6153`, `+0.0557`). That class has only `117` boxes (`≈ 39` per fold), so treat it as suggestive rather than established, but it is evidence that the custom two-stage stack is not uniformly behind.
- **Closing exactly these two classes closes the entire gap.** Restoring Spaghetti to the official `0.1458` adds `+0.0128` mAP50 and restoring Layer cracking to `0.4619` adds `+0.0170`, giving `0.4727` against a `0.4735` target. No other class needs to move at all.
- **Spaghetti is intrinsically hard for every system, not just the custom one** (`0.1458` for the best available reference, its worst class by a wide margin, `± 0.0215` fold spread). The measured reason is not what the class name suggests; see the next subsection. This is also the most plausible origin of the aggregate precision deficit: Spaghetti is `87.8%` of all boxes, so `0.0819` AP50 with recall already at parity implies most excess false positives fall on tangle images.

The two official systems differ per class only on Warping (`0.0810`, inside the `± 0.0274`–`0.0503` fold spread at `39` boxes per fold) and Spaghetti (`0.0169`); everywhere else they are within `0.014`. Their `0.0124` aggregate difference is therefore concentrated in the least reliable class, which reinforces treating them as indistinguishable.

### Why Spaghetti Is Structurally Hard (measured from the frozen folds)

Computed directly from the frozen label files (`cv-data/roboflow-3d-print-fail-v1/fold_*/valid/labels`, `3421` images, `8889` boxes; box sizes expressed at the `960` px training geometry):

| Class | Boxes | Images | Boxes/img | Median W × H (px) | Median side | Side < 32 px | Same-class GT pairs with IoU > 0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spaghetti | 7804 | 3173 | 2.46 | 174 × 186 | 180 | 0.4% | 0.0% |
| Layer cracking | 285 | 99 | 2.88 | 118 × 50 | 78 | 4.2% | 0.5% |
| Over extrusion | 464 | 125 | 3.71 | 180 × 89 | 122 | 0.2% | 0.0% |
| Stringing | 219 | 79 | 2.77 | 137 × 109 | 120 | 1.4% | 0.0% |
| Warping | 117 | 88 | 1.33 | 398 × 65 | 166 | 0.0% | 0.0% |

Four candidate explanations can then be ruled out or in directly:

1. **Not a small-object problem.** Spaghetti's median box is `174 × 186` px — about `18%` of the `960` frame — and only `0.4%` of its boxes have a side under `32` px. A P2 head, higher input resolution, tiled inference, or SAHI-style slicing is therefore not justified for this class, and Phase 3 already measured resolution to be flat.
2. **Not an instance-separation problem.** Same-class ground-truth boxes within one image have a median pairwise IoU of `0.000`, only `0.0%` of pairs exceed `0.3`, and just `4` of `7804` boxes are nested. Heavy mutual overlap and duplicate annotation are both absent.
3. **Not a cross-class ambiguity within images.** Spaghetti never co-occurs with any other class in these folds: `3173` images are Spaghetti-only and the `248` remaining defect images contain no Spaghetti at all. The five-class label set effectively partitions into two disjoint subsets, so a Spaghetti prediction is never competing against a co-located Stringing or Over extrusion box.
4. **It is an extent-definition problem.** `7804` boxes are spread over `3173` amorphous tangle images at `2.46` boxes per image (maximum `15`), and because no two of them overlap they partition the mass into disjoint crops whose extent is set by the annotator rather than by any object boundary. Reaching IoU `0.5` therefore requires reproducing one annotator's particular cut of a diffuse phenomenon, which bounds AP50 for *any* detector — which is exactly what the best available system shows at `0.1458`.

The most elongated class is not the limiting one: Warping is `398 × 65` px (aspect `6.1`) and the custom model already leads there. Elongated-structure modelling is not the binding constraint either.

**Diagnostic that separates label-defined classes from the label-arbitrary one** (secondary view; the five-class number remains the headline):

| Aggregate over | YOLO11n @ 960 | Custom Faster R-CNN | Custom ÷ official |
| --- | ---: | ---: | ---: |
| All five classes (headline) | 0.4735 | 0.4429 | 93.5% |
| Four well-defined classes (Spaghetti excluded) | 0.5554 | 0.5332 | 96.0% |

On the four classes whose ground truth is well defined, the custom Faster R-CNN is within `0.0222` of the strongest official reference. Report that view next to the five-class number as a diagnostic, never instead of it.

**Corrected Phase 4 implication.** The direction this document carried before the measurement — dense small-object handling for Spaghetti — is not supported and is withdrawn.

- **Spaghetti:** no resolution, P2, tiling, or small-object work. The levers that match the evidence are extent-aware box regression (IoU-aware or decoupled regression heads) and a label-extent audit before either. Its realistic per-class ceiling is near `0.15`, so matching the official value adds at most `+0.0128` mAP50.
- **Layer cracking is where geometry and data scarcity both bite:** a `118 × 50` px median (aspect `2.4`, `4.2%` of boxes under `32` px) across only `285` boxes in `99` images. This is the genuine thin-object target and the genuine scarcity target, worth `+0.0170` if brought to the official `0.4619`.
- **Over extrusion, Stringing, and Warping** are already within `0.033`, ahead, or inside fold noise. Leave them alone.
- Per-class AP50 is satisfied for both references and the Phase 3 decision rule is fully closed.

### Protocol Differences To State, Not Hide

- **Augmentation is deliberately not matched.** The official references use the full Ultralytics stack; the selected custom recipes use only `letterbox + affine` (YOLO26) and `letterbox + hflip + affine` (Faster R-CNN). Closing that recipe gap is the point of the benchmark, so the comparison is "mature stack versus current custom stack", not a single-variable ablation.
- **Checkpoint selection is deliberately not matched.** Ultralytics selects `best.pt` by fitness, which weights mAP50-95 at `0.9`, while the custom practical runs select `best.pt` by project mAP50. A mAP50-selected official reference would require saving every epoch and re-scoring it with project metrics; that machinery is not implemented and is not needed for the Phase 3 gate.
- **Both references retain COCO pretraining.** They are a practical engineering target, not an architecture-only comparison with the local scratch implementations, and they must never be described as improving the submitted strict-scratch study.

## Phase 4: First Experiment — Class-Aware NMS-IoU Sweep (evaluation only)

**Why this precedes any retraining.** Phase 3 leaves the custom Faster R-CNN `0.0306` mAP50 below the target, with the shortfall concentrated in Spaghetti and Layer cracking. Two measured facts make a post-processing sweep the cheapest legitimate first test of that gap:

- In the frozen folds, same-class ground-truth boxes essentially never overlap: median pairwise IoU is `0.000` in every class, the worst class (Layer cracking) has only `0.5%` of pairs above `0.3`, Spaghetti has `0.0%`, and just `4` of `7804` boxes are nested. A class-aware NMS threshold in `0.40`–`0.50` therefore cannot merge genuinely distinct instances of the same class.
- The adopted threshold is `0.70`, which leaves duplicate detections of one tangle in the ranked list. Each surviving duplicate is a false positive ranked above a true positive, which lowers precision and lowers AP50 directly — and Spaghetti is `87.8%` of all boxes and the class with annotator-chosen extent, so it is where duplicates are most likely.

**Scope.** Evaluation only, on the existing selected checkpoints. No training, no new data, no confidence-threshold change, no fold changes, no candidate public-test data.

**Commands.** `run_phase4_nms_iou_sweep.bat` runs eight evaluation points (`0.40` / `0.50` / `0.60` / `0.70` for `runs/faster_rcnn/post_submission_letterbox_hflip_affine` and `runs/yolo26/post_submission_letterbox_affine`), each writing its own aggregate file so the published headline JSON is never overwritten. The `0.70` point is the control and runs last: it must reproduce mAP50 `0.4429` (Faster R-CNN) and `0.2320` (custom YOLO26). If it does not, the sweep is not comparable and must be stopped rather than interpreted.

**Decision rule, fixed before running.**

- Adopt a lower threshold only if it raises five-class mAP50 on at least two of three folds **and** does not decrease Spaghetti AP50.
- Do not adopt a threshold merely because it raises precision: recall at `0.25` must stay within `0.02` of its current value, otherwise the change is a trade rather than a gain.
- Record the whole sweep even if it is flat or negative. A negative result rules out post-processing as the source of the gap and sends Phase 4 fully into box regression and training, which is itself worth knowing before spending GPU time on transfer learning.

**Honest expectation.** AP50 matching claims the single highest-scoring unclaimed detection per ground-truth box, so duplicates usually cost less than the raw precision deficit implies. A move on the order of `+0.005` mAP50 is plausible; anything near `+0.03` is not. The purpose of the sweep is to bound the effect, not to close the gap.

## Phase 4: Second Experiment — Augmentation Composition, Mosaic, and close_mosaic (both models)

**Why this is the first training experiment.** The selected custom recipes train with a single augmentation component at a time: both models use `--online-augmentation affine`, so the adopted policy is `letterbox` resize plus a gentle `5°`/`0.05`/`0.10` warp and nothing else. (`hflip` in the Faster R-CNN run-root name is historical; it is not stacked on the affine run.) The official YOLO26 recipe instead stacks mosaic `1.0`, HSV, `fliplr 0.5`, `scale 0.5`, and `translate 0.1`, and disables mosaic for the final 8–10 epochs via `close_mosaic`. None of those interactions has been tested here, and mosaic — implemented and unit-tested as Component 5 — has never been trained on a full three-fold CV.

**What changes in the code.** `--online-augmentation` now accepts `+`-joined components in fixed Ultralytics order (`mosaic`, `affine`, `hflip`, `hsv`), so a faithful recipe is expressible, for example `mosaic+hflip+hsv` or `mosaic+affine+hflip+hsv`. `--close-mosaic-epochs N` switches the training dataset to the non-mosaic remainder for the final `N` epochs; validation always stays `none`.

**Scope.** Training only, on the frozen 3-fold grouped development folds at 960 px. No data, label, fold, threshold, architecture, loss, optimizer, inference, or candidate-test change. Both models run the same policy so the comparison stays paired.

**Commands.** `run_phase4_mosaic_composition.bat` runs one policy per architecture against that architecture's selected baseline: custom YOLO26 `letterbox + mosaic + hflip + hsv` and Faster R-CNN `letterbox + mosaic + hflip + hsv`, each with `--close-mosaic-epochs 10`. The mosaic component uses the gentle `mosaic` preset (no per-quadrant warp), matching Ultralytics `mosaic 1.0` composition; the never-scored `mosaic_medium` and `mosaic_strong` presets are not used.

**Decision rule, fixed before running.**

- Adopt only if mAP50 rises on at least two of three folds for that architecture and Spaghetti AP50 does not fall.
- Report per-class AP50 for all five classes. A recipe that merely relocates the same Spaghetti and Layer-cracking deficit is not an improvement.
- Keep and record a negative result too: it bounds the augmentation-recipe contribution to the `0.0306` Faster R-CNN gap and the `0.2415` custom YOLO26 gap.

**Honest expectation.** Mosaic raises the effective object count and shrinks object scale, which is the wrong direction for Spaghetti's `174 × 186` px boxes and may cost Spaghetti AP50. The plausible upside is Layer cracking and Warping robustness, on the order of a few thousandths of mAP50.

**Outcome (2026-11-15): rejected for both models.** The first attempt was invalidated by a quadrant-cropping defect in `apply_mosaic` — the frame was warped into a half-size quadrant without a downscale, so labels pointed at the wrong pixels — and all four runs were retrained after the fix. Custom YOLO26 fell to mAP50 `0.1553 ± 0.0124` from `0.2320 ± 0.0113` and Faster R-CNN to `0.2452 ± 0.0447` from `0.4429 ± 0.0084`; mAP50 was lower on all six folds, and Spaghetti AP50 fell for YOLO26 (`0.0528` vs `0.0576`) while rising only `+0.0055` for Faster R-CNN, so the pre-registered rule fails on both architectures. Neither selected recipe changes. The expected mechanism held: halving object scale costs the thin classes first (Layer cracking `−0.0986` and `−0.1883`), and because the treatment also replaced `affine` and added `hsv` — the latter already rejected in Phase 2 on both models — the loss cannot be credited to mosaic alone. This bounds the augmentation-recipe contribution to the remaining gap and closes Phase 4's training branch in favour of gradient accumulation and pretrained initialization.

## Ordered Roadmap

| Phase | Work | Output / decision gate |
| --- | --- | --- |
| 0 | Freeze the submitted baseline metadata and create a separate post-submission run namespace. | Historical and practical studies are visibly separate. |
| 1 | Implement the shared letterbox and coordinate-transform foundation. | **Completed & Adopted:** Full 3-fold grouped CV completed. YOLO26 mAP50 improved to `0.1985 ± 0.0063` (+25.2%); Faster R-CNN maintained parity at `0.3280 ± 0.0282`. Letterbox adopted as standard geometry. |
| 2 | Add a controlled Ultralytics-style train-only augmentation recipe: HSV/color jitter, horizontal flip, random affine or perspective, and optional Mosaic/MixUp with correct box filtering. Start with one component at a time. | **Component 1 rejected:** HSV lowered YOLO26 mAP50 $0.1985 \rightarrow 0.1795$ and Faster R-CNN $0.3280 \rightarrow 0.3179$. **Component 2:** hflip adopted for Faster R-CNN ($0.3280 \rightarrow 0.3394$), rejected for YOLO26 ($0.1985 \rightarrow 0.1890$). **Component 3 (affine) adopted for both after fixing a warp matrix-direction bug that invalidated the first runs:** YOLO26 mAP50 $0.1985 \rightarrow 0.2320$, Faster R-CNN $0.3394 \rightarrow 0.4429$, all five classes improved for both, and both models became more stable. Selected recipes: Faster R-CNN `letterbox + hflip + affine`, YOLO26 `letterbox + affine`. Component 4 executed: three affine-strength presets (`affine`, `affine_medium`, `affine_strong`) implemented and passing regression tests (19 augmentation checks, 7 geometry checks green). Evaluation against each architecture current best recipe pending. Perspective, mosaic, and mix-up deferred until Component 4 is recorded. Component 5 implemented and unit-tested mosaic without a three-fold score; Phase 4 now scores it composed (`run_phase4_mosaic_composition.bat`). |
| 2a | Component 4: affine-strength presets | **Implemented:** `affine`, `affine_medium`, `affine_strong` with `degrees`/`translate`/`scale` swept one variable at a time. 19 augmentation checks and 7 geometry checks green. **Evaluated** against each architecture current best recipe on the frozen 3-fold grouped CV at 960 px. Selected recipe: `affine` for both models (YOLO26 letterbox + `affine` mAP50 0.2320 ± 0.0113; Faster R-CNN letterbox + hflip + `affine` mAP50 0.4429 ± 0.0084). Each step up in strength degraded mAP50. Perspective and mix-up remain unimplemented; mosaic is implemented and is scored composed in Phase 4. |
| 3 | Run an Ultralytics parity benchmark on the frozen existing grouped folds using pretrained YOLO11n or YOLO26n. | **Completed 2026-11-05:** geometry-matched `960` px references reached YOLO11n mAP50 `0.4735 ± 0.0326` and YOLO26n `0.4611 ± 0.0186`. The resolution ablation is flat against the `640` references (`0.4799` / `0.4592`, all deltas within one SD), so geometry was not a confound. Phase 4 target fixed at `0.4735`; custom Faster R-CNN already sits at `0.4429` (93.5%) with recall parity and a precision deficit concentrated in Spaghetti and Layer cracking (those two classes are `97%` of the gap). |
| 4 | Add transfer learning to the custom architectures. Prefer architecture-compatible detection pretraining for the custom YOLO26 and a modern pretrained backbone for Faster R-CNN; document every imported weight source. | Phase 4 opens with the evaluation-only NMS-IoU sweep (`run_phase4_nms_iou_sweep.bat`), justified by the measurement that same-class ground-truth boxes essentially never overlap (at most `0.5%` of same-class pairs above IoU `0.3`). Then compare pretrained custom models against their frozen scratch baselines and the `0.4735` YOLO11n-at-`960` parity target, with Layer cracking as the first training sub-goal; the gap to close is score calibration, duplicate suppression, and hard negatives, not localization or operating recall. |
| 5 | Freeze practical-model settings and report same-fold grouped-development results with an explicit selection-aligned limitation. | No candidate public-test use and no independent-generalization claim. |

## Architecture-Specific Direction

- **Custom YOLO26:** after the letterbox/augmentation foundation, prioritize detector-level pretraining and a mature loss/assignment/augmentation recipe. Loading official Ultralytics weights directly is only valid if the model architecture and parameter mapping are verified; otherwise describe the result as an Ultralytics-based detector, not the exact custom model.
- **Custom Faster R-CNN:** retain the selected scheduled scratch configuration as the historical baseline. For the practical track, use a proven pretrained backbone and modern detection augmentation only after the common transform foundation is validated. The earlier ImageNet transfer result remains historical evidence, not proof that all transfer-learning variants fail.

## Tracking Template

For every practical experiment, append a row or section with:

- study ID and Git commit;
- dataset manifest and split policy;
- initialization / pretrained-weight source;
- transform and augmentation recipe;
- model, optimizer, schedule, resolution, batch size, and seed;
- checkpoint-selection rule;
- grouped-development metrics and per-class metrics;
- whether the result advances to the next phase.

## Immediate Next Steps

1. **Phase 1 complete:** Letterbox baseline established and adopted for both Custom YOLO26 and Custom Faster R-CNN on frozen 3-fold grouped development data.
2. **Phase 2 Component 1 (HSV jitter): completed and rejected for both models** on the frozen 3-fold grouped CV over the adopted letterbox base. Custom YOLO26 mAP50 $0.1985 \pm 0.0063 \rightarrow 0.1795 \pm 0.0229$ and mAP50-95 $0.0691 \pm 0.0053 \rightarrow 0.0610 \pm 0.0086$; Custom Faster R-CNN mAP50 $0.3280 \pm 0.0282 \rightarrow 0.3179 \pm 0.0188$ and mAP50-95 $0.1198 \pm 0.0097 \rightarrow 0.1162 \pm 0.0044$. The policy remains in the codebase as a tested negative result and is not used by the selected models.
3. **Component 2 (completed, split verdict):** train-only horizontal flip was evaluated on the frozen 3-fold grouped CV. Faster R-CNN **adopts** hflip (mAP50 $0.3280 \pm 0.0282 \rightarrow 0.3394 \pm 0.0326$; mAP50-95 $0.1198 \pm 0.0097 \rightarrow 0.1343 \pm 0.0188$; four of five classes improved). YOLO26 **rejects** hflip (mAP50 $0.1985 \pm 0.0063 \rightarrow 0.1890 \pm 0.0190$; mAP50-95 $0.0691 \pm 0.0053 \rightarrow 0.0662 \pm 0.0097$), because the gains on spaghetti and stringing did not offset losses on layer cracking and warping. The selected Faster R-CNN recipe is now letterbox + hflip; the selected YOLO26 recipe remains letterbox-only. Because the mAP50-95 spread widened for Faster R-CNN, treat the gain as promising rather than decisive on three folds.
4. **Component 3 (completed, adopted for both):** the controlled 3-fold affine re-run after the matrix-direction fix is the strongest single result in Phase 2. **Custom YOLO26** mAP50 $0.1985 \pm 0.0063 \rightarrow \mathbf{0.2320 \pm 0.0113}$ and mAP50-95 $0.0691 \pm 0.0053 \rightarrow \mathbf{0.0891 \pm 0.0075}$ (epochs 50 / 48 / 49). **Custom Faster R-CNN** mAP50 $0.3394 \pm 0.0326 \rightarrow \mathbf{0.4429 \pm 0.0084}$ and mAP50-95 $0.1343 \pm 0.0188 \rightarrow \mathbf{0.1713 \pm 0.0080}$ (epochs 40 / 42 / 50). Per-class mean AP50 improved for **all five classes in both models** — YOLO26 spaghetti $0.0336 \rightarrow 0.0576$, layer cracking $0.1510 \rightarrow 0.1856$, over extrusion $0.3432 \rightarrow 0.3751$, stringing $0.0884 \rightarrow 0.1636$, warping $0.3763 \rightarrow 0.3780$; Faster R-CNN spaghetti $0.0599 \rightarrow 0.0819$, layer cracking $0.2603 \rightarrow 0.3770$, over extrusion $0.5799 \rightarrow 0.6951$, stringing $0.2437 \rightarrow 0.3896$, warping $0.5532 \rightarrow 0.6710$. Both models also became **more stable**: Faster R-CNN mAP50 SD fell $\pm 0.0326 \rightarrow \pm 0.0084$ with all three folds (0.4477 / 0.4332 / 0.4479) above every control fold, and YOLO26's SD stayed small at $\pm 0.0113$. The fold-3 weakness that previously made the hflip gain look marginal is gone. **Selected recipes: Faster R-CNN `letterbox + hflip + affine`; YOLO26 `letterbox + affine`.**
5. **Component 4 (completed, results recorded):** the three affine-strength presets were evaluated one at a time against each architecture current best recipe on the frozen 3-fold grouped CV at 960 px. Selected recipe: `affine` for both models (Custom YOLO26 letterbox + `affine` mAP50 0.2320 ± 0.0113; Custom Faster R-CNN letterbox + hflip + `affine` mAP50 0.4429 ± 0.0084). Each step up in strength (`affine_medium`, then `affine_strong`) degraded mAP50 monotonically, so the simplest strength is selected. Perspective and mix-up remain deferred until their own components are scored. Run roots: `runs/yolo26/post_submission_letterbox_affine*`, `runs/faster_rcnn/post_submission_letterbox_hflip_affine*`, `runs/faster_rcnn/post_submission_letterbox_hflip_affine_medium`, `runs/faster_rcnn/post_submission_letterbox_hflip_affine_strong`.
6. **Component 5 (implemented, tested, never scored):** mosaic presets were added to the shared augmentation module and covered by alignment/reproducibility tests, and a fold-1 smoke confirmed geometry and label alignment for both architectures. No three-fold score was ever recorded, so mosaic remains an available policy rather than part of the selected recipe and no strength ordering is claimed. Phase 4 scores it composed with `hflip` and `hsv` under `close_mosaic`.
7. **Phase 3 (completed 2026-11-05): completed and target fixed.** The geometry-matched `960` px parity benchmark finished on the remote GPU host via `run_phase3_parity_benchmark.bat`. **YOLO11n** reached mAP50 $0.4735 \pm 0.0326$, mAP50-95 $0.1924 \pm 0.0180$, precision $0.3793 \pm 0.0292$, and recall $0.2393 \pm 0.0190$ (folds 0.4878 / 0.4362 / 0.4964). **YOLO26n** reached mAP50 $0.4611 \pm 0.0186$, mAP50-95 $0.1923 \pm 0.0067$, precision $0.3942 \pm 0.0400$, and recall $0.2172 \pm 0.0122$ (folds 0.4579 / 0.4810 / 0.4443). Against the `640` references ($0.4799$ / $0.4592$) the mAP50 changes are $-0.0064$ and $+0.0019$, both inside one sample SD, so image geometry was not a confound and the two official systems are statistically indistinguishable at this fold count. **Phase 4 target: YOLO11n at `960` px, mAP50 $0.4735 \pm 0.0326$.** Custom Faster R-CNN sits at $0.4429$ (93.5% of target) with recall matching to $0.0009$ but precision $0.1944$ lower. Per-class AP50 is now transcribed for both references: the custom model is at parity on Over extrusion and Stringing, ahead on Warping ($0.6710$ vs $0.6153$), and behind only on Spaghetti ($0.0819$ vs $0.1458$) and Layer cracking ($0.3770$ vs $0.4619$), which together are $97\%$ of the gap. Matching just those two classes yields $0.4727$ against the $0.4735$ target. No candidate public-test data was used and the Phase 3 decision rule is fully closed.
8. **Phase 4 next:** run `run_phase4_nms_iou_sweep.bat` first (evaluation only, eight points; the `0.70` control must reproduce `0.4429` / `0.2320`), then `run_phase4_mosaic_composition.bat` (composed mosaic + `hflip` + `hsv` with `close_mosaic 10`, one paired run per architecture); **mosaic composition rejected 2026-11-15** on both models after the quadrant-geometry fix, so no recipe changes. The remaining Phase 4 training items are ordered: `nbs=64` gradient accumulation, then pretrained initialization for both custom models. Then begin transfer learning on the custom architectures against the $0.4735$ target, starting with Layer cracking — thin `118 × 50` px boxes and only `285` instances across `99` images — and explicitly not with small-object work on Spaghetti, whose median box is `174 × 186` px and whose diffuse extent bounds AP50 for every system at about `0.15`. All results remain selection-aligned grouped-development evidence and must not be presented as improving the submitted strict-scratch study.

9. **Ultralytics feature inventory (post-Phase-3 audit, recorded so it is not re-derived).** Implemented and exercised: letterbox/other resize modes, `hflip`, `affine` strengths, mosaic presets, HSV, photometric, EMA, AMP, warmup, cosine and plateau scheduling, early stopping, mAP50/loss checkpoint selection, P2, positive class weighting, balanced sampling, `--fraction`, and hflip TTA — each with its recorded verdict. Implemented in the custom YOLO26 only: dual one2many/one2one heads, NMS-free top-k inference, DFL and DFL-free regression (`--reg-max`), focal BCE, and the Ultralytics `TaskAlignedAssigner`. Implemented in Faster R-CNN only: torchvision transfer with `--backbone-weights imagenet` and `--backbone-lr-multiplier`. **Absent and deliberately not prioritized:** STAL, `multi_scale`, label smoothing, dropout, `rect` training, `copy_paste`, `mixup`/`cutmix`, `MuSGD`, `ProgLoss`, `close_mosaic` (now added), gradient accumulation/`nbs`, YOLO26 pretrained initialization, full-COCO Faster R-CNN initialization, export, DDP, `torch.compile`, and training plots. STAL and `multi_scale` are skipped because measured boxes are large and Ultralytics itself ships `multi_scale 0.0`; label smoothing and dropout are at their Ultralytics defaults of off; the remaining training-side gaps are the ranked Phase 4 shortlist (mosaic recipe, gradient accumulation, pretrained initialization).
