# Post-Submission Practical Improvement Plan

## Purpose

Continue the detector work as a separate practical-performance study. The submitted strict-scratch results remain frozen and must not be replaced, relabeled, or compared as though they used the same protocol.

Current submitted grouped-CV development baselines at 960 pixels:

| System | mAP50 |
| --- | ---: |
| Custom YOLO26, selected raw scratch configuration | 0.1586 |
| Custom Faster R-CNN, selected raw scheduled scratch configuration | 0.3258 |
| Official pretrained Ultralytics YOLO26n reference | 0.4592 |
| Official pretrained Ultralytics YOLO11n reference | 0.4799 |

## Requirements In Execution Order

Reaching a credible Ultralytics-level practical system requires all three items below. They are separate from, and do not change, the submitted strict-scratch study.

The dataset, labels, grouped folds, and public-test boundary are frozen for this study. The active work is therefore the Ultralytics-style training stack first, followed by detector pretraining; data improvement is explicitly deferred.

| Execution order | Requirement | Why It Is Needed | Plan Phases | Status |
| --- | --- | --- | --- | --- |
| 1. First code implementation | Mature training stack | Official Ultralytics systems combine correct image geometry, strong train-only augmentation, tuned optimization, losses, normalization, and postprocessing. | Phases 1-3 | **Active now:** implement shared letterbox geometry, then controlled augmentation and an Ultralytics parity benchmark. |
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

## Ordered Roadmap

| Phase | Work | Output / decision gate |
| --- | --- | --- |
| 0 | Freeze the submitted baseline metadata and create a separate post-submission run namespace. | Historical and practical studies are visibly separate. |
| 1 | Implement the shared letterbox and coordinate-transform foundation. | **Completed & Adopted:** Full 3-fold grouped CV completed. YOLO26 mAP50 improved to `0.1985 ± 0.0063` (+25.2%); Faster R-CNN maintained parity at `0.3280 ± 0.0282`. Letterbox adopted as standard geometry. |
| 2 | Add a controlled Ultralytics-style train-only augmentation recipe: HSV/color jitter, horizontal flip, random affine or perspective, and optional Mosaic/MixUp with correct box filtering. Start with one component at a time. | **Active now:** Implement Component 1 (train-only HSV photometric jitter) with deterministic unit tests and run a controlled full-CV comparison on top of the letterbox base. |
| 3 | Run an Ultralytics parity benchmark on the frozen existing grouped folds using pretrained YOLO11n or YOLO26n. | Establish the target under matching fold inputs, image geometry, and project evaluation code. |
| 4 | Add transfer learning to the custom architectures. Prefer architecture-compatible detection pretraining for the custom YOLO26 and a modern pretrained backbone for Faster R-CNN; document every imported weight source. | Compare pretrained custom models against their frozen scratch baselines and the parity benchmark. |
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
2. **Phase 2 (Augmentation Foundation):** Implement Component 1 of the train-only augmentation pipeline (HSV/photometric color jitter in training dataset loaders only, preserving bounding boxes and keeping validation strictly deterministic).
3. Add unit tests in `tests/` verifying HSV transform bounds, seed reproducibility, and label/bounding box coordinate invariance.
4. Run full 3-fold grouped CV for Custom YOLO26 and Custom Faster R-CNN with Component 1 on top of the adopted letterbox geometry before proceeding to geometric transforms (horizontal flip, random affine, mosaic).
