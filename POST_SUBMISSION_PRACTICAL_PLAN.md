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

## Ordered Roadmap

| Phase | Work | Output / decision gate |
| --- | --- | --- |
| 0 | Freeze the submitted baseline metadata and create a separate post-submission run namespace. | Historical and practical studies are visibly separate. |
| 1 | Implement the shared letterbox and coordinate-transform foundation. | **Completed:** tests, K-fold dry runs, and two CUDA geometry smokes passed; stretch remains historical default. |
| 2 | Add a controlled Ultralytics-style train-only augmentation recipe: HSV/color jitter, horizontal flip, random affine or perspective, and optional Mosaic/MixUp with correct box filtering. Start with one component at a time. | A deterministic transform test suite and one pre-registered recipe per architecture. |
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

1. Run one fixed post-submission grouped-CV letterbox baseline per architecture, using distinct run roots and no other setting changes.
2. Compare each letterbox result with its matching historical stretch baseline; do not adopt letterbox unless the full three-fold result supports it.
3. If letterbox is retained or its trade-off is documented, implement exactly one controlled Ultralytics-style train-only augmentation component with transform tests before beginning a full CV run.
4. Do not introduce pretrained weights until the geometry and first augmentation component have a documented full-CV outcome.
