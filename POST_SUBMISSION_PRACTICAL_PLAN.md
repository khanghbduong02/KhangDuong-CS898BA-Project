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

## Practical Performance Requirements

Reaching a credible Ultralytics-level practical system requires all three items below. They are separate from, and do not change, the submitted strict-scratch study.

| Requirement | Why It Is Needed | Plan Phases | Status |
| --- | --- | --- | --- |
| 1. Detector pretraining | Large-scale detection pretraining provides transferable visual features and localization priors that the current 3D-print dataset cannot supply from scratch. | Phase 4 | Planned after the transform and augmentation foundation is validated. |
| 2. Mature training stack | Official Ultralytics systems combine correct image geometry, strong train-only augmentation, tuned optimization, losses, normalization, and postprocessing. | Phases 1-3 | **First implementation starts here:** shared letterbox geometry, then controlled augmentation and a parity benchmark. |
| 3. More independent, consistent data | Sparse minority examples, ambiguous boxes, and repeatedly reused folds limit both real-world generalization and trustworthy model selection. | Phase 5 | Planned before any final practical-performance claim. |

The first engineering task is requirement 2, not because it is the largest eventual gain, but because it is the shared foundation needed to train and evaluate both custom architectures correctly before introducing pretrained weights.

## Rules For The New Study

- Use a distinct `post_submission` run root and record every setting, checkpoint, dataset manifest, and result.
- Treat the existing grouped folds as development data only. Do not use the contaminated candidate public test split for tuning or final claims.
- Before reporting a final practical result, reserve or collect a group-disjoint holdout that remains untouched until all model choices are frozen.
- Clearly label pretrained weights, Ultralytics components, and external data. This is transfer learning and practical engineering, not strict scratch learning.
- Keep the original submitted checkpoints and metrics unchanged as historical baselines.

## First Implementation: Shared Letterbox And Transform Foundation

**Implement this first.** Both custom datasets currently stretch each image directly to a square. Add a shared transform module that:

1. letterboxes images to a fixed canvas while preserving aspect ratio;
2. maps boxes and predictions between source, letterboxed, and model coordinates;
3. exposes deterministic train and validation behavior;
4. is used by both YOLO26 and Faster R-CNN training, evaluation, and demo code; and
5. has synthetic box-round-trip and image/label alignment tests.

Why first: it removes avoidable geometric distortion, provides the coordinate contract required for Ultralytics-style augmentation, and lets both architectures use the same well-tested image geometry. It is more foundational than another learning-rate or batch-size sweep.

**Acceptance gate:** existing historical behavior remains reproducible through an explicit `stretch` mode, and letterbox round trips recover source boxes within one pixel after clipping.

## Ordered Roadmap

| Phase | Work | Output / decision gate |
| --- | --- | --- |
| 0 | Freeze the submitted baseline metadata and create a separate post-submission run namespace. | Historical and practical studies are visibly separate. |
| 1 | Implement the shared letterbox and coordinate-transform foundation. | Tests pass; stretch mode remains available for historical reproduction. |
| 2 | Add a controlled Ultralytics-style train-only augmentation recipe: HSV/color jitter, horizontal flip, random affine or perspective, and optional Mosaic/MixUp with correct box filtering. Start with one component at a time. | A deterministic transform test suite and one pre-registered recipe per architecture. |
| 3 | Run an Ultralytics parity benchmark on the new practical protocol using pretrained YOLO11n or YOLO26n. | Establish a realistic target under identical splits, image geometry, and evaluation code. |
| 4 | Add transfer learning to the custom architectures. Prefer architecture-compatible detection pretraining for the custom YOLO26 and a modern pretrained backbone for Faster R-CNN; document every imported weight source. | Compare pretrained custom models against their frozen scratch baselines and the parity benchmark. |
| 5 | Improve data credibility: adjudicate difficult annotations, add independent source groups, and reserve a group-disjoint final holdout before selecting a final practical model. | Final practical claim uses data that was not repeatedly used for model selection. |

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

1. Create the shared `letterbox` transform API and synthetic tests.
2. Add an explicit `--resize-mode stretch|letterbox` argument defaulting to `stretch` for historical reproducibility.
3. Run a one-epoch smoke test for both architectures in `letterbox` mode.
4. Only then launch one fixed post-submission grouped-CV letterbox baseline per architecture.
