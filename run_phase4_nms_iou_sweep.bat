@echo off
rem Phase 4 step 1: class-aware NMS-IoU sweep on the existing selected checkpoints.
rem Evaluation only. No training, no new data, no threshold-on-confidence change.
rem
rem Justification (measured from the frozen folds): same-class ground-truth boxes
rem essentially never overlap (median pairwise IoU 0.000 in every class; at most
rem 0.5% of same-class pairs above IoU 0.3, and 0.0% for Spaghetti), so a
rem class-aware NMS threshold of 0.40-0.50 cannot merge distinct instances while it
rem does suppress duplicate detections of the same tangle. The adopted threshold is
rem 0.70. Spaghetti dominates the box count (87.8%) and is the class whose extent is
rem annotator-chosen, which is exactly where duplicate suppression should pay.
rem
rem The 0.70 point is listed last so it is the control and so the run root's per-fold
rem artifacts end up matching the published five-class numbers again. Every point
rem writes its own aggregate file, so nothing overwrites the headline result.
cd /d "%~dp0"

set PYTHON=python
set DEVICE=cuda
set DATA_ROOT=cv-data/roboflow-3d-print-fail-v1
set FRCNN_ROOT=runs/faster_rcnn/post_submission_letterbox_hflip_affine
set YOLO26_ROOT=runs/yolo26/post_submission_letterbox_affine

echo ===== Faster R-CNN: NMS IoU 0.40 =====
%PYTHON% eval_faster_rcnn_kfold_cv.py --data-root %DATA_ROOT% --run-root %FRCNN_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --nms-iou 0.40 --output %FRCNN_ROOT%/nms_sweep_iou_0p40.json
if errorlevel 1 goto :failed

echo ===== Faster R-CNN: NMS IoU 0.50 =====
%PYTHON% eval_faster_rcnn_kfold_cv.py --data-root %DATA_ROOT% --run-root %FRCNN_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --nms-iou 0.50 --output %FRCNN_ROOT%/nms_sweep_iou_0p50.json
if errorlevel 1 goto :failed

echo ===== Faster R-CNN: NMS IoU 0.60 =====
%PYTHON% eval_faster_rcnn_kfold_cv.py --data-root %DATA_ROOT% --run-root %FRCNN_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --nms-iou 0.60 --output %FRCNN_ROOT%/nms_sweep_iou_0p60.json
if errorlevel 1 goto :failed

echo ===== Faster R-CNN: NMS IoU 0.70 (control) =====
%PYTHON% eval_faster_rcnn_kfold_cv.py --data-root %DATA_ROOT% --run-root %FRCNN_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --nms-iou 0.70 --output %FRCNN_ROOT%/nms_sweep_iou_0p70.json
if errorlevel 1 goto :failed

echo ===== Custom YOLO26: NMS IoU 0.40 =====
%PYTHON% eval_yolo26_kfold_cv.py --data-root %DATA_ROOT% --run-root %YOLO26_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --inference-branch one2many --postprocess class_aware_nms --nms-iou 0.40 --output %YOLO26_ROOT%/nms_sweep_iou_0p40.json
if errorlevel 1 goto :failed

echo ===== Custom YOLO26: NMS IoU 0.50 =====
%PYTHON% eval_yolo26_kfold_cv.py --data-root %DATA_ROOT% --run-root %YOLO26_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --inference-branch one2many --postprocess class_aware_nms --nms-iou 0.50 --output %YOLO26_ROOT%/nms_sweep_iou_0p50.json
if errorlevel 1 goto :failed

echo ===== Custom YOLO26: NMS IoU 0.60 =====
%PYTHON% eval_yolo26_kfold_cv.py --data-root %DATA_ROOT% --run-root %YOLO26_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --inference-branch one2many --postprocess class_aware_nms --nms-iou 0.60 --output %YOLO26_ROOT%/nms_sweep_iou_0p60.json
if errorlevel 1 goto :failed

echo ===== Custom YOLO26: NMS IoU 0.70 (control) =====
%PYTHON% eval_yolo26_kfold_cv.py --data-root %DATA_ROOT% --run-root %YOLO26_ROOT% --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25 --inference-branch one2many --postprocess class_aware_nms --nms-iou 0.70 --output %YOLO26_ROOT%/nms_sweep_iou_0p70.json
if errorlevel 1 goto :failed

echo.
echo NMS-IoU sweep finished. Eight aggregate files were written:
echo   %FRCNN_ROOT%\nms_sweep_iou_0p40.json .. 0p70.json
echo   %YOLO26_ROOT%\nms_sweep_iou_0p40.json .. 0p70.json
echo The 0p70 file must reproduce the published five-class numbers; if it does not,
echo the sweep is not comparable and should be stopped.
exit /b 0

:failed
echo.
echo NMS-IoU sweep stopped with an error. Fix the cause and re-run this file;
echo completed points only cost evaluation time to repeat.
exit /b 1
