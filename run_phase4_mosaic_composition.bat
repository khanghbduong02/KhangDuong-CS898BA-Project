@echo off
rem Phase 4 step 2: composed augmentation, paired across both custom models.
rem Registered in POST_SUBMISSION_PRACTICAL_PLAN.md before this file existed.
rem
rem Policy: mosaic (Ultralytics mosaic 1.0 composition, no per-quadrant warp)
rem + horizontal flip + HSV jitter, with the mosaic component disabled for the
rem final 10 of 50 epochs (Ultralytics close_mosaic). Letterbox resize base and
rem every other setting match each model's selected baseline recipe exactly.
rem Training only, frozen 3-fold grouped development folds, no candidate test.
rem
rem Adoption rule (fixed in the plan before running): mAP50 must rise on at
rem least 2 of 3 folds and Spaghetti AP50 must not fall.
cd /d "%~dp0"

set PYTHON=python
set DEVICE=0
set DATA_ROOT=cv-data/roboflow-3d-print-fail-v1
set YOLO26_ROOT=runs/yolo26/post_submission_letterbox_mosaic_hflip_hsv
set FRCNN_ROOT=runs/faster_rcnn/post_submission_letterbox_mosaic_hflip_hsv

echo ===== Custom YOLO26: mosaic+hflip+hsv, close_mosaic 10 =====
%PYTHON% run_yolo26_kfold_cv.py --data-root %DATA_ROOT% --run-root %YOLO26_ROOT% --epochs 50 --batch-size 8 --imgsz 960 --workers 0 --seed 42 --device %DEVICE% --scale n --focal-gamma 2 --class-positive-weight-power 0.25 --checkpoint-selection map50 --reduce-lr-patience 0 --reduce-lr-cooldown 0 --early-stopping-patience 0 --ema-decay 0 --resize-mode letterbox --online-augmentation mosaic+hflip+hsv --close-mosaic-epochs 10
if errorlevel 1 goto :failed

echo ===== Custom Faster R-CNN: mosaic+hflip+hsv, close_mosaic 10 =====
%PYTHON% run_faster_rcnn_kfold_cv.py --data-root %DATA_ROOT% --run-root %FRCNN_ROOT% --epochs 50 --batch-size 2 --imgsz 960 --workers 0 --seed 42 --device %DEVICE% --scale s --class-positive-weight-power 0.25 --checkpoint-selection map50 --lr-schedule cosine --warmup-epochs 3 --warmup-start-factor 0.1 --cosine-final-factor 0.02 --reduce-lr-patience 0 --reduce-lr-cooldown 0 --early-stopping-patience 0 --ema-decay 0 --resize-mode letterbox --online-augmentation mosaic+hflip+hsv --close-mosaic-epochs 10
if errorlevel 1 goto :failed

echo ===== Evaluate Custom YOLO26 =====
%PYTHON% eval_yolo26_kfold_cv.py --data-root %DATA_ROOT% --run-root %YOLO26_ROOT% --imgsz 960 --batch-size 8 --workers 0 --device %DEVICE% --conf-thresh 0.25
if errorlevel 1 goto :failed

echo ===== Evaluate Custom Faster R-CNN =====
%PYTHON% eval_faster_rcnn_kfold_cv.py --data-root %DATA_ROOT% --run-root %FRCNN_ROOT% --imgsz 960 --batch-size 2 --workers 0 --device %DEVICE% --conf-thresh 0.25
if errorlevel 1 goto :failed

echo All Phase 4 mosaic runs and evaluations finished.
exit /b 0

:failed
echo Phase 4 mosaic run failed. See the last command above.
exit /b 1
