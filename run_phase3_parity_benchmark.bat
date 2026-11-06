@echo off
rem Phase 3: official Ultralytics parity benchmark at the custom models' adopted
rem 960 px letterbox geometry. Frozen group-disjoint folds 1-3, official
rem COCO-pretrained weights, project detection_metrics.py after Ultralytics NMS.
rem Paths resolve relative to this file, so the checkout can live anywhere.
rem Re-running this file resumes: completed folds are skipped by the runners.
cd /d "%~dp0"

set PYTHON=python
set DEVICE=0
set DATA_ROOT=cv-data/roboflow-3d-print-fail-v1

echo ===== YOLO11n reference: train folds 1-3 =====
%PYTHON% run_ultralytics_kfold_cv.py --data-root %DATA_ROOT% --run-root runs/ultralytics/post_submission_parity_yolo11n_imgsz960 --weights yolo11n.pt --folds 1 2 3 --epochs 50 --imgsz 960 --batch-size 8 --workers 0 --seed 42 --device %DEVICE% --optimizer auto --patience 100
if errorlevel 1 goto :failed

echo ===== YOLO11n reference: evaluate folds 1-3 =====
%PYTHON% eval_ultralytics_kfold_cv.py --data-root %DATA_ROOT% --run-root runs/ultralytics/post_submission_parity_yolo11n_imgsz960 --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25
if errorlevel 1 goto :failed

echo ===== YOLO26n reference: train folds 1-3 =====
%PYTHON% run_ultralytics_kfold_cv.py --data-root %DATA_ROOT% --run-root runs/ultralytics/post_submission_parity_yolo26n_imgsz960 --weights yolo26n.pt --folds 1 2 3 --epochs 50 --imgsz 960 --batch-size 8 --workers 0 --seed 42 --device %DEVICE% --optimizer auto --patience 100
if errorlevel 1 goto :failed

echo ===== YOLO26n reference: evaluate folds 1-3 =====
%PYTHON% eval_ultralytics_kfold_cv.py --data-root %DATA_ROOT% --run-root runs/ultralytics/post_submission_parity_yolo26n_imgsz960 --folds 1 2 3 --split valid --imgsz 960 --batch-size 8 --device %DEVICE% --conf-thresh 0.25
if errorlevel 1 goto :failed

echo.
echo Phase 3 benchmark finished. Aggregate metrics are in:
echo   runs/ultralytics/post_submission_parity_yolo11n_imgsz960/cv_evaluation_valid_project_metrics_conf_0p25.json
echo   runs/ultralytics/post_submission_parity_yolo26n_imgsz960/cv_evaluation_valid_project_metrics_conf_0p25.json
exit /b 0

:failed
echo.
echo Phase 3 benchmark stopped with an error. Fix the cause and re-run this file to resume completed folds.
exit /b 1
