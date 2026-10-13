@echo off
cd /d "d:\Study\WSU\Master\1.3\Computer Vision & Image Analysis\Project"
python run_yolo26_kfold_cv.py --data-root cv-data/roboflow-3d-print-fail-v1 --run-root runs/yolo26/post_submission_letterbox_affine_medium --epochs 50 --batch-size 8 --imgsz 960 --workers 0 --seed 42 --device cuda --scale n --focal-gamma 2 --class-positive-weight-power 0.25 --checkpoint-selection map50 --reduce-lr-patience 0 --reduce-lr-cooldown 0 --early-stopping-patience 0 --ema-decay 0 --resize-mode letterbox --online-augmentation affine_medium
echo EXIT=%ERRORLEVEL%
