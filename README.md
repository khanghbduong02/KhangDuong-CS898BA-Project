# Automated Detection and Classification of 3D Printing Failures Using Image Processing and Deep Learning

## Project Overview

This project aims to automatically detect and classify failures in 3D printing images using computer vision and deep learning.

The system will:

* Detect failure locations using bounding boxes.
* Classify failure types.
* Investigate the impact of image processing techniques on object detection performance.

## Dataset

Source: [Roboflow](https://universe.roboflow.com/purvi-rathore-5amqh/3d-print-failure-detection-efvsh)

Number of Images:

* Training: 2696
* Validation: 1524
* Testing: 329

Classes:

* Spaghetti
* Stringing
* Warping

## Proposed Methods

### Object Detection Models

* YOLO
* Faster R-CNN

### Image Processing Techniques

* LAB Color Space
* CLAHE Contrast Enhancement
* Canny Edge Detection

## Evaluation Metrics

* mAP50
* mAP50-95
* Precision
* Recall
* IoU

## Current Status

Project Pitch and Scope Approval Stage.

## Getting Started: Image Processing Pipeline

This project compares three preprocessing tracks before object detection:

* Baseline: RGB input only.
* Method 1: RGB -> LAB -> CLAHE.
* Method 2: RGB -> LAB -> CLAHE -> Canny edge enhancement.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Preprocessed Dataset Variants

Run from the project root:

```bash
python preprocess_dataset.py --input-dir roboflow-data --output-dir processed-data
```

This generates:

* `processed-data/baseline/`
* `processed-data/clahe/`
* `processed-data/clahe_canny/`

Each variant includes:

* `train/images`, `train/labels`
* `valid/images`, `valid/labels`
* `test/images`, `test/labels`
* `data.yaml`

### 3. Train and Compare Models

Use each variant's `data.yaml` for YOLO and Faster R-CNN experiments. Compare:

* mAP50
* mAP50-95
* Precision
* Recall
* IoU

### Notes

* Labels are copied unchanged from the original Roboflow dataset.
* Canny settings are configurable in the script CLI.
* The CLAHE + Canny method overlays edges on the CLAHE-enhanced image to keep visual context while emphasizing defect boundaries.
