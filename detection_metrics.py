from __future__ import annotations

from typing import Any, Dict, List

import torch


def xywhn_to_xyxy(labels_xywhn: torch.Tensor, imgsz: int) -> torch.Tensor:
    if labels_xywhn.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    x = labels_xywhn[:, 0] * imgsz
    y = labels_xywhn[:, 1] * imgsz
    w = labels_xywhn[:, 2] * imgsz
    h = labels_xywhn[:, 3] * imgsz
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return torch.stack((x1, y1, x2, y2), dim=1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-9)


def compute_ap(tp: List[float], conf: List[float], num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0

    order = torch.tensor(conf).argsort(descending=True)
    tp_sorted = torch.tensor(tp, dtype=torch.float32)[order]
    fp_sorted = 1.0 - tp_sorted

    tpc = torch.cumsum(tp_sorted, dim=0)
    fpc = torch.cumsum(fp_sorted, dim=0)

    recall = tpc / max(float(num_gt), 1e-9)
    precision = tpc / (tpc + fpc).clamp(min=1e-9)

    mrec = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    mpre = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0)[0], dims=[0])

    indices = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap.item())


def match_per_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresh: float,
) -> torch.Tensor:
    if pred_boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    order = pred_scores.argsort(descending=True)
    sorted_boxes = pred_boxes[order]
    sorted_labels = pred_labels[order]

    matched_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
    sorted_tp = torch.zeros((sorted_boxes.shape[0],), dtype=torch.float32)

    if gt_boxes.numel() == 0:
        return sorted_tp

    ious = box_iou(sorted_boxes, gt_boxes)
    for i in range(sorted_boxes.shape[0]):
        same_class = gt_labels == sorted_labels[i]
        if not same_class.any():
            continue

        candidate_ious = ious[i].clone()
        candidate_ious[~same_class] = -1.0
        best_iou, best_j = candidate_ious.max(dim=0)
        if best_iou >= iou_thresh and not matched_gt[best_j]:
            sorted_tp[i] = 1.0
            matched_gt[best_j] = True

    # AP later sorts confidences globally. Restore the original prediction order
    # so each true-positive flag remains paired with its own confidence value.
    tp = torch.zeros_like(sorted_tp)
    tp[order] = sorted_tp
    return tp


def _update_confusion_matrix(
    confusion: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_thresh: float,
    conf_thresh: float,
) -> None:
    bg_idx = confusion.shape[0] - 1

    pred_mask = pred_scores >= conf_thresh
    pred_boxes = pred_boxes[pred_mask]
    pred_scores = pred_scores[pred_mask]
    pred_labels = pred_labels[pred_mask]

    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return

    if pred_boxes.numel() == 0:
        for true_class in gt_labels.tolist():
            true_class = int(true_class)
            if 0 <= true_class < bg_idx:
                confusion[true_class, bg_idx] += 1
        return

    if gt_boxes.numel() == 0:
        for pred_class in pred_labels.tolist():
            pred_class = int(pred_class)
            if 0 <= pred_class < bg_idx:
                confusion[bg_idx, pred_class] += 1
        return

    ious = box_iou(pred_boxes, gt_boxes)
    candidate_pairs = torch.nonzero(ious >= iou_thresh, as_tuple=False)

    matched_pred = torch.zeros((pred_boxes.shape[0],), dtype=torch.bool)
    matched_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)

    if candidate_pairs.numel() > 0:
        pair_scores = ious[candidate_pairs[:, 0], candidate_pairs[:, 1]]
        order = pair_scores.argsort(descending=True)
        candidate_pairs = candidate_pairs[order]

        for pair in candidate_pairs:
            pred_i = int(pair[0].item())
            gt_j = int(pair[1].item())
            if matched_pred[pred_i] or matched_gt[gt_j]:
                continue

            pred_class = int(pred_labels[pred_i].item())
            true_class = int(gt_labels[gt_j].item())
            if 0 <= true_class < bg_idx and 0 <= pred_class < bg_idx:
                confusion[true_class, pred_class] += 1
            matched_pred[pred_i] = True
            matched_gt[gt_j] = True

    for gt_j in range(gt_boxes.shape[0]):
        if not matched_gt[gt_j]:
            true_class = int(gt_labels[gt_j].item())
            if 0 <= true_class < bg_idx:
                confusion[true_class, bg_idx] += 1

    for pred_i in range(pred_boxes.shape[0]):
        if not matched_pred[pred_i]:
            pred_class = int(pred_labels[pred_i].item())
            if 0 <= pred_class < bg_idx:
                confusion[bg_idx, pred_class] += 1


def _compute_class_ap(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    class_id: int,
    iou_thresh: float,
) -> float:
    class_tp: List[float] = []
    class_conf: List[float] = []
    num_gt = 0

    for pred, tgt in zip(predictions, targets):
        pred_mask = pred["labels"] == class_id
        gt_mask = tgt["labels"] == class_id

        p_boxes = pred["boxes"][pred_mask]
        p_scores = pred["scores"][pred_mask]
        g_boxes = tgt["boxes"][gt_mask]
        g_labels = tgt["labels"][gt_mask]

        num_gt += int(g_boxes.shape[0])
        if p_boxes.numel() == 0:
            continue

        tp = match_per_image(
            p_boxes,
            p_scores,
            pred_labels=torch.full((p_boxes.shape[0],), class_id, dtype=torch.long),
            gt_boxes=g_boxes,
            gt_labels=g_labels,
            iou_thresh=iou_thresh,
        )
        class_tp.extend(tp.tolist())
        class_conf.extend(p_scores.tolist())

    return compute_ap(class_tp, class_conf, num_gt)


def _compute_class_pr(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    class_id: int,
    conf_thresh: float,
) -> Dict[str, float]:
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    gt_total = 0

    for pred, tgt in zip(predictions, targets):
        pred_mask = (pred["labels"] == class_id) & (pred["scores"] >= conf_thresh)
        gt_mask = tgt["labels"] == class_id

        p_boxes = pred["boxes"][pred_mask]
        p_scores = pred["scores"][pred_mask]
        p_labels = pred["labels"][pred_mask]
        g_boxes = tgt["boxes"][gt_mask]
        g_labels = tgt["labels"][gt_mask]

        gt_total += int(g_boxes.shape[0])

        tp = match_per_image(p_boxes, p_scores, p_labels, g_boxes, g_labels, iou_thresh=0.50)
        tp_count = float(tp.sum().item())
        tp_total += tp_count
        fp_total += float(max(p_boxes.shape[0] - int(tp_count), 0))
        fn_total += float(max(g_boxes.shape[0] - int(tp_count), 0))

    precision = tp_total / max(tp_total + fp_total, 1e-9)
    recall = tp_total / max(tp_total + fn_total, 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "num_gt": float(gt_total),
    }


def compute_detection_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    conf_thresh: float,
) -> Dict[str, Any]:
    iou_thresholds = [0.50 + 0.05 * i for i in range(10)]
    per_class: List[Dict[str, float]] = []

    for class_id in range(num_classes):
        ap_values: List[float] = []
        for iou_t in iou_thresholds:
            ap = _compute_class_ap(predictions, targets, class_id, iou_t)
            ap_values.append(ap)

        valid_ap = [v for v in ap_values if v == v]
        pr = _compute_class_pr(predictions, targets, class_id, conf_thresh)

        per_class.append(
            {
                "class_id": float(class_id),
                "ap50": ap_values[0] if ap_values[0] == ap_values[0] else 0.0,
                "ap50_95": (sum(valid_ap) / len(valid_ap)) if valid_ap else 0.0,
                "precision": pr["precision"],
                "recall": pr["recall"],
                "num_gt": pr["num_gt"],
            }
        )

    present_classes = [c for c in per_class if c["num_gt"] > 0]
    map50 = sum(c["ap50"] for c in present_classes) / max(len(present_classes), 1)
    map50_95 = sum(c["ap50_95"] for c in present_classes) / max(len(present_classes), 1)

    # Micro precision/recall across all classes at IoU50.
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    for pred, tgt in zip(predictions, targets):
        conf_mask = pred["scores"] >= conf_thresh
        p_boxes = pred["boxes"][conf_mask]
        p_scores = pred["scores"][conf_mask]
        p_labels = pred["labels"][conf_mask]
        g_boxes = tgt["boxes"]
        g_labels = tgt["labels"]

        tp = match_per_image(p_boxes, p_scores, p_labels, g_boxes, g_labels, iou_thresh=0.50)
        tp_count = float(tp.sum().item())
        tp_total += tp_count
        fp_total += float(max(p_boxes.shape[0] - int(tp_count), 0))
        fn_total += float(max(g_boxes.shape[0] - int(tp_count), 0))

    precision = tp_total / max(tp_total + fp_total, 1e-9)
    recall = tp_total / max(tp_total + fn_total, 1e-9)

    confusion = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int64)
    for pred, tgt in zip(predictions, targets):
        _update_confusion_matrix(
            confusion=confusion,
            pred_boxes=pred["boxes"],
            pred_scores=pred["scores"],
            pred_labels=pred["labels"],
            gt_boxes=tgt["boxes"],
            gt_labels=tgt["labels"],
            iou_thresh=0.50,
            conf_thresh=conf_thresh,
        )

    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
