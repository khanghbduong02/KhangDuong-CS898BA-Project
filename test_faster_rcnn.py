from __future__ import annotations

import torch

from models.faster_rcnn import (
    RPN_FORCE_MATCH_TOPK,
    _assign_levels,
    assign_rpn_targets,
    build_faster_rcnn,
)


def test_fpn_level_assignment() -> None:
    """Check the canonical P3--P6 assignment boundaries."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 112.0, 112.0],
            [0.0, 0.0, 224.0, 224.0],
            [0.0, 0.0, 448.0, 448.0],
            [0.0, 0.0, 896.0, 896.0],
        ]
    )
    assert _assign_levels(boxes).tolist() == [0, 1, 2, 3]


def test_rpn_target_assignment() -> None:
    """Ensure each separated target receives forced positive RPN anchors."""
    anchors = torch.tensor(
        [
            [0.0, 0.0, 32.0, 32.0],
            [2.0, 0.0, 34.0, 32.0],
            [0.0, 2.0, 32.0, 34.0],
            [2.0, 2.0, 34.0, 34.0],
            [96.0, 96.0, 128.0, 128.0],
            [98.0, 96.0, 130.0, 128.0],
            [96.0, 98.0, 128.0, 130.0],
            [98.0, 98.0, 130.0, 130.0],
            [200.0, 200.0, 232.0, 232.0],
        ]
    )
    targets = torch.tensor(
        [
            [0.0, 0.0, 32.0, 32.0],
            [96.0, 96.0, 128.0, 128.0],
        ]
    )
    labels, matched_gt = assign_rpn_targets(anchors, targets)
    assert matched_gt is not None
    assert (labels == 1).sum().item() >= 2 * RPN_FORCE_MATCH_TOPK
    for target_index in range(targets.shape[0]):
        assert ((labels == 1) & (matched_gt == target_index)).sum().item() >= RPN_FORCE_MATCH_TOPK

    empty_labels, empty_matches = assign_rpn_targets(anchors, torch.zeros((0, 4)))
    assert empty_matches is None
    assert torch.equal(empty_labels, torch.zeros_like(empty_labels))


def test_model_train_validation_and_inference() -> None:
    """Exercise finite loss/backprop, frozen validation BatchNorm, and NMS output."""
    model = build_faster_rcnn(
        nc=2,
        scale="s",
        min_size=128,
        max_size=128,
        class_positive_weights=torch.tensor([1.0, 1.5]),
        score_threshold=0.001,
        nms_threshold=0.70,
        max_detections=30,
    )
    images = [torch.rand(3, 128, 128), torch.rand(3, 128, 128)]
    targets = [
        {
            "boxes": torch.tensor([[12.0, 15.0, 75.0, 92.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
        },
        {
            "boxes": torch.tensor([[25.0, 30.0, 100.0, 110.0]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.long),
        },
    ]

    model.train()
    losses = model(images, targets, compute_losses=True)
    assert set(losses) == {"loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg"}
    total_loss = sum(losses.values())
    assert torch.isfinite(total_loss)
    total_loss.backward()

    running_mean_before = model.backbone.stem[1].running_mean.detach().clone()
    model.eval()
    with torch.no_grad():
        validation_losses = model(images, targets, compute_losses=True)
        predictions = model(images)

    assert torch.isfinite(sum(validation_losses.values()))
    assert torch.equal(running_mean_before, model.backbone.stem[1].running_mean)
    assert len(predictions) == len(images)
    for prediction in predictions:
        assert set(prediction) == {"boxes", "labels", "scores"}
        assert prediction["boxes"].shape[1] == 4
        assert prediction["labels"].dtype == torch.long
        assert prediction["boxes"].shape[0] <= 30


def main() -> None:
    test_fpn_level_assignment()
    print("fpn_level_assignment: passed")
    test_rpn_target_assignment()
    print("rpn_target_assignment: passed")
    test_model_train_validation_and_inference()
    print("faster_rcnn_model_smoke: passed")


if __name__ == "__main__":
    main()
