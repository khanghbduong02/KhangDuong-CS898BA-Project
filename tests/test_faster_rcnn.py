from __future__ import annotations

import torch
from torchvision.models import resnet18

from inference_tta import merge_hflip_predictions
from models.faster_rcnn import (
    P2_ANCHOR_SIZES,
    P2_ANCHOR_STRIDES,
    RPN_FORCE_MATCH_TOPK,
    FPN,
    ResNetBackbone,
    _assign_levels,
    assign_rpn_targets,
    build_faster_rcnn,
    generate_anchors,
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


def test_torchvision_resnet_mapping() -> None:
    """Verify ResNet-18 state keys map exactly into the local small backbone."""
    source = resnet18(weights=None)
    backbone = ResNetBackbone([64, 128, 256, 512], [2, 2, 2, 2])
    backbone.load_torchvision_resnet_state(source.state_dict())
    assert torch.equal(backbone.stem[0].weight, source.conv1.weight)
    assert torch.equal(backbone.stage2[0].shortcut[0].weight, source.layer2[0].downsample[0].weight)


def test_p2_fpn_assignment_and_model() -> None:
    """Verify the optional small-object P2 path produces P2--P6 consistently."""
    backbone = ResNetBackbone([64, 128, 256, 512], [2, 2, 2, 2])
    fpn = FPN(backbone.out_channels_with_p2, use_p2=True)
    with torch.no_grad():
        c2, c3, c4, c5 = backbone(torch.rand(1, 3, 128, 128))
        features = fpn(c2, c3, c4, c5)
    assert [tuple(feature.shape[-2:]) for feature in features] == [
        (32, 32),
        (16, 16),
        (8, 8),
        (4, 4),
        (2, 2),
    ]
    anchors = generate_anchors(features, strides=P2_ANCHOR_STRIDES, sizes=P2_ANCHOR_SIZES)
    assert anchors.shape == (4092, 4)

    boxes = torch.tensor(
        [
            [0.0, 0.0, 32.0, 32.0],
            [0.0, 0.0, 112.0, 112.0],
            [0.0, 0.0, 224.0, 224.0],
            [0.0, 0.0, 448.0, 448.0],
            [0.0, 0.0, 896.0, 896.0],
        ]
    )
    assert _assign_levels(boxes, num_levels=5, min_level=2).tolist() == [0, 1, 2, 3, 4]

    model = build_faster_rcnn(
        nc=2,
        scale="s",
        min_size=128,
        max_size=128,
        use_p2=True,
        score_threshold=0.001,
        max_detections=30,
    )
    images = [torch.rand(3, 128, 128), torch.rand(3, 128, 128)]
    targets = [
        {
            "boxes": torch.tensor([[12.0, 15.0, 40.0, 45.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
        },
        {
            "boxes": torch.tensor([[25.0, 30.0, 60.0, 70.0]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.long),
        },
    ]
    model.train()
    losses = model(images, targets, compute_losses=True)
    assert torch.isfinite(sum(losses.values()))
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    assert model.inference_settings()["use_p2"] is True
    assert len(predictions) == len(images)


def test_hflip_prediction_merge() -> None:
    """Merge original and flipped per-class predictions without cross-class suppression."""
    original = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0]]),
            "scores": torch.tensor([0.9, 0.7]),
            "labels": torch.tensor([1, 2]),
        }
    ]
    flipped = [
        {
            "boxes": torch.tensor([[70.0, 10.0, 90.0, 30.0], [40.0, 10.0, 60.0, 30.0]]),
            "scores": torch.tensor([0.8, 0.6]),
            "labels": torch.tensor([1, 1]),
        }
    ]
    merged = merge_hflip_predictions(
        original,
        flipped,
        image_widths=[100],
        nms_iou=0.70,
        max_detections=300,
    )[0]
    assert merged["labels"].tolist() == [1, 2, 1]
    assert torch.allclose(merged["scores"], torch.tensor([0.9, 0.7, 0.6]))
    assert torch.allclose(
        merged["boxes"],
        torch.tensor([[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0], [40.0, 10.0, 60.0, 30.0]]),
    )


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
    test_torchvision_resnet_mapping()
    print("torchvision_resnet_mapping: passed")
    test_p2_fpn_assignment_and_model()
    print("p2_fpn_assignment_and_model: passed")
    test_hflip_prediction_merge()
    print("hflip_prediction_merge: passed")
    test_model_train_validation_and_inference()
    print("faster_rcnn_model_smoke: passed")


if __name__ == "__main__":
    main()
