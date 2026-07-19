"""Custom Faster R-CNN built entirely with local PyTorch modules.

No pretrained weights or high-level torchvision detector classes are used.
Only low-level operations from ``torchvision.ops`` are used for IoU, NMS,
RoI Align, and clipping. Training labels use standard zero-indexed YOLO class
IDs at the dataset boundary; this model internally reserves class ``0`` for
background and shifts foreground IDs by one.

Architecture
------------
Backbone : ResNet-style CNN with BasicBlocks   (scale: s / m / l)
Neck     : Feature Pyramid Network (FPN)       — 4 levels P3–P6
RPN      : Region Proposal Network             — 3 anchors / location / level
RoI      : RoI Align 7×7 (torchvision.ops)
Head     : 2× FC → cls (nc+1) + class-agnostic box regression
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, clip_boxes_to_image, nms, roi_align


# ---------------------------------------------------------------------------
# Scale configs: (channels per stage, blocks per stage)
# ---------------------------------------------------------------------------

SCALE_CONFIG: Dict[str, Tuple[List[int], List[int]]] = {
    "s": ([64, 128, 256, 512],  [2, 2, 2, 2]),   # ResNet-18 depth
    "m": ([64, 128, 256, 512],  [3, 4, 6, 3]),   # ResNet-50 depth
    "l": ([96, 192, 384, 768],  [3, 4, 6, 3]),   # wider channels
}

# FPN / anchor hyperparameters
FPN_CHANNELS   = 256
ANCHOR_STRIDES = [8,   16,   32,   64]   # feature stride per FPN level P3–P6
ANCHOR_SIZES   = [32,  64,  128,  256]   # base anchor size per level
ANCHOR_RATIOS  = [0.5, 1.0, 2.0]        # anchor height/width aspect ratios
NUM_ANCHORS    = len(ANCHOR_RATIOS)      # anchors per spatial location per level

# RPN hyperparameters
RPN_FG_IOU         = 0.7
RPN_BG_IOU         = 0.3
RPN_BATCH          = 256
RPN_POS_FRAC       = 0.5
RPN_PRE_NMS_TRAIN  = 2000
RPN_POST_NMS_TRAIN = 1000
RPN_PRE_NMS_TEST   = 1000
RPN_POST_NMS_TEST  = 300
RPN_NMS_THRESH     = 0.7

# Detection head hyperparameters
ROI_OUTPUT_SIZE = 7
ROI_SAMPLES     = 512
ROI_POS_FRAC    = 0.25
ROI_FG_IOU      = 0.5
BOX_WEIGHTS     = (10.0, 10.0, 5.0, 5.0)   # encode/decode weights for head

# Inference post-processing
DEFAULT_SCORE_THRESH   = 0.001
DEFAULT_NMS_THRESH     = 0.70
DEFAULT_DETECTIONS_MAX = 300


# ===========================================================================
# Backbone
# ===========================================================================

class BasicBlock(nn.Module):
    """Two-conv residual block without bottleneck."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut: Optional[nn.Sequential] = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out      = self.relu(self.bn1(self.conv1(x)))
        out      = self.bn2(self.conv2(out))
        identity = self.shortcut(x) if self.shortcut is not None else x
        return self.relu(out + identity)


def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock(in_ch, out_ch, stride)]
    for _ in range(1, n_blocks):
        layers.append(BasicBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class ResNetBackbone(nn.Module):
    """4-stage ResNet-style backbone; returns (C3, C4, C5) at strides 8, 16, 32."""

    def __init__(self, channels: List[int], blocks: List[int]) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels

        # Stem: stride-2 conv + stride-2 maxpool → total stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 7, 2, 3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.stage1 = _make_stage(c1, c1, blocks[0], stride=1)   # stride 4  (C2)
        self.stage2 = _make_stage(c1, c2, blocks[1], stride=2)   # stride 8  (C3)
        self.stage3 = _make_stage(c2, c3, blocks[2], stride=2)   # stride 16 (C4)
        self.stage4 = _make_stage(c3, c4, blocks[3], stride=2)   # stride 32 (C5)

        self.out_channels = [c2, c3, c4]   # channels of C3, C4, C5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.stem(x)
        x  = self.stage1(x)
        c3 = self.stage2(x)    # stride 8
        c4 = self.stage3(c3)   # stride 16
        c5 = self.stage4(c4)   # stride 32
        return c3, c4, c5


# ===========================================================================
# Feature Pyramid Network
# ===========================================================================

class FPN(nn.Module):
    """Top-down FPN producing P3, P4, P5, P6 with equal channel width."""

    def __init__(self, in_channels: List[int], out_ch: int = FPN_CHANNELS) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.smooth  = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, 1, 1) for _ in in_channels])
        self.p6_conv = nn.Conv2d(in_channels[-1], out_ch, 3, 2, 1)   # P6 from C5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor
    ) -> List[torch.Tensor]:
        l3, l4, l5 = self.lateral[0](c3), self.lateral[1](c4), self.lateral[2](c5)
        # Top-down merge
        t5 = l5
        t4 = l4 + F.interpolate(t5, size=l4.shape[-2:], mode="nearest")
        t3 = l3 + F.interpolate(t4, size=l3.shape[-2:], mode="nearest")
        return [
            self.smooth[0](t3),   # P3  stride 8
            self.smooth[1](t4),   # P4  stride 16
            self.smooth[2](t5),   # P5  stride 32
            self.p6_conv(c5),     # P6  stride 64
        ]


# ===========================================================================
# Anchor generation
# ===========================================================================

def _level_anchors(
    feat_h: int, feat_w: int, stride: int, size: int,
    ratios: List[float], device: torch.device,
) -> torch.Tensor:
    """XYXY anchors for one FPN level. Shape: (feat_h * feat_w * len(ratios), 4)."""
    half = size / 2.0
    base = torch.tensor(
        [[-half / math.sqrt(r), -half * math.sqrt(r),
           half / math.sqrt(r),  half * math.sqrt(r)] for r in ratios],
        dtype=torch.float32, device=device,
    )  # (A, 4)
    xs = (torch.arange(feat_w, device=device).float() + 0.5) * stride
    ys = (torch.arange(feat_h, device=device).float() + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    centres = torch.stack([gx, gy, gx, gy], dim=-1).reshape(-1, 4)   # (H*W, 4)
    return (centres.unsqueeze(1) + base.unsqueeze(0)).reshape(-1, 4)  # (H*W*A, 4)


def generate_anchors(
    features: List[torch.Tensor],
    strides:  List[int]   = ANCHOR_STRIDES,
    sizes:    List[int]   = ANCHOR_SIZES,
    ratios:   List[float] = ANCHOR_RATIOS,
) -> torch.Tensor:
    """All anchors concatenated across FPN levels. Shape: (N_total, 4) XYXY."""
    device = features[0].device
    parts  = []
    for feat, stride, size in zip(features, strides, sizes):
        _, _, fh, fw = feat.shape
        parts.append(_level_anchors(fh, fw, stride, size, ratios, device))
    return torch.cat(parts, dim=0)


# ===========================================================================
# Box encode / decode
# ===========================================================================

def encode_boxes(
    proposals: torch.Tensor,
    gt_boxes:  torch.Tensor,
    weights:   Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    wx, wy, ww, wh = weights
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
    gx = gt_boxes[:, 0] + 0.5 * gw
    gy = gt_boxes[:, 1] + 0.5 * gh
    return torch.stack([
        wx * (gx - px) / pw,
        wy * (gy - py) / ph,
        ww * torch.log(gw / pw),
        wh * torch.log(gh / ph),
    ], dim=1)


def decode_boxes(
    anchors:  torch.Tensor,
    deltas:   torch.Tensor,
    weights:  Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    clip_val: float = math.log(1000.0 / 16.0),
) -> torch.Tensor:
    wx, wy, ww, wh = weights
    pw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    ph = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    px = anchors[:, 0] + 0.5 * pw
    py = anchors[:, 1] + 0.5 * ph
    gx = px + pw * (deltas[:, 0] / wx)
    gy = py + ph * (deltas[:, 1] / wy)
    gw = pw * torch.exp((deltas[:, 2] / ww).clamp(max=clip_val))
    gh = ph * torch.exp((deltas[:, 3] / wh).clamp(max=clip_val))
    return torch.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], dim=1)


# ===========================================================================
# RPN
# ===========================================================================

class RPNHead(nn.Module):
    """Shared 3×3 conv → objectness logit + box-delta heads."""

    def __init__(self, in_ch: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.obj  = nn.Conv2d(in_ch, num_anchors, 1)
        self.box  = nn.Conv2d(in_ch, num_anchors * 4, 1)
        for m in [self.conv, self.obj, self.box]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        obj_maps, box_maps = [], []
        for f in features:
            t = F.relu(self.conv(f), inplace=True)
            obj_maps.append(self.obj(t))
            box_maps.append(self.box(t))
        return obj_maps, box_maps


class RPN(nn.Module):

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.head = RPNHead(in_ch, NUM_ANCHORS)

    # ---- proposal generation ----------------------------------------

    def _proposals_single(
        self,
        anchors:    torch.Tensor,
        obj_scores: torch.Tensor,
        box_deltas: torch.Tensor,
        img_h: int,
        img_w: int,
        training: bool,
    ) -> torch.Tensor:
        pre_nms  = RPN_PRE_NMS_TRAIN  if training else RPN_PRE_NMS_TEST
        post_nms = RPN_POST_NMS_TRAIN if training else RPN_POST_NMS_TEST

        scores = obj_scores.sigmoid()
        if scores.numel() > pre_nms:
            scores, idx = scores.topk(pre_nms)
            anchors    = anchors[idx]
            box_deltas = box_deltas[idx]

        proposals = decode_boxes(anchors, box_deltas)
        proposals = clip_boxes_to_image(proposals, (img_h, img_w))
        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]
        valid = (w >= 1.0) & (h >= 1.0)
        proposals, scores = proposals[valid], scores[valid]
        keep = nms(proposals, scores, RPN_NMS_THRESH)[:post_nms]
        return proposals[keep]

    # ---- RPN loss ---------------------------------------------------

    def _loss(
        self,
        anchors: torch.Tensor,
        obj_all: torch.Tensor,
        box_all: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = anchors.device
        B = len(targets)

        inside = (
            (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= img_w) & (anchors[:, 3] <= img_h)
        )
        valid_a   = anchors[inside]
        obj_valid = obj_all[:, inside]
        box_valid = box_all[:, inside]
        V = valid_a.shape[0]

        if V == 0:
            return obj_all.sum() * 0.0, box_all.sum() * 0.0

        loss_obj = anchors.new_zeros(())
        loss_box = anchors.new_zeros(())

        for i in range(B):
            gt       = targets[i]["boxes"].to(device)
            obj_i    = obj_valid[i]
            box_i    = box_valid[i]
            labels   = torch.full((V,), -1, dtype=torch.long, device=device)
            matched_gt: Optional[torch.Tensor] = None

            if gt.numel() > 0:
                iou = box_iou(valid_a, gt)
                max_iou, best_gt = iou.max(dim=1)
                matched_gt = best_gt
                labels[max_iou >= RPN_FG_IOU] = 1
                labels[max_iou <  RPN_BG_IOU] = 0
                labels[iou.argmax(dim=0)]      = 1   # guarantee each GT has anchor
            else:
                labels[:] = 0

            pos_idx = (labels == 1).nonzero(as_tuple=False).view(-1)
            neg_idx = (labels == 0).nonzero(as_tuple=False).view(-1)
            n_pos   = min(pos_idx.numel(), int(RPN_BATCH * RPN_POS_FRAC))
            n_neg   = min(neg_idx.numel(), RPN_BATCH - n_pos)

            if pos_idx.numel() > n_pos:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:n_pos]]
            if neg_idx.numel() > n_neg:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:n_neg]]

            sampled     = torch.cat([pos_idx, neg_idx])
            target_obj  = obj_i.new_zeros(sampled.numel())
            target_obj[:n_pos] = 1.0

            if sampled.numel() > 0:
                loss_obj = loss_obj + F.binary_cross_entropy_with_logits(
                    obj_i[sampled], target_obj, reduction="sum"
                ) / RPN_BATCH

            if n_pos > 0 and matched_gt is not None:
                loss_box = loss_box + F.smooth_l1_loss(
                    box_i[pos_idx],
                    encode_boxes(valid_a[pos_idx], gt[matched_gt[pos_idx]]),
                    beta=1.0 / 9, reduction="sum",
                ) / RPN_BATCH

        return loss_obj / B, loss_box / B

    # ---- forward ----------------------------------------------------

    def forward(
        self,
        features: List[torch.Tensor],
        anchors:  torch.Tensor,
        img_h:    int,
        img_w:    int,
        targets:  Optional[List[Dict[str, torch.Tensor]]] = None,
        compute_losses: bool = False,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        if compute_losses and targets is None:
            raise ValueError("RPN loss computation requires targets")

        obj_maps, box_maps = self.head(features)
        B = features[0].shape[0]

        obj_flat, box_flat = [], []
        for obj_m, box_m in zip(obj_maps, box_maps):
            _, A, H, W = obj_m.shape
            obj_flat.append(obj_m.permute(0, 2, 3, 1).reshape(B, -1))
            box_flat.append(box_m.permute(0, 2, 3, 1).reshape(B, H * W * A, 4))

        obj_all = torch.cat(obj_flat, dim=1)   # (B, N_total)
        box_all = torch.cat(box_flat, dim=1)   # (B, N_total, 4)

        proposals = [
            self._proposals_single(
                anchors,
                obj_all[i],
                box_all[i],
                img_h,
                img_w,
                training=compute_losses,
            )
            for i in range(B)
        ]

        losses: Dict[str, torch.Tensor] = {}
        if compute_losses and targets is not None:
            l_obj, l_box = self._loss(anchors, obj_all, box_all, targets, img_h, img_w)
            losses["loss_objectness"]  = l_obj
            losses["loss_rpn_box_reg"] = l_box

        return proposals, losses


# ===========================================================================
# RoI Align helper
# ===========================================================================

def _assign_levels(boxes: torch.Tensor, num_levels: int = 4) -> torch.Tensor:
    """Map boxes to P3--P6 using the canonical FPN scale-to-level formula."""
    areas  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    scales = areas.clamp(min=1e-6).sqrt()
    canonical_levels = torch.floor(4.0 + torch.log2(scales / 224.0 + 1e-6)).long()
    return (canonical_levels - 3).clamp(0, num_levels - 1)


def roi_align_multilevel(
    features:          List[torch.Tensor],
    proposals_per_img: List[torch.Tensor],
    strides:           List[int]  = ANCHOR_STRIDES,
    output_size:       int        = ROI_OUTPUT_SIZE,
) -> torch.Tensor:
    """Pool features from the matching FPN level for every proposal."""
    device = features[0].device
    C = features[0].shape[1]

    rois_list: List[torch.Tensor] = []
    for img_idx, props in enumerate(proposals_per_img):
        if props.numel() == 0:
            continue
        idx_col = props.new_full((props.shape[0], 1), float(img_idx))
        rois_list.append(torch.cat([idx_col, props], dim=1))

    if not rois_list:
        return features[0].new_zeros(0, C, output_size, output_size)

    rois   = torch.cat(rois_list, dim=0)      # (N_total, 5)
    boxes  = rois[:, 1:]
    levels = _assign_levels(boxes, num_levels=len(features))

    out = features[0].new_zeros(rois.shape[0], C, output_size, output_size)
    for lvl, (feat, stride) in enumerate(zip(features, strides)):
        mask = levels == lvl
        if not mask.any():
            continue
        out[mask] = roi_align(
            feat, rois[mask],
            output_size=(output_size, output_size),
            spatial_scale=1.0 / stride,
            sampling_ratio=2,
        )
    return out


# ===========================================================================
# Detection Head
# ===========================================================================

class DetectionHead(nn.Module):
    """Two-layer FC head: cls (nc+1) + class-agnostic box regression."""

    def __init__(self, in_ch: int, roi_size: int, num_classes: int) -> None:
        super().__init__()
        flat = in_ch * roi_size * roi_size
        self.fc1       = nn.Linear(flat, 1024)
        self.fc2       = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.box_pred  = nn.Linear(1024, 4)            # class-agnostic

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.box_pred.weight,  std=0.001)
        for layer in [self.fc1, self.fc2, self.cls_score, self.box_pred]:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x.flatten(1)), inplace=True)
        x = F.relu(self.fc2(x),            inplace=True)
        return self.cls_score(x), self.box_pred(x)   # (N, nc+1), (N, 4)


# ===========================================================================
# Full Faster R-CNN
# ===========================================================================

class FasterRCNN(nn.Module):

    def __init__(
        self,
        backbone:    ResNetBackbone,
        fpn:         FPN,
        rpn:         RPN,
        head:        DetectionHead,
        num_classes: int,
        min_size:    int = 640,
        max_size:    int = 640,
        class_weights: Optional[torch.Tensor] = None,
        score_threshold: float = DEFAULT_SCORE_THRESH,
        nms_threshold: float = DEFAULT_NMS_THRESH,
        max_detections: int = DEFAULT_DETECTIONS_MAX,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Faster R-CNN requires background plus at least one foreground class")
        self.backbone    = backbone
        self.fpn         = fpn
        self.rpn         = rpn
        self.head        = head
        self.num_classes = num_classes   # includes background (index 0)
        self.min_size    = min_size
        self.max_size    = max_size

        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        )
        self.register_buffer(
            "pixel_std",  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        )
        if class_weights is None:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32).reshape(-1)
        if class_weights.numel() != num_classes:
            raise ValueError(
                f"Expected {num_classes} classification weights, got {class_weights.numel()}"
            )
        if not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
            raise ValueError("Classification weights must be finite and greater than zero")
        self.register_buffer("classification_weights", class_weights)
        self.set_inference_settings(score_threshold, nms_threshold, max_detections)

    def set_inference_settings(
        self,
        score_threshold: float,
        nms_threshold: float,
        max_detections: int,
    ) -> None:
        """Set the per-class candidate filtering and NMS policy used in evaluation mode."""
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0.0 < nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be in (0, 1]")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_detections = int(max_detections)

    def inference_settings(self) -> Dict[str, float | int | str]:
        """Return serializable inference metadata for checkpoints and reports."""
        return {
            "postprocess": "per_class_nms",
            "score_threshold": self.score_threshold,
            "nms_threshold": self.nms_threshold,
            "max_detections": self.max_detections,
        }

    def _normalise(
        self, images: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, int, int]:
        if not images:
            raise ValueError("Faster R-CNN requires at least one image")
        expected_shape: Optional[Tuple[int, int]] = None
        for image in images:
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("Each image must have shape (3, height, width)")
            shape = (int(image.shape[1]), int(image.shape[2]))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError("All images in a batch must share the same spatial dimensions")

        normed = [(img - self.pixel_mean) / self.pixel_std for img in images]
        batch  = torch.stack(normed, dim=0)
        return batch, batch.shape[2], batch.shape[3]

    # ---- head training pass -------------------------------------------

    def _head_train(
        self,
        features:  List[torch.Tensor],
        proposals: List[torch.Tensor],
        targets:   List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        device = features[0].device
        sampled_props:  List[torch.Tensor] = []
        sampled_labels: List[torch.Tensor] = []
        sampled_deltas: List[torch.Tensor] = []

        for props, tgt in zip(proposals, targets):
            gt_boxes  = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device)

            if gt_boxes.numel() == 0 and props.numel() == 0:
                continue

            # Append GT boxes to guarantee positives
            if gt_boxes.numel() > 0:
                all_props = torch.cat([props, gt_boxes], dim=0) if props.numel() > 0 else gt_boxes
            else:
                all_props = props

            N = all_props.shape[0]
            labels   = torch.zeros(N, dtype=torch.long, device=device)
            box_tgt  = all_props.new_zeros(N, 4)
            best_gt: Optional[torch.Tensor] = None

            if gt_boxes.numel() > 0:
                iou = box_iou(all_props, gt_boxes)
                max_iou, best_gt = iou.max(dim=1)
                fg = max_iou >= ROI_FG_IOU
                labels[fg] = gt_labels[best_gt[fg]]

            fg_mask = labels > 0
            fg_idx  = fg_mask.nonzero(as_tuple=False).view(-1)
            bg_idx  = (~fg_mask).nonzero(as_tuple=False).view(-1)

            n_fg = min(fg_idx.numel(), int(ROI_SAMPLES * ROI_POS_FRAC))
            n_bg = min(bg_idx.numel(), ROI_SAMPLES - n_fg)

            if fg_idx.numel() > n_fg:
                fg_idx = fg_idx[torch.randperm(fg_idx.numel(), device=device)[:n_fg]]
            if bg_idx.numel() > n_bg:
                bg_idx = bg_idx[torch.randperm(bg_idx.numel(), device=device)[:n_bg]]

            keep = torch.cat([fg_idx, bg_idx])

            if n_fg > 0 and best_gt is not None:
                box_tgt[fg_idx] = encode_boxes(
                    all_props[fg_idx], gt_boxes[best_gt[fg_idx]],
                    weights=BOX_WEIGHTS,
                )

            sampled_props.append(all_props[keep])
            sampled_labels.append(labels[keep])
            sampled_deltas.append(box_tgt[keep])

        if not sampled_props:
            zero = features[0].sum() * 0.0
            return {"loss_classifier": zero, "loss_box_reg": zero}

        pooled     = roi_align_multilevel(features, sampled_props)
        cls_logits, box_preds = self.head(pooled)

        all_labels = torch.cat(sampled_labels)
        all_deltas = torch.cat(sampled_deltas)

        loss_cls = F.cross_entropy(
            cls_logits,
            all_labels,
            weight=self.classification_weights.to(dtype=cls_logits.dtype),
        )

        fg2 = all_labels > 0
        if fg2.sum() > 0:
            loss_box = F.smooth_l1_loss(
                box_preds[fg2], all_deltas[fg2], beta=1.0, reduction="mean"
            )
        else:
            loss_box = box_preds.sum() * 0.0

        return {"loss_classifier": loss_cls, "loss_box_reg": loss_box}

    # ---- inference post-processing ------------------------------------

    def _postprocess(
        self,
        features:  List[torch.Tensor],
        proposals: List[torch.Tensor],
        img_h:     int,
        img_w:     int,
    ) -> List[Dict[str, torch.Tensor]]:
        empty = {
            "boxes":  features[0].new_zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long, device=features[0].device),
            "scores": features[0].new_zeros(0),
        }
        if all(p.numel() == 0 for p in proposals):
            return [dict(empty) for _ in proposals]

        pooled = roi_align_multilevel(features, proposals)
        cls_logits, box_preds = self.head(pooled)
        cls_probs = F.softmax(cls_logits, dim=1)

        results: List[Dict[str, torch.Tensor]] = []
        offset = 0
        for props in proposals:
            n = props.shape[0]
            device = props.device

            if n == 0:
                results.append(dict(empty))
                continue

            probs  = cls_probs[offset:offset + n]
            deltas = box_preds[offset:offset + n]
            offset += n

            all_boxes:  List[torch.Tensor] = []
            all_scores: List[torch.Tensor] = []
            all_labels: List[torch.Tensor] = []

            for cls in range(1, self.num_classes):
                sc   = probs[:, cls]
                keep = sc > self.score_threshold
                if not keep.any():
                    continue
                boxes = decode_boxes(props[keep], deltas[keep], weights=BOX_WEIGHTS)
                boxes = clip_boxes_to_image(boxes, (img_h, img_w))
                sc_k  = sc[keep]
                knms  = nms(boxes, sc_k, self.nms_threshold)
                all_boxes.append(boxes[knms])
                all_scores.append(sc_k[knms])
                all_labels.append(
                    torch.full((knms.numel(),), cls, dtype=torch.long, device=device)
                )

            if all_boxes:
                b = torch.cat(all_boxes)
                s = torch.cat(all_scores)
                l = torch.cat(all_labels)
                if s.numel() > self.max_detections:
                    top = s.topk(self.max_detections).indices
                    b, s, l = b[top], s[top], l[top]
                results.append({"boxes": b, "scores": s, "labels": l})
            else:
                results.append(dict(empty))

        return results

    # ---- main forward -------------------------------------------------

    def forward(
        self,
        images:  List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        *,
        compute_losses: bool = False,
    ) -> Dict[str, torch.Tensor] | List[Dict[str, torch.Tensor]]:
        """Return losses when requested, otherwise class-aware-NMS detections.

        ``compute_losses=True`` is independent of ``self.training`` so callers
        can evaluate validation loss in ``eval`` mode without updating BatchNorm
        running statistics.
        """
        if compute_losses and targets is None:
            raise ValueError("Loss computation requires targets")
        batch, img_h, img_w = self._normalise(images)
        c3, c4, c5          = self.backbone(batch)
        features            = self.fpn(c3, c4, c5)
        anchors             = generate_anchors(features)

        proposals, rpn_losses = self.rpn(
            features, anchors, img_h, img_w,
            targets if compute_losses else None,
            compute_losses=compute_losses,
        )

        if compute_losses:
            assert targets is not None
            return {**rpn_losses, **self._head_train(features, proposals, targets)}

        return self._postprocess(features, proposals, img_h, img_w)


# ===========================================================================
# Factory
# ===========================================================================

def build_faster_rcnn(
    nc:       int,
    scale:    str = "m",
    min_size: int = 640,
    max_size: int = 640,
    class_positive_weights: Optional[torch.Tensor] = None,
    score_threshold: float = DEFAULT_SCORE_THRESH,
    nms_threshold: float = DEFAULT_NMS_THRESH,
    max_detections: int = DEFAULT_DETECTIONS_MAX,
) -> FasterRCNN:
    """Build a fully custom Faster R-CNN.

    Args:
        nc:       Number of *foreground* classes (0 = background).
        scale:    Model size — ``'s'``, ``'m'``, or ``'l'``.
        min_size: Minimum image side length.
        max_size: Maximum image side length.
        class_positive_weights: Optional positive-class weights, one per
            zero-indexed foreground class. Background always has weight 1.
        score_threshold: Candidate score floor before per-class NMS.
        nms_threshold: Per-class NMS IoU threshold.
        max_detections: Maximum detections retained per image after NMS.

    Returns:
        A :class:`FasterRCNN` with randomly initialised weights, no external
        downloads of any kind.
    """
    if nc <= 0:
        raise ValueError("nc must be positive")
    if min_size <= 0 or max_size <= 0:
        raise ValueError("min_size and max_size must be positive")
    if scale not in SCALE_CONFIG:
        raise ValueError(f"Unknown scale '{scale}'. Choose from {list(SCALE_CONFIG.keys())}.")

    if class_positive_weights is None:
        class_weights = torch.ones(nc + 1, dtype=torch.float32)
    else:
        foreground_weights = (
            torch.as_tensor(class_positive_weights)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1)
        )
        if foreground_weights.numel() != nc:
            raise ValueError(
                f"Expected {nc} foreground class weights, got {foreground_weights.numel()}"
            )
        class_weights = torch.cat((torch.ones(1, dtype=torch.float32), foreground_weights))

    channels, blocks = SCALE_CONFIG[scale]
    backbone = ResNetBackbone(channels, blocks)
    fpn      = FPN(backbone.out_channels, FPN_CHANNELS)
    rpn      = RPN(FPN_CHANNELS)
    head     = DetectionHead(FPN_CHANNELS, ROI_OUTPUT_SIZE, nc + 1)

    return FasterRCNN(
        backbone,
        fpn,
        rpn,
        head,
        num_classes=nc + 1,
        min_size=min_size,
        max_size=max_size,
        class_weights=class_weights,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        max_detections=max_detections,
    )
