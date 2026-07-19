from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


SCALE_CONFIG = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}
STRIDES = (8.0, 16.0, 32.0)


def class_aware_nms(
    decoded: torch.Tensor,
    num_classes: int,
    score_threshold: float = 0.001,
    iou_threshold: float = 0.70,
    max_detections: int = 300,
) -> list[torch.Tensor]:
    """Apply class-aware NMS to raw decoded one-to-one predictions.

    `decoded` has shape ``(batch, 4 + nc, anchors)`` and stores pixel-space
    ``xyxy`` boxes followed by sigmoid class scores. This deliberately runs
    before the legacy global top-k selection so a duplicated high-score anchor
    cannot consume the limited detection budget ahead of a distinct object.
    Each returned tensor has shape ``(detections, 6)`` with columns
    ``x1, y1, x2, y2, score, class_id``.
    """
    if decoded.ndim != 3 or decoded.shape[1] != 4 + num_classes:
        raise ValueError(
            f"Expected decoded predictions with shape (batch, {4 + num_classes}, anchors), "
            f"got {tuple(decoded.shape)}"
        )
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("NMS score threshold must be in [0, 1]")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be in (0, 1]")
    if max_detections <= 0:
        raise ValueError("NMS max detections must be positive")

    try:
        from torchvision.ops import batched_nms
    except ImportError as exc:
        raise RuntimeError(
            "Class-aware NMS requires torchvision.ops.batched_nms. Install a torchvision build matching PyTorch."
        ) from exc

    results: list[torch.Tensor] = []
    for image_predictions in decoded.float():
        boxes = image_predictions[:4].transpose(0, 1).contiguous()
        scores = image_predictions[4:]
        valid_boxes = (
            torch.isfinite(boxes).all(dim=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        candidate_indices = torch.nonzero(
            (scores >= score_threshold) & valid_boxes.unsqueeze(0),
            as_tuple=False,
        )

        if candidate_indices.numel() == 0:
            results.append(torch.zeros((0, 6), dtype=torch.float32, device=decoded.device))
            continue

        class_ids = candidate_indices[:, 0].to(dtype=torch.long)
        anchor_indices = candidate_indices[:, 1]
        candidate_boxes = boxes[anchor_indices]
        candidate_scores = scores[class_ids, anchor_indices]
        finite_scores = torch.isfinite(candidate_scores)
        candidate_boxes = candidate_boxes[finite_scores]
        candidate_scores = candidate_scores[finite_scores]
        class_ids = class_ids[finite_scores]

        if candidate_scores.numel() == 0:
            results.append(torch.zeros((0, 6), dtype=torch.float32, device=decoded.device))
            continue

        kept_indices = batched_nms(candidate_boxes, candidate_scores, class_ids, iou_threshold)[:max_detections]
        kept_boxes = candidate_boxes[kept_indices]
        kept_scores = candidate_scores[kept_indices].unsqueeze(1)
        kept_classes = class_ids[kept_indices].to(dtype=kept_boxes.dtype).unsqueeze(1)
        results.append(torch.cat((kept_boxes, kept_scores, kept_classes), dim=1))
    return results


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def make_divisible(v: float, divisor: int = 8) -> int:
    return int((v + divisor / 2) // divisor * divisor)


def depth_gain(n: int, depth_mult: float) -> int:
    return max(int(round(n * depth_mult)), 1)


def make_anchors(feats: List[torch.Tensor], strides: torch.Tensor, offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    anchor_points = []
    stride_tensor = []
    for feat, stride in zip(feats, strides):
        _, _, height, width = feat.shape
        sx = torch.arange(width, device=feat.device, dtype=feat.dtype) + offset
        sy = torch.arange(height, device=feat.device, dtype=feat.dtype) + offset
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((grid_x, grid_y), dim=0).reshape(2, -1))
        stride_tensor.append(torch.full((1, height * width), stride, device=feat.device, dtype=feat.dtype))
    return torch.cat(anchor_points, dim=1), torch.cat(stride_tensor, dim=1)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = False) -> torch.Tensor:
    left_top, right_bottom = distance[:, :2], distance[:, 2:]
    x1y1 = anchor_points - left_top
    x2y2 = anchor_points + right_bottom
    if xywh:
        center = (x1y1 + x2y2) / 2
        size = x2y2 - x1y1
        return torch.cat((center, size), dim=1)
    return torch.cat((x1y1, x2y2), dim=1)


class DistributionIntegral(nn.Module):
    """Convert four discrete distance distributions into expected box distances."""

    def __init__(self, reg_max: int) -> None:
        super().__init__()
        if reg_max <= 1:
            raise ValueError("DistributionIntegral requires reg_max greater than one")
        self.reg_max = reg_max
        self.register_buffer("project", torch.arange(reg_max, dtype=torch.float32), persistent=True)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Return expected left/top/right/bottom distances from `(B, 4*R, A)` logits."""
        batch_size, channels, anchors = distances.shape
        expected_channels = 4 * self.reg_max
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} distributional box channels, got {channels}"
            )
        probabilities = distances.view(batch_size, 4, self.reg_max, anchors).softmax(dim=2)
        project = self.project.to(dtype=distances.dtype).view(1, 1, self.reg_max, 1)
        return (probabilities * project).sum(dim=2)


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        d: int = 1,
        act: bool | nn.Module = True
    ) -> None:
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, k[0], 1)
        self.cv2 = Conv(hidden, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, hidden, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k(C3):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        hidden = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(hidden, hidden, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = max(1, num_heads)
        self.head_dim = dim // self.num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * self.num_heads
        hidden = dim + nh_kd * 2
        self.qkv = Conv(dim, hidden, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x)
        q, k, v = qkv.view(batch_size, self.num_heads, self.key_dim * 2 + self.head_dim, tokens).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q * self.scale).transpose(-2, -1) @ k
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(batch_size, channels, height, width)
        x = x + self.pe(v.reshape(batch_size, channels, height, width))
        return self.proj(x)


class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=max(1, num_heads))
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C3k2(C2f):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g, e=1.0),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g, e=1.0)
            for _ in range(n)
        )


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1, act=False)
        self.cv2 = Conv(hidden * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError("C2PSA expects matching input and output channels.")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Detect(nn.Module):
    dynamic = False
    max_det = 300

    def __init__(self, nc: int = 80, reg_max: int = 1, end2end: bool = True, ch: Tuple[int, int, int] = ()) -> None:
        super().__init__()
        if reg_max <= 0:
            raise ValueError("reg_max must be positive")
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + self.reg_max * 4
        self.end2end = end2end
        self.register_buffer("stride", torch.tensor(STRIDES, dtype=torch.float32), persistent=False)
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.shape: tuple[int, int, int, int] | None = None

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DistributionIntegral(reg_max) if reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

        self.bias_init()

    def forward_head(
        self,
        x: List[torch.Tensor],
        box_head: nn.ModuleList,
        cls_head: nn.ModuleList,
    ) -> Dict[str, torch.Tensor]:
        batch_size = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(batch_size, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(batch_size, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def raw_tensor(self, preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((preds["boxes"], preds["scores"]), dim=1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        one2many = self.forward_head(feats, self.cv2, self.cv3)
        detached_feats = [feat.detach() for feat in feats]
        one2one = self.forward_head(detached_feats, self.one2one_cv2, self.one2one_cv3)

        outputs: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            "one2many": one2many,
            "one2one": one2one,
            "one_to_many": self.raw_tensor(one2many),
        }

        if self.training:
            outputs["one_to_one"] = self.raw_tensor(one2one)
            return outputs  # type: ignore[return-value]

        decoded = self._inference(one2one)
        outputs["decoded"] = decoded
        outputs["one_to_one"] = self.postprocess(decoded.permute(0, 2, 1))
        return outputs  # type: ignore[return-value]

    def _inference(self, preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        boxes = self._get_decode_boxes(preds)
        return torch.cat((boxes, preds["scores"].sigmoid()), dim=1)

    def _get_decode_boxes(self, preds: Dict[str, torch.Tensor]) -> torch.Tensor:
        shape = preds["feats"][0].shape
        if self.dynamic or self.shape != shape:
            anchors, strides = make_anchors(preds["feats"], self.stride.to(preds["boxes"].device), 0.5)
            self.anchors = anchors
            self.strides = strides
            self.shape = shape

        distances = self.dfl(preds["boxes"])
        return dist2bbox(distances, self.anchors.unsqueeze(0), xywh=False) * self.strides.unsqueeze(0)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores = preds.split((4, self.nc), dim=-1)
        top_scores, class_ids, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat((boxes, top_scores, class_ids), dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, anchors, nc = scores.shape
        k = min(max_det, anchors)
        original_index = scores.max(dim=-1)[0].topk(k, dim=1)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=original_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = original_index[torch.arange(batch_size, device=scores.device)[..., None], index // nc]
        return scores.unsqueeze(-1), (index % nc).float().unsqueeze(-1), idx

    def bias_init(self) -> None:
        for stride, box_head, cls_head in zip(self.stride.tolist(), self.cv2, self.cv3):
            box_head[-1].bias.data[:] = 2.0
            cls_head[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

        if self.end2end:
            for stride, box_head, cls_head in zip(self.stride.tolist(), self.one2one_cv2, self.one2one_cv3):
                box_head[-1].bias.data[:] = 2.0
                cls_head[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


@dataclass
class YOLO26Config:
    nc: int = 3
    scale: str = "n"
    topk: int = 300
    reg_max: int = 1


class YOLO26(nn.Module):
    """Local YOLO26 reimplementation aligned to the official yolo26.yaml layout."""

    def __init__(self, cfg: YOLO26Config) -> None:
        super().__init__()
        if cfg.scale not in SCALE_CONFIG:
            valid = ", ".join(sorted(SCALE_CONFIG.keys()))
            raise ValueError(f"Unknown scale '{cfg.scale}'. Expected one of: {valid}")

        depth_mult, width_mult, max_channels = SCALE_CONFIG[cfg.scale]

        def width(channels: int) -> int:
            return min(make_divisible(channels * width_mult), max_channels)

        c1 = width(64)
        c2 = width(128)
        c3 = width(256)
        c4 = width(512)
        c5 = width(1024)

        n2 = depth_gain(2, depth_mult)
        n1 = depth_gain(1, depth_mult)

        self.b0 = Conv(3, c1, 3, 2)
        self.b1 = Conv(c1, c2, 3, 2)
        self.b2 = C3k2(c2, c3, n=n2, c3k=False, e=0.25)
        self.b3 = Conv(c3, c3, 3, 2)
        self.b4 = C3k2(c3, c4, n=n2, c3k=False, e=0.25)
        self.b5 = Conv(c4, c4, 3, 2)
        self.b6 = C3k2(c4, c4, n=n2, c3k=True)
        self.b7 = Conv(c4, c5, 3, 2)
        self.b8 = C3k2(c5, c5, n=n2, c3k=True)
        self.b9 = SPPF(c5, c5, 5, 3, True)
        self.b10 = C2PSA(c5, c5, n=n2)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.h13 = C3k2(c5 + c4, c4, n=n2, c3k=True)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.h16 = C3k2(c4 + c4, c3, n=n2, c3k=True)
        self.h17 = Conv(c3, c3, 3, 2)
        self.h19 = C3k2(c3 + c4, c4, n=n2, c3k=True)
        self.h20 = Conv(c4, c4, 3, 2)
        self.h22 = C3k2(c4 + c5, c5, n=n1, c3k=True, e=0.5, attn=True)

        self.detect = Detect(nc=cfg.nc, reg_max=cfg.reg_max, end2end=True, ch=(c3, c4, c5))
        self.detect.max_det = cfg.topk

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        x6 = self.b6(x5)
        x7 = self.b7(x6)
        x8 = self.b8(x7)
        x9 = self.b9(x8)
        x10 = self.b10(x9)

        x11 = self.up1(x10)
        x12 = torch.cat((x11, x6), dim=1)
        x13 = self.h13(x12)

        x14 = self.up2(x13)
        x15 = torch.cat((x14, x4), dim=1)
        x16 = self.h16(x15)

        x17 = self.h17(x16)
        x18 = torch.cat((x17, x13), dim=1)
        x19 = self.h19(x18)

        x20 = self.h20(x19)
        x21 = torch.cat((x20, x10), dim=1)
        x22 = self.h22(x21)

        return self.detect([x16, x19, x22])


def build_yolo26(nc: int = 3, scale: str = "n", topk: int = 300, reg_max: int = 1) -> YOLO26:
    return YOLO26(YOLO26Config(nc=nc, scale=scale, topk=topk, reg_max=reg_max))
