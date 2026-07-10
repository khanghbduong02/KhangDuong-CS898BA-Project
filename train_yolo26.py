from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from models.yolo26_torch import build_yolo26


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STRIDES = (8, 16, 32)


class YoloDetectionDataset(Dataset):
    def __init__(self, split_root: Path, imgsz: int, fraction: float = 1.0) -> None:
        self.images_dir = split_root / "images"
        self.labels_dir = split_root / "labels"
        self.imgsz = imgsz

        image_paths = [
            path
            for path in sorted(self.images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if 0.0 < fraction < 1.0:
            keep = max(1, int(len(image_paths) * fraction))
            image_paths = image_paths[:keep]

        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        target_w = self.imgsz
        target_h = self.imgsz
        if target_w < original_w or target_h < original_h:
            interpolation = cv2.INTER_AREA
        elif target_w > original_w or target_h > original_h:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LINEAR

        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        labels = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                labels.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)
        return image_tensor, labels_tensor


def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def build_dense_targets(
    targets: Sequence[torch.Tensor],
    nc: int,
    imgsz: int,
    device: torch.device,
    assign_radius: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_sizes = [imgsz // stride for stride in STRIDES]
    offsets = []
    running = 0
    for grid_size in grid_sizes:
        offsets.append(running)
        running += grid_size * grid_size

    batch_size = len(targets)
    total_cells = running
    cls_target = torch.zeros((batch_size, total_cells, nc), device=device)
    box_target = torch.zeros((batch_size, total_cells, 4), device=device)
    pos_mask = torch.zeros((batch_size, total_cells), dtype=torch.bool, device=device)

    for batch_index, sample_targets in enumerate(targets):
        if sample_targets.numel() == 0:
            continue

        for target in sample_targets:
            cls_id = int(target[0].item())
            x_center, y_center, width, height = target[1:].tolist()
            max_side = max(width, height)

            if max_side < 0.10:
                scale_index = 0
            elif max_side < 0.25:
                scale_index = 1
            else:
                scale_index = 2

            grid_size = grid_sizes[scale_index]
            cell_x = min(grid_size - 1, max(0, int(x_center * grid_size)))
            cell_y = min(grid_size - 1, max(0, int(y_center * grid_size)))

            for dy in range(-assign_radius, assign_radius + 1):
                for dx in range(-assign_radius, assign_radius + 1):
                    neighbor_x = cell_x + dx
                    neighbor_y = cell_y + dy
                    if not (0 <= neighbor_x < grid_size and 0 <= neighbor_y < grid_size):
                        continue

                    flat_index = offsets[scale_index] + neighbor_y * grid_size + neighbor_x

                    if pos_mask[batch_index, flat_index] and cls_target[batch_index, flat_index, cls_id] == 0:
                        continue

                    pos_mask[batch_index, flat_index] = True
                    cls_target[batch_index, flat_index, cls_id] = 1.0
                    box_target[batch_index, flat_index] = torch.tensor([x_center, y_center, width, height], device=device)

    return cls_target, box_target, pos_mask


@dataclass
class LossBreakdown:
    total: torch.Tensor
    cls_loss: torch.Tensor
    box_loss: torch.Tensor
    reg_loss: torch.Tensor


def build_batch_targets(targets: Sequence[torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    batch_idx_list: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    bboxes_list: list[torch.Tensor] = []

    for batch_index, target in enumerate(targets):
        if target.numel() == 0:
            continue
        target = target.to(device)
        count = target.shape[0]
        batch_idx_list.append(torch.full((count,), batch_index, dtype=torch.long, device=device))
        cls_list.append(target[:, 0:1])
        bboxes_list.append(target[:, 1:5])

    if not batch_idx_list:
        return {
            "batch_idx": torch.zeros((0,), dtype=torch.long, device=device),
            "cls": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }

    return {
        "batch_idx": torch.cat(batch_idx_list, dim=0),
        "cls": torch.cat(cls_list, dim=0),
        "bboxes": torch.cat(bboxes_list, dim=0),
    }


class BranchDetectionLoss:
    def __init__(
        self,
        nc: int,
        strides: Tuple[int, int, int],
        device: torch.device,
        box_gain: float,
        cls_gain: float,
        reg_gain: float,
        tal_topk: int,
    ) -> None:
        self.device = device
        self.nc = nc
        self.reg_max = 1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.reg_gain = reg_gain
        self.stride = torch.tensor(strides, dtype=torch.float32, device=device)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=nc,
            alpha=0.5,
            beta=6.0,
            stride=list(strides),
        )
        self.bbox_loss = BboxLoss(self.reg_max).to(device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def __call__(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBreakdown:
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = dist2bbox(pred_distri, anchor_points, xywh=False)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp(min=1.0)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        box_loss = torch.zeros((), device=self.device)
        reg_loss = torch.zeros((), device=self.device)
        if fg_mask.sum():
            box_loss, reg_loss = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        total = self.box_gain * box_loss + self.cls_gain * cls_loss + self.reg_gain * reg_loss
        return LossBreakdown(
            total=total,
            cls_loss=(self.cls_gain * cls_loss).detach(),
            box_loss=(self.box_gain * box_loss).detach(),
            reg_loss=(self.reg_gain * reg_loss).detach(),
        )


class E2EDetectLoss:
    def __init__(
        self,
        nc: int,
        strides: Tuple[int, int, int],
        device: torch.device,
        box_gain: float,
        cls_gain: float,
        reg_gain: float,
        one2many_topk: int,
        one2one_topk: int,
    ) -> None:
        self.one2many = BranchDetectionLoss(nc, strides, device, box_gain, cls_gain, reg_gain, tal_topk=one2many_topk)
        self.one2one = BranchDetectionLoss(nc, strides, device, box_gain, cls_gain, reg_gain, tal_topk=one2one_topk)

    def __call__(self, outputs: dict[str, Any], batch: dict[str, torch.Tensor]) -> LossBreakdown:
        loss_many = self.one2many(outputs["one2many"], batch)
        loss_one = self.one2one(outputs["one2one"], batch)
        return LossBreakdown(
            total=loss_many.total + loss_one.total,
            cls_loss=loss_many.cls_loss + loss_one.cls_loss,
            box_loss=loss_many.box_loss + loss_one.box_loss,
            reg_loss=loss_many.reg_loss + loss_one.reg_loss,
        )


def compute_loss(
    outputs: dict[str, Any],
    targets: Sequence[torch.Tensor],
    nc: int,
    criterion: E2EDetectLoss,
    device: torch.device,
) -> LossBreakdown:
    if outputs.get("one2many") is None or outputs.get("one2one") is None:
        raise RuntimeError("Model outputs must include one2many and one2one branches for YOLO26-style E2E loss.")

    batch = build_batch_targets(targets, device=device)
    return criterion(outputs, batch)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    criterion: E2EDetectLoss,
    device: torch.device,
    nc: int,
    use_amp: bool,
) -> Tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_cls = 0.0
    total_box = 0.0
    num_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = [target.to(device) for target in targets]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = compute_loss(
                    outputs,
                    targets,
                    nc=nc,
                    criterion=criterion,
                    device=device,
                )

            if training:
                scaler.scale(loss.total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.total.detach().item())
        total_cls += float(loss.cls_loss.item())
        total_box += float((loss.box_loss + loss.reg_loss).item())
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return total_loss / num_batches, total_cls / num_batches, total_box / num_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom YOLO26 model from scratch.")
    parser.add_argument("--data-root", type=Path, default=Path("processed-data/baseline"), help="Dataset variant root containing train/valid/test splits")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="AdamW weight decay")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--fraction", type=float, default=1.0, help="Train/validation subset fraction for quick runs")
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. cuda or cuda:0")
    parser.add_argument("--scale", type=str, default="n", help="YOLO26 scale variant")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of detection classes")
    parser.add_argument("--box-gain", type=float, default=7.5, help="Box IoU loss multiplier")
    parser.add_argument("--cls-gain", type=float, default=0.5, help="Classification BCE loss multiplier")
    parser.add_argument("--reg-gain", type=float, default=1.5, help="Regression term multiplier (DFL-style slot; L1 when reg_max=1)")
    parser.add_argument("--one2many-topk", type=int, default=10, help="Task-aligned top-k for one-to-many assignment")
    parser.add_argument("--one2one-topk", type=int, default=1, help="Task-aligned top-k for one-to-one assignment")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/yolo26"), help="Output directory for checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch.")

    device = torch.device(args.device)
    train_root = args.data_root / "train"
    valid_root = args.data_root / "valid"
    if not train_root.exists() or not valid_root.exists():
        raise FileNotFoundError(f"Expected train/valid folders under {args.data_root}")

    train_dataset = YoloDetectionDataset(train_root, imgsz=args.imgsz, fraction=args.fraction)
    valid_dataset = YoloDetectionDataset(valid_root, imgsz=args.imgsz, fraction=args.fraction)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = build_yolo26(nc=args.num_classes, scale=args.scale).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = E2EDetectLoss(
        nc=args.num_classes,
        strides=STRIDES,
        device=device,
        box_gain=args.box_gain,
        cls_gain=args.cls_gain,
        reg_gain=args.reg_gain,
        one2many_topk=args.one2many_topk,
        one2one_topk=args.one2one_topk,
    )
    use_amp = device.type == "cuda"

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best.pt"
    last_path = args.save_dir / "last.pt"

    best_val = float("inf")
    print(f"Training on device={device}, train_images={len(train_dataset)}, valid_images={len(valid_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_cls, train_box = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            nc=args.num_classes,
            use_amp=use_amp,
        )
        val_loss, val_cls, val_box = run_epoch(
            model,
            valid_loader,
            optimizer=None,
            scaler=scaler,
            criterion=criterion,
            device=device,
            nc=args.num_classes,
            use_amp=use_amp,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_cls={train_cls:.4f} train_box={train_box:.4f} "
            f"val_loss={val_loss:.4f} val_cls={val_cls:.4f} val_box={val_box:.4f}"
        )

    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()