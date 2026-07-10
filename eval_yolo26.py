from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.yolo26_torch import build_yolo26
from train_yolo26 import E2EDetectLoss, STRIDES, YoloDetectionDataset, collate_fn, run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the custom YOLO26 model on valid or test split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved checkpoint")
    parser.add_argument("--data-root", type=Path, default=Path("processed-data/baseline"), help="Dataset variant root containing train/valid/test splits")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="valid", help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--fraction", type=float, default=1.0, help="Subset fraction for quick evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device, e.g. cuda or cuda:0")
    parser.add_argument("--scale", type=str, default="n", help="YOLO26 scale variant")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of detection classes")
    parser.add_argument("--box-gain", type=float, default=7.5, help="Box IoU loss multiplier")
    parser.add_argument("--cls-gain", type=float, default=0.5, help="Classification BCE loss multiplier")
    parser.add_argument("--reg-gain", type=float, default=1.5, help="Regression term multiplier")
    parser.add_argument("--one2many-topk", type=int, default=10, help="Task-aligned top-k for one-to-many assignment")
    parser.add_argument("--one2one-topk", type=int, default=1, help="Task-aligned top-k for one-to-one assignment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch.")

    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Expected split folder at {split_root}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = build_yolo26(nc=args.num_classes, scale=args.scale).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = YoloDetectionDataset(split_root, imgsz=args.imgsz, fraction=args.fraction)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

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
    loss, cls_loss, box_loss = run_epoch(
        model,
        loader,
        optimizer=None,
        scaler=scaler,
        criterion=criterion,
        device=device,
        nc=args.num_classes,
        use_amp=device.type == "cuda",
    )

    print(
        f"split={args.split} images={len(dataset)} loss={loss:.4f} cls_loss={cls_loss:.4f} box_loss={box_loss:.4f}"
    )


if __name__ == "__main__":
    main()