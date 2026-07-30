from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from inference_tta import unflip_decoded_xyxy
from models.yolo26_torch import DistributionIntegral, build_yolo26, class_aware_nms
from train_yolo26 import E2EDetectLoss, build_positive_class_weights, compute_loss


def test_class_aware_nms() -> None:
	"""Check duplicate same-class boxes are suppressed but different classes remain."""
	decoded = torch.tensor(
		[
			[
				[0.0, 0.0, 0.0],
				[0.0, 0.0, 0.0],
				[10.0, 10.0, 10.0],
				[10.0, 10.0, 10.0],
				[0.90, 0.80, 0.0001],
				[0.0001, 0.0001, 0.70],
			]
		],
		dtype=torch.float32,
	)
	detections = class_aware_nms(
		decoded,
		num_classes=2,
		score_threshold=0.001,
		iou_threshold=0.70,
		max_detections=300,
	)[0]
	assert detections.shape == (2, 6)
	assert torch.allclose(detections[:, 4], torch.tensor([0.90, 0.70]))
	assert detections[:, 5].long().tolist() == [0, 1]


def test_distribution_integral() -> None:
	"""Check DFL logits decode to their expected discrete distance bins."""
	reg_max = 4
	logits = torch.full((1, 4 * reg_max, 1), -20.0)
	for side, bin_index in enumerate((0, 1, 2, 3)):
		logits[0, side * reg_max + bin_index, 0] = 20.0
	distances = DistributionIntegral(reg_max)(logits)
	expected = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
	assert torch.allclose(distances, expected, atol=1e-4)


def test_hflip_decoded_unflip() -> None:
	"""Horizontal-flip decoded boxes must map back while retaining class scores."""
	flipped = torch.tensor(
		[[[70.0, 20.0], [10.0, 11.0], [90.0, 40.0], [30.0, 31.0], [0.9, 0.8], [0.1, 0.2]]]
	)
	restored = unflip_decoded_xyxy(flipped, image_width=100)
	expected = torch.tensor(
		[[[10.0, 60.0], [10.0, 11.0], [30.0, 80.0], [30.0, 31.0], [0.9, 0.8], [0.1, 0.2]]]
	)
	assert torch.equal(restored, expected)
	assert torch.equal(unflip_decoded_xyxy(restored, image_width=100), flipped)


def test_neutral_class_weights_allow_missing_smoke_classes() -> None:
	"""A neutral class-weight setting supports deliberately small smoke subsets."""
	with tempfile.TemporaryDirectory() as temporary_directory:
		labels_dir = Path(temporary_directory) / "labels"
		labels_dir.mkdir()
		image_path = Path(temporary_directory) / "sample.jpg"
		(labels_dir / "sample.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
		dataset = SimpleNamespace(image_paths=[image_path], labels_dir=labels_dir)
		weights, counts = build_positive_class_weights(dataset, num_classes=2, power=0.0)

	assert counts == [1, 0]
	assert torch.equal(weights, torch.ones(2))


def main() -> None:
	model = build_yolo26(nc=3, scale="n", topk=300)
	model.eval()

	x = torch.randn(1, 3, 640, 640)
	with torch.no_grad():
		outputs = model(x)

	print("one_to_many:", tuple(outputs["one_to_many"].shape))
	print("one_to_one:", tuple(outputs["one_to_one"].shape))
	assert outputs["one_to_many"].shape == (1, 7, 8400)
	assert outputs["one_to_one"].shape == (1, 300, 6)
	decoded_one2many = model.detect.decode_branch(outputs["one2many"])
	decoded_one2one = model.detect.decode_branch(outputs["one2one"])
	assert decoded_one2many.shape == (1, 7, 8400)
	assert torch.allclose(decoded_one2one, outputs["decoded"])

	dfl_model = build_yolo26(nc=5, scale="n", topk=300, reg_max=16)
	dfl_model.eval()
	with torch.no_grad():
		dfl_outputs = dfl_model(x)
	assert dfl_outputs["one_to_many"].shape == (1, 69, 8400)
	assert dfl_outputs["decoded"].shape == (1, 9, 8400)
	assert dfl_outputs["one_to_one"].shape == (1, 300, 6)
	assert torch.isfinite(dfl_outputs["decoded"]).all()
	assert dfl_model.detect.decode_branch(dfl_outputs["one2many"]).shape == (1, 9, 8400)
	print("distributional_head:", tuple(dfl_outputs["one_to_many"].shape))

	p2_model = build_yolo26(nc=3, scale="n", topk=300, use_p2=True)
	p2_model.eval()
	with torch.no_grad():
		p2_outputs = p2_model(x)
	assert tuple(p2_model.detect.stride.tolist()) == (4.0, 8.0, 16.0, 32.0)
	assert p2_outputs["one_to_many"].shape == (1, 7, 34000)
	assert p2_outputs["one_to_one"].shape == (1, 300, 6)
	assert p2_model.detect.decode_branch(p2_outputs["one2many"]).shape == (1, 7, 34000)
	print("p2_head:", tuple(p2_outputs["one_to_many"].shape))

	p2_model.train()
	p2_training_outputs = p2_model(x)
	p2_criterion = E2EDetectLoss(
		nc=3,
		strides=(4, 8, 16, 32),
		device=x.device,
		box_gain=7.5,
		cls_gain=0.5,
		reg_gain=1.5,
		one2many_topk=10,
		one2one_topk=1,
	)
	p2_loss = compute_loss(
		p2_training_outputs,
		[torch.tensor([[1.0, 0.5, 0.5, 0.1, 0.1]])],
		nc=3,
		criterion=p2_criterion,
		device=x.device,
	)
	assert torch.isfinite(p2_loss.total)
	p2_loss.total.backward()
	print("p2_training_loss: passed")

	test_class_aware_nms()
	print("class_aware_nms: passed")
	test_distribution_integral()
	print("distribution_integral: passed")
	test_hflip_decoded_unflip()
	print("hflip_decoded_unflip: passed")
	test_neutral_class_weights_allow_missing_smoke_classes()
	print("neutral_smoke_class_weights: passed")


if __name__ == "__main__":
	main()