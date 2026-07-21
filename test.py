from __future__ import annotations

import torch

from models.yolo26_torch import DistributionIntegral, build_yolo26, class_aware_nms


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

	test_class_aware_nms()
	print("class_aware_nms: passed")
	test_distribution_integral()
	print("distribution_integral: passed")


if __name__ == "__main__":
	main()