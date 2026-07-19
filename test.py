from __future__ import annotations

import torch

from models.yolo26_torch import build_yolo26, class_aware_nms


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


def main() -> None:
	model = build_yolo26(nc=3, scale="n", topk=300)
	model.eval()

	x = torch.randn(1, 3, 640, 640)
	with torch.no_grad():
		outputs = model(x)

	print("one_to_many:", tuple(outputs["one_to_many"].shape))
	print("one_to_one:", tuple(outputs["one_to_one"].shape))
	test_class_aware_nms()
	print("class_aware_nms: passed")


if __name__ == "__main__":
	main()