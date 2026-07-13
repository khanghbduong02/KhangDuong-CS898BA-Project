from __future__ import annotations

import torch

from models.yolo26_torch import build_yolo26


def main() -> None:
	model = build_yolo26(nc=3, scale="n", topk=300)
	model.eval()

	x = torch.randn(1, 3, 640, 640)
	with torch.no_grad():
		outputs = model(x)

	print("one_to_many:", tuple(outputs["one_to_many"].shape))
	print("one_to_one:", tuple(outputs["one_to_one"].shape))


if __name__ == "__main__":
	main()