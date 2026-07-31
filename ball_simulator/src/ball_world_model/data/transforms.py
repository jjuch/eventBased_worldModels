from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import v2

from ball_world_model.config import ImageConfig


class FrameTransform:
    def __init__(self, config: ImageConfig) -> None:
        operations: list[object] = [
            v2.ToImage(),
            v2.Resize((config.height, config.width), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]

        if config.normalization == "imagenet":
            operations.append(
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                )
            )

        self.transform = v2.Compose(operations)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB"))


def denormalise_rgb(tensor: torch.Tensor, normalisation: str) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if normalisation == "imagenet":
        mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
        tensor = tensor * std + mean
    return tensor.clamp(0.0, 1.0)