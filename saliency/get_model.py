from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image

from .vlm_saliency import build_method, compute_similarity


def _load_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def _resize_image_for_model(image: Image.Image, target_size: int) -> Image.Image:
    if image.size == (target_size, target_size):
        return image
    return image.resize((target_size, target_size), Image.Resampling.BICUBIC)


def _infer_model_input_size(model_name: str) -> int:
    # CLIP defaults: most variants are 224, ViT-L/14@336px uses 336.
    name = model_name.strip()
    if "@336" in name:
        return 336
    return 224


class SaliencyMethodRunner:
    def __init__(
        self,
        method_name: str,
        model_name: str = "ViT-B/16",
        target_device: Optional[str] = None,
        resize_input_size: Optional[int] = None,
    ):
        self.method_name = method_name
        self.model_name = model_name
        self.target_device = target_device
        self.resize_input_size = resize_input_size if resize_input_size is not None else _infer_model_input_size(model_name)
        self.method = build_method(method_name, model_name=model_name, device=target_device)

    def __call__(self, image: Image.Image | str | Path, texts: Sequence[str] | str) -> Dict[str, Any]:
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)

        original_image = _load_image(image)
        processed_image = _resize_image_for_model(original_image, self.resize_input_size)

        maps = self.method.explain(processed_image, text_list)
        sims = compute_similarity(
            image=processed_image,
            texts=text_list,
            model_name=self.model_name,
            target_device=self.target_device,
        )

        results: Dict[str, Dict[str, float | np.ndarray]] = {}
        for text in text_list:
            results[text] = {
                "similarity": float(sims[text]),
                "map": maps[text],
            }

        return {
            "method": self.method_name,
            "model": self.model_name,
            "model_input_size": self.resize_input_size,
            "original_size": original_image.size,
            "processed_size": processed_image.size,
            "processed_image": processed_image,
            "results": results,
        }


def get_model(
    method_name: str,
    model_name: str = "ViT-B/16",
    target_device: Optional[str] = None,
    resize_input_size: Optional[int] = None,
) -> SaliencyMethodRunner:
    return SaliencyMethodRunner(
        method_name=method_name,
        model_name=model_name,
        target_device=target_device,
        resize_input_size=resize_input_size,
    )
