from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image

from .vlm_saliency import build_method, compute_similarity


class SaliencyMethodRunner:
    def __init__(self, method_name: str, model_name: str = "ViT-B/16", target_device: Optional[str] = None):
        self.method_name = method_name
        self.model_name = model_name
        self.target_device = target_device
        self.method = build_method(method_name, model_name=model_name, device=target_device)

    def __call__(self, image: Image.Image | str | Path, texts: Sequence[str] | str) -> Dict[str, Any]:
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)

        maps = self.method.explain(image, text_list)
        sims = compute_similarity(
            image=image,
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
            "results": results,
        }


def get_model(method_name: str, model_name: str = "ViT-B/16", target_device: Optional[str] = None) -> SaliencyMethodRunner:
    return SaliencyMethodRunner(method_name=method_name, model_name=model_name, target_device=target_device)
