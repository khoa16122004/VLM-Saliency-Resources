from __future__ import annotations

from email.mime import image
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from saliency import get_model


def _to_float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_pil(image_float: np.ndarray) -> Image.Image:
    clipped = np.clip(image_float * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clipped, mode="RGB")


def _safe_ratio(numer: float, denom: float, eps: float = 1e-8) -> float:
    return float(numer / max(denom, eps))


def _sample_position(
    allowed_mask: np.ndarray,
    patch_h: int,
    patch_w: int,
    rng: np.random.Generator,
    ) -> Tuple[int, int]:
    h, w = allowed_mask.shape
    valid: list[Tuple[int, int]] = []
    for y in range(0, h - patch_h + 1):
        for x in range(0, w - patch_w + 1):
            region = allowed_mask[y : y + patch_h, x : x + patch_w]
            if region.mean() > 0.8:
                valid.append((y, x))
    if valid:
        return valid[int(rng.integers(0, len(valid)))]
    # Fallback: if mask is too strict, sample anywhere.
    y = int(rng.integers(0, h - patch_h + 1))
    x = int(rng.integers(0, w - patch_w + 1))
    return y, x


def _eval_candidate(
	base_img: np.ndarray,
	patch: np.ndarray,
	pos: Tuple[int, int],
	text: str,
	runner,
	original_sim: float,
	sim_tol: float,
	sim_weight: float,
) -> Dict[str, Any]:
    y, x = pos
    ph, pw = patch.shape[:2]

    adv = base_img.copy()
    adv[y : y + ph, x : x + pw, :] = np.clip(adv[y : y + ph, x : x + pw, :] + patch, 0.0, 1.0)
    adv_pil = _to_pil(adv)

    out = runner(adv_pil, text)
    payload = out["results"][text]
    sim = float(payload["similarity"])
    sal = np.asarray(payload["map"], dtype=np.float32)

    patch_mass = float(sal[y : y + ph, x : x + pw].sum())
    total_mass = float(sal.sum())
    attract_ratio = _safe_ratio(patch_mass, total_mass)

    sim_drop = max(0.0, original_sim - sim)
    sim_abs = abs(sim - original_sim)
    feasible = sim_drop <= sim_tol
    score = attract_ratio - sim_weight * sim_abs

    return {
        "image": adv,
        "image_pil": adv_pil,
        "sim": sim,
        "sim_drop": sim_drop,
        "attract_ratio": attract_ratio,
        "score": score,
        "feasible": feasible,
        "saliency_map": sal,
    }


def search_patch_es(
    image: Image.Image,
    text: str,
    *,
    method_name: str = "gradeclip",
    model_name: str = "ViT-B/16",
    target_device: str = "cpu",
    patch_ratio: float = 0.18,
    steps: int = 120,
    sigma_init: float = 0.08,
    sim_tolerance: float = 0.02,
    sim_weight: float = 0.4,
    top_quantile: float = 0.80,
    seed: int = 0,
    ) -> Dict[str, Any]:
    """1+1 ES to find an additive patch that attracts saliency while preserving similarity.

    Inputs are only image and text; model/method and hyper-parameters are optional.
    """
    rng = np.random.default_rng(seed)
    runner = get_model(method_name, model_name=model_name, target_device=target_device)

    base_pil = image.convert("RGB")
    base_img = _to_float_rgb(base_pil)
    h, w = base_img.shape[:2]
    ph = max(8, int(h * patch_ratio))
    pw = max(8, int(w * patch_ratio))
    ph = min(ph, h)
    pw = min(pw, w)

    base_out = runner(base_pil, text)
    base_payload = base_out["results"][text]
    original_sim = float(base_payload["similarity"])
    base_sal = np.asarray(base_payload["map"], dtype=np.float32)

    thresh = float(np.quantile(base_sal, top_quantile))
    important = base_sal >= thresh
    allowed = ~important

    pos = _sample_position(allowed, ph, pw, rng)
    patch = rng.normal(0.0, sigma_init, size=(ph, pw, 3)).astype(np.float32)
    sigma = float(sigma_init)

    best = _eval_candidate(
        base_img=base_img,
        patch=patch,
        pos=pos,
        text=text,
        runner=runner,
        original_sim=original_sim,
        sim_tol=sim_tolerance,
        sim_weight=sim_weight,
    )

    history = []
    for step in range(steps):
        cand_patch = patch + rng.normal(0.0, sigma, size=patch.shape).astype(np.float32)
        cand_patch = np.clip(cand_patch, -0.5, 0.5)

        # Occasionally resample position to escape local traps.
        if step > 0 and step % 20 == 0:
            cand_pos = _sample_position(allowed, ph, pw, rng)
        else:
            cand_pos = pos

        cand = _eval_candidate(
            base_img=base_img,
            patch=cand_patch,
            pos=cand_pos,
            text=text,
            runner=runner,
            original_sim=original_sim,
            sim_tol=sim_tolerance,
            sim_weight=sim_weight,
        )

        accept = False
        if cand["feasible"] and not best["feasible"]:
            accept = True
        elif cand["feasible"] == best["feasible"] and cand["score"] > best["score"]:
            accept = True

        if accept:
            patch = cand_patch
            pos = cand_pos
            best = cand
            sigma *= 1.05
        else:
            sigma *= 0.99

        history.append(
            {
                "step": step,
                "score": float(best["score"]),
                "sim": float(best["sim"]),
                "sim_drop": float(best["sim_drop"]),
                "attract_ratio": float(best["attract_ratio"]),
                "sigma": float(sigma),
            }
        )

    y, x = pos
    return {
        "original_similarity": original_sim,
        "best_similarity": float(best["sim"]),
        "similarity_drop": float(best["sim_drop"]),
        "attract_ratio": float(best["attract_ratio"]),
        "score": float(best["score"]),
        "feasible": bool(best["feasible"]),
        "position": (int(y), int(x)),
        "patch_size": (int(ph), int(pw)),
        "patched_image": best["image_pil"],
        "saliency_map": best["saliency_map"],
        "history": history,
    }

