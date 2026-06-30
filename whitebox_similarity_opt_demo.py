import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from tqdm import tqdm

from saliency import get_model, overlay_heatmap
from saliency.vlm_saliency import compute_similarity, initialize_backends, clipmodel, device
import clip


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def _tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).astype(np.uint8)
    return Image.fromarray(x)


def _clip_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def optimize_image_for_text(
    image: Image.Image,
    target_text: str,
    show_text: str,
    model_name: str,
    steps: int,
    lr: float,
    eps: float,
) -> Image.Image:
    initialize_backends(model_name=model_name)
    # `image` is expected to be pre-resized in main() to CLIP native resolution.

    x0 = _pil_to_tensor01(image).to(device)
    x = x0.clone().detach().requires_grad_(True)

    text_tokens = clip.tokenize([target_text, show_text]).to(device)
    with torch.no_grad():
        text_feats = F.normalize(clipmodel.encode_text(text_tokens), dim=-1)
        target_feat = text_feats[0:1]
        show_feat = text_feats[1:2]

    pbar = tqdm(range(steps), desc="White-box attack", unit="step")
    for _ in pbar:
        x_norm = _clip_normalize(x)

        img_feat = F.normalize(clipmodel.encode_image(x_norm), dim=-1)
        sim_target = (img_feat @ target_feat.T).mean()
        sim_show = (img_feat @ show_feat.T).mean()
        sim_total = sim_target + sim_show
        pbar.set_postfix(
            sim_target=f"{float(sim_target.detach()):.4f}",
            sim_show=f"{float(sim_show.detach()):.4f}",
            sim_total=f"{float(sim_total.detach()):.4f}",
        )

        if x.grad is not None:
            x.grad.zero_()
        sim_total.backward()

        with torch.no_grad():
            x += lr * x.grad.sign()
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = x.clamp(0.0, 1.0)
        x = x.detach().requires_grad_(True)

    return _tensor01_to_pil(x)


def _fit_to_cell(img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    ratio = min(cell_w / img.width, cell_h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
    ox = (cell_w - new_w) // 2
    oy = (cell_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def build_before_after_grid(
    image_before: Image.Image,
    image_after: Image.Image,
    texts: List[str],
    methods: List[str],
    overlays_before: Dict[str, Dict[str, Image.Image]],
    overlays_after: Dict[str, Dict[str, Image.Image]],
    out_file: Path,
) -> None:
    pad = 12
    header_h = 64
    left_w = 200
    cell_w, cell_h = 280, 210

    # Columns: input_before, input_after, and for each method (before, after)
    col_titles = ["input_before", "input_after"]
    for m in methods:
        col_titles.append(f"{m}_before")
        col_titles.append(f"{m}_after")

    n_rows = len(texts)
    n_cols = len(col_titles)

    canvas_w = left_w + n_cols * cell_w + (n_cols + 2) * pad
    canvas_h = header_h + n_rows * cell_h + (n_rows + 2) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        header_font = ImageFont.truetype("arial.ttf", 24)
        row_font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        header_font = ImageFont.load_default()
        row_font = ImageFont.load_default()

    for c, title in enumerate(col_titles):
        x = left_w + pad * (c + 1) + cell_w * c
        draw.text((x + 4, pad + 10), title, fill=(20, 20, 20), font=header_font)

    img_before_cell = _fit_to_cell(image_before, cell_w, cell_h)
    img_after_cell = _fit_to_cell(image_after, cell_w, cell_h)

    for r, text in enumerate(texts):
        y = header_h + pad * (r + 1) + cell_h * r
        draw.text((pad + 6, y + cell_h // 2 - 14), text, fill=(20, 20, 20), font=row_font)

        c0x = left_w + pad
        c1x = left_w + pad * 2 + cell_w
        canvas.paste(img_before_cell, (c0x, y))
        canvas.paste(img_after_cell, (c1x, y))

        col = 2
        for m in methods:
            xb = left_w + pad * (col + 1) + cell_w * col
            xa = left_w + pad * (col + 2) + cell_w * (col + 1)
            canvas.paste(_fit_to_cell(overlays_before[m][text], cell_w, cell_h), (xb, y))
            canvas.paste(_fit_to_cell(overlays_after[m][text], cell_w, cell_h), (xa, y))
            col += 2

    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="White-box optimize image-text similarity and compare saliency before/after")
    p.add_argument("--image", type=str, default="test_imgs/cat_dog_car.jpg")
    p.add_argument("--target-text", type=str, default="X")
    p.add_argument("--show-text", type=str, default="dog")
    p.add_argument("--methods", nargs="+", type=str, default=["maskclip", "clipsurgery", "gradeclip"])
    p.add_argument("--model", type=str, default="ViT-B/16")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--eps", type=float, default=0.08)
    p.add_argument("--output-dir", type=str, default="outputs/whitebox_similarity")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_input = Image.open(args.image).convert("RGB")

    initialize_backends(model_name=args.model)
    attack_res = int(clipmodel.visual.input_resolution)
    image_before = image_input.resize((attack_res, attack_res), Image.Resampling.BICUBIC)
    texts = [args.show_text]

    print(f"Attack target text: {args.target_text}")
    print(f"Show text: {args.show_text}")

    image_after = optimize_image_for_text(
        image=image_before,
        target_text=args.target_text,
        show_text=args.show_text,
        model_name=args.model,
        steps=args.steps,
        lr=args.lr,
        eps=args.eps,
    )

    print(f"Working resolution (attack + explain): {image_before.size[0]}x{image_before.size[1]}")

    sim_before = compute_similarity(image_before, texts, model_name=args.model)
    sim_after = compute_similarity(image_after, texts, model_name=args.model)

    print("Similarity before/after:")
    for t in texts:
        print(f"  text={t:>10s} | before={sim_before[t]:.4f} | after={sim_after[t]:.4f} | delta={sim_after[t]-sim_before[t]:+.4f}")

    overlays_before: Dict[str, Dict[str, Image.Image]] = {}
    overlays_after: Dict[str, Dict[str, Image.Image]] = {}

    for m in args.methods:
        runner = get_model(m, model_name=args.model)
        out_before = runner(image_before, texts)
        out_after = runner(image_after, texts)

        overlays_before[m] = {}
        overlays_after[m] = {}

        for text in texts:
            sal_b = out_before["results"][text]["map"]
            sal_a = out_after["results"][text]["map"]
            overlays_before[m][text] = overlay_heatmap(image_before, sal_b)
            overlays_after[m][text] = overlay_heatmap(image_after, sal_a)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    image_before.save(out_dir / f"{stem}_before.png")
    image_after.save(out_dir / f"{stem}_after.png")

    grid_path = out_dir / f"{stem}_whitebox_before_after_grid.png"
    build_before_after_grid(
        image_before=image_before,
        image_after=image_after,
        texts=texts,
        methods=list(args.methods),
        overlays_before=overlays_before,
        overlays_after=overlays_after,
        out_file=grid_path,
    )

    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
