import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import get_imagenet_dataloader


@torch.no_grad()
def classify_with_class_name_similarity(
    model,
    loader,
    class_names: List[str],
    device: str,
) -> tuple[int, float, Dict[str, List[str]]]:
    valid_ids = [i for i, name in enumerate(class_names) if name]
    text_list = [class_names[i] for i in valid_ids]
    text_tokens = clip.tokenize(text_list).to(device)
    text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

    id_to_col = {class_id: col for col, class_id in enumerate(valid_ids)}
    correct_by_class: Dict[str, List[str]] = {name: [] for name in text_list}

    total = 0
    correct = 0

    for images, class_ids, class_names_batch, img_paths, _ in tqdm(loader):
        images = images.to(device, non_blocking=True)
        class_ids = torch.as_tensor(class_ids, device=device, dtype=torch.long)

        image_features = F.normalize(model.encode_image(images), dim=-1)
        logits = image_features @ text_features.T
        pred_cols = logits.argmax(dim=-1)

        gt_cols = torch.tensor([id_to_col.get(int(cid), -1) for cid in class_ids.tolist()], device=device, dtype=torch.long)
        mask = pred_cols.eq(gt_cols) & gt_cols.ge(0)

        if isinstance(class_names_batch, str):
            class_names_batch = [class_names_batch]
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        for idx in mask.nonzero(as_tuple=True)[0].tolist():
            class_name = str(class_names_batch[idx]).replace("_", " ")
            if class_name not in correct_by_class:
                correct_by_class[class_name] = []
            correct_by_class[class_name].append(os.path.abspath(img_paths[idx]))

        correct += int(mask.sum().item())
        total += images.size(0)

    acc = float(correct) / float(total) if total > 0 else 0.0
    return correct, acc, correct_by_class


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.pretrained, device=device)
    model.eval()

    loader, class_names = get_imagenet_dataloader(
        img_dir=args.img_dir,
        annotations_file=args.annotations_file,
        batch_size=args.batch_size,
        transform=preprocess,
        num_workers=args.num_workers,
        shuffle=False,
    )

    correct, acc, correct_by_class = classify_with_class_name_similarity(
        model=model,
        loader=loader,
        class_names=class_names,
        device=device,
    )

    print(f"Top-1 accuracy: {acc:.4f}")
    print(f"Number of correct samples: {correct}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.pretrained.replace("/", "")
    output_path = out_dir / f"clip_{model_name}_imagenet_correct_paths.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(correct_by_class, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP class-name similarity classification")
    parser.add_argument("--pretrained", type=str, default="ViT-B/32", help="CLIP backbone name")
    parser.add_argument("--img_dir", type=str, default="/datastore/elo/quanphm/dataset/ImageNet1K/val", help="ImageNet val folder")
    parser.add_argument("--annotations_file", type=str, default="imgnet1k_label.json", help="ImageNet folder-to-class mapping JSON")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--output_dir", type=str, default="right", help="Output directory for JSON")
    args = parser.parse_args()
    main(args)
