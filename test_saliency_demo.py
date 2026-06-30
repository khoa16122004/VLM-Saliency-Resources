from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from saliency import get_model, overlay_heatmap
from saliency.vlm_saliency import compute_similarity


def _fit_to_cell(img: Image.Image, cell_size: tuple[int, int]) -> Image.Image:
    cell_w, cell_h = cell_size
    ratio = min(cell_w / img.width, cell_h / img.height)
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
    ox = (cell_w - new_w) // 2
    oy = (cell_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def _build_comparison_grid(
    image: Image.Image,
    texts: list[str],
    methods: list[str],
    overlays: dict[str, dict[str, Image.Image]],
    save_path: Path,
) -> None:
    header_h = 58
    left_w = 210
    pad = 12
    cell_w, cell_h = 320, 240

    n_rows = len(texts)
    n_cols = 1 + len(methods)

    canvas_w = left_w + n_cols * cell_w + (n_cols + 2) * pad
    canvas_h = header_h + n_rows * cell_h + (n_rows + 2) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        header_font = ImageFont.truetype("arial.ttf", 28)
        row_font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        # Fallback when Arial is unavailable in the runtime environment.
        header_font = ImageFont.load_default()
        row_font = ImageFont.load_default()

    # Column headers
    col_titles = ["input"] + methods
    for c, title in enumerate(col_titles):
        x = left_w + pad * (c + 1) + cell_w * c
        y = pad
        draw.text((x + 6, y + 10), title, fill=(20, 20, 20), font=header_font)

    input_cell = _fit_to_cell(image, (cell_w, cell_h))

    # Rows: one text prompt per row
    for r, text in enumerate(texts):
        y = header_h + pad * (r + 1) + cell_h * r

        draw.text((pad + 6, y + cell_h // 2 - 14), text, fill=(20, 20, 20), font=row_font)

        x_input = left_w + pad
        canvas.paste(input_cell, (x_input, y))

        for c, method in enumerate(methods, start=1):
            x = left_w + pad * (c + 1) + cell_w * c
            ov = overlays[method][text]
            canvas.paste(_fit_to_cell(ov, (cell_w, cell_h)), (x, y))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    image_path = root / "test_imgs" / "cat_dog_car.jpg"
    output_root = root / "outputs" / "saliency_demo"
    output_root.mkdir(parents=True, exist_ok=True)

    texts = [
        "dog",
        'cat',
        "car",      
    ]
    methods = ["maskclip", "clipsurgery", "gradeclip"]

    image = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path}")
    print(f"Texts: {texts}")

    overlays: dict[str, dict[str, Image.Image]] = {}
    for method_name in methods:
        print(f"Running method: {method_name}")
        runner = get_model(method_name)
        outputs = runner(image, texts)

        overlays[method_name] = {}

        for text, payload in outputs["results"].items():
            sal = payload["map"]
            overlays[method_name][text] = overlay_heatmap(image, sal, channel="jet")

    shared_sim = compute_similarity(image=image, texts=texts)
    for text in texts:
        print(f"Shared CLIP similarity | text={text} | score={shared_sim[text]:.4f}")

    grid_path = output_root / "dog_car_column_comparison.png"
    _build_comparison_grid(image, texts, methods, overlays, grid_path)
    print(f"Saved comparison grid: {grid_path}")


if __name__ == "__main__":
    main()