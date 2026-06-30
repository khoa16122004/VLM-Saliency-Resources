from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from saliency import get_model, overlay_heatmap
from saliency.vlm_saliency import BackendNotAvailableError, compute_similarity


def _saliency_order(saliency_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = saliency_map.shape
    ys, xs = np.unravel_index(np.argsort(-saliency_map.reshape(-1)), (h, w))
    return ys, xs


def _deletion_image(base: np.ndarray, ys: np.ndarray, xs: np.ndarray, n_pixels: int) -> np.ndarray:
    out = base.copy()
    if n_pixels > 0:
        out[ys[:n_pixels], xs[:n_pixels], :] = 0
    return out


def _insertion_image(base: np.ndarray, ys: np.ndarray, xs: np.ndarray, n_pixels: int) -> np.ndarray:
    out = np.zeros_like(base)
    if n_pixels > 0:
        out[ys[:n_pixels], xs[:n_pixels], :] = base[ys[:n_pixels], xs[:n_pixels], :]
    return out


def _score_for_text(image_arr: np.ndarray, text: str, model_name: str) -> float:
    image = Image.fromarray(image_arr.astype(np.uint8))
    sims = compute_similarity(image=image, texts=[text], model_name=model_name)
    return float(sims[text])


def _concat_strip(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("images must be non-empty")
    return np.concatenate(images, axis=1)


def _plot_demo(
    method_order: list[str],
    deletion_samples_by_method: dict[str, list[np.ndarray]],
    insertion_samples_by_method: dict[str, list[np.ndarray]],
    deletion_scores_samples_by_method: dict[str, list[float]],
    insertion_scores_samples_by_method: dict[str, list[float]],
    sample_steps: list[int],
    deletion_curves: dict[str, list[float]],
    insertion_curves: dict[str, list[float]],
    target_text: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(20, 9), dpi=120)
    outer = fig.add_gridspec(2, 2, width_ratios=[3.1, 1.15], wspace=0.08, hspace=0.22)

    left_top = outer[0, 0].subgridspec(1, len(method_order), wspace=0.03)
    for i, method in enumerate(method_order):
        ax = fig.add_subplot(left_top[0, i])
        strip = _concat_strip(deletion_samples_by_method[method])
        ax.imshow(strip)
        last_score = deletion_scores_samples_by_method[method][-1]
        ax.set_title(f"{method} | Sim@last: {last_score:.3f}", fontsize=11, pad=4)
        ax.set_xlabel("Deletion steps: " + ", ".join(str(s) for s in sample_steps), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.33, 0.96, "Deletion Process", fontsize=16, ha="center")

    left_bottom = outer[1, 0].subgridspec(1, len(method_order), wspace=0.03)
    for i, method in enumerate(method_order):
        ax = fig.add_subplot(left_bottom[0, i])
        strip = _concat_strip(insertion_samples_by_method[method])
        ax.imshow(strip)
        last_score = insertion_scores_samples_by_method[method][-1]
        ax.set_title(f"{method} | Sim@last: {last_score:.3f}", fontsize=11, pad=4)
        ax.set_xlabel("Insertion steps: " + ", ".join(str(s) for s in sample_steps), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.33, 0.50, "Insertion Process", fontsize=16, ha="center")

    steps_all = np.arange(len(next(iter(deletion_curves.values()))))
    ax_del = fig.add_subplot(outer[0, 1])
    colors = {
        "selfattn": "#7f7f7f",
        "rollout": "#d62728",
        "gradcam": "#1f77b4",
        "game": "#17becf",
        "maskclip": "#9467bd",
        "clipsurgery": "#ff7f0e",
        "m2ib": "#bcbd22",
        "gradeclip_wo_ksim": "#8c564b",
        "gradeclip": "#2ca02c",
    }
    for method in method_order:
        ax_del.plot(
            steps_all,
            deletion_curves[method],
            color=colors.get(method, None),
            linewidth=2.0,
            label=method,
        )
    ax_del.set_title("(a) Deletion Step")
    ax_del.set_xlabel("Step")
    ax_del.set_ylabel(f"Similarity to '{target_text}'")
    ax_del.grid(True, alpha=0.3)
    ax_del.legend(fontsize=9)

    ax_ins = fig.add_subplot(outer[1, 1])
    for method in method_order:
        ax_ins.plot(
            steps_all,
            insertion_curves[method],
            color=colors.get(method, None),
            linewidth=2.0,
            label=method,
        )
    ax_ins.set_title("(b) Insertion Step")
    ax_ins.set_xlabel("Step")
    ax_ins.set_ylabel(f"Similarity to '{target_text}'")
    ax_ins.grid(True, alpha=0.3)
    ax_ins.legend(fontsize=9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_three_row_comparison(
    method_order: list[str],
    image_resized: np.ndarray,
    saliency_maps: dict[str, np.ndarray],
    similarity_scores: dict[str, float],
    contrast_scores: dict[str, float],
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 10), dpi=120)
    grid = fig.add_gridspec(len(method_order), 3, wspace=0.05, hspace=0.22)

    base_pil = Image.fromarray(image_resized.astype(np.uint8))
    for r, method in enumerate(method_order):
        hm = saliency_maps[method]
        overlay = np.asarray(overlay_heatmap(base_pil, hm, channel="jet"), dtype=np.uint8)

        ax0 = fig.add_subplot(grid[r, 0])
        ax0.imshow(image_resized)
        ax0.set_title(f"{method} | input", fontsize=11)
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = fig.add_subplot(grid[r, 1])
        ax1.imshow(hm, cmap="jet", vmin=0.0, vmax=1.0)
        ax1.set_title(
            f"heatmap | sim={similarity_scores[method]:.3f} | contrast={contrast_scores[method]:.3f}",
            fontsize=10,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(grid[r, 2])
        ax2.imshow(overlay)
        ax2.set_title("overlay", fontsize=11)
        ax2.set_xticks([])
        ax2.set_yticks([])

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    image_path = root / "test_imgs" / "ostrich.jpg"
    output_root = root / "outputs" / "saliency_demo"
    output_root.mkdir(parents=True, exist_ok=True)

    target_text = "sky"
    methods = [
        "selfattn",
        "rollout",
        "gradcam",
        "game",
        "maskclip",
        "clipsurgery",
        "m2ib",
        "gradeclip_wo_ksim",
        "gradeclip",
    ]
    model_name = "ViT-B/16"
    total_steps = 50
    sample_gap = 10

    image_raw = Image.open(image_path).convert("RGB")
    deletion_curves: dict[str, list[float]] = {}
    insertion_curves: dict[str, list[float]] = {}
    deletion_samples_by_method: dict[str, list[np.ndarray]] = {}
    insertion_samples_by_method: dict[str, list[np.ndarray]] = {}
    deletion_scores_samples_by_method: dict[str, list[float]] = {}
    insertion_scores_samples_by_method: dict[str, list[float]] = {}
    saliency_maps_by_method: dict[str, np.ndarray] = {}
    similarity_scores_by_method: dict[str, float] = {}
    contrast_scores_by_method: dict[str, float] = {}
    sample_steps = list(range(0, total_steps + 1, sample_gap))

    outputs_ref = None
    available_methods: list[str] = []
    for method_name in methods:
        try:
            runner = get_model(method_name, model_name=model_name)
            outputs = runner(image_raw, [target_text])
        except (BackendNotAvailableError, RuntimeError, ModuleNotFoundError) as exc:
            print(f"[skip] {method_name}: {exc}")
            continue

        available_methods.append(method_name)
        if outputs_ref is None:
            outputs_ref = outputs

        image_resized = np.asarray(outputs["processed_image"], dtype=np.uint8)
        saliency_map = np.asarray(outputs["results"][target_text]["map"], dtype=np.float32)
        saliency_map = np.nan_to_num(saliency_map, nan=0.0, posinf=1.0, neginf=0.0)
        saliency_map = np.clip(saliency_map, 0.0, 1.0)
        saliency_maps_by_method[method_name] = saliency_map
        similarity_scores_by_method[method_name] = float(outputs["results"][target_text]["similarity"])
        contrast_scores_by_method[method_name] = float(np.std(saliency_map))

        ys, xs = _saliency_order(saliency_map)
        area = saliency_map.shape[0] * saliency_map.shape[1]
        pixels_per_step = max(1, area // total_steps)

        deletion_curve: list[float] = []
        insertion_curve: list[float] = []
        deletion_samples: list[np.ndarray] = []
        insertion_samples: list[np.ndarray] = []
        deletion_scores_samples: list[float] = []
        insertion_scores_samples: list[float] = []

        for step in range(total_steps + 1):
            n_pixels = min(area, step * pixels_per_step)

            img_del = _deletion_image(image_resized, ys, xs, n_pixels)
            img_ins = _insertion_image(image_resized, ys, xs, n_pixels)

            score_del = _score_for_text(img_del, target_text, model_name)
            score_ins = _score_for_text(img_ins, target_text, model_name)

            deletion_curve.append(score_del)
            insertion_curve.append(score_ins)

            if step in sample_steps:
                deletion_samples.append(img_del)
                insertion_samples.append(img_ins)
                deletion_scores_samples.append(score_del)
                insertion_scores_samples.append(score_ins)

        deletion_curves[method_name] = deletion_curve
        insertion_curves[method_name] = insertion_curve
        deletion_samples_by_method[method_name] = deletion_samples
        insertion_samples_by_method[method_name] = insertion_samples
        deletion_scores_samples_by_method[method_name] = deletion_scores_samples
        insertion_scores_samples_by_method[method_name] = insertion_scores_samples

    if not available_methods:
        raise RuntimeError("No method could run. Please check backend dependencies.")

    out_path = output_root / "deletion_insertion_similarity_demo.png"
    _plot_demo(
        method_order=available_methods,
        deletion_samples_by_method=deletion_samples_by_method,
        insertion_samples_by_method=insertion_samples_by_method,
        deletion_scores_samples_by_method=deletion_scores_samples_by_method,
        insertion_scores_samples_by_method=insertion_scores_samples_by_method,
        sample_steps=sample_steps,
        deletion_curves=deletion_curves,
        insertion_curves=insertion_curves,
        target_text=target_text,
        save_path=out_path,
    )

    comparison_path = output_root / "comparison_three_rows.png"
    _plot_three_row_comparison(
        method_order=available_methods,
        image_resized=np.asarray(outputs_ref["processed_image"], dtype=np.uint8),
        saliency_maps=saliency_maps_by_method,
        similarity_scores=similarity_scores_by_method,
        contrast_scores=contrast_scores_by_method,
        save_path=comparison_path,
    )

    print(f"Image: {image_path}")
    print(f"Methods requested: {methods}")
    print(f"Methods used: {available_methods}")
    print(f"Model: {model_name}")
    print(f"Model input size: {outputs_ref['model_input_size']}")
    print(f"Processed image size: {outputs_ref['processed_size']}")
    print("Heatmap contrast (std):")
    for method_name in available_methods:
        print(f"  - {method_name}: {contrast_scores_by_method[method_name]:.4f}")
    print(f"Saved demo figure: {out_path}")
    print(f"Saved 3-row comparison: {comparison_path}")


if __name__ == "__main__":
    main()
