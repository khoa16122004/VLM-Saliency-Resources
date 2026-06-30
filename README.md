# VLM-Saliency-Resources

## Quick Python Flow

A quick guide to run the saliency pipeline from image loading to result export.

### 1) Load an image and create a runner

```python
from PIL import Image
from saliency import get_model, overlay_heatmap

# 1) Load image
image = Image.open("test_imgs/ostrich.jpg").convert("RGB")

# 2) Select method + model
method_name = "gradeclip"   # for example: gradeclip, maskclip, gradcam, selfattn, rollout, game, m2ib, clipsurgery
model_name = "ViT-B/16"     # or ViT-B/32, ViT-L/14, ViT-L/14@336px

# 3) Create runner instance
runner = get_model(method_name, model_name=model_name)
```

### 2) Run and get outputs

```python
texts = ["ostrich", "sky", "tree"]

# Run once
outputs = runner(image, texts)

print(outputs["method"])              # active method
print(outputs["model"])               # active model
print(outputs["processed_size"])      # resized image size used by the model

for text, payload in outputs["results"].items():
    sim = payload["similarity"]        # image-text similarity
    sal = payload["map"]               # saliency map as numpy array [H, W]
    print(text, sim, sal.shape)
```

### 3) Save overlay images

```python
from pathlib import Path

out_dir = Path("outputs/saliency_demo")
out_dir.mkdir(parents=True, exist_ok=True)

processed_image = outputs["processed_image"]

for text, payload in outputs["results"].items():
    sal = payload["map"]
    overlay = overlay_heatmap(processed_image, sal, channel="jet")
    safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    overlay.save(out_dir / f"overlay_{method_name}_{safe}.png")
```

## Run multiple methods in one script

```python
from PIL import Image
from saliency import get_model

image = Image.open("test_imgs/ostrich.jpg").convert("RGB")
texts = ["ostrich"]
model_name = "ViT-B/16"

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

for method_name in methods:
    try:
        runner = get_model(method_name, model_name=model_name)
        outputs = runner(image, texts)
        print(method_name, outputs["results"]["ostrich"]["similarity"])
    except Exception as exc:
        # some methods require extra backends (GAME, M2IB, CLIP_Surgery)
        print(f"skip {method_name}: {exc}")
```

## Notes

- The runner resizes the image to the model input size first, then all steps (map, similarity, overlay) run on that resized image.
- If you use methods with external backends (GAME, M2IB, CLIP_Surgery), install/clone those dependencies first.