# VLM-Saliency-Resources

## Quick Python Flow

Huong dan nhanh de su dung pipeline saliency tu luc doc anh den luc lay ket qua.

### 1) Doc anh va tao runner

```python
from PIL import Image
from saliency import get_model, overlay_heatmap

# 1. Doc anh
image = Image.open("test_imgs/ostrich.jpg").convert("RGB")

# 2. Chon method + model
method_name = "gradeclip"   # vi du: gradeclip, maskclip, gradcam, selfattn, rollout, game, m2ib, clipsurgery
model_name = "ViT-B/16"     # hoac ViT-B/32, ViT-L/14, ViT-L/14@336px

# 3. Tao runner (instance class)
runner = get_model(method_name, model_name=model_name)
```

### 2) Chay va lay ket qua

```python
texts = ["ostrich", "sky", "tree"]

# Chay 1 lan
outputs = runner(image, texts)

print(outputs["method"])              # method dang dung
print(outputs["model"])               # model dang dung
print(outputs["processed_size"])      # kich thuoc anh sau resize theo model

for text, payload in outputs["results"].items():
	sim = payload["similarity"]        # similarity image-text
	sal = payload["map"]               # saliency map numpy [H, W]
	print(text, sim, sal.shape)
```

### 3) Luu overlay ra file

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

## Chay nhieu method trong 1 doan

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
		# mot so method can backend rieng (GAME, M2IB, CLIP_Surgery)
		print(f"skip {method_name}: {exc}")
```

## Ghi chu nhanh

- Anh duoc resize theo input size cua model ben trong runner, sau do toan bo flow (map, similarity, overlay) dung anh da resize.
- Neu dung methods can backend ngoai (GAME, M2IB, CLIP_Surgery), can clone/cai dat dung dependencies truoc.