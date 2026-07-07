import argparse
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize as np_resize
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

import clip

try:
    import Game_MM_CLIP.clip as mm_clip
except Exception:
    mm_clip = None

try:
    import CLIP_Surgery.clip as surgery_clip
except Exception:
    surgery_clip = None

ClipWrapper = None
vision_heatmap_iba = None


def _try_import_m2ib_symbols() -> Tuple[Optional[object], Optional[object]]:
    # 1) Try package-style import first.
    try:
        from M2IB.scripts.clip_wrapper import ClipWrapper as _ClipWrapper
        from M2IB.scripts.methods import vision_heatmap_iba as _vision_heatmap_iba

        return _ClipWrapper, _vision_heatmap_iba
    except Exception:
        pass

    # 2) Try local-clone layout where modules import via `from scripts...`.
    # Add <workspace>/M2IB to sys.path so `import scripts.*` resolves.
    candidates = [
        Path.cwd() / "M2IB",
        Path(__file__).resolve().parents[1] / "M2IB",
    ]
    for m2ib_root in candidates:
        if not (m2ib_root / "scripts").exists():
            continue
        m2ib_root_str = str(m2ib_root)
        if m2ib_root_str not in sys.path:
            sys.path.insert(0, m2ib_root_str)
        try:
            from scripts.clip_wrapper import ClipWrapper as _ClipWrapper
            from scripts.methods import vision_heatmap_iba as _vision_heatmap_iba

            return _ClipWrapper, _vision_heatmap_iba
        except Exception:
            continue

    return None, None


ClipWrapper, vision_heatmap_iba = _try_import_m2ib_symbols()

try:
    from transformers import CLIPTokenizerFast
except Exception:
    CLIPTokenizerFast = None


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_transform = Compose([ToTensor(), Normalize(_CLIP_MEAN, _CLIP_STD)])


# Global backends to stay close to generate_emap.py usage style.
device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel = None
preprocess = None
mm_clipmodel = None
surgery_model = None
m2ib_model = None
clip_tokenizer = None


class BackendNotAvailableError(RuntimeError):
    pass


def _require_backend(obj, backend_name: str, install_hint: str) -> None:
    if obj is None:
        raise BackendNotAvailableError(f"{backend_name} backend is not available. {install_hint}")


def initialize_backends(
    model_name: str = "ViT-B/16",
    target_device: Optional[str] = None,
    load_game: bool = False,
    load_surgery: bool = False,
    load_m2ib: bool = False,
) -> None:
    global device, clipmodel, preprocess, mm_clipmodel, surgery_model, m2ib_model, clip_tokenizer

    device = target_device or ("cuda" if torch.cuda.is_available() else "cpu")

    if clipmodel is None or preprocess is None:
        clipmodel, preprocess = clip.load(model_name, device=device)
        clipmodel.eval()

    if load_game and mm_clipmodel is None:
        _require_backend(mm_clip, "GAME", "Clone Transformer-MM-Explainability and expose Game_MM_CLIP on PYTHONPATH.")
        mm_clipmodel, _ = mm_clip.load(model_name, device=device, jit=False)
        mm_clipmodel.eval()

    if load_surgery and surgery_model is None:
        _require_backend(surgery_clip, "CLIPSurgery", "Clone CLIP_Surgery and expose it on PYTHONPATH.")
        surgery_name = f"CS-{model_name}"
        surgery_model, _ = surgery_clip.load(surgery_name, device=device)
        surgery_model.eval()

    if load_m2ib and m2ib_model is None:
        _require_backend(ClipWrapper, "M2IB", "Clone M2IB repo and expose M2IB.scripts on PYTHONPATH.")
        _require_backend(CLIPTokenizerFast, "M2IB tokenizer", "Install transformers: pip install transformers")
        m2ib_model = ClipWrapper(clipmodel)
        clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")


def imgprocess_keepsize(img: Image.Image, patch_size: Tuple[int, int] = (16, 16), scale_factor: float = 1.0) -> torch.Tensor:
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph
    img = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)(img).convert("RGB")
    return _transform(img)


def generate_masks(input_size, N, s, p1):
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype("float32")

    masks = np.empty((N, *input_size))
    for i in range(N):
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        masks[i, :, :] = np_resize(grid[i], up_size, order=1, mode="reflect", anti_aliasing=False)[
            x : x + input_size[0], y : y + input_size[1]
        ]
    masks = masks.reshape(-1, 1, *input_size)
    return torch.tensor(masks)


def rise(model, image, txt_embedding, target_device, N=2000, s=8, p1=0.5):
    input_size = image.shape[-2:]
    masks = generate_masks(input_size, N, s, p1)
    batch_size = 50
    preds = []
    masked = image * masks
    with torch.no_grad():
        for i in range(0, N, batch_size):
            image_features = model.encode_image(masked[i : min(i + batch_size, N)].to(target_device))
            image_features = F.normalize(image_features, dim=-1)
            preds.append((image_features @ txt_embedding.T).cpu())
            del image_features
    preds = torch.cat(preds, dim=0)
    sal = (preds * masks.reshape(N, -1)).sum(0).reshape(*input_size)
    sal = sal / N / p1
    return sal


def m2ib_clip_map(model, image, texts, target_device, vbeta=0.1, vvar=1, vlayer=9, tbeta=0.1, tvar=1, tlayer=9):
    del tbeta, tvar, tlayer
    _require_backend(vision_heatmap_iba, "M2IB", "Clone M2IB repo and expose M2IB.scripts on PYTHONPATH.")
    _require_backend(clip_tokenizer, "M2IB tokenizer", "Install transformers: pip install transformers")
    text_ids = torch.tensor([clip_tokenizer.encode(texts, add_special_tokens=True)]).to(target_device)
    vmap = vision_heatmap_iba(text_ids, image, model, vlayer, vbeta, vvar)
    return vmap


def clip_surgery_map(model, image, texts, target_device):
    _require_backend(surgery_clip, "CLIPSurgery", "Clone CLIP_Surgery and expose it on PYTHONPATH.")
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = surgery_clip.encode_text_with_prompt_ensemble(model, texts, target_device)
        similarity = surgery_clip.clip_feature_surgery(image_features, text_features)
        similarity_map = surgery_clip.get_similarity_map(similarity[:, 1:, :], image.shape[-2:])
    return similarity_map


def mm_interpret(image, texts, model, target_device, start_layer=-1, start_layer_text=-1, flag="image", rollout=False):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    del logits_per_text
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(target_device) * logits_per_image)
    model.zero_grad()

    if flag == "image":
        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

        if start_layer == -1:
            start_layer = len(image_attn_blocks) - 1

        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(target_device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        attentions = []
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            avg_heads = (cam.sum(dim=0) / cam.shape[0]).detach()
            attentions.append(avg_heads.unsqueeze(0))
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]
        dim = int(image_relevance[0].numel() ** 0.5)
        image_relevance = image_relevance.reshape(batch_size, dim, dim)
        if rollout:
            return attentions
        return image_relevance

    if flag == "text":
        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

        if start_layer_text == -1:
            start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(target_device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        attentions = []
        for i, blk in enumerate(text_attn_blocks):
            if i < start_layer_text:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            avg_heads = (cam.sum(dim=0) / cam.shape[0]).detach()
            attentions.append(avg_heads.unsqueeze(0))
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text

        if rollout:
            return attentions
        return text_relevance

    raise ValueError(f"Unsupported flag: {flag}")


def compute_rollout_attention(all_layer_matrices, start_layer=0, flag="image"):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True) for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    if flag == "text":
        return joint_attention
    if flag == "image":
        joint_attention = joint_attention[:, 0, 1:]
        dim = int(joint_attention[0].numel() ** 0.5)
        return joint_attention.reshape(batch_size, dim, dim)
    raise ValueError(f"Unsupported flag: {flag}")


def attention_layer(q, k, v, num_heads=1, attn_mask=None):
    tgt_len, bsz, embed_dim = q.shape
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if attn_mask is not None:
        attn_output_weights += attn_mask
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_heads = torch.bmm(attn_output_weights, v)
    attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
    attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
    return attn_output, attn_output_weights


def clip_encode_dense(x):
    initialize_backends()
    clip_inres = clipmodel.visual.input_resolution
    clip_ksize = clipmodel.visual.conv1.kernel_size

    # Keep input on the exact device/dtype expected by CLIP visual stem.
    conv1_weight = clipmodel.visual.conv1.weight
    x = x.to(device=conv1_weight.device, dtype=conv1_weight.dtype)
    x = clipmodel.visual.conv1(x)
    feah, feaw = x.shape[-2:]

    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    class_embedding = clipmodel.visual.class_embedding.to(x.dtype)
    x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

    pos_embedding = clipmodel.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h = clip_inres // clip_ksize[0]
    pos_w = clip_inres // clip_ksize[1]
    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode="bicubic", align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
    x = x + pos_embedding
    x = clipmodel.visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x_in = torch.nn.Sequential(*clipmodel.visual.transformer.resblocks[:-1])(x)

    targetTR = clipmodel.visual.transformer.resblocks[-1]
    x_before_attn = targetTR.ln_1(x_in)

    linear = torch._C._nn.linear
    q, k, v = linear(x_before_attn, targetTR.attn.in_proj_weight, targetTR.attn.in_proj_bias).chunk(3, dim=-1)
    attn_output, attn = attention_layer(q, k, v, 1)
    attn_output.retain_grad()
    x_after_attn = linear(attn_output, targetTR.attn.out_proj.weight, targetTR.attn.out_proj.bias)

    x = x_after_attn + x_in
    x_out = x + targetTR.mlp(targetTR.ln_2(x))

    x = x_out.permute(1, 0, 2)
    x = clipmodel.visual.ln_post(x)
    x = x @ clipmodel.visual.proj

    with torch.no_grad():
        qkv = torch.stack((q, k, v), dim=0)
        qkv = linear(qkv, targetTR.attn.out_proj.weight, targetTR.attn.out_proj.bias)
        q_out, k_out, v_out = qkv[0], qkv[1], qkv[2]

        v_final = v_out + x_in
        v_final = v_final + targetTR.mlp(targetTR.ln_2(v_final))
        v_final = v_final.permute(1, 0, 2)
        v_final = clipmodel.visual.ln_post(v_final)
        v_final = v_final @ clipmodel.visual.proj

    return x, v_final[:, 1:], x_in, v, q_out, k_out, attn, attn_output, (feah, feaw)


def grad_eclip(c, q_out, k_out, v, att_output, map_size, withksim=True):
    grad = torch.autograd.grad(c, att_output, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        raise RuntimeError("grad_eclip could not obtain gradient from attention output.")
    grad = grad.detach()
    grad_cls = grad[:1, 0, :]
    if withksim:
        q_cls = q_out[:1, 0, :]
        k_patch = k_out[1:, 0, :]
        q_cls = F.normalize(q_cls, dim=-1)
        k_patch = F.normalize(k_patch, dim=-1)
        cosine_qk = (q_cls * k_patch).sum(-1)
        cosine_qk = (cosine_qk - cosine_qk.min()) / (cosine_qk.max() - cosine_qk.min() + 1e-8)
        emap_lastv = F.relu_((grad_cls * v[1:, 0, :] * cosine_qk[:, None]).detach().sum(-1))
    else:
        emap_lastv = F.relu_((grad_cls * v[1:, 0, :]).detach().sum(-1))
    return emap_lastv.reshape(*map_size)


def grad_cam(c, layer_feat, map_size):
    grad = torch.autograd.grad(c, layer_feat, retain_graph=True)[0]
    grad = grad.detach()
    grad_weight = grad.mean(0, keepdim=True)
    grad_cam_map = F.relu_((grad_weight * layer_feat[1:, 0, :]).detach().sum(-1))
    return grad_cam_map.reshape(*map_size)


def mask_clip(txt_feats, v_final, k_out, map_size):
    v_final = F.normalize(v_final, dim=-1)
    cosine_v = (v_final @ txt_feats)[0].transpose(1, 0)
    k_cls = k_out[:1, 0, :]
    k_patch = k_out[1:, 0, :]
    k_cls = F.normalize(k_cls, dim=-1)
    k_patch = F.normalize(k_patch, dim=-1)
    cosine_qk = (k_cls * k_patch).sum(-1)

    sim_v = cosine_v * cosine_qk[None, :]
    return sim_v.detach().reshape(-1, *map_size)


def _normalize_map(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    heatmap = torch.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = heatmap - heatmap.min()
    denom = heatmap.max().clamp(min=eps)
    heatmap = heatmap / denom
    return torch.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)


def _resize_to_image(heatmap: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    w, h = image_size
    heat = heatmap[None, None, ...]
    heat = F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return _normalize_map(heat)


def _resize_raw_to_image(heatmap: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    w, h = image_size
    heat = heatmap[None, None, ...]
    return F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)[0, 0]


def _normalize_map_percentile(heatmap: torch.Tensor, q_low: float = 0.01, q_high: float = 0.99, eps: float = 1e-8) -> torch.Tensor:
    heatmap = torch.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    flat = heatmap.flatten()
    low = torch.quantile(flat, q_low)
    high = torch.quantile(flat, q_high)
    heatmap = (heatmap - low) / (high - low + eps)
    heatmap = heatmap.clamp(0.0, 1.0)
    return torch.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)


def _smooth_map(heatmap: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return heatmap
    pad = kernel_size // 2
    x = heatmap[None, None, ...]
    x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    return x[0, 0]


def _jet_colormap(hm: np.ndarray) -> np.ndarray:
    x = np.clip(hm, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(image: Image.Image, saliency_map: np.ndarray, alpha: float = 0.45, channel: str = "jet") -> Image.Image:
    img = np.asarray(image.convert("RGB")).astype(np.float32)

    if isinstance(saliency_map, torch.Tensor):
        hm = saliency_map.detach().float().cpu().numpy()
    else:
        hm = np.asarray(saliency_map)

    hm = np.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)
    hm = hm.astype(np.float32, copy=False)
    hm = np.clip(hm, 0.0, 1.0)
    if channel.lower() != "jet":
        raise ValueError(f"Unsupported channel: {channel}. Only 'jet' is supported.")
    color = (_jet_colormap(hm) * 255.0).astype(np.float32)
    out = np.clip((1 - alpha) * img + alpha * color, 0, 255)
    out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
    return Image.fromarray(out)


def compute_similarity(
    image: Image.Image | str | Path,
    texts: Sequence[str],
    model_name: str = "ViT-B/16",
    target_device: Optional[str] = None,
) -> Dict[str, float]:
    initialize_backends(model_name=model_name, target_device=target_device)

    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    else:
        pil = image.convert("RGB")

    image_t = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = F.normalize(clipmodel.encode_image(image_t), dim=-1)
        text_tokens = clip.tokenize(list(texts)).to(device)
        text_features = F.normalize(clipmodel.encode_text(text_tokens), dim=-1)
        sims = (image_features @ text_features.T).squeeze(0).detach().cpu().float().numpy()

    return {text: float(sims[i]) for i, text in enumerate(texts)}


class CLIPSaliencyBase:
    def __init__(self, model_name: str = "ViT-B/16", target_device: Optional[str] = None):
        initialize_backends(model_name=model_name, target_device=target_device)
        self.device = device

    def _load_image(self, image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class GradECLIPSaliency(CLIPSaliencyBase):
    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        pil = self._load_image(image)
        img_keepsized = imgprocess_keepsize(pil).to(self.device).unsqueeze(0)
        outputs, _, _, v, q_out, k_out, _, att_output, map_size = clip_encode_dense(img_keepsized)

        text_tokens = clip.tokenize(list(texts)).to(self.device)
        with torch.no_grad():
            text_features = F.normalize(clipmodel.encode_text(text_tokens), dim=-1)
        img_embedding = F.normalize(outputs[:, 0], dim=-1)
        cosines = (img_embedding @ text_features.T)[0]

        maps: Dict[str, np.ndarray] = {}
        for i, text in enumerate(texts):
            emap = grad_eclip(cosines[i], q_out, k_out, v, att_output, map_size)
            emap = _resize_to_image(emap, pil.size)
            maps[text] = emap.detach().cpu().numpy()
        return maps


class GradECLIPNoKSimSaliency(CLIPSaliencyBase):
    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        pil = self._load_image(image)
        outputs, _, _, v, q_out, k_out, _, att_output, map_size = clip_encode_dense(imgprocess_keepsize(pil).to(self.device).unsqueeze(0))

        text_tokens = clip.tokenize(list(texts)).to(self.device)
        with torch.no_grad():
            text_features = F.normalize(clipmodel.encode_text(text_tokens), dim=-1)
        img_embedding = F.normalize(outputs[:, 0], dim=-1)
        cosines = (img_embedding @ text_features.T)[0]

        maps: Dict[str, np.ndarray] = {}
        for i, text in enumerate(texts):
            emap = grad_eclip(cosines[i], q_out, k_out, v, att_output, map_size, withksim=False)
            maps[text] = _resize_to_image(emap, pil.size).detach().cpu().numpy()
        return maps


class GradCAMSaliency(CLIPSaliencyBase):
    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        pil = self._load_image(image)
        outputs, _, last_input, _, _, _, _, _, map_size = clip_encode_dense(imgprocess_keepsize(pil).to(self.device).unsqueeze(0))

        text_tokens = clip.tokenize(list(texts)).to(self.device)
        with torch.no_grad():
            text_features = F.normalize(clipmodel.encode_text(text_tokens), dim=-1)
        img_embedding = F.normalize(outputs[:, 0], dim=-1)
        cosines = (img_embedding @ text_features.T)[0]

        maps: Dict[str, np.ndarray] = {}
        for i, text in enumerate(texts):
            emap = grad_cam(cosines[i], last_input, map_size)
            maps[text] = _resize_to_image(emap, pil.size).detach().cpu().numpy()
        return maps


class SelfAttentionSaliency(CLIPSaliencyBase):
    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        pil = self._load_image(image)
        _, _, _, _, _, _, attn, _, map_size = clip_encode_dense(imgprocess_keepsize(pil).to(self.device).unsqueeze(0))
        emap = attn[0, :1, 1:].detach().reshape(*map_size)
        emap = _resize_to_image(emap, pil.size).detach().cpu().numpy()
        return {text: emap.copy() for text in texts}


class MaskCLIPSaliency(CLIPSaliencyBase):
    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        pil = self._load_image(image)
        img_keepsized = imgprocess_keepsize(pil).to(self.device).unsqueeze(0)
        _, v_final, _, _, _, k_out, _, _, map_size = clip_encode_dense(img_keepsized)

        text_tokens = clip.tokenize(list(texts)).to(self.device)
        with torch.no_grad():
            text_features = F.normalize(clipmodel.encode_text(text_tokens), dim=-1)
        txt_feats = text_features.T

        maps_tensor = mask_clip(txt_feats, v_final, k_out, map_size)
        maps: Dict[str, np.ndarray] = {}
        for i, text in enumerate(texts):
            emap = _resize_to_image(maps_tensor[i], pil.size)
            maps[text] = emap.detach().cpu().numpy()
        return maps


class CLIPSurgerySaliency(CLIPSaliencyBase):
    def __init__(self, model_name: str = "ViT-B/16", target_device: Optional[str] = None):
        initialize_backends(model_name=model_name, target_device=target_device, load_surgery=True)
        self.device = device

    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        pil = self._load_image(image)
        img_clip = preprocess(pil).to(self.device).unsqueeze(0)

        # Keep behavior close to Grad-ECLIP eval scripts: target text in front plus fixed distractors.
        prompt_bank = [
            "airplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book", "bottle",
            "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth", "computer", "cow",
            "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food", "grass", "ground",
            "horse", "keyboard", "light", "motorbike", "mountain", "mouse", "person", "plate", "platform",
            "potted plant", "road", "rock", "sheep", "shelves", "sidewalk", "sign", "sky", "snow", "sofa",
            "table", "track", "train", "tree", "truck", "tv monitor", "wall", "water", "window", "wood",
        ]
        texts_list = list(texts)
        surgery_texts = texts_list + [p for p in prompt_bank if p not in set(texts_list)]

        sim_map = clip_surgery_map(surgery_model, img_clip, surgery_texts, self.device)
        text_to_idx = {t: i for i, t in enumerate(surgery_texts)}

        maps: Dict[str, np.ndarray] = {}
        for text in texts_list:
            emap_raw = sim_map[0, :, :, text_to_idx[text]]
            emap = _resize_raw_to_image(emap_raw, pil.size)
            emap = _normalize_map_percentile(emap, q_low=0.01, q_high=0.99)
            emap = _smooth_map(emap, kernel_size=5)
            emap = _normalize_map(emap)

            # Fallback when surgery output collapses to near-constant values.
            if float((emap.max() - emap.min()).item()) < 1e-4:
                emap = _normalize_map(_resize_raw_to_image(torch.abs(emap_raw), pil.size))

            maps[text] = emap.detach().cpu().numpy()
        return maps


class M2IBSaliency(CLIPSaliencyBase):
    def __init__(self, model_name: str = "ViT-B/16", target_device: Optional[str] = None):
        initialize_backends(model_name=model_name, target_device=target_device, load_m2ib=True)
        self.device = device

    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        pil = self._load_image(image)
        img_clip = preprocess(pil).to(self.device).unsqueeze(0)

        maps: Dict[str, np.ndarray] = {}
        for text in texts:
            emap = m2ib_clip_map(m2ib_model, img_clip, text, self.device)
            if not isinstance(emap, torch.Tensor):
                emap = torch.tensor(emap)
            if emap.ndim == 3:
                emap = emap[0]
            emap = emap.to(self.device)
            emap = _resize_to_image(emap, pil.size)
            maps[text] = emap.detach().cpu().numpy()
        return maps


class GAMESaliency(CLIPSaliencyBase):
    def __init__(self, model_name: str = "ViT-B/16", target_device: Optional[str] = None):
        initialize_backends(model_name=model_name, target_device=target_device, load_game=True)
        self.device = device

    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        _require_backend(mm_clip, "GAME", "Clone Transformer-MM-Explainability and expose Game_MM_CLIP on PYTHONPATH.")
        pil = self._load_image(image)
        img_clip = preprocess(pil).to(self.device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(list(texts)).to(self.device)
        relevance = mm_interpret(model=mm_clipmodel, image=img_clip, texts=text_tokenized, target_device=self.device)

        maps: Dict[str, np.ndarray] = {}
        for i, text in enumerate(texts):
            emap = _resize_to_image(relevance[i], pil.size)
            maps[text] = emap.detach().cpu().numpy()
        return maps


class RolloutSaliency(CLIPSaliencyBase):
    def __init__(self, model_name: str = "ViT-B/16", target_device: Optional[str] = None):
        initialize_backends(model_name=model_name, target_device=target_device, load_game=True)
        self.device = device

    def explain(self, image: Image.Image | str | Path, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        _require_backend(mm_clip, "GAME", "Clone Transformer-MM-Explainability and expose Game_MM_CLIP on PYTHONPATH.")
        pil = self._load_image(image)
        img_clip = preprocess(pil).to(self.device).unsqueeze(0)
        text_tokenized = mm_clip.tokenize(list(texts)).to(self.device)
        attentions = mm_interpret(model=mm_clipmodel, image=img_clip, texts=text_tokenized, target_device=self.device, rollout=True)
        relevance = compute_rollout_attention(attentions)

        maps: Dict[str, np.ndarray] = {}
        for i, text in enumerate(texts):
            emap = _resize_to_image(relevance[i], pil.size)
            maps[text] = emap.detach().cpu().numpy()
        return maps


SAL_METHODS = {
    "selfattn": SelfAttentionSaliency,
    "rollout": RolloutSaliency,
    "gradcam": GradCAMSaliency,
    "game": GAMESaliency,
    "gradeclip": GradECLIPSaliency,
    "gradeclip_wo_ksim": GradECLIPNoKSimSaliency,
    "maskclip": MaskCLIPSaliency,
    "clipsurgery": CLIPSurgerySaliency,
    "m2ib": M2IBSaliency,
}


def build_method(method: str, model_name: str = "ViT-B/16", device: Optional[str] = None) -> CLIPSaliencyBase:
    key = method.lower()
    if key not in SAL_METHODS:
        raise ValueError(f"Unsupported method: {method}. Available: {list(SAL_METHODS.keys())}")
    return SAL_METHODS[key](model_name=model_name, target_device=device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful Grad-Eclip style saliency demo")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--texts", type=str, nargs="+", required=True)
    parser.add_argument("--method", type=str, default="gradeclip", choices=list(SAL_METHODS.keys()))
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--output-dir", type=str, default="outputs/saliency")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method = build_method(args.method, model_name=args.model)
    image = Image.open(args.image).convert("RGB")
    maps = method.explain(image, args.texts)

    stem = Path(args.image).stem
    for text, sal in maps.items():
        safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")[:80]
        np.save(output_dir / f"{stem}_{args.method}_{safe}.npy", sal)
        overlay = overlay_heatmap(image, sal)
        overlay.save(output_dir / f"{stem}_{args.method}_{safe}.png")

if __name__ == "__main__":
    main()
