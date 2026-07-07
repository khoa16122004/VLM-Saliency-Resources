import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip

import saliency.vlm_saliency as vlm


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def pil_to_tensor_01(image: Image.Image, size: int = 224) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x


def tensor_01_to_pil(x: torch.Tensor) -> Image.Image:
    arr = x.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def normalize_for_clip(x01: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.to(device=x01.device, dtype=x01.dtype)
    std = CLIP_STD.to(device=x01.device, dtype=x01.dtype)
    return (x01 - mean) / std


def attention_layer(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tgt_len, bsz, embed_dim = q.shape
    scaling = float(embed_dim) ** -0.5
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz, embed_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz, embed_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz, embed_dim).transpose(0, 1)

    attn_weights = torch.bmm(q, k.transpose(1, 2))
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

    return attn_output, attn_weights


def encode_image_dense(x01: torch.Tensor):
    model = vlm.clipmodel
    x = normalize_for_clip(x01)

    conv_w = model.visual.conv1.weight
    x = x.to(device=conv_w.device, dtype=conv_w.dtype)
    x = model.visual.conv1(x)
    feah, feaw = x.shape[-2:]

    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    class_embedding = model.visual.class_embedding.to(x.dtype)
    x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype), x], dim=1)

    clip_inres = model.visual.input_resolution
    clip_ksize = model.visual.conv1.kernel_size
    pos_embedding = model.visual.positional_embedding.to(x.dtype)
    tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
    pos_h = clip_inres // clip_ksize[0]
    pos_w = clip_inres // clip_ksize[1]
    img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
    img_pos = F.interpolate(img_pos, size=(feah, feaw), mode="bicubic", align_corners=False)
    img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
    x = x + torch.cat((tok_pos[None, ...], img_pos), dim=1)
    x = model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x_in = torch.nn.Sequential(*model.visual.transformer.resblocks[:-1])(x)
    tr = model.visual.transformer.resblocks[-1]
    x_before_attn = tr.ln_1(x_in)

    linear = torch._C._nn.linear
    q, k, v = linear(x_before_attn, tr.attn.in_proj_weight, tr.attn.in_proj_bias).chunk(3, dim=-1)
    attn_output, attn = attention_layer(q, k, v)
    x_after_attn = linear(attn_output, tr.attn.out_proj.weight, tr.attn.out_proj.bias)

    x = x_after_attn + x_in
    x_out = x + tr.mlp(tr.ln_2(x))
    x = x_out.permute(1, 0, 2)
    x = model.visual.ln_post(x)
    x = x @ model.visual.proj
    return x, q, k, v, attn_output, attn, (feah, feaw)


def _normalize_map(hm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    hm = torch.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
    hm = hm - hm.min()
    hm = hm / hm.max().clamp(min=eps)
    return torch.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)


def gradeclip_heatmap(image01: torch.Tensor, text_tokens: torch.Tensor, create_graph: bool = False):
    if not image01.requires_grad:
        image01 = image01.detach().clone().requires_grad_(True)

    img_out, q, k, v, attn_output, _, map_size = encode_image_dense(image01)
    image_features = F.normalize(img_out[:, 0], dim=-1)
    with torch.no_grad():
        text_features = F.normalize(vlm.clipmodel.encode_text(text_tokens), dim=-1)
    score = (image_features @ text_features.T)[0, 0]

    q_cls = q[0, 0, :]
    k_patch = k[1:, 0, :]
    v_patch = v[1:, 0, :]
    cdim = float(q_cls.shape[0])

    logits = (k_patch * q_cls.unsqueeze(0)).sum(-1) / math.sqrt(cdim)
    lam = (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)

    grad_attn = torch.autograd.grad(
        score,
        attn_output,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )[0]
    if grad_attn is None:
        raise RuntimeError("Could not compute gradient dS/d(attn_output) for Grad-ECLIP heatmap.")
    w = grad_attn[0, 0, :]

    token_h = F.relu((w.unsqueeze(0) * (lam.unsqueeze(-1) * v_patch)).sum(-1))
    h_patch = token_h.reshape(*map_size)
    return score, _normalize_map(h_patch), h_patch


def soft_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(x.detach().flatten(), q)


def upsample_mask(mask_patch: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask_patch[None, None, ...], size=size_hw, mode="bilinear", align_corners=False)[0, 0]


def cosine_sim_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).squeeze(0)


def make_baseline(i0: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "black":
        return torch.zeros_like(i0)
    if mode == "mean":
        return torch.ones_like(i0) * i0.mean(dim=(-2, -1), keepdim=True)
    if mode == "blur":
        x = i0
        for _ in range(3):
            x = F.avg_pool2d(x, kernel_size=9, stride=1, padding=4)
        return x
    raise ValueError(f"Unsupported baseline mode: {mode}")


def attack_gradeclip_faithfulness(
    i0: torch.Tensor,
    text_tokens: torch.Tensor,
    eps: float = 8.0 / 255.0,
    alpha: float = 1.0 / 255.0,
    n_iter: int = 80,
    k_ratio: float = 0.2,
    lambdas: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5),
    baseline_mode: str = "mean",
    tau_soft: float = 10.0,
):
    l1, l2, l3, l4 = lambdas
    s0, h0, _ = gradeclip_heatmap(i0, text_tokens, create_graph=False)
    baseline = make_baseline(i0, mode=baseline_mode)

    delta = torch.zeros_like(i0, requires_grad=True)
    logs = []

    for it in range(n_iter):
        i_t = (i0 + delta).clamp(0.0, 1.0)
        s_t, h_t, _ = gradeclip_heatmap(i_t, text_tokens, create_graph=True)

        th_top = soft_quantile(h_t, 1.0 - k_ratio)
        m_top = torch.sigmoid(tau_soft * (h_t - th_top))

        th_low = soft_quantile(h_t, k_ratio)
        m_low = torch.sigmoid(tau_soft * (th_low - h_t))

        h_img, w_img = i_t.shape[-2:]
        m_top_px = upsample_mask(m_top, (h_img, w_img)).unsqueeze(0)
        m_low_px = upsample_mask(m_low, (h_img, w_img)).unsqueeze(0)

        i_del_top = i_t * (1.0 - m_top_px) + baseline * m_top_px
        i_del_low = i_t * (1.0 - m_low_px) + baseline * m_low_px

        s_del_top, _, _ = gradeclip_heatmap(i_del_top, text_tokens, create_graph=True)
        s_del_low, _, _ = gradeclip_heatmap(i_del_low, text_tokens, create_graph=True)

        l_pred = (s_t - s0) ** 2
        l_map = 1.0 - cosine_sim_flat(h_t, h0)
        l_break = F.relu(s0 - s_del_top)
        l_invert = -(s0 - s_del_low)
        loss = l1 * l_pred + l2 * l_map + l3 * l_break + l4 * l_invert

        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_((i0 + delta).clamp(0.0, 1.0) - i0)
        delta.requires_grad_(True)

        if it % 10 == 0 or it == n_iter - 1:
            logs.append(
                {
                    "iter": int(it),
                    "loss": float(loss.detach().cpu().item()),
                    "score": float(s_t.detach().cpu().item()),
                    "score_del_top": float(s_del_top.detach().cpu().item()),
                    "score_del_low": float(s_del_low.detach().cpu().item()),
                    "l_pred": float(l_pred.detach().cpu().item()),
                    "l_map": float(l_map.detach().cpu().item()),
                    "l_break": float(l_break.detach().cpu().item()),
                    "l_invert": float(l_invert.detach().cpu().item()),
                }
            )

    i_adv = (i0 + delta.detach()).clamp(0.0, 1.0)
    s_adv, h_adv, _ = gradeclip_heatmap(i_adv, text_tokens, create_graph=False)
    return i_adv, h0.detach(), h_adv.detach(), float(s0.item()), float(s_adv.item()), logs


def jet_colormap(hm: np.ndarray) -> np.ndarray:
    x = np.clip(hm, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_heatmap_image(hm: torch.Tensor, out_path: Path) -> None:
    hm_np = hm.detach().float().cpu().numpy()
    hm_np = np.clip(hm_np, 0.0, 1.0)
    color = (jet_colormap(hm_np) * 255.0).astype(np.uint8)
    Image.fromarray(color, mode="RGB").save(out_path)


def save_logs(logs: list, out_path: Path, s0: float, s_adv: float) -> None:
    lines = [
        f"S0={s0:.6f}",
        f"S_adv={s_adv:.6f}",
        "iter,loss,score,score_del_top,score_del_low,l_pred,l_map,l_break,l_invert",
    ]
    for row in logs:
        lines.append(
            f"{row['iter']},{row['loss']:.8f},{row['score']:.8f},{row['score_del_top']:.8f},{row['score_del_low']:.8f},"
            f"{row['l_pred']:.8f},{row['l_map']:.8f},{row['l_break']:.8f},{row['l_invert']:.8f}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-ECLIP faithfulness attack demo on one image/text pair")
    parser.add_argument("--image", type=str, default="test_imgs/ostrich.jpg")
    parser.add_argument("--text", type=str, default="ostrich")
    parser.add_argument("--output_dir", type=str, default="outputs/gradeclip_attack_ostrich")
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--alpha", type=float, default=1.0 / 255.0)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--k_ratio", type=float, default=0.2)
    parser.add_argument("--baseline_mode", type=str, default="mean", choices=["mean", "black", "blur"])
    args = parser.parse_args()

    vlm.initialize_backends(model_name=args.model)
    vlm.clipmodel.eval()
    vlm.clipmodel.float()
    for p in vlm.clipmodel.parameters():
        p.requires_grad_(False)

    dev = next(vlm.clipmodel.parameters()).device

    image = Image.open(args.image).convert("RGB")
    i0 = pil_to_tensor_01(image, size=224).to(dev)
    text_tokens = clip.tokenize([args.text]).to(dev)

    i_adv, h0, h_adv, s0, s_adv, logs = attack_gradeclip_faithfulness(
        i0,
        text_tokens,
        eps=args.eps,
        alpha=args.alpha,
        n_iter=args.iters,
        k_ratio=args.k_ratio,
        baseline_mode=args.baseline_mode,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_path = out_dir / "original.png"
    adv_path = out_dir / "adversarial.png"
    hm0_path = out_dir / "heatmap_original.png"
    hmadv_path = out_dir / "heatmap_adversarial.png"
    log_path = out_dir / "attack_log.csv"

    tensor_01_to_pil(i0).save(orig_path)
    tensor_01_to_pil(i_adv).save(adv_path)
    save_heatmap_image(h0, hm0_path)
    save_heatmap_image(h_adv, hmadv_path)
    save_logs(logs, log_path, s0=s0, s_adv=s_adv)

    print(f"Saved original image: {orig_path}")
    print(f"Saved adversarial image: {adv_path}")
    print(f"Saved original heatmap: {hm0_path}")
    print(f"Saved adversarial heatmap: {hmadv_path}")
    print(f"Saved attack log: {log_path}")
    print(f"Similarity original={s0:.6f}, adversarial={s_adv:.6f}, abs_delta={abs(s_adv - s0):.6f}")


if __name__ == "__main__":
    main()
