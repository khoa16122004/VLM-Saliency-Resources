# Adversarial Attack on Grad-ECLIP Faithfulness (IMD/Deletion-Insertion)

## Mục tiêu

Grad-ECLIP (Zhao et al., 2025) claim heatmap của nó có **faithfulness** cao, đo bằng
metric **IMD = Insertion AUC − Deletion AUC** (xóa/thêm pixel theo thứ tự quan trọng
trong heatmap, quan sát sự thay đổi của similarity/accuracy).

Mục tiêu của thuật toán này: tạo ra một **perturbation nhỏ, không đổi score dự đoán, không
đổi rõ heatmap về mặt hình dạng**, nhưng làm cho:

- Xóa đúng những vùng mà Grad-ECLIP nói là "quan trọng nhất" (top-k heatmap) → similarity
  **không giảm** (hoặc tăng) — phá vỡ giả định "vùng quan trọng ⇒ xóa thì rớt điểm".
- (Tùy chọn) Xóa những vùng mà heatmap nói là "không quan trọng" (bottom-k) → similarity
  **giảm mạnh hơn** — đảo ngược quan hệ nhân quả.

Nếu attack thành công với perturbation nhỏ (imperceptible), điều đó cho thấy **Deletion/Insertion/IMD
là proxy dễ bị đánh lừa**, và faithfulness cao trên benchmark không đảm bảo tính robust của explanation.

---

## 1. Kiến trúc cần implement trước: Grad-ECLIP forward

### 1.1. Model

- Dùng CLIP ViT-B/16 (OpenAI hoặc OpenCLIP), có thể lấy qua HuggingFace `transformers`
  (`CLIPModel`, `CLIPProcessor`) hoặc `open_clip`.
- Cần **truy cập được** vào layer attention cuối của Vision Transformer (image encoder):
  giá trị `q_cls`, `k_i`, `v_i` trước khi concat multi-head, và output `o_cls^(0)`.
- Cách dễ nhất: **hook** vào module attention cuối (`model.vision_model.encoder.layers[-1].self_attn`)
  để lấy `value` projection output (`v_i`) và tính lại attention theo公 công thức single-head
  softmax như paper (gộp toàn bộ channel, không chia head) — xem công thức (7).

### 1.2. Công thức Grad-ECLIP (image encoder, layer cuối, single-head)

Cho ảnh $I$, text $T$:

```
F_I = image_embedding(I)          # sau linear projection + normalize
F_T = text_embedding(T)           # normalize
S = cosine(F_I, F_T)              # matching score (scalar)

# Lấy attention layer cuối cùng, forward lại attention "single head":
q_cls              # query của token [CLS], shape [C]
k_i, v_i           # key/value của tất cả token patch, shape [N, C]

attn_logits_i = q_cls . k_i^T / sqrt(C)          # [N]
lambda_i = (attn_logits_i - min) / (max - min)   # "loosen" 0-1 normalization, KHÔNG dùng softmax

o_cls = sum_i softmax(attn_logits_i) * v_i        # forward thật (dùng để tính S)

w_c = d S / d o_cls[c]             # gradient của score theo output attention layer (autograd)

H_i = ReLU( sum_c w_c * lambda_i * v_i[c] )        # heatmap thô, 1 giá trị / patch token

# Reshape H (N patches -> H_grid x W_grid), interpolate (bilinear) lên kích thước ảnh gốc
```

> **Chú ý quan trọng**: `lambda_i` dùng 0-1 normalization (min-max) trên toàn bộ `attn_logits`,
> KHÔNG dùng softmax. Đây là điểm khác biệt cốt lõi so với raw attention / Grad-CAM.

### 1.3. API cần code

```python
def gradeclip_heatmap(model, image_tensor, text_tokens, create_graph=False):
    """
    Trả về:
      S: scalar tensor, matching score (có grad tới image_tensor)
      H: tensor [Hgrid, Wgrid] hoặc đã upsample [H_img, W_img], heatmap Grad-ECLIP
         (có grad tới image_tensor nếu create_graph=True, để dùng cho attack bậc 2)
    """
```

- `create_graph=True` là **bắt buộc** khi cần lan truyền gradient của attack loss qua chính `H`
  (vì `H` chứa `w_c = dS/do_cls`, tức đạo hàm bậc 1 — attack cần đạo hàm của `H` theo ảnh,
  tức đạo hàm bậc 2 của `S`). Dùng `torch.autograd.grad(..., create_graph=True)`.

---

## 2. Thuật toán tấn công (PGD nhiều thành phần loss)

### 2.1. Input / Output

**Input:**
- `I0`: ảnh gốc, tensor `[3,H,W]`, giá trị trong `[0,1]`.
- `T`: text prompt (đã tokenize).
- `model`: CLIP đã freeze toàn bộ tham số (`requires_grad_(False)` cho weights, chỉ `I` có grad).
- `eps`: biên độ nhiễu L∞ (thử `2/255`, `4/255`, `8/255`).
- `alpha`: step size PGD (thường `eps/10`).
- `n_iter`: số bước (thử 100–300).
- `k_ratio`: tỉ lệ patch bị coi là "top-k quan trọng" (ví dụ 0.2 = 20% patch).
- `lambda1..4`: hệ số trọng số 4 loss (xem dưới), mặc định `1.0, 1.0, 1.0, 0.5`.
- `baseline_mode`: cách "xóa" pixel khi mô phỏng deletion — `"blur"` (Gaussian blur mạnh) hoặc
  `"mean"` (điền bằng mean pixel dataset) hoặc `"black"`. Paper gốc dùng blur/mean tùy metric;
  mặc định dùng **mean pixel** cho đơn giản, có thể để tham số hóa.

**Output:**
- `I_adv`: ảnh sau attack.
- Log các giá trị loss, `S0`, `S(I_adv)`, cosine(H0, H_adv), Deletion/Insertion AUC trước/sau.

### 2.2. Pseudocode chi tiết

```python
def attack_gradeclip_faithfulness(
    model, I0, T,
    eps=8/255, alpha=1/255, n_iter=200,
    k_ratio=0.2, lambdas=(1.0, 1.0, 1.0, 0.5),
    baseline_mode="mean", tau_soft=10.0, seed=0
):
    l1, l2, l3, l4 = lambdas
    device = I0.device
    torch.manual_seed(seed)

    # ---- 0. Baselines & targets ----
    with torch.no_grad():
        S0, H0 = gradeclip_heatmap(model, I0, T, create_graph=False)
        H0 = H0.detach()
        S0 = S0.detach()

    baseline_img = make_baseline(I0, mode=baseline_mode)  # [3,H,W], no grad needed

    delta = torch.zeros_like(I0, requires_grad=True)

    for it in range(n_iter):
        I_t = torch.clamp(I0 + delta, 0.0, 1.0)

        # ---- 1. Forward Grad-ECLIP với graph để lấy đạo hàm bậc 2 ----
        S_t, H_t = gradeclip_heatmap(model, I_t, T, create_graph=True)

        # ---- 2. Soft top-k / bottom-k mask (differentiable) ----
        # dùng soft-threshold quanh percentile để giữ khả vi
        thresh_top = soft_quantile(H_t, 1 - k_ratio)      # giá trị ngưỡng ở percentile (1-k)
        M_top = torch.sigmoid(tau_soft * (H_t - thresh_top))   # ~1 nếu H_t lớn (quan trọng)

        thresh_low = soft_quantile(H_t, k_ratio)
        M_low = torch.sigmoid(tau_soft * (thresh_low - H_t))   # ~1 nếu H_t nhỏ (không quan trọng)

        # upsample mask patch-grid -> pixel-grid nếu H_t đang ở patch resolution
        M_top_px = upsample_mask(M_top, size=I_t.shape[-2:])
        M_low_px = upsample_mask(M_low, size=I_t.shape[-2:])

        # ---- 3. Ảnh "đã xóa" vùng top-k / bottom-k (soft, khả vi) ----
        I_del_top = I_t * (1 - M_top_px) + baseline_img * M_top_px
        I_del_low = I_t * (1 - M_low_px) + baseline_img * M_low_px

        S_del_top, _ = gradeclip_heatmap(model, I_del_top, T, create_graph=True)
        S_del_low, _ = gradeclip_heatmap(model, I_del_low, T, create_graph=True)

        # ---- 4. Các thành phần loss ----
        L_pred   = (S_t - S0) ** 2
        L_map    = 1 - cosine_sim_flat(H_t, H0)
        # muốn xóa top-k mà score KHÔNG giảm (lý tưởng: S_del_top >= S0)
        L_break  = torch.relu(S0 - S_del_top)          # >0 nếu score có giảm -> muốn ép về 0
        # muốn xóa bottom-k mà score giảm MẠNH hơn xóa top-k (đảo thứ tự quan trọng)
        L_invert = torch.relu(S0 - S_del_low) * (-1.0)  # tối thiểu hoá số âm = tối đa hoá độ giảm
        # (có thể đổi thành: L_invert = -(S0 - S_del_low), tùy convention)

        L = l1 * L_pred + l2 * L_map + l3 * L_break + l4 * L_invert

        # ---- 5. Backprop tới delta, PGD step ----
        grad = torch.autograd.grad(L, delta, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
            # đảm bảo I0+delta hợp lệ trong [0,1]
            delta.copy_(torch.clamp(I0 + delta, 0, 1) - I0)
        delta.requires_grad_(True)

        if it % 20 == 0:
            log_progress(it, L, S_t, L_pred, L_map, L_break, L_invert)

    I_adv = torch.clamp(I0 + delta.detach(), 0, 1)
    return I_adv, {"S0": S0, "H0": H0}
```

### 2.3. Hàm phụ cần code

```python
def soft_quantile(x, q):
    """Ước lượng percentile q của tensor x (dùng torch.quantile, không cần grad qua chính threshold
    -- có thể .detach() threshold value để tránh phức tạp; gradient chính vẫn chảy qua x trong sigmoid)."""
    return torch.quantile(x.detach().flatten(), q)

def upsample_mask(mask_patch, size):
    """Nearest hoặc bilinear upsample mask từ patch-grid (ví dụ 14x14 cho ViT-B/16, 224/16)
    lên kích thước ảnh gốc (H,W)."""
    m = mask_patch.view(1, 1, *patch_grid_shape)
    return F.interpolate(m, size=size, mode="bilinear", align_corners=False).squeeze()

def cosine_sim_flat(a, b):
    a, b = a.flatten(), b.flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze()

def make_baseline(I0, mode="mean"):
    if mode == "mean":
        return torch.ones_like(I0) * I0.mean(dim=(-2, -1), keepdim=True)
    if mode == "black":
        return torch.zeros_like(I0)
    if mode == "blur":
        return gaussian_blur(I0, kernel_size=31, sigma=10)
```

> **Ghi chú kỹ thuật autograd**: `gradeclip_heatmap` bên trong nó đã gọi
> `torch.autograd.grad(S, o_cls, create_graph=True)[0]` để lấy `w_c`. Khi hàm này được gọi lại
> trong attack loop với `create_graph=True`, toàn bộ graph (kể cả graph của đạo hàm bậc 1) phải
> được giữ để `torch.autograd.grad(L, delta, ...)` ở bước 5 lan truyền được qua `w_c`. Cần cẩn thận
> **không** gọi `.detach()` nhầm chỗ trong `gradeclip_heatmap`.

---

## 3. Đánh giá kết quả (evaluation script riêng, KHÔNG dùng soft mask)

Sau khi có `I_adv`, đánh giá bằng đúng protocol **cứng** (hard top-k, không sigmoid) giống paper
để so sánh công bằng với số liệu Table I của paper gốc.

```python
def evaluate_faithfulness(model, I, T, n_steps=100, baseline_mode="mean"):
    """
    Trả về Deletion AUC, Insertion AUC, IMD = Insertion - Deletion,
    dùng heatmap Grad-ECLIP tính TRÊN ẢNH I (I có thể là I0 hoặc I_adv).
    """
    with torch.no_grad():
        S_full, H = gradeclip_heatmap(model, I, T, create_graph=False)
    order = torch.argsort(H.flatten(), descending=True)  # patch index từ quan trọng nhất

    baseline = make_baseline(I, baseline_mode)
    del_curve, ins_curve = [], []
    N = order.numel()
    for step in range(n_steps + 1):
        frac = step / n_steps
        n_del = int(frac * N)
        mask_del = index_to_pixel_mask(order[:n_del], I.shape[-2:])
        mask_ins = index_to_pixel_mask(order[:n_del], I.shape[-2:])

        I_del = I * (1 - mask_del) + baseline * mask_del
        I_ins = baseline * (1 - mask_ins) + I * mask_ins

        with torch.no_grad():
            s_del, _ = gradeclip_heatmap(model, I_del, T, create_graph=False)
            s_ins, _ = gradeclip_heatmap(model, I_ins, T, create_graph=False)
        del_curve.append(s_del.item())
        ins_curve.append(s_ins.item())

    Deletion = float(np.mean(del_curve))
    Insertion = float(np.mean(ins_curve))
    IMD = Insertion - Deletion
    return {"Deletion": Deletion, "Insertion": Insertion, "IMD": IMD}
```

Chạy hàm này **hai lần**: một lần trên `I0` (baseline, để đối chiếu số liệu paper), một lần trên
`I_adv` (attack). So sánh.

---

## 4. Metric báo cáo cuối cùng

Với mỗi ảnh test, xuất ra bảng:

| Metric | I0 (gốc) | I_adv (sau attack) |
|---|---|---|
| Similarity S(I,T) | S0 | S(I_adv,T) ≈ S0 (nếu attack tốt) |
| Cosine(heatmap gốc, heatmap ảnh này) | 1.0 | cos(H0, H_adv) — càng gần 1 càng "map giống" |
| Deletion AUC | (thấp, giống paper) | cao bất thường nếu attack thành công |
| Insertion AUC | (cao, giống paper) | thấp bất thường nếu attack thành công |
| IMD | dương, giống Table I | ≈0 hoặc âm nếu attack thành công |
| L∞(I_adv - I0) | 0 | ≤ eps |

Chạy trên **N ảnh** (ví dụ 100–500 ảnh từ ImageNet validation hoặc MS-COCO Karpathy split, đúng
tập paper dùng ở §IV-B1) và lấy **trung bình + độ lệch chuẩn** của từng đại lượng, cộng paired
t-test giữa IMD(I0) và IMD(I_adv) để báo cáo ý nghĩa thống kê.

Vẽ thêm biểu đồ: **IMD sau attack vs. eps** (2/255, 4/255, 8/255, 16/255) — đường cong "adversarial
fragility" cho thấy cần nhiễu bao nhỏ để phá vỡ faithfulness.

---

## 5. Ablation / mở rộng (tùy chọn, làm sau khi pipeline chính chạy được)

1. **So sánh độ mong manh giữa các XAI method**: chạy attack tương tự (thay `gradeclip_heatmap`
   bằng `gradcam_heatmap`, `game_heatmap`, `rollout_heatmap`) để xem method nào dễ bị tấn công hơn.
2. **Universal / transferable perturbation**: học một `delta` chung cho nhiều ảnh, kiểm tra có
   transfer được attack faithfulness sang ảnh chưa thấy không.
3. **Defense baseline**: SmoothGrad-style — tính `H_smooth = E_{noise~N(0,sigma)} [H(I+noise)]`
   trước khi đưa vào Deletion/Insertion, kiểm tra attack có còn hiệu quả không.
4. **Kiểm tra trên text encoder**: áp dụng đúng logic tương tự để tấn công phần **textual explanation**
   (word importance heatmap) — xáo trộn embedding của 1-2 token ít quan trọng để đảo thứ tự quan
   trọng của các từ mà không đổi câu / không đổi matching score.

---

## 6. Checklist implement cho agent

- [ ] Load CLIP ViT-B/16 (freeze weights).
- [ ] Implement `gradeclip_heatmap(model, image, text, create_graph)` đúng công thức (6)-(20) paper.
- [ ] Viết unit test: so sánh heatmap của bạn với Fig.1(i)/Fig.3(i) trên 1-2 ảnh mẫu (định tính,
      nhìn bằng mắt) trước khi chạy attack.
- [ ] Implement `make_baseline`, `soft_quantile`, `upsample_mask`, `cosine_sim_flat`.
- [ ] Implement attack loop `attack_gradeclip_faithfulness` với double-backward hoạt động
      (test bằng `torch.autograd.gradcheck` trên tensor nhỏ nếu nghi ngờ đúng sai).
- [ ] Implement `evaluate_faithfulness` (hard version, không soft mask) để tái lập số liệu
      Table I của paper trên vài chục ảnh làm sanity check trước khi attack.
- [ ] Chạy attack trên bộ test nhỏ (10-20 ảnh) với `eps=8/255`, kiểm tra định tính:
      ảnh nhìn có khác gì không, heatmap trước/sau có giống không, IMD có sụp không.
- [ ] Scale lên bộ test lớn hơn (100-500 ảnh), xuất bảng + t-test + biểu đồ IMD vs eps.
