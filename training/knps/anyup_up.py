"""AnyUp 특징 업샘플러 (오프라인 단독 포팅).

원본: "AnyUp: Universal Feature Upsampling", Wimmer et al., ICLR 2026 (arXiv 2510.12764)
      https://github.com/wimmerth/anyup  (코드·가중치 CC-BY-4.0, Copyright (c) the AnyUp authors)
라이선스: Creative Commons Attribution 4.0 International. 재배포 시 위 출처 표기 필요 (NOTICE 에 한 줄 추가 권장).

원본에서 바꾼 점 (CC-BY 4.0 의 '변경 사항 표시'):
  * torch.hub / 인터넷 / NATTEN / matplotlib 의존 제거. 순수 PyTorch 단일 파일.
  * 원본의 nn.MultiheadAttention(need_weights=True) 호출을 같은 수식(헤드별 softmax → 헤드 평균 → 가공 안 한 V 에 곱함)의
    명시적 계산으로 바꿈. state_dict 키·모양은 원본과 100% 같아서 공식 가중치(anyup_paper.pth)를 그대로 읽음.
    (원본 CrossAttentionBlock 과의 수치 일치는 scratchpad/deep/anyup_repo/test_anyup.py 에서 확인)
  * key 를 만드는 특징과 실제로 섞을 값(value) 특징을 따로 줄 수 있게 함 (values=...). 기본은 원본과 같이 같은 특징.
  * frozen=True 이면 주의(attention) 가중치를 no_grad 로 계산하고 value 쪽으로만 기울기를 흘림 (학습 메모리 절약).
  * 창(window) 마스크를 모듈 안 사전에 캐시 (버퍼가 아니므로 state_dict 에 들어가지 않음).
  * 메모리: 특징 무관 층(LFU)은 표본 단위로, 주의는 질의 단위로 자동 분할 (값 동일, TTA 묶음 32 에서 GPU 메모리 폭증 방지).
  * MPS 는 나누어떨어지지 않는 adaptive_avg_pool 을 지원하지 않아 같은 값의 행렬곱으로 대체 (CUDA/CPU 는 원래 함수).
  * 참고(공식 코드의 특성, 그대로 유지): LFU 의 softmax 묶음이 논문 식 1 과 달리 채널·기저가 섞여 있음 (아래 주석).

사용:
  up = FeatUpsampler(weights="assets/anyup.pth")          # 파일 없으면 경고 후 무작위 초기화
  hr = up(img_imagenet_norm, lr_feats, (64, 64))           # (B,C,64,64)
  hr = up(img, key_feats, (64, 64), values=fused_feats)    # key 는 ViT 원특징, 섞는 값은 융합 특징
가중치 내려받기 (인터넷 되는 곳에서 한 번):
  python -m knps.anyup_up --download
"""
import hashlib
import math
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ASSET = Path(__file__).resolve().parent.parent / "assets" / "anyup.pth"
URLS = {
    # 논문 실험에 쓴 모델 (DINOv2 ViT-S/14 로만 학습). 논문 Tab.6: DINOv3 ViT-S+ 에도 그대로 일반화
    "paper": ("https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth",
              "9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2"),
    # 여러 백본(DINOv2-S, DINOv2-R-S, CLIP-B, SigLIP-B, ViT-B)으로 학습. 처음 보는 특징에 약간 더 강함 (Tab.7)
    "multi": ("https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth",
              "b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59"),
}


# ----------------------------------------------------------------------------- 원본 층 (구조·키 동일)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8, pad_mode="zeros", norm_fn=None,
                 activation_fn=nn.SiLU, use_conv_shortcut=False):
        super().__init__()
        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else (lambda c: nn.Identity())
        p = kernel_size // 2
        self.block = nn.Sequential(
            N(in_channels), activation_fn(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
            N(out_channels), activation_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=p, padding_mode=pad_mode, bias=False),
        )
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
                         if use_conv_shortcut or in_channels != out_channels else nn.Identity())

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class LearnedFeatureUnification(nn.Module):
    """특징 무관 층 (논문 식 1): 채널마다 같은 k×k 기저 M 개로 합성곱 → 기저 축 softmax → 채널 평균.

    입력 채널 수 N 에 상관없이 M 채널이 나옴 (그래서 384/768/256 어떤 특징이든 받음).
    """

    def __init__(self, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))

    BUDGET = 2 ** 26   # 한 번에 만드는 (표본 수 × c × M × h × w) 원소 상한. TTA 묶음 32 에서 GPU 메모리 폭증 방지

    def forward(self, features):
        b, c, h, w = features.shape
        step = max(1, self.BUDGET // (c * self.out_channels * h * w))
        if b > step:   # 표본마다 독립이므로 묶음을 나눠도 값이 같음
            return torch.cat([self._one(features[s:s + step]) for s in range(0, b, step)], 0)
        return self._one(features)

    def _one(self, features):
        b, c, h, w = features.shape
        k, p = self.kernel_size, self.kernel_size // 2
        x = F.conv2d(F.pad(features, (p, p, p, p), value=0), self.basis.repeat(c, 1, 1, 1), groups=c)
        mask = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
        denom = F.conv2d(F.pad(mask, (p, p, p, p), value=0), torch.ones(1, 1, k, k, dtype=x.dtype, device=x.device))
        # 주의: 공식 코드 그대로. conv 출력 채널 순서는 (입력채널 i, 기저 j) → i*M+j 인데 이를 (M, c) 로 view 하므로
        # 논문 식 1 의 '기저 축 softmax' 와 달리 채널·기저가 섞인 묶음에 softmax 가 걸림. 공개 가중치가 이 방식으로
        # 학습되었으므로 바꾸지 않음 (바꾸면 가중치와 어긋남)
        x = (x / denom).view(b, self.out_channels, c, h, w)
        return F.softmax(x, dim=1).mean(dim=2)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim: int, theta: int = 100):
        super().__init__()
        self.dim, self.theta = dim, theta
        self.freqs = nn.Parameter(torch.empty(2, dim))
        f1 = theta ** torch.linspace(0, -1, dim // 4)
        f1 = torch.cat([f1, f1])
        f2 = torch.zeros(2, dim)
        f2[0, : dim // 2] = f1
        f2[1, -dim // 2:] = f1
        self.freqs.data.copy_(f2 * 2 * math.pi)

    def forward(self, x, coords):
        angle = coords @ self.freqs.to(coords.dtype)
        return x * angle.cos() + _rotate_half(x) * angle.sin()


def _coords(h, w, device, dtype):
    x = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx, yy), -1).view(1, h * w, 2)


def _window_mask(H, W, h, w, ratio, device):
    """(H*W, h*w) bool, True = 볼 수 없음 (원본 compute_attention_mask 와 같은 식)."""
    r = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    c = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    pr, pc = torch.meshgrid(r, c, indexing="ij")
    r0 = ((pr - ratio).clamp(0, 1) * h).floor().long().reshape(-1, 1)
    r1 = ((pr + ratio).clamp(0, 1) * h).ceil().long().reshape(-1, 1)
    c0 = ((pc - ratio).clamp(0, 1) * w).floor().long().reshape(-1, 1)
    c1 = ((pc + ratio).clamp(0, 1) * w).ceil().long().reshape(-1, 1)
    rows = torch.arange(h, device=device)
    cols = torch.arange(w, device=device)
    ok = ((rows >= r0) & (rows < r1)).unsqueeze(2) & ((cols >= c0) & (cols < c1)).unsqueeze(1)
    return ~ok.reshape(H * W, h * w)


def _pool_matrix(n_in, n_out, device, dtype):
    """adaptive_avg_pool 1D 와 같은 (n_out, n_in) 평균 행렬 (구간 [floor(i*n/m), ceil((i+1)*n/m)))."""
    i = torch.arange(n_out, device=device)
    s = (i * n_in) // n_out
    e = ((i + 1) * n_in + n_out - 1) // n_out
    j = torch.arange(n_in, device=device)
    m = ((j[None] >= s[:, None]) & (j[None] < e[:, None])).to(dtype)
    return m / m.sum(1, keepdim=True)


def _adaptive_pool(x, size):
    """F.adaptive_avg_pool2d 와 같은 값. MPS 는 나누어떨어지지 않는 크기(256→28 등)를 지원하지 않아 행렬곱으로 대신함."""
    H, W = x.shape[-2:]
    h, w = int(size[0]), int(size[1])
    if x.device.type != "mps" or (H % h == 0 and W % w == 0):
        return F.adaptive_avg_pool2d(x, (h, w))
    ph = _pool_matrix(H, h, x.device, x.dtype)
    pw = _pool_matrix(W, w, x.device, x.dtype)
    return torch.einsum("hH,bcHW,wW->bchw", ph, x, pw)


class CrossAttention(nn.Module):
    """원본과 같은 키 (norm_q, norm_k, attention.in_proj_*, attention.out_proj.*).

    out_proj 와 in_proj 의 V 부분은 원본에서도 쓰이지 않음 (헤드 평균 주의를 가공 안 한 V 에 바로 곱함 → 출력은
    입력 특징의 볼록 결합 = 특징 공간 보존). 키 호환을 위해 nn.MultiheadAttention 을 그릇으로만 둠.
    """

    def __init__(self, qk_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.attention = nn.MultiheadAttention(embed_dim=qk_dim, num_heads=num_heads, dropout=0.0, batch_first=True)

    def weights(self, q, k, mask):
        """q (B,Q,D), k (B,K,D), mask (Q,K) bool(True=차단) → 헤드 평균 주의 (B,Q,K)."""
        D = q.shape[-1]
        H, d = self.num_heads, D // self.num_heads
        w_q, w_k, _ = self.attention.in_proj_weight.chunk(3)
        b_q, b_k, _ = self.attention.in_proj_bias.chunk(3)
        q = F.linear(self.norm_q(q), w_q, b_q)
        k = F.linear(self.norm_k(k), w_k, b_k)
        B, Q, K = q.shape[0], q.shape[1], k.shape[1]
        q = q.view(B, Q, H, d).transpose(1, 2) * (d ** -0.5)
        k = k.view(B, K, H, d).transpose(1, 2)
        logit = q @ k.transpose(-1, -2)                                   # (B,H,Q,K)
        if mask is not None:
            logit = logit.masked_fill(mask, float("-inf"))
        return F.softmax(logit.float(), dim=-1).mean(1)                   # (B,Q,K) float32


class CrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio=0.1):
        super().__init__()
        self.cross_attn = CrossAttention(qk_dim, num_heads)
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self._masks = {}

    def mask(self, H, W, h, w, device):
        if self.window_ratio <= 0:
            return None
        key = (H, W, h, w, str(device))
        if key not in self._masks:
            self._masks[key] = _window_mask(H, W, h, w, self.window_ratio, device)
        return self._masks[key]

    def forward(self, q, k, v, q_chunk=None, frozen=False):
        q = self.conv2d(q)
        b, _, H, W = q.shape
        _, _, h, w = k.shape
        c = v.shape[1]
        mask = self.mask(H, W, h, w, q.device)
        q = q.permute(0, 2, 3, 1).reshape(b, H * W, -1)
        k = k.permute(0, 2, 3, 1).reshape(b, h * w, -1)
        v = v.permute(0, 2, 3, 1).reshape(b, h * w, c)
        Q = H * W
        heads = self.cross_attn.num_heads
        # q_chunk 를 안 주면 (묶음 × 헤드 × 질의 × 키) 주의 행렬이 2^26 원소를 넘지 않게 질의를 나눔 (값은 같음)
        step = int(q_chunk) if q_chunk else max(256, (2 ** 26) // max(1, b * heads * h * w))
        outs = []
        for s in range(0, Q, step):
            e = min(Q, s + step)
            m = None if mask is None else mask[s:e]
            if frozen:
                with torch.no_grad():
                    a = self.cross_attn.weights(q[:, s:e], k, m)
            else:
                a = self.cross_attn.weights(q[:, s:e], k, m)
            outs.append(a.to(v.dtype) @ v)                                # (B,chunk,C) — V 로는 항상 기울기
        return torch.cat(outs, 1).view(b, H, W, c).permute(0, 3, 1, 2)


class AnyUp(nn.Module):
    """원본 anyup.model.AnyUp (use_natten=False) 과 같은 구조·키. 기본값은 공개 가중치의 설정."""

    def __init__(self, input_dim=3, qk_dim=128, kernel_size=1, kernel_size_lfu=5, window_ratio=0.1, num_heads=4):
        super().__init__()
        self.qk_dim = qk_dim
        self._rb = dict(kernel_size=1, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm, activation_fn=nn.SiLU)
        self.image_encoder = self._enc(input_dim, kernel_size)
        self.key_encoder = self._enc(qk_dim, 1)
        self.query_encoder = self._enc(qk_dim, 1)
        self.key_features_encoder = nn.Sequential(LearnedFeatureUnification(qk_dim, kernel_size_lfu),
                                                  *[ResBlock(qk_dim, qk_dim, **self._rb) for _ in range(2)])
        self.cross_decode = CrossAttentionBlock(qk_dim, num_heads, window_ratio)
        self.aggregation = self._enc(2 * qk_dim, 3)
        self.rope = RoPE(qk_dim)

    def _enc(self, cin, k):
        return nn.Sequential(nn.Conv2d(cin, self.qk_dim, k, padding=k // 2, padding_mode="reflect", bias=False),
                             *[ResBlock(self.qk_dim, self.qk_dim, **self._rb) for _ in range(2)])

    def attention_inputs(self, image, feats, out_size):
        enc = self.image_encoder(image)
        b, d, H, W = enc.shape
        enc = enc.permute(0, 2, 3, 1).reshape(b, H * W, d)
        enc = self.rope(enc, _coords(H, W, enc.device, enc.dtype))
        enc = enc.view(b, H, W, d).permute(0, 3, 1, 2)
        q = _adaptive_pool(self.query_encoder(enc), out_size)
        k = _adaptive_pool(self.key_encoder(enc), feats.shape[-2:])
        k = self.aggregation(torch.cat([k, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1))
        return q, k

    def forward(self, image, features, output_size=None, values=None, q_chunk=None, frozen=False):
        output_size = tuple(output_size) if output_size is not None else tuple(image.shape[-2:])
        v = features if values is None else values
        assert v.shape[-2:] == features.shape[-2:], "values 와 key 특징의 격자 크기가 같아야 합니다"
        if frozen:
            with torch.no_grad():
                q, k = self.attention_inputs(image, features.detach(), output_size)
        else:
            q, k = self.attention_inputs(image, features, output_size)
        return self.cross_decode(q, k, v, q_chunk=q_chunk, frozen=frozen)


# ----------------------------------------------------------------------------- 우리 모델용 래퍼
class FeatUpsampler(nn.Module):
    """forward(guidance_img_normalized, feats, out_hw, values=None) → (B, C_values, *out_hw).

    guidance_img_normalized: ImageNet mean/std 로 정규화된 RGB (AnyUp 학습 조건). 크기 자유
    feats: key 를 만들 저해상도 특징 (B,C,h,w). AnyUp 은 DINOv2 마지막 층(norm 적용) 특징으로 학습됨
    values: 실제로 섞을 특징 (B,C',h,w). None 이면 feats
    guide_size: 안내 영상을 이 크기로 줄여서 넣음 (계산량은 안내 영상 화소 수에 비례. None 이면 그대로)
    frozen: True 면 AnyUp 파라미터를 얼리고 주의 가중치를 no_grad 로 계산 (value 로만 기울기)
    weights: 공식 state_dict 경로. None 이면 무작위 초기화 (체크포인트에서 읽을 때 — 인터넷·파일 불필요)
    """

    def __init__(self, weights: Optional[str] = None, frozen: bool = True, guide_size: Optional[int] = None,
                 q_chunk: Optional[int] = None, window_ratio: float = 0.1):
        super().__init__()
        self.net = AnyUp(window_ratio=window_ratio)
        self.frozen = bool(frozen)
        self.guide_size = int(guide_size) if guide_size else None
        self.q_chunk = q_chunk
        if weights is not None:
            p = Path(weights)
            if p.is_file():
                self.net.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            else:
                warnings.warn(f"AnyUp 가중치 파일이 없습니다: {p} → 무작위 초기화 "
                              f"(인터넷 되는 곳에서 `python -m knps.anyup_up --download` 로 받으세요)")
        if self.frozen:
            for prm in self.net.parameters():
                prm.requires_grad_(False)

    def forward(self, guidance_img_normalized, feats, out_hw, values=None):
        g = guidance_img_normalized
        if self.guide_size and g.shape[-1] != self.guide_size:
            mode = "area" if g.shape[-1] > self.guide_size else "bilinear"
            g = F.interpolate(g, size=(self.guide_size, self.guide_size), mode=mode,
                              **({} if mode == "area" else {"align_corners": False}))
        return self.net(g, feats, tuple(out_hw), values=values, q_chunk=self.q_chunk, frozen=self.frozen)


def download_weights(dst=ASSET, which: str = "paper", force: bool = False) -> Path:
    """공식 가중치를 한 번 내려받아 dst 에 저장 (sha256 확인). 추론 서버에서는 부르지 않음."""
    dst = Path(dst)
    url, sha = URLS[which]
    if dst.is_file() and not force:
        if hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
            return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    torch.hub.download_url_to_file(url, str(tmp), progress=True)
    got = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if got != sha:
        tmp.unlink()
        raise RuntimeError(f"AnyUp 가중치 해시 불일치: {got}")
    tmp.replace(dst)
    return dst


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--which", default="paper", choices=list(URLS))
    ap.add_argument("--dst", default=str(ASSET))
    a = ap.parse_args()
    if a.download:
        p = download_weights(a.dst, a.which)
        print(p, p.stat().st_size, "bytes")
