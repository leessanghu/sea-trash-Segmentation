"""SoMA (Yun et al., CVPR 2025, arXiv 2412.04077): 사전학습 가중치의 '가장 작은' 특이성분만 학습하는 LoRA 변형.

  W = U Σ V^T  (SVD)
  B = U[:, -r:] · sqrt(Σ[-r:] / s),  A = sqrt(Σ[-r:] / s) · V^T[-r:, :]        (s = alpha / r, 논문 식 (4)(5) + 공식 코드의 Σ/s)
  W_res = W - s·B·A  (얼림)  →  학습 시작 순간 W_res + s·B·A == W  (사전학습 모델과 출력이 같음, 논문 식 (6))

즉 큰 특이성분(일반화에 중요한 '세계 지식')은 W_res 에 얼려 두고, 작은 r 개 성분(문맥·텍스처 특화)만 B, A 로 고칩니다.
추가 규칙(논문 4.2, 공식 코드 start_lora_idx): 앞쪽 freeze_blocks 개 블록은 어댑터 없이 통째로 얼림.
공식 코드처럼 (1) 합쳐진 qkv 는 q/k/v 를 따로 SVD (공식 변환 스크립트가 qkv 를 q,k,v 로 쪼갬),
(2) 어댑터가 붙은 Linear 의 bias 도 학습 (SoraConfig bias="all") 합니다.

lora.py 와 같은 parametrize 방식이라 timm EvaAttention(DINOv3) 처럼 weight 를 직접 읽는 경우에도 동작하고,
저장 시 lora.merged_state_dict() 가 그대로 W_res + s·B·A 를 일반 weight 로 합쳐 줍니다 (predict/package 변경 없음).
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from .lora import TARGETS, merged_state_dict  # noqa: F401  (merged_state_dict 는 parametrize 일반형이라 SoMA 에도 그대로 씀)


class _SoMAWeight(nn.Module):
    """weight 파라미터화: W_res(original, 얼림) + s · B @ A.  chunks=3 이면 qkv 를 q/k/v 블록별로 따로 분해."""

    def __init__(self, w: torch.Tensor, r: int, alpha: float, chunks: int = 1):
        super().__init__()
        out_f, in_f = w.shape
        assert out_f % chunks == 0, (out_f, chunks)
        o = out_f // chunks
        assert r <= min(o, in_f), f"rank {r} > min({o}, {in_f})"
        self.scale = alpha / r
        self.chunks = chunks
        w64 = w.detach().to("cpu", torch.float64).reshape(chunks, o, in_f)
        U, S, Vh = torch.linalg.svd(w64, full_matrices=False)          # S 내림차순 → 끝 r 개가 minor
        sq = (S[:, -r:] / self.scale).sqrt()                            # 공식 코드: Sr /= scaling
        B = (U[:, :, -r:] * sq[:, None, :]).float()                     # (c, o, r)
        A = (sq[:, :, None] * Vh[:, -r:, :]).float()                    # (c, r, in)
        self.B = nn.Parameter(B.to(w.device))
        self.A = nn.Parameter(A.to(w.device))
        # 얼릴 잔차. fp32 로 반올림된 B, A 로 빼야 초기 출력이 정확히 같음 (논문: SVD 수치오차 방지용 W - BA 정의)
        self.w_res_init = (w64 - self.scale * (B.double() @ A.double())).reshape(out_f, in_f).to(w.dtype)
        self.sigma_minor = S[:, -r:].float()                            # 기록용 (분석·로그)

    def delta(self) -> torch.Tensor:
        d = self.B @ self.A                                             # (c, o, in)
        return self.scale * d.reshape(-1, d.shape[-1])

    def forward(self, w_res):
        return w_res + self.delta()


def apply_soma(vit: nn.Module, r: int = 16, alpha: float = 16, freeze_blocks: int = 0, targets=TARGETS,
               split_qkv: bool = True, train_bias: bool = True, **_ignored) -> int:
    """vit 전체를 얼리고, freeze_blocks 이후 블록의 대상 Linear 에 SoMA 어댑터를 붙입니다. 붙인 개수를 반환.

    alpha: 초기화가 Σ/s 로 보정되므로 초기 출력에는 영향 없음. 학습 중 유효 보폭만 바뀜 (공식 설정은 alpha = r, s = 1).
    split_qkv: 합쳐진 qkv 를 q/k/v 로 나눠 각각 r 개 minor 성분 (공식 구현과 같음). False 면 qkv 전체를 한 행렬로 SVD.
    train_bias: 어댑터가 붙은 Linear 의 bias 도 학습 (공식 SoraConfig bias="all"; 논문 Params* 4.9M 은 bias 포함 값).
    """
    for p in vit.parameters():
        p.requires_grad_(False)
    n = 0
    for name, mod in list(vit.named_modules()):
        parts = name.split(".")
        if not (isinstance(mod, nn.Linear) and parts[0] == "blocks" and parts[-1] in targets):
            continue
        if int(parts[1]) < freeze_blocks:
            continue                                                    # 앞쪽 블록: 어댑터 없이 완전 동결
        chunks = 3 if (split_qkv and parts[-1] == "qkv" and mod.out_features % 3 == 0) else 1
        p = _SoMAWeight(mod.weight, r, alpha, chunks)
        w_res = p.w_res_init
        del p.w_res_init
        parametrize.register_parametrization(mod, "weight", p)
        orig = mod.parametrizations.weight.original
        with torch.no_grad():
            orig.copy_(w_res)
        orig.requires_grad_(False)
        if train_bias and mod.bias is not None:
            mod.bias.requires_grad_(True)
        n += 1
    return n


def awd_factor(step: int, total: int, min_ratio: float = 0.0) -> float:
    """Annealed Weight Decay (논문 4.3, 기본 cosine): wd(step) = wd0 · awd_factor.  1 → min_ratio.

    공식 설정: wd0 = 0.05, eta_min = 1e-5 (min_ratio = 2e-4), 전체 학습 반복에 걸쳐 cosine.
    """
    t = min(max(step / max(total, 1), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def soma_modules(model: nn.Module):
    """(이름, _SoMAWeight) 목록. 분석·디버그용."""
    out = []
    for name, mod in model.named_modules():
        if parametrize.is_parametrized(mod, "weight"):
            for p in mod.parametrizations.weight:
                if isinstance(p, _SoMAWeight):
                    out.append((name, p))
    return out
