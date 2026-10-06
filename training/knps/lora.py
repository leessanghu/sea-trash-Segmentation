"""ViT 백본용 LoRA / DoRA (peft 없이 직접 구현).

원래 가중치 W0 는 얼리고, 작은 보정만 학습합니다.
  LoRA: W = W0 + (alpha/r)·B·A
  DoRA: W = m · (W0 + (alpha/r)·B·A) / ||W0 + (alpha/r)·B·A||  (출력 채널마다 크기 m 과 방향을 따로 학습)

nn.Linear 를 감싸지 않고 weight 자체를 바꾸는 방식(parametrize)이라, timm 이 F.linear(x, self.qkv.weight, ...) 처럼
가중치를 직접 읽는 경우(DINOv3 의 EvaAttention)에도 보정이 빠지지 않습니다.
저장할 때 merged_state_dict() 로 보정을 원래 가중치에 합치므로 제출 모델은 일반 ViT 와 같습니다 (추론 속도·의존성 그대로).
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

# fc1_g / fc1_x: SwiGLU MLP (DINOv3 ViT-S+/H+, timm SwiGLU). 빠뜨리면 S+ 는 MLP 확장층에 보정이 안 붙음
TARGETS = ("qkv", "proj", "fc1", "fc1_g", "fc1_x", "fc2")


class _LoRAWeight(nn.Module):
    def __init__(self, w0: torch.Tensor, r: int, alpha: float, dora: bool):
        super().__init__()
        out_f, in_f = w0.shape
        self.scale = alpha / r
        self.dora = dora
        self.A = nn.Parameter(torch.empty(r, in_f))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_f, r))       # 0 으로 시작 → 처음에는 원래 모델과 똑같음
        if dora:
            self.m = nn.Parameter(w0.detach().norm(dim=1, keepdim=True).clone())

    def forward(self, w0):
        w = w0 + self.scale * (self.B @ self.A)
        if self.dora:
            # 논문 권장대로 크기 정규화 항은 역전파에서 상수로 취급 (메모리 절약, 값은 같음)
            w = self.m * w / w.norm(dim=1, keepdim=True).detach()
        return w


def apply_lora(vit: nn.Module, r: int = 16, alpha: float = 32, dora: bool = False, targets=TARGETS) -> int:
    """vit 의 모든 파라미터를 얼리고, 블록 안의 대상 Linear 에 LoRA/DoRA 를 붙입니다. 붙인 개수를 반환."""
    for p in vit.parameters():
        p.requires_grad_(False)
    n = 0
    for name, mod in list(vit.named_modules()):
        if isinstance(mod, nn.Linear) and name.startswith("blocks.") and name.split(".")[-1] in targets:
            parametrize.register_parametrization(mod, "weight", _LoRAWeight(mod.weight, r, alpha, dora))
            mod.parametrizations.weight.original.requires_grad_(False)
            n += 1
    return n


def merged_state_dict(model: nn.Module) -> dict:
    """보정을 원래 가중치에 합친 '일반 모델'의 state_dict (LoRA/DoRA 키 없음). 학습 중인 모델은 건드리지 않음.

    주의: parametrize.remove_parametrizations 를 복사본에 쓰면 안 됩니다. 복사본과 원본이 같은 ParametrizedLinear
    클래스를 공유해서, 복사본에서 지운 weight 속성이 원본에서도 사라집니다 (두 번째 저장·다음 학습에서 오류).
    """
    merged = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if parametrize.is_parametrized(mod, "weight"):
                merged[name] = mod.weight.detach().clone()      # W0 + 보정 (DoRA 면 크기·방향 반영)
    tag = ".parametrizations.weight."
    sd = {}
    for k, v in model.state_dict().items():
        if tag in k:
            if k.endswith(tag + "original"):
                base = k[: -len(tag + "original")]
                sd[base + ".weight"] = merged[base]
            continue                                             # A, B, m 은 버림
        sd[k] = v.detach()
    return sd
