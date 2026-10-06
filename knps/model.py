"""모델 정의와 체크포인트 입출력, D4 TTA 추론.

모든 모델은 [0,1] 범위 RGB 텐서 (B,3,H,W) 를 받아 (분할 로짓 (B,1,H,W), 분류 로짓 (B,)) 을 돌려줍니다.
정규화는 모델 안에서 하므로 추론 코드는 모델 종류를 몰라도 됩니다.

spec 예시
  {"kind": "smp", "arch": "unet", "encoder": "resnet34"}
  {"kind": "dino", "vit": "vit_small_patch14_dinov2.lvd142m", "dino_size": 448}
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class SmpSegCls(nn.Module):
    """segmentation_models_pytorch 모델 + 인코더 최심층 풀링 분류 헤드(aux_params)."""

    ARCHS = {"unet": "Unet", "unetpp": "UnetPlusPlus", "fpn": "FPN", "upernet": "UPerNet", "manet": "MAnet"}

    def __init__(self, arch="unet", encoder="resnet34", pretrained=True, in_scale=1):
        super().__init__()
        import segmentation_models_pytorch as smp

        cls = getattr(smp, self.ARCHS[arch])
        self.in_scale = int(in_scale)   # 2 이면 입력을 2배로 키워 작은 쓰레기를 더 크게 보고, 출력은 원래 크기로 평균 축소
        self.norm = _Normalize()
        self.net = cls(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                       in_channels=3, classes=1, aux_params={"classes": 1, "dropout": 0.2})

    def forward(self, x):
        x = self.norm(x)
        if self.in_scale > 1:
            x = F.interpolate(x, scale_factor=self.in_scale, mode="bilinear", align_corners=False)
        seg, cls = self.net(x)
        if self.in_scale > 1:
            seg = F.avg_pool2d(seg, self.in_scale)
        return seg, cls[:, 0]

    def backbone_parameters(self):
        return list(self.net.encoder.parameters())


def _cba(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DinoUNet(nn.Module):
    """DINOv2 ViT 특징(의미) + 얕은 CNN 스템(1~4배 해상도 세부)을 합친 소형 U-Net.

    ViT 패치(14px)가 0.3m 영상에서 너무 굵기 때문에 입력을 dino_size 로 키워 넣고,
    가는 띠 모양 경계는 원해상도 CNN 스템에서 복원합니다.
    """

    def __init__(self, vit="vit_small_patch14_dinov2.lvd142m", dino_size=448, layers=(2, 5, 8, 11), pretrained=True):
        super().__init__()
        import timm

        self.norm = _Normalize()
        self.dino_size = int(dino_size)
        self.layers = tuple(layers)
        self.vit = timm.create_model(vit, pretrained=pretrained, img_size=self.dino_size, num_classes=0)
        c = self.vit.embed_dim
        self.fuse = nn.Sequential(nn.Conv2d(c * len(self.layers), 256, 1, bias=False), nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True), _cba(256, 256))
        self.stem1 = nn.Sequential(_cba(3, 32), _cba(32, 32))
        self.stem2 = nn.Sequential(_cba(32, 64, 2), _cba(64, 64))
        self.stem4 = nn.Sequential(_cba(64, 96, 2), _cba(96, 96))
        self.dec4 = nn.Sequential(_cba(256 + 96, 128), _cba(128, 128))
        self.dec2 = nn.Sequential(_cba(128 + 64, 64), _cba(64, 64))
        self.dec1 = nn.Sequential(_cba(64 + 32, 32), _cba(32, 32))
        self.head = nn.Conv2d(32, 1, 1)
        self.cls = nn.Sequential(nn.Linear(c * 2, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        x = self.norm(x)
        h, w = x.shape[-2:]
        xv = F.interpolate(x, size=(self.dino_size, self.dino_size), mode="bilinear", align_corners=False)
        final, inter = self.vit.forward_intermediates(xv, indices=list(self.layers), norm=True, output_fmt="NCHW")
        npt = self.vit.num_prefix_tokens
        tok = torch.cat([final[:, 0], final[:, npt:].mean(1)], dim=1)
        f = self.fuse(torch.cat(inter, dim=1))
        s1 = self.stem1(x)
        s2 = self.stem2(s1)
        s4 = self.stem4(s2)
        f = F.interpolate(f, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec4(torch.cat([f, s4], 1))
        d = F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([d, s2], 1))
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat([d, s1], 1))
        return self.head(d), self.cls(tok)[:, 0]

    def backbone_parameters(self):
        return list(self.vit.parameters())


def license_notes(spec: dict) -> list:
    """제출·양도 전에 확인이 필요한 사전학습 가중치 라이선스 메모."""
    name = str(spec.get("vit", "")) + str(spec.get("encoder", ""))
    if "dinov3" in name:
        return ["DINOv3 License (Meta): 재배포 시 라이선스 사본 동봉 + 'Built with DINOv3' 표기 의무. "
                "수상 시 모델 양도와 충돌하지 않는지 운영진 확인 필요"]
    return []


def build_model(spec: dict, pretrained: bool = False) -> nn.Module:
    kind = spec.get("kind", "smp")
    if kind == "smp":
        return SmpSegCls(spec.get("arch", "unet"), spec.get("encoder", "resnet34"), pretrained, spec.get("in_scale", 1))
    if kind == "dino":
        m = DinoUNet(spec.get("vit", "vit_small_patch14_dinov2.lvd142m"), spec.get("dino_size", 448),
                     tuple(spec.get("layers", (2, 5, 8, 11))), pretrained)
        if spec.get("lora"):   # 학습 중에만 (저장할 때 합쳐서 spec 에서 지우므로 추론 모델은 일반 DINO)
            from .lora import apply_lora
            apply_lora(m.vit, **spec["lora"])
        return m
    raise ValueError(f"알 수 없는 모델 종류: {kind}")


def init_from_baseline(model: SmpSegCls, baseline_path) -> list:
    """주최 측 UNet-ResNet18(2클래스 softmax) 가중치를 1채널 시그모이드 모델로 옮깁니다.

    softmax 의 debris 확률 p1 = sigmoid(z1 - z0) 이므로 분할 헤드는 (w1 - w0, b1 - b0) 로 정확히 변환됩니다.
    분류 헤드는 새로 학습합니다. 옮기지 못한 키 목록을 반환합니다.
    """
    sd = torch.load(baseline_path, map_location="cpu", weights_only=True)["state_dict"]
    new = {}
    for k, v in sd.items():
        if k.startswith("segmentation_head.0."):
            v = v[1] - v[0]
            v = v.unsqueeze(0)
        new["net." + k] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    return [k for k in missing if not k.startswith("net.classification_head")] + list(unexpected)


def save_checkpoint(path, model, spec, meta=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if spec.get("lora"):   # LoRA/DoRA 보정을 원래 가중치에 합쳐 일반 모델로 저장 (tune·package·predict 는 그대로)
        from .lora import merged_state_dict
        sd = {k: v.cpu() for k, v in merged_state_dict(model).items()}
        meta = {**(meta or {}), "lora": spec["lora"]}
        spec = {k: v for k, v in spec.items() if k != "lora"}
    else:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"spec": spec, "state_dict": sd, "meta": meta or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["spec"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["spec"]


def _d4(x, k, flip):
    if flip:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k, dims=[-2, -1])


def _d4_inv(x, k, flip):
    x = torch.rot90(x, -k, dims=[-2, -1])
    if flip:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def predict_probs(models, x, tta: int = 8, amp: bool = True, weights=None, scales=(1.0,)):
    """x: (B,3,H,W) [0,1]. 모델·TTA 평균 (debris 확률맵 (B,H,W), 분류 확률 (B,)) 을 반환합니다.

    tta: 1(원본), 2(좌우반전 추가), 4(회전 4), 8(D4 전체)
    weights: 모델별 가중치 (없으면 모두 1). tune.py --search-weights 가 찾아 config 에 저장합니다
    scales: 크기 바꾼 TTA. 예: (1.0, 1.25) 면 원본과 1.25배로 키운 입력의 평균 (확률맵은 원래 크기로 되돌림)
    """
    weights = [1.0] * len(models) if weights is None else [float(w) for w in weights]
    ops = [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True), (3, False), (3, True)]
    ops = {1: ops[:1], 2: ops[:2], 4: [(k, False) for k in range(4)], 8: ops}[tta]
    use_amp = amp and x.is_cuda
    seg_sum = torch.zeros(x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
    cls_sum = torch.zeros(x.shape[0], device=x.device)
    n = 0.0
    h, w = x.shape[-2:]
    for s in scales:
        xs = x if s == 1 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        for m, wgt in zip(models, weights):
            if wgt <= 0:
                continue
            for k, flip in ops:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    seg, cls = m(_d4(xs, k, flip))
                prob = _d4_inv(torch.sigmoid(seg.float()), k, flip)
                if s != 1:
                    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
                seg_sum += wgt * prob[:, 0]
                cls_sum += wgt * torch.sigmoid(cls.float())
                n += wgt
    return seg_sum / n, cls_sum / n
