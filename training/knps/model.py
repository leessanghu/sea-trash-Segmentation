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


# DINOv3 위성 모델(SAT-493M, ViT-L/7B 만 공개)은 정규화 값이 다름 (공식 README·timm pretrained_cfg)
SAT493M_MEAN = (0.430, 0.411, 0.296)
SAT493M_STD = (0.213, 0.156, 0.143)


class _Normalize(nn.Module):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

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

    TAP_NORMS = ("ln", "bn")

    def __init__(self, vit="vit_small_patch14_dinov2.lvd142m", dino_size=448, layers=(2, 5, 8, 11), pretrained=True,
                 tap_norm="ln", up="bilinear", up_trainable=False, up_guide=None, up_key="last"):
        super().__init__()
        import timm

        self.norm = _Normalize(*((SAT493M_MEAN, SAT493M_STD) if "sat493m" in vit else (IMAGENET_MEAN, IMAGENET_STD)))
        self.dino_size = int(dino_size)
        self.layers = tuple(int(i) for i in layers)
        self.vit = timm.create_model(vit, pretrained=pretrained, img_size=self.dino_size, num_classes=0)
        depth = len(self.vit.blocks)
        if not self.layers or any(not 0 <= i < depth for i in self.layers) or len(set(self.layers)) != len(self.layers):
            raise ValueError(f"layers={self.layers} 가 {vit} 의 블록 수 {depth} (0..{depth - 1}) 와 맞지 않습니다")
        # tap_norm (DINOv3 논문 App. A.2): 중간층에는 특정 채널이 모든 패치에서 매우 큰 값(feature-dimension outlier)이 있음.
        #   "ln": 모든 탭에 마지막 LayerNorm 적용 (timm norm=True, 기존 동작·기존 체크포인트 호환)
        #   "bn": 탭은 그대로 두고 탭마다 BatchNorm (논문이 중간층에 권장한 방식. 마지막 LN 파라미터는 중간층엔 최적이 아님)
        if tap_norm not in self.TAP_NORMS:
            raise ValueError(f"tap_norm 은 {self.TAP_NORMS} 중 하나: {tap_norm}")
        self.tap_norm = tap_norm
        # up="anyup": 융합 특징(ViT 격자 32×32/28×28 → 스템 1/4 격자 64×64) 확대를 쌍선형 대신 AnyUp(ICLR'26)으로.
        # 주의 가중치는 [원해상도 정규화 영상 + ViT 마지막 층 특징(AnyUp 학습 조건과 같음)] 으로 만들고,
        # 그 가중치로 융합 특징(256ch)을 섞음 (출력 = 저해상 융합 특징의 볼록 결합). 가중치는 체크포인트 안에 저장됨
        self.up_key = up_key
        self.up = None
        if up == "anyup":
            from .anyup_up import ASSET, FeatUpsampler
            # pretrained=False(load_checkpoint 경로)면 파일·인터넷 없이 무작위 초기화 후 체크포인트 값으로 덮어씀
            self.up = FeatUpsampler(weights=str(ASSET) if pretrained else None, frozen=not up_trainable,
                                    guide_size=up_guide)
        elif up != "bilinear":
            raise ValueError(f"알 수 없는 up: {up}")
        c = self.vit.embed_dim
        if tap_norm == "bn":
            self.tap_bn = nn.ModuleList([nn.BatchNorm2d(c) for _ in self.layers])
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
        final, inter = self.vit.forward_intermediates(xv, indices=list(self.layers), norm=self.tap_norm == "ln",
                                                      output_fmt="NCHW")
        if self.tap_norm == "bn":
            inter = [bn(t) for bn, t in zip(self.tap_bn, inter)]
        npt = self.vit.num_prefix_tokens   # DINOv3: CLS + 레지스터 4 = 5 (레지스터는 분할 특징에서 제외됨)
        tok = torch.cat([final[:, 0], final[:, npt:].mean(1)], dim=1)
        f = self.fuse(torch.cat(inter, dim=1))
        s1 = self.stem1(x)
        s2 = self.stem2(s1)
        s4 = self.stem4(s2)
        if self.up is not None:
            key = inter[-1] if self.up_key == "last" else f
            f = self.up(x, key, s4.shape[-2:], values=f)
        else:
            f = F.interpolate(f, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec4(torch.cat([f, s4], 1))
        d = F.interpolate(d, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([d, s2], 1))
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat([d, s1], 1))
        return self.head(d), self.cls(tok)[:, 0]

    def backbone_parameters(self):
        return list(self.vit.parameters())

    def up_parameters(self):
        """학습되는 AnyUp 파라미터 (up_trainable 일 때만. train.py 가 낮은 학습률 그룹으로 분리)."""
        return [] if self.up is None else [p for p in self.up.parameters() if p.requires_grad]


def license_notes(spec: dict) -> list:
    """제출·양도 전에 확인이 필요한 사전학습 가중치 라이선스 메모."""
    name = str(spec.get("vit", "")) + str(spec.get("encoder", ""))
    notes = []
    if "dinov3" in name:
        notes.append("DINOv3 License (Meta): 재배포 시 라이선스 사본 동봉 + 'Built with DINOv3' 표기 의무. "
                     "수상 시 모델 양도와 충돌하지 않는지 운영진 확인 필요")
    if spec.get("up") == "anyup":
        notes.append("AnyUp (Wimmer et al., ICLR 2026, github.com/wimmerth/anyup) 코드·가중치 CC-BY-4.0: 출처 표기 필요")
    return notes


def build_model(spec: dict, pretrained: bool = False) -> nn.Module:
    kind = spec.get("kind", "smp")
    if kind == "smp":
        return SmpSegCls(spec.get("arch", "unet"), spec.get("encoder", "resnet34"), pretrained, spec.get("in_scale", 1))
    if kind == "dino":
        m = DinoUNet(spec.get("vit", "vit_small_patch14_dinov2.lvd142m"), spec.get("dino_size", 448),
                     tuple(spec.get("layers", (2, 5, 8, 11))), pretrained, spec.get("tap_norm", "ln"),
                     up=spec.get("up", "bilinear"), up_trainable=bool(spec.get("up_trainable", False)),
                     up_guide=spec.get("up_guide"), up_key=spec.get("up_key", "last"))
        if spec.get("lora"):   # 학습 중에만 (저장할 때 합쳐서 spec 에서 지우므로 추론 모델은 일반 DINO)
            cfg = dict(spec["lora"])
            if cfg.pop("mode", "lora") == "soma":   # SoMA: minor 특이성분 초기화 + 앞쪽 블록 동결 (knps/soma.py)
                from .soma import apply_soma
                apply_soma(m.vit, **cfg)
            else:
                from .lora import apply_lora
                apply_lora(m.vit, **cfg)
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
    if spec.get("lora"):   # LoRA/DoRA/SoMA 보정을 원래 가중치에 합쳐 일반 모델로 저장 (tune·package·predict 는 그대로)
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


@torch.no_grad()
def predict_probs_dual(models, x, tta: int = 8, amp: bool = True, w_det=None, w_seg=None, scales=(1.0,)):
    """판정용·마스크용 앙상블을 한 번의 추론으로 따로 평균합니다 (모델마다 forward 는 한 번).

    반환 (판정 확률맵, 판정 분류확률, 마스크 확률맵). 대회 점수 = 0.5 F1(패치 판정) + 0.5 형상(마스크)이라
    판정은 오탐·놓침에 강한 멤버로, 마스크는 경계가 정확한 멤버로 따로 가중할 수 있습니다.
    w_det == w_seg 이면 predict_probs 와 같은 값 (판정 확률맵 = 마스크 확률맵).
    """
    w_det = [1.0] * len(models) if w_det is None else [float(w) for w in w_det]
    w_seg = list(w_det) if w_seg is None else [float(w) for w in w_seg]
    ops = [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True), (3, False), (3, True)]
    ops = {1: ops[:1], 2: ops[:2], 4: [(k, False) for k in range(4)], 8: ops}[tta]
    use_amp = amp and x.is_cuda
    b, h, w = x.shape[0], x.shape[-2], x.shape[-1]
    det_sum = torch.zeros(b, h, w, device=x.device)
    seg_sum = torch.zeros(b, h, w, device=x.device)
    cls_sum = torch.zeros(b, device=x.device)
    nd = ns = 0.0
    for s in scales:
        xs = x if s == 1 else F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        for m, wd, wsg in zip(models, w_det, w_seg):
            if wd <= 0 and wsg <= 0:
                continue
            for k, flip in ops:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    seg, cls = m(_d4(xs, k, flip))
                prob = _d4_inv(torch.sigmoid(seg.float()), k, flip)
                if s != 1:
                    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
                if wd > 0:
                    det_sum += wd * prob[:, 0]
                    cls_sum += wd * torch.sigmoid(cls.float())
                    nd += wd
                if wsg > 0:
                    seg_sum += wsg * prob[:, 0]
                    ns += wsg
    return det_sum / nd, cls_sum / nd, seg_sum / ns
