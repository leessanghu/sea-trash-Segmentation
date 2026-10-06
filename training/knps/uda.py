"""2단계 자기학습(UDA): EMA teacher 의사라벨 + DACS 섞기 + MIC 가리기 (이진 분할 + 패치 분류 헤드용).

근거 논문 (scratchpad/deep/papers, 코드 scratchpad/deep/code 에서 확인)
  - VFM-UDA / VFM-UDA++ (Englert 2024/2025, tue-mps/vfm-uda[-plusplus]):
      EMA α=0.999 (초반 min(1-1/(t+1), α)), teacher 는 좌우반전 평균, 확률 0.968 이상 화소 비율을
      배치 전체의 '품질' 스칼라로 곱함, DACS(ClassMix), 마스크 비율 0.7, 손실마다 따로 역전파.
  - MIC (Hoyer 2023, lhoyer/MIC): 64px 블록(1024 입력 기준, 반해상도면 32), r=0.7, λ_M=1,
      정규화 '후' 0 으로 채움(= 평균색), 색 증강 후 가림. 희소 객체(검출)는 r=0.5 로 낮춤.
  - DACS (Tranheden 2021): 소스 정답의 '클래스 절반' 화소를 대상 영상에 붙임. 의사라벨만 쓰면
    소수 클래스가 사라짐(GTA→CS: 소스만 32.9 → 의사라벨만 23.0 → DACS 52.1 mIoU).

우리 문제에 맞춘 변경 (논문에 없는 것은 '변경'으로 표시)
  - 이진이라 '클래스 절반' = 쓰레기 영역(팽창 dilate px). 소스에 쓰레기가 없으면 CutMix 사각형으로 대체.
  - [변경] 품질 가중치를 양성/음성 따로 계산 (논문은 전체 화소 중 확신 비율 하나 → 이진·희소 양성에선
    배경이 늘 99% 라 의미가 없음). 확신 문턱도 양성은 낮게(class-balanced), 사이 구간은 무시.
  - [변경] 음성 의사라벨 가중치 상한 neg_cap (전부 음성으로 쏠리는 확증 편향 완화).
  - [변경] 예측 양성 주변 neg_ring px 는 음성으로 쓰지 않음 (평가 정답은 우리 예측보다 넓은 쪽이라
    주변을 음성으로 가르치면 마스크가 좁아지는 쪽으로 끌림).
  - [변경] MIC 에서 가려진 블록 안의 양성 의사라벨은 무시 (작은 쓰레기는 문맥만으로 추론 불가).
  - [변경] MIC 순전파는 BatchNorm 이동평균을 갱신하지 않음 (회색 블록이 추론용 BN 통계를 오염시키지 않게).
  - [변경] pos_mode='ignore': 쓰레기가 없도록 골라 둔 배경 풀(bg_ports 등)에서는 teacher 양성을 정답으로
    쓰지 않고 무시만 함 (그 양성은 대부분 오탐일 것이므로 강화하면 안 됨).
  - [변경] teacher 패치 확률로 대상 배치를 고르는 양성 과표집(oversample).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .augment import AUG2, d4, random_shadow, rhm, sat_degrade, scale_jitter, scene_color
from .data import CROP, _imread
from .model import IMAGENET_MEAN

DEFAULT_CFG = {
    # 의사라벨
    "thr_pos": 0.75,         # 이 확률 이상이면 확신 양성 (논문 0.968 은 19클래스 softmax 기준이라 이진·희소 양성엔 너무 높음)
    "thr_neg": 0.10,         # 이 확률 이하면 확신 음성 (= 음성 확신도 0.9). 사이 구간은 손실에서 제외
    "pos_keep": None,        # 예: 0.5 → 배치 안 예측양성(p>0.5) 화소의 상위 50% 는 thr_pos 보다 낮아도 양성으로 (CBST식 클래스 균형 문턱)
    "min_pos_px": 10,        # 영상 안 확신 양성 화소가 이보다 적으면 그 양성은 무시 (점 잡음)
    "pos_mode": "label",     # 'label' | 'ignore' (쓰레기 없는 배경 풀)
    "pos_weight": 1.0,       # 양성 의사라벨 화소 가중치 배수
    "neg_cap": 0.5,          # 음성 의사라벨 화소 가중치 상한 (품질 q_neg 와 min)
    "neg_ring": 4,           # 예측 양성 주변 이 폭(px)은 음성 의사라벨에서 제외 (평가 정답이 우리 예측보다 넓음)
    "teacher_tta": 2,        # 1 원본 | 2 좌우반전 평균 (VFM-UDA) | 4 회전 4
    # 손실 가중치
    "lam": 1.0,              # 대상 손실 전체 배수 (--uda-weight)
    "start": 0,              # 이 전역 스텝부터 켬 (--uda-start)
    "ramp": 100,             # 켠 뒤 lam 을 0→1 로 올리는 스텝 수
    "mix_weight": 1.0,
    "mic_weight": 1.0,       # MIC 논문 λ_M=1 최적 (0.5: -1.1, 2.0: -0.5, 10: 기준 이하)
    "cls_weight": 0.5,       # 대상 패치 분류 soft-target 손실
    "src_cls_weight": 0.5,   # 소스 분류 손실 (train.py --cls-weight)
    # DACS
    "mix_dilate": 3,         # 붙이는 쓰레기 영역 팽창(px): 가장자리 문맥 포함
    "mix_cutmix": True,      # 소스에 쓰레기가 없으면 사각형 CutMix 로 대체
    # MIC
    "mic_patch": 32,         # 블록 크기(px, 256 입력). MIC 표8: b 를 너무 키우면(이미지 1/4) 손해
    "mic_ratio": 0.5,        # 희소 객체라 0.7 대신 0.5 (MIC 검출 설정과 같은 이유)
    "mic_ignore_masked_pos": True,
    "bn_update": ("src", "mix"),   # BN 이동평균을 갱신할 순전파
    # 분류 soft-target 문턱
    "cls_thr_pos": 0.7,
    "cls_thr_neg": 0.2,
    # 양성 과표집 (로더가 tgt_bs*oversample 장을 주면 teacher 패치 확률로 tgt_bs 장을 고름)
    "oversample": 1,
    "oversample_top": 0.5,   # 고른 것 중 teacher 패치 확률 상위 비율 (나머지는 무작위 → 음성 장면 유지)
    "tgt_bs": 0,             # 0 이면 들어온 대상 배치 크기 그대로
}


def make_cfg(over=None, **kw):
    cfg = dict(DEFAULT_CFG)
    cfg.update(over or {})
    cfg.update({k: v for k, v in kw.items() if v is not None})
    unknown = set(cfg) - set(DEFAULT_CFG)
    if unknown:
        raise KeyError(f"알 수 없는 UDA 설정: {sorted(unknown)}")
    return cfg


# --------------------------------------------------------------------------------------------
# 비라벨 데이터
# --------------------------------------------------------------------------------------------
def _parse_dirs(dirs):
    """'a,b:2' 또는 리스트 → [(Path, weight)]. 폴더에 images/ 가 있으면 그 안을 씀."""
    parts = dirs.split(",") if isinstance(dirs, str) else list(dirs)
    out = []
    for part in [str(s).strip() for s in parts if str(s).strip()]:
        path, _, w = part.rpartition(":")
        try:
            weight = float(w)
        except ValueError:
            path, weight = part, 1.0
        p = Path(path)
        out.append((p / "images" if (p / "images").is_dir() else p, weight))
    return out


class UnlabeledDataset(Dataset):
    """라벨 없는 영상 풀. 반환: (x_weak, x_strong, valid, index)

    x_weak  : teacher 입력. 무작위 256 자르기 + D4 + 약한 스케일 지터 + (확률적) 위성 열화(sat_degrade).
              위성 열화를 teacher 쪽에도 거는 이유: 대상(평가) 영상이 0.3m 위성이라 풀(항공 정사영상)을
              위성처럼 보이게 한 상태에서 의사라벨을 만드는 것이 목적에 맞음.
    x_strong: student 입력. x_weak 에 화소 위치를 바꾸지 않는 광학 증강만 추가
              (그림자, [AUG2] RHM, 장면 색 보정, 잡음) → 의사라벨과 화소 정렬 유지.
    valid   : 스케일 축소·패딩으로 생긴 테두리 = 0 (의사라벨 손실에서 제외).
    weights : 폴더별 가중치를 반영한 표본 가중치 (WeightedRandomSampler 용).
    """

    SKIP = ("contact_sheet", "sheet", "preview")   # 폴더에 섞인 미리보기 그림 (예: bg_naip_train2/contact_sheet_100.jpg)

    def __init__(self, dirs, crop=CROP, p_scale=0.3, p_degrade_weak=0.5, p_shadow=0.3, p_color=0.9,
                 p_noise=0.3, max_per_dir=None, seed=0, exts=(".png",)):
        self.crop = int(crop)
        self.p_scale, self.p_degrade_weak = p_scale, p_degrade_weak
        self.p_shadow, self.p_color, self.p_noise = p_shadow, p_color, p_noise
        self.files, weights, self.sources = [], [], []
        rs = np.random.default_rng(seed)
        for d, w in _parse_dirs(dirs):
            fs = sorted(p for p in d.glob("*") if p.suffix.lower() in exts
                        and not any(k in p.name.lower() for k in self.SKIP))
            if not fs:
                raise FileNotFoundError(f"비라벨 폴더에 영상이 없습니다: {d}")
            if max_per_dir and len(fs) > max_per_dir:
                fs = [fs[i] for i in sorted(rs.choice(len(fs), max_per_dir, replace=False))]
            self.files += fs
            weights += [w / len(fs)] * len(fs)
            self.sources.append((str(d), len(fs), w))
        self.weights = torch.tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.files)

    def _crop(self, img, rng):
        h, w = img.shape[:2]
        c = self.crop
        valid = np.zeros((max(h, c), max(w, c)), np.uint8)   # 0 = 유효, 2 = 패딩 (scale_jitter 규약과 같음)
        if h < c or w < c:
            py, px = max(c - h, 0), max(c - w, 0)
            img = np.pad(img, ((0, py), (0, px), (0, 0)), mode="reflect")
            valid[h:, :] = 2
            valid[:, w:] = 2
            h, w = img.shape[:2]
        y = int(rng.integers(0, h - c + 1))
        x = int(rng.integers(0, w - c + 1))
        return img[y:y + c, x:x + c], valid[y:y + c, x:x + c]

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,)).item()))
        img = _imread(self.files[i])
        img, lab = self._crop(img, rng)
        img = img.astype(np.float32) / 255.0
        img, lab = d4(img, lab, rng)
        if rng.random() < self.p_scale:
            img, lab = scale_jitter(img, lab, rng)
        weak = sat_degrade(img, rng) if rng.random() < self.p_degrade_weak else np.ascontiguousarray(img)
        strong = weak
        if rng.random() < self.p_shadow:
            strong = random_shadow(strong, rng)
        if AUG2 and rng.random() < 0.5:
            strong = rhm(strong, rng)
        if rng.random() < self.p_color:
            strong = scene_color(strong, rng)
        if rng.random() < self.p_noise:
            strong = strong + rng.normal(0, rng.uniform(0.003, 0.015), strong.shape).astype(np.float32)
        strong = np.clip(strong, 0, 1).astype(np.float32)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        valid = torch.from_numpy((lab != 2).astype(np.float32))[None]
        return to_t(np.clip(weak, 0, 1).astype(np.float32)), to_t(strong), valid, i


# --------------------------------------------------------------------------------------------
# 의사라벨
# --------------------------------------------------------------------------------------------
class EnsembleTeacher(nn.Module):
    """고정 앙상블 teacher (예: v5 멤버 6개). 확률 평균을 다시 로짓으로 돌려 teacher_predict 에 그대로 씀.

    train.py 의 기본 teacher 는 ema.model(단일 멤버) 입니다. 단일 멤버는 real_val 양성 화소의 59% 를
    확신 음성(p<=0.1)으로 봐서(scratchpad/deep/uda_test/gt_neg.py, v5 model0) 앙상블이 더 안전할 수 있습니다.
    이 경우 teacher 가 갱신되지 않으므로 '자기학습'이 아니라 '앙상블 증류 + 섞기/가리기'가 됩니다.
    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        ps, pc = 0.0, 0.0
        for m in self.models:
            s, c = m(x)
            ps = ps + torch.sigmoid(s.float())
            pc = pc + torch.sigmoid(c.float())
        n = len(self.models)
        return torch.logit((ps / n).clamp(1e-6, 1 - 1e-6)), torch.logit((pc / n).clamp(1e-6, 1 - 1e-6))


@torch.no_grad()
def teacher_predict(teacher, x, tta=2, amp=False):
    """teacher 확률맵 (B,1,H,W) 과 패치 확률 (B,). 확률(시그모이드 후)을 평균 (VFM-UDA 의 softmax 평균과 같은 방식)."""
    was = teacher.training
    teacher.eval()
    ops = {1: [(0, False)], 2: [(0, False), (0, True)], 4: [(k, False) for k in range(4)]}[int(tta)]
    ps, pc = 0.0, 0.0
    for k, flip in ops:
        xi = torch.flip(x, dims=[-1]) if flip else x
        xi = torch.rot90(xi, k, dims=[-2, -1])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            seg, cls = teacher(xi)
        p = torch.sigmoid(seg.float())
        p = torch.rot90(p, -k, dims=[-2, -1])
        ps = ps + (torch.flip(p, dims=[-1]) if flip else p)
        pc = pc + torch.sigmoid(cls.float())
    teacher.train(was)
    return ps / len(ops), pc / len(ops)


@torch.no_grad()
def binary_pseudo_label(teacher_prob, thr_pos=0.75, thr_neg=0.10, valid=None, pos_keep=None, min_pos_px=0,
                        neg_cap=0.5, pos_weight=1.0, pos_mode="label", neg_ring=0):
    """이진 의사라벨. teacher_prob (B,1,H,W) [0,1].

    반환 (target, weight, quality)
      target : 확신 양성 1, 나머지 0
      weight : 확신 양성 = q_pos*pos_weight, 확신 음성 = min(q_neg, neg_cap), 사이 구간·무효 = 0
      quality: dict(q_pos, q_neg, pos_rate, pos_img, ign_rate: (B,) 텐서 / thr_pos: 실제 쓴 양성 문턱(float))
        q_pos = 예측양성(p>0.5) 중 확신 양성 비율, q_neg = 예측음성 중 확신 음성 비율 (MIC 의 q 를 클래스별로)
        pos_rate = 유효 화소 중 양성 의사라벨 비율, pos_img = 양성 의사라벨이 있는 영상인지(0/1)
    """
    p = teacher_prob.float()
    B = p.shape[0]
    v = torch.ones_like(p) if valid is None else (valid > 0.5).float()
    thr = float(thr_pos)
    if pos_keep:   # 클래스 균형 문턱: 예측양성의 상위 pos_keep 은 thr_pos 보다 낮아도 확신 양성 (단 0.5 아래로는 안 내려감)
        pp = p[(p > 0.5) & (v > 0)]
        if pp.numel() > 0:
            q = torch.quantile(pp[:200000].float(), 1 - float(pos_keep)).item()
            thr = min(thr, max(0.5, q))
    conf_pos = (p >= thr) & (v > 0)
    conf_neg = (p <= thr_neg) & (v > 0)
    if neg_ring > 0:   # 예측 양성(p>0.5) 주변 neg_ring px 는 음성으로 쓰지 않음: 마스크를 좁히는 방향(LB 와 반대)으로 끌지 않게
        near = F.max_pool2d((p > 0.5).float(), 2 * neg_ring + 1, 1, neg_ring) > 0
        conf_neg = conf_neg & ~near
    if min_pos_px > 0:
        few = conf_pos.flatten(1).sum(1) < min_pos_px
        conf_pos = conf_pos & ~few.view(B, 1, 1, 1)
    pred_pos = ((p > 0.5) & (v > 0)).flatten(1).sum(1).float()
    pred_neg = ((p <= 0.5) & (v > 0)).flatten(1).sum(1).float()
    q_pos = torch.where(pred_pos > 0, conf_pos.flatten(1).sum(1) / pred_pos.clamp(min=1), torch.ones_like(pred_pos))
    q_neg = torch.where(pred_neg > 0, conf_neg.flatten(1).sum(1) / pred_neg.clamp(min=1), torch.ones_like(pred_neg))
    w_pos = (q_pos * pos_weight).view(B, 1, 1, 1)
    w_neg = q_neg.clamp(max=neg_cap).view(B, 1, 1, 1)
    if pos_mode == "label":
        target = conf_pos.float()
        weight = conf_pos.float() * w_pos + conf_neg.float() * w_neg
    elif pos_mode == "ignore":
        target = torch.zeros_like(p)
        weight = conf_neg.float() * w_neg
    else:
        raise ValueError(pos_mode)
    nv = v.flatten(1).sum(1).clamp(min=1)
    quality = {
        "q_pos": q_pos, "q_neg": q_neg, "thr_pos": thr,
        "pos_rate": conf_pos.flatten(1).sum(1) / nv,
        "pos_img": conf_pos.flatten(1).any(1).float(),
        "ign_rate": ((weight == 0) & (v > 0)).flatten(1).sum(1) / nv,
    }
    return target, weight, quality


# --------------------------------------------------------------------------------------------
# DACS (이진판 ClassMix) / MIC
# --------------------------------------------------------------------------------------------
def _cutmix_box(h, w, gen, device, lo=0.25, hi=0.5):
    area = float(torch.empty(1).uniform_(lo, hi, generator=gen))
    ar = float(torch.empty(1).uniform_(0.5, 2.0, generator=gen))
    bh = int(min(h, round((area * h * w * ar) ** 0.5)))
    bw = int(min(w, round((area * h * w / ar) ** 0.5)))
    y = int(torch.randint(0, h - bh + 1, (1,), generator=gen))
    x = int(torch.randint(0, w - bw + 1, (1,), generator=gen))
    m = torch.zeros(1, h, w, device=device)
    m[:, y:y + bh, x:x + bw] = 1
    return m


@torch.no_grad()
def dacs_mix(src_x, src_y, src_w, tgt_x, tgt_pl, tgt_w, dilate=3, cutmix=True, min_src_px=1, gen=None):
    """소스 쓰레기 영역(+dilate px 테두리)을 대상 영상에 붙임. 라벨·가중치도 같은 마스크로 섞음.

    DACS 는 소스 정답 클래스의 절반을 붙임. 이진이면 '쓰레기' 한 클래스를 붙이는 것이 같은 역할
    (소스 배경 클래스를 고르면 사실상 소스 전체를 붙이게 돼 대상이 거의 안 남음).
    대상 i 마다 소스는 쓰레기가 있는 것 중에서 돌아가며 고르고, 하나도 없으면 cutmix=True 일 때
    무작위 소스의 사각형(면적 25~50%)을 붙임 (DACS 표3: CutMix 로 바꿔도 48.7 mIoU, 클래스 소멸 없음).
    반환 (x_mix, y_mix, w_mix, mask (B,1,H,W), src_index (B,) long, 붙인 영역의 소스 양성 화소 수 (B,))
    """
    Bt, _, H, W = tgt_x.shape
    dev = tgt_x.device
    pos_px = (src_y * src_w).flatten(1).sum(1)
    pos_idx = torch.nonzero(pos_px >= min_src_px).flatten().tolist()
    gen = gen or torch.Generator().manual_seed(int(torch.randint(0, 2 ** 31, (1,))))
    order = [pos_idx[int(k)] for k in torch.randperm(len(pos_idx), generator=gen)] if pos_idx else []
    masks, sidx = [], []
    for i in range(Bt):
        if order:
            j = order[i % len(order)]
            m = (src_y[j] > 0.5).float()
            if dilate > 0:
                m = F.max_pool2d(m[None], 2 * dilate + 1, 1, dilate)[0]
        else:
            j = int(torch.randint(0, src_x.shape[0], (1,), generator=gen))
            m = _cutmix_box(H, W, gen, dev) if cutmix else torch.zeros(1, H, W, device=dev)
        masks.append(m)
        sidx.append(j)
    mask = torch.stack(masks)
    sidx = torch.tensor(sidx, device=dev, dtype=torch.long)
    sx, sy, sw = src_x[sidx], src_y[sidx], src_w[sidx]
    x_mix = mask * sx + (1 - mask) * tgt_x
    y_mix = mask * sy + (1 - mask) * tgt_pl
    w_mix = mask * sw + (1 - mask) * tgt_w
    pasted_pos = (mask * sy * sw).flatten(1).sum(1)
    return x_mix, y_mix, w_mix, mask, sidx, pasted_pos


@torch.no_grad()
def mic_mask(x, patch=32, ratio=0.7, fill=None, gen=None):
    """MIC 블록 마스크: patch x patch 블록을 확률 ratio 로 가림. 반환 (가린 영상, keep (B,1,H,W) 1=보임).

    모델 입력이 [0,1] 이고 정규화가 모델 안에 있으므로, MIC/VFM-UDA++ 처럼 '정규화 후 0' 과 같아지도록
    ImageNet 평균색으로 채움 (검정 0 으로 채우면 우리 도메인에선 그림자·어두운 바다처럼 보임).
    """
    B, C, H, W = x.shape
    gh, gw = -(-H // patch), -(-W // patch)
    r = torch.rand((B, 1, gh, gw), generator=gen).to(x.device)
    keep = (r > ratio).float()
    keep = F.interpolate(keep, scale_factor=patch, mode="nearest")[:, :, :H, :W]
    if fill is None:
        fill = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * keep + fill * (1 - keep), keep


# --------------------------------------------------------------------------------------------
# 손실
# --------------------------------------------------------------------------------------------
def pl_loss(logits, y, w, dice=True):
    """가중 BCE(전체 화소 평균 = MIC 식 '(loss*weight).mean()', 그래서 품질 가중치의 크기가 그대로 반영됨)
    + 양성 의사라벨이 있는 영상만 Dice (그 영상의 양성 평균 가중치를 곱함)."""
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
    if not dice:
        return bce
    m = (w > 0).float()
    t = (y * m).flatten(1)
    has = t.sum(1) >= 1
    if not has.any():
        return bce
    p = (torch.sigmoid(logits) * m).flatten(1)
    inter = (p * t).sum(1)
    d = 1 - (2 * inter + 1) / (p.sum(1) + t.sum(1) + 1)
    wp = ((w * y).flatten(1).sum(1) / t.sum(1).clamp(min=1)).detach()
    return bce + (d * wp)[has].mean()


class _BNStatsFrozen:
    """이 순전파에서는 BatchNorm 이 배치 통계로 정규화하되 이동평균(running stats)은 갱신하지 않게 함."""

    def __init__(self, model, on=True):
        self.mods = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)] if on else []

    def __enter__(self):
        self.saved = [m.momentum for m in self.mods]
        for m in self.mods:
            m.momentum = 0.0
        return self

    def __exit__(self, *exc):
        for m, mo in zip(self.mods, self.saved):
            m.momentum = mo
        return False


def _cl(x):
    return x.contiguous(memory_format=torch.channels_last)


def uda_step(student, teacher, batch_src, batch_tgt, seg_loss_fn, cfg, backward=None, amp=False, step=None):
    """한 스텝의 손실 계산 (+ backward 를 주면 항목별로 따로 역전파 → 최대 메모리 ≈ 순전파 1회분).

    batch_src: (x, y, w, patch_label)  SegDataset 출력 (device 위)
    batch_tgt: (x_weak, x_strong, valid, idx)  UnlabeledDataset 출력 (device 로 옮김)
    seg_loss_fn(logits, y, w): train.seg_loss (BCE+Dice, 채점 제외 가중치)
    backward: 예) lambda l: scaler.scale(l).backward(). None 이면 out['loss'] 를 직접 역전파.
    반환 dict: loss, src, src_cls, mix, mic, cls_uda (float 또는 텐서), lam, 의사라벨 통계.
    """
    cfg = make_cfg(cfg)
    xs, ys, ws, lab = batch_src
    dev = xs.device
    xw, xst, valid = (t.to(dev, non_blocking=True) for t in batch_tgt[:3])
    lam = float(cfg["lam"])
    if step is not None and cfg["ramp"] > 0:
        lam *= float(np.clip((step - cfg["start"] + 1) / cfg["ramp"], 0.0, 1.0))
    out = {"lam": lam}
    upd = set(cfg["bn_update"])

    def _bw(loss):
        if backward is not None:
            backward(loss)
            return loss.detach()
        return loss

    # 1) 소스 지도학습 (기존 train.py 와 같은 손실)
    with _BNStatsFrozen(student, "src" not in upd), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        seg, cls = student(_cl(xs))
    l_src = seg_loss_fn(seg.float(), ys, ws)
    valid_lab = lab >= 0
    l_scls = (F.binary_cross_entropy_with_logits(cls.float()[valid_lab], lab[valid_lab]) if valid_lab.any()
              else cls.float().sum() * 0)
    l_source = _bw(l_src + cfg["src_cls_weight"] * l_scls)
    out.update(src=l_src.detach(), src_cls=l_scls.detach())
    del seg, cls

    # 2) teacher 의사라벨 (+ 양성 과표집)
    prob, pcls = teacher_predict(teacher, _cl(xw), cfg["teacher_tta"], amp)
    tb = int(cfg["tgt_bs"]) or xw.shape[0]
    if xw.shape[0] > tb:
        k = int(round(tb * cfg["oversample_top"]))
        order = torch.argsort(pcls, descending=True)
        top, rest = order[:k], order[k:]
        rest = rest[torch.randperm(len(rest), device=rest.device)[:tb - k]]
        sel = torch.cat([top, rest])
        xw, xst, valid, prob, pcls = xw[sel], xst[sel], valid[sel], prob[sel], pcls[sel]
    y_t, w_t, q = binary_pseudo_label(prob, cfg["thr_pos"], cfg["thr_neg"], valid, cfg["pos_keep"],
                                      cfg["min_pos_px"], cfg["neg_cap"], cfg["pos_weight"], cfg["pos_mode"],
                                      cfg["neg_ring"])

    # 3) DACS 섞기 + 분류 soft-target
    x_m, y_m, w_m, mmask, sidx, pasted = dacs_mix(xs, ys, ws, xst, y_t, w_t, cfg["mix_dilate"], cfg["mix_cutmix"])
    pasted_pos = pasted >= max(cfg["min_pos_px"], 1)
    soft = pcls.clone()
    cw = ((pcls >= cfg["cls_thr_pos"]) | (pcls <= cfg["cls_thr_neg"])).float()
    if cfg["pos_mode"] == "ignore":
        cw = cw * (pcls <= cfg["cls_thr_neg"]).float()
    soft = torch.where(pasted_pos, torch.ones_like(soft), soft)
    cw = torch.where(pasted_pos, torch.ones_like(cw), cw)
    with _BNStatsFrozen(student, "mix" not in upd), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        seg_m, cls_m = student(_cl(x_m))
    l_mix = pl_loss(seg_m.float(), y_m, w_m)
    l_cls = (F.binary_cross_entropy_with_logits(cls_m.float(), soft, reduction="none") * cw).mean()
    _bw(lam * (cfg["mix_weight"] * l_mix + cfg["cls_weight"] * l_cls))
    out.update(mix=l_mix.detach(), cls_uda=l_cls.detach())
    del seg_m, cls_m

    # 4) MIC: 가린 대상 영상 → 전체 영상 의사라벨과 일치
    l_mic = torch.zeros((), device=dev)
    if cfg["mic_weight"] > 0:
        x_k, keep = mic_mask(xst, cfg["mic_patch"], cfg["mic_ratio"])
        w_k = w_t
        if cfg["mic_ignore_masked_pos"]:
            w_k = w_t * (1 - (1 - keep) * (y_t > 0.5).float())
        with _BNStatsFrozen(student, "mic" not in upd), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            seg_k, _ = student(_cl(x_k))
        l_mic = pl_loss(seg_k.float(), y_t, w_k)
        _bw(lam * cfg["mic_weight"] * l_mic)
        del seg_k
    out["mic"] = l_mic.detach()

    if backward is None:
        out["loss"] = l_source + lam * (cfg["mix_weight"] * l_mix + cfg["cls_weight"] * l_cls
                                        + cfg["mic_weight"] * l_mic)
    else:
        out["loss"] = (l_source + lam * (cfg["mix_weight"] * out["mix"] + cfg["cls_weight"] * out["cls_uda"]
                                         + cfg["mic_weight"] * out["mic"])).detach()
    out.update({
        "pl_pos_rate": float(q["pos_rate"].mean()), "pl_pos_img": float(q["pos_img"].mean()),
        "q_pos": float(q["q_pos"].mean()), "q_neg": float(q["q_neg"].mean()), "thr_pos": q["thr_pos"],
        "pl_ign_rate": float(q["ign_rate"].mean()), "teacher_cls": float(pcls.mean()),
        "teacher_pmax": float(prob.flatten(1).amax(1).mean()),
        "mix_pos_img": float(pasted_pos.float().mean()),
    })
    return out


class UDAMonitor:
    """의사라벨 통계 누적 + '전부 음성으로 쏠림' 경고.

    처음 window 스텝의 양성 의사라벨 영상 비율을 기준으로 잡고, 최근 window 평균이 그 collapse 배 밑으로
    떨어지면 warn=True. (DACS 표3: 섞기 없이 의사라벨만 쓰면 소수 클래스가 0 으로 사라짐)
    """

    KEYS = ("src", "src_cls", "mix", "mic", "cls_uda", "pl_pos_rate", "pl_pos_img", "q_pos", "q_neg",
            "pl_ign_rate", "teacher_cls", "teacher_pmax", "mix_pos_img", "thr_pos")

    def __init__(self, window=50, collapse=0.3):
        self.window, self.collapse = window, collapse
        self.hist = {k: [] for k in self.KEYS}
        self.base = None
        self.warn = False

    def update(self, out):
        for k in self.KEYS:
            if k in out:
                v = out[k]
                self.hist[k].append(float(v.item() if torch.is_tensor(v) else v))
        h = self.hist["pl_pos_img"]
        if self.base is None and len(h) >= self.window:
            self.base = float(np.mean(h[:self.window]))
        if self.base and len(h) >= 2 * self.window:
            self.warn = float(np.mean(h[-self.window:])) < self.collapse * self.base

    def mean(self, k, last=None):
        h = self.hist[k][-last:] if last else self.hist[k]
        return float(np.mean(h)) if h else float("nan")

    def summary(self, last=None):
        s = (f"uda mix {self.mean('mix', last):.4f} mic {self.mean('mic', last):.4f} cls {self.mean('cls_uda', last):.4f}"
             f" | PL 양성화소 {self.mean('pl_pos_rate', last):.3%} 양성영상 {self.mean('pl_pos_img', last):.1%}"
             f" q+ {self.mean('q_pos', last):.2f} q- {self.mean('q_neg', last):.2f} 무시 {self.mean('pl_ign_rate', last):.1%}"
             f" teacher패치 {self.mean('teacher_cls', last):.3f}")
        if self.warn:
            s += f"  [경고] 양성 의사라벨이 초기({self.base:.1%})의 {self.collapse:.0%} 미만 → 전부 음성 쏠림 의심"
        return s
