r"""분할+분류 멀티태스크 학습.

예:
  python train.py --model unet_r18_base      # synth_train / synth_val 로 학습 → <데이터폴더>/runs/unet_r18_base
  python train.py --model dino_s --bs 24
  python train.py --model unet_r34 --train C:/sea_trash_data/synth_train:1,C:/sea_trash_data/real_train:3 \
                  --val C:/sea_trash_data/real_val,C:/sea_trash_data/synth_val
  (<데이터폴더>: 로컬 C:\sea_trash_data, 코랩 /content/data. 환경변수 SEA_TRASH_DATA 로 변경)

--val 에 여러 폴더를 주면 첫 번째 폴더 점수로 최고 모델을 고릅니다 (실제 라벨 검증셋이 있으면 맨 앞에).
검증 점수는 대회 산식(0.5*macroF1 + 0.5*형상)이며, 판정 임계값 t_img 만 빠르게 맞춘 값입니다.
전체 후처리 탐색은 학습 후 tune.py 로 합니다.
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from knps.data import TINY_PX, build_train, contest_gts, dumped_drop, list_items
from knps.postprocess import DEFAULT_CFG as _DCFG
from knps.evaluate import QUICK_GRID, infer_items, search_cfg
from knps.model import build_model, init_from_baseline, save_checkpoint
from knps.soma import awd_factor
from knps.paths import data_path, ensure_outside_onedrive

PRESETS = {
    # 주최 측 베이스라인 가중치(실제 양성 4장 반영)에서 시작
    "unet_r18_base": {"kind": "smp", "arch": "unet", "encoder": "resnet18"},
    "unet_r34": {"kind": "smp", "arch": "unet", "encoder": "resnet34"},
    "unet_r50": {"kind": "smp", "arch": "unet", "encoder": "resnet50"},
    "unetpp_r34": {"kind": "smp", "arch": "unetpp", "encoder": "resnet34"},
    "unet_effb3": {"kind": "smp", "arch": "unet", "encoder": "efficientnet-b3"},
    "dino_s": {"kind": "dino", "vit": "vit_small_patch14_dinov2.lvd142m", "dino_size": 448},
    "dino_b": {"kind": "dino", "vit": "vit_base_patch14_dinov2.lvd142m", "dino_size": 448},
    # ConvNeXt-T (timm convnext_tiny.in12k_ft_in1k, Apache-2.0, ImageNet 계열): 작은 점에 강한 CNN
    "unet_cnxt_t": {"kind": "smp", "arch": "unet", "encoder": "tu-convnext_tiny"},
    # DINOv3 ViT-S/16 (Meta DINOv3 License). 상업 이용은 되지만 재배포 시 라이선스 동봉·"Built with DINOv3" 표기 의무.
    # 수상 시 모델 양도 조건과 맞는지 운영진 확인 전에는 최종 제출에서 빼 둘 수 있도록 별도 앙상블로 관리하세요
    "dino3_s": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 448},
    # 작은 쓰레기 대응: 입력을 2배(256→512)로 키워 넣고 출력을 256 으로 되돌림 (계산량 약 4배)
    "unet_r18_base_x2": {"kind": "smp", "arch": "unet", "encoder": "resnet18", "in_scale": 2},
    "unet_cnxt_t_x2": {"kind": "smp", "arch": "unet", "encoder": "tu-convnext_tiny", "in_scale": 2},
    # 원래 가중치는 얼리고 작은 보정만 학습 (사전학습 특징을 더 많이 보존 → 드론→위성 차이에 강할 것으로 기대, 앙상블 다양성).
    # 저장할 때 보정을 합치므로 제출 모델은 일반 DINO 와 같음. 백본을 엇갈려 써서 서로 덜 닮게 함
    "dino3_s_lora": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 448,
                     "lora": {"r": 16, "alpha": 32, "dora": False}},
    "dino_s_dora": {"kind": "dino", "vit": "vit_small_patch14_dinov2.lvd142m", "dino_size": 448,
                    "lora": {"r": 16, "alpha": 32, "dora": True}},
    # [exp] Plan A "같은 계열 키우기": ViT-Base, 해상도↑ (scratchpad/test_models*.py 로 빌드·역전파 확인, HF 게이트 없음)
    "dino3_b": {"kind": "dino", "vit": "vit_base_patch16_dinov3.lvd1689m", "dino_size": 448},       # DINOv3 License
    "dino_b_dora": {"kind": "dino", "vit": "vit_base_patch14_dinov2.lvd142m", "dino_size": 448,
                    "lora": {"r": 16, "alpha": 32, "dora": True}},
    # [exp] SoMA (Yun+ CVPR'25, knps/soma.py): 가장 작은 r 개 특이성분만 학습 + 앞 freeze_blocks 블록 통째 동결
    #       + adapter 그룹에 annealed weight decay(--awd-wd → 0, cosine). 저장 시 합쳐지므로 제출 모델은 일반 DINO.
    #       freeze_blocks=3: 논문 규칙 "헤드로 가는 첫(최고해상도) 특징을 만드는 블록까지 동결" → layers[0]=2 → 블록 0~2
    "dino3_s_soma": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 448,
                     "lora": {"r": 16, "alpha": 32, "mode": "soma", "freeze_blocks": 3}},
    "dino_s_soma": {"kind": "dino", "vit": "vit_small_patch14_dinov2.lvd142m", "dino_size": 448,
                    "lora": {"r": 16, "alpha": 32, "mode": "soma", "freeze_blocks": 3}},
    "dino3_b_soma": {"kind": "dino", "vit": "vit_base_patch16_dinov3.lvd1689m", "dino_size": 448,   # DINOv3 License
                     "lora": {"r": 16, "alpha": 32, "mode": "soma", "freeze_blocks": 3}},
    "dino3_s_560": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 560},   # DINOv3 License
    "dino_s_518": {"kind": "dino", "vit": "vit_small_patch14_dinov2.lvd142m", "dino_size": 518},
    # [exp] DINOv3 백본 선택 (scratchpad/deep/dinov3_presets.patch). 근거: DINOv3 논문 Tab.14·Fig.11·App.A.2 +
    # 실제 연안 타일 고정특징 선형탐침(scratchpad/deep/probe). 탭은 모두 (2,5,8,11) = 공식 FOUR_EVEN_INTERVALS 이자 탐침 최고
    # 해상도: 256px 타일 → 512 (32x32 패치 = 논문 밀집평가 표준 1024 토큰, 고해상 적응 최소 전역크롭 512). FLOPs x1.20
    # 탐침 AP: S@448 대비 +0.009 / +0.014 (타일셋 2개, 8접힘 중 7 양수). S@256(확대 없음)은 -0.13 → 확대는 필수
    "dino3_s_512_lora": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 512,
                         "lora": {"r": 16, "alpha": 32, "dora": False}},
    "dino3_s_512": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 512},
    # 큰 백본: 논문 Tab.14 ADE20k +4.8(S→B). 탐침은 +0.018 / -0.012 로 불확실 (고정특징 한계). 448 유지 (B@512 이득 없음,
    # FLOPs x2.9 vs x3.7). 체크포인트 353MB(fp32)
    "dino3_b_lora": {"kind": "dino", "vit": "vit_base_patch16_dinov3.lvd1689m", "dino_size": 448,
                     "lora": {"r": 16, "alpha": 32, "dora": False}},
    # ViT-S+/16: 깊이 12·폭 384 로 S 와 같고 MLP 만 SwiGLU(29M, 123MB). 논문 Tab.14 ADE20k 47.0(S)→48.8(S+)→51.8(B) 이지만
    # 우리 탐침에서는 S 보다 낮음 (AP -0.011 / -0.036) → 주력이 아닌 다양성용. LoRA 대상에 fc1_g/fc1_x 추가 필요 (knps/lora.py TARGETS)
    "dino3_splus_lora": {"kind": "dino", "vit": "vit_small_plus_patch16_dinov3.lvd1689m", "dino_size": 448,
                         "lora": {"r": 16, "alpha": 32, "dora": False}},
    # 선택: 더 높은 해상도. 탐침 +0.018 / +0.026 AP (8접힘 중 7 양수)로 가장 좋지만 FLOPs x1.68, MPS 실측 추론 ~x2.1,
    # 학습 메모리 ~x1.8 → T4 에서는 --bs 를 줄여야 함
    "dino3_s_640_lora": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 640,
                         "lora": {"r": 16, "alpha": 32, "dora": False}},
    "unet_cnxt_s_x2": {"kind": "smp", "arch": "unet", "encoder": "tu-convnext_small", "in_scale": 2},
    # [exp] Plan B "새 계열로 다양성"
    "unet_mit_b3_x2": {"kind": "smp", "arch": "unet", "encoder": "mit_b3", "in_scale": 2},
    "fpn_mit_b2_x2": {"kind": "smp", "arch": "fpn", "encoder": "mit_b2", "in_scale": 2},
    "unet_effv2_s_x2": {"kind": "smp", "arch": "unet", "encoder": "tu-tf_efficientnetv2_s", "in_scale": 2},
    # [Plan C/C2 2026-10-06] CNN 판정 멤버 강화 + 경계가 정확한 CNN(UNet++ 촘촘한 스킵)으로 형상 보강
    "unet_r34_x2": {"kind": "smp", "arch": "unet", "encoder": "resnet34", "in_scale": 2},
    "unet_r50_x2": {"kind": "smp", "arch": "unet", "encoder": "resnet50", "in_scale": 2},
    "unet_effb3_x2": {"kind": "smp", "arch": "unet", "encoder": "efficientnet-b3", "in_scale": 2},
    "unetpp_r34_x2": {"kind": "smp", "arch": "unetpp", "encoder": "resnet34", "in_scale": 2},
    "unetpp_effb3_x2": {"kind": "smp", "arch": "unetpp", "encoder": "efficientnet-b3", "in_scale": 2},
    "unetpp_r50_x2": {"kind": "smp", "arch": "unetpp", "encoder": "resnet50", "in_scale": 2},
    # [exp] AnyUp(ICLR'26, CC-BY-4.0) 특징 업샘플러: ViT 융합 특징을 1/4 격자로 키울 때 쌍선형 대신 영상 안내 창 주의.
    # 공식 가중치(assets/anyup.pth, 3.5MB)로 시작해 얼려 둠(up_trainable=False). 가중치는 체크포인트 안에 저장되어 추론은 오프라인.
    # up_guide=128: 안내 영상을 256→128 로 줄여 넣음 (출력 64×64 기준 guide256 과 코사인 1.0000, FLOP 증가 +68%→+26%)
    "dino3_s_lora_up": {"kind": "dino", "vit": "vit_small_patch16_dinov3.lvd1689m", "dino_size": 448,
                        "lora": {"r": 16, "alpha": 32, "dora": False}, "up": "anyup", "up_trainable": False, "up_guide": 128},
    "dino_s_dora_up": {"kind": "dino", "vit": "vit_small_patch14_dinov2.lvd142m", "dino_size": 448,
                       "lora": {"r": 16, "alpha": 32, "dora": True}, "up": "anyup", "up_trainable": False, "up_guide": 128},
}
BASELINE = Path("assets/model/unet_r18_debris_lite.pt")


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        msd = model.state_dict()
        for k, e in self.model.state_dict().items():
            if e.dtype.is_floating_point:
                e.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                e.copy_(msd[k])


# ---- [exp] 실험 스위치 (환경변수로 켬. 기본은 원래 v5 레시피와 같음) ----
EXP_GT_DILATE = int(os.environ.get("EXP_GT_DILATE", "0"))   # 학습 정답을 n 화소 팽창 (평가 정답이 더 넓다는 가설)
EXP_LOSS = os.environ.get("EXP_LOSS", "dice")              # dice | tolf (대회 형상점수를 흉내 낸 1화소 허용 F)


def tol_f_loss(logits, y, w=None, beta=1.0, eps=1.0):
    """1화소 허용 F-score(대회 형상점수)의 미분가능 대리. scratchpad/tolloss.py 에서 하드 마스크로 대회 metric 과 일치 확인."""
    if w is None:
        w = torch.ones_like(y)
    p = torch.sigmoid(logits) * w
    t = y * w
    pd = F.max_pool2d(p, 3, 1, 1)
    td = F.max_pool2d(t, 3, 1, 1)
    prec = ((p * td).flatten(1).sum(1) + eps) / (p.flatten(1).sum(1) + eps)
    rec = ((t * pd).flatten(1).sum(1) + eps) / (t.flatten(1).sum(1) + eps)
    b2 = beta * beta
    f = (1 + b2) * prec * rec / (b2 * prec + rec)
    pos = t.flatten(1).sum(1) > 0
    return (1 - f[pos]).mean() if pos.any() else logits.sum() * 0


def seg_loss(logits, y, w=None):
    """BCE + Dice(쓰레기 있는 패치만). w: 손실에 쓰는 화소(채점 제외 화소 = 0)."""
    if w is None:
        w = torch.ones_like(y)
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum().clamp(min=1)
    p = (torch.sigmoid(logits) * w).flatten(1)
    t = (y * w).flatten(1)
    pos = t.sum(1) > 0
    if not pos.any():
        return bce
    if EXP_LOSS == "tolf":
        return bce + tol_f_loss(logits, y, w)
    inter = (p * t).sum(1)
    dice = 1 - (2 * inter + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + dice[pos].mean()


def resume_state(out, spec, model, ema, opt, scaler, steps):
    """끊긴 학습 상태를 불러옵니다. 반환 (다음 시작 에폭, 전역 스텝, 최고 점수)."""
    rp, lp = out / "resume.pt", out / "last.pt"
    if rp.exists():
        st = torch.load(rp, map_location="cpu", weights_only=False)
        if st["spec"] != spec:
            sys.exit(f"--model 이 저장된 학습과 다릅니다: 저장 {st['spec']} / 지금 {spec}")
        model.load_state_dict(st["model"])
        ema.model.load_state_dict(st["ema"])
        ema.step = st["ema_step"]
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        print(f"이어서 학습: {st['epoch']}에폭 완료 상태에서 재개 (옵티마이저·학습률 일정 포함), best {st['best']:.4f}")
        return st["epoch"], st["step"], st["best"]
    if lp.exists():
        # resume.pt 를 저장하기 전 버전으로 학습한 경우: EMA 가중치와 에폭만 알 수 있음
        ck = torch.load(lp, map_location="cpu", weights_only=False)
        if ck["spec"] != spec:
            sys.exit(f"--model 이 저장된 학습과 다릅니다: 저장 {ck['spec']} / 지금 {spec}")
        model.load_state_dict(ck["state_dict"])
        ema.model.load_state_dict(ck["state_dict"])
        ep = int(ck["meta"]["epoch"])
        ema.step = ep * steps
        best = -1.0
        if (out / "best.pt").exists():
            v = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["meta"].get("val") or []
            best = float(v[0]) if v else -1.0
        print(f"이어서 학습: last.pt 에서 {ep}에폭 완료 상태로 재개. 예전 버전 체크포인트라 "
              f"옵티마이저 상태는 새로 시작합니다(학습률 일정은 이어짐). best {best:.4f}")
        return ep, ep * steps, best
    sys.exit(f"이어서 할 체크포인트가 없습니다: {out}")


def fixed_score(probs, cls, gts, neg_weight):
    """고정 후처리로 대회 산식 채점. score/f1/shape 를 돌려줍니다."""
    from knps.metric import patch_shape_score
    from knps.postprocess import postprocess
    tp = fn = fp = tn = 0
    shapes = []
    for p, c, g in zip(probs, cls, gts):
        m = postprocess(p.astype(np.float32), float(c), FIXED_CFG)
        pp = int(m.sum()) >= 50
        if g.any():
            tp, fn = tp + pp, fn + (not pp)
            shapes.append(patch_shape_score(m, g) if pp else 0.0)
        else:
            fp, tn = fp + pp, tn + (not pp)
    fp, tn = fp * neg_weight, tn * neg_weight
    f1 = (2 * tp / max(2 * tp + fp + fn, 1e-9) + 2 * tn / max(2 * tn + fn + fp, 1e-9)) / 2
    sh = float(np.mean(shapes)) if shapes else 0.0
    return {"score": 0.5 * f1 + 0.5 * sh, "f1": f1, "shape": sh}


FIXED_CFG = {**_DCFG, "t_pix": 0.5, "t_img": 0.4, "alpha": 0.25, "min_cc": 10, "close": 0,
             "rescue_px": 80, "dilate": 1, "min_area": 0}   # 리더보드 0.7079 설정


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_r18_base", choices=list(PRESETS))
    ap.add_argument("--train", default=None, help="'폴더:가중치' 를 쉼표로. 기본: <데이터폴더>/synth_train")
    ap.add_argument("--val", default=None, help="검증 폴더(쉼표로). 첫 번째로 모델 선택. 기본: <데이터폴더>/synth_val")
    ap.add_argument("--paste", default="", help="실제 쓰레기 조각 온라인 붙여넣기: '조각폴더,배경폴더1,배경폴더2,...'")
    ap.add_argument("--paste-weight", type=float, default=0.25, help="붙여넣기 소스의 샘플 비중")
    ap.add_argument("--test-pos-ratio", type=float, default=0.0,
                    help="테스트 양성 비율 추정치(예: 0.64). 주면 에폭마다의 검증 점수(최고 에폭 선택 기준)를 "
                         "검증셋과 테스트의 비율 차이만큼 음성 가중치로 보정")
    ap.add_argument("--out", default=None, help="기본: <데이터폴더>/runs/<model>")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500, help="에폭당 스텝")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=None, help="기본: smp 0.3, dino 0.1, LoRA/DoRA/SoMA 1.0 (SoMA 논문은 0.5)")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--awd-wd", type=float, default=0.05,
                    help="SoMA 전용: 어댑터(백본) 그룹 weight decay 시작값. cosine 으로 --awd-min 배까지 줄임 (논문 4.3, 공식 0.05)")
    ap.add_argument("--awd-min", type=float, default=0.0, help="SoMA AWD 최종 비율 (공식 코드 eta_min 1e-5/0.05 = 2e-4)")
    ap.add_argument("--cls-weight", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2),
                    help="데이터 증강 워커 수. 기본: CPU 코어 수(최대 8). 무료 코랩 T4 는 2, L4·A100 은 더 많음")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--init", default="", help="가중치만 가져와 새로 학습할 체크포인트(.pt, spec 포함). 학습률 일정은 처음부터")
    ap.add_argument("--reapply-lora", action="store_true",
                    help="[DiWA] --init 체크포인트(합쳐진 일반 DINO) 위에 --model 프리셋의 LoRA/DoRA 를 다시 붙여 학습")
    ap.add_argument("--resume", action="store_true",
                    help="--out 폴더의 끊긴 학습을 이어서 (같은 명령에 --resume 만 붙이면 됨)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny-px", type=int, default=TINY_PX,
                    help="외딴 작은 쓰레기 묶음 기준(px). 이보다 작으면 학습에서 채점 제외, 검증에서는 정답에서 빼고 "
                         "그것만 있던 패치는 평가 제외 (대회 정답 단위 = 수 m 이상 집적대). 0 이면 끔")
    ap.add_argument("--confuser-boost", type=float, default=2.0,
                    help="confusers.csv 꼬리표(밝은 자갈·조개/포말/방파제)가 붙은 음성 타일을 이 배수로 더 자주 뽑음. 1 이면 끔")
    ap.add_argument("--dumped", action="store_true",
                    help="도로·건물 20m 이내 쓰레기(투기물 의심)를 학습에서 채점 제외, 검증에서는 정답에서 뺌 (masks_dumped)")
    # ---- ADELE (잡음 라벨 교정, knps/adele.py) ----
    ap.add_argument("--adele", action="store_true",
                    help="ADELE: 합성 폴더별 학습 IoU 곡선이 꺾이면(early learning 끝) EMA 교사 확신 화소로 합성 라벨을 즉석 교정")
    ap.add_argument("--adele-groups", default="cp_pos_train,cp_band_train,synth_train,cp_typed_train,"
                                              "cp_typed_train_b,cp_typed_train_c",
                    help="교정할(잡음이 있다고 보는) 학습 폴더 이름들, 쉼표로. real_*/bg_* 는 넣어도 무시. 붙여넣기 소스는 'paste'")
    ap.add_argument("--adele-tau", type=float, default=0.8, help="교정 확신도 τ (쓰레기 라벨→배경: 교사 P(배경)>τ)")
    ap.add_argument("--adele-tau-add", type=float, default=None, help="배경 라벨→쓰레기 τ. 기본: --adele-tau")
    ap.add_argument("--adele-add-mode", default="pos_patch", choices=["none", "pos_patch", "all"],
                    help="배경→쓰레기 교정 범위. pos_patch: 원래 쓰레기가 있는 패치에서만 (공식 class constraint)")
    ap.add_argument("--adele-r", type=float, default=0.9, help="교정 시작 기준: 기울기 상대 변화 > r (논문 0.9)")
    ap.add_argument("--adele-every", type=int, default=50, help="학습 IoU 를 몇 스텝마다 한 점으로 기록할지")
    ap.add_argument("--adele-fallback", type=float, default=0.6,
                    help="곡선 맞춤이 끝내 안 되는 그룹은 전체 스텝의 이 비율에서 강제로 교정 시작")
    ap.add_argument("--adele-force-onset", type=int, default=-1, help="(시험용) 이 스텝부터 모든 대상 그룹 교정 시작")
    ap.add_argument("--adele-teacher-scales", default="1.0",
                    help="교정에 쓰는 EMA 교사 출력의 축척들(평균). 논문은 0.7,1.0,1.5 (교사 추론 3배)")
    ap.add_argument("--adele-ms", type=float, default=0.0,
                    help="다중 축척 일관성 손실 가중치 λ (논문 1). 0 이면 끔. 학생 순전파·역전파가 약 3.7배")
    ap.add_argument("--adele-ms-scales", default="0.7,1.0,1.5", help="일관성 손실 축척 (크기는 32 배수로 반올림)")
    ap.add_argument("--adele-verbose", action="store_true", help="스텝마다 교정 화소 수 출력")
    # [exp] 2단계 자기학습(UDA): EMA teacher(=ema.model) 의사라벨 + DACS 섞기 + MIC 가리기 (knps/uda.py)
    ap.add_argument("--uda", default="",
                    help="비라벨 영상 폴더(쉼표, '폴더:가중치'). 주면 자기학습을 켭니다. 권장: --init <체크포인트> "
                         "--epochs 3~4 --steps 500 --lr 1.5e-4 (0.3x). 테스트셋은 절대 넣지 말 것")
    ap.add_argument("--uda-weight", type=float, default=1.0, help="대상 손실(섞기+MIC+분류) 전체 배수")
    ap.add_argument("--uda-start", type=int, default=0, help="이 전역 스텝부터 UDA 를 켬 (그 전엔 기존 학습과 같음)")
    ap.add_argument("--uda-bs", type=int, default=0, help="비라벨 배치 크기 (0 이면 --bs)")
    ap.add_argument("--uda-cfg", default="",
                    help="knps.uda.DEFAULT_CFG 덮어쓰기 JSON. 예: '{\"thr_pos\":0.8,\"pos_mode\":\"ignore\",\"oversample\":2}'")
    a = ap.parse_args()
    if a.uda and a.adele:
        sys.exit("--uda 와 --adele 은 함께 쓸 수 없습니다 (둘 다 teacher 로 라벨을 바꿈). 하나씩 A/B 하세요")
    a.train = a.train or f"{data_path('synth_train')}:1"
    if a.val is None:
        a.val = str(data_path("synth_val")) if data_path("synth_val").is_dir() else ""
    a.out = a.out or str(data_path("runs", a.model))
    ensure_outside_onedrive(a.out)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")  # 로컬 맥북 MPS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(PRESETS[a.model])

    print(f"[exp] EXP_LOSS={EXP_LOSS} EXP_GT_DILATE={EXP_GT_DILATE}", flush=True)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        spec = ck["spec"]
        model = build_model(spec, pretrained=False)
        model.load_state_dict(ck["state_dict"])
        print("init from", a.init)
        lora = PRESETS[a.model].get("lora")
        if a.reapply_lora and lora and spec.get("kind") == "dino" and not spec.get("lora"):
            # [DiWA] 공유 초기화(합쳐 저장된 일반 DINO)에서 다시 LoRA/DoRA 로 학습. B=0 이라 시작 함수는 init 과 같음
            if {k: v for k, v in PRESETS[a.model].items() if k != "lora"} != spec:
                sys.exit(f"--reapply-lora: init spec {spec} 이 --model {a.model} 과 다릅니다")
            from knps.lora import apply_lora
            apply_lora(model.vit, **lora)
            spec = {**spec, "lora": lora}
            print(f"  init 위에 {'DoRA' if lora.get('dora') else 'LoRA'} r={lora['r']} 다시 붙임")
    elif a.model.startswith("unet_r18_base"):
        model = build_model(spec, pretrained=False)
        left = init_from_baseline(model, BASELINE)
        print("init from baseline", BASELINE, "미이관 키:", left)
    else:
        model = build_model(spec, pretrained=not a.no_pretrained)
    model = model.to(device).to(memory_format=torch.channels_last)

    soma = bool(spec.get("lora")) and spec["lora"].get("mode") == "soma"
    mult = a.backbone_lr_mult
    if mult is None:   # LoRA/DoRA/SoMA 는 보정만 학습하므로 백본 학습률을 낮추지 않음 (SoMA 공식 설정은 0.5: --backbone-lr-mult 0.5)
        mult = 1.0 if spec.get("lora") else (0.1 if spec["kind"] == "dino" else 0.3)
    if spec.get("lora"):
        n_tr = sum(p.numel() for p in model.backbone_parameters() if p.requires_grad)
        kind = "SoMA" if soma else ("DoRA" if spec["lora"].get("dora") else "LoRA")
        print(f"  {kind} r={spec['lora']['r']}: 백본 학습 파라미터 {n_tr / 1e6:.2f}M (lr x{mult})"
              + (f", AWD {a.awd_wd}→{a.awd_wd * a.awd_min:g}" if soma else ""))
    bb = {id(p) for p in model.backbone_parameters()}
    upp = {id(p) for p in model.up_parameters()} if hasattr(model, "up_parameters") else set()
    groups = [{"params": [p for p in model.parameters() if id(p) in bb], "lr": a.lr * mult},
              {"params": [p for p in model.parameters() if id(p) not in bb and id(p) not in upp], "lr": a.lr}]
    if soma:
        groups[0]["weight_decay"] = a.awd_wd   # 어댑터 그룹만 AWD (헤드는 --wd 고정)
    if upp:   # 학습하는 AnyUp: 사전학습 업샘플러가 망가지지 않게 낮은 학습률 (논문 처음 학습 2e-4 의 절반 수준)
        groups.append({"params": [p for p in model.parameters() if id(p) in upp], "lr": a.lr * 0.2})
        print(f"  AnyUp 학습 파라미터 {sum(p.numel() for p in model.up_parameters()) / 1e6:.2f}M (lr x0.2)")
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    base_lrs = [g["lr"] for g in groups]
    total = a.epochs * a.steps
    warm = min(500, total // 10)

    ds, w, info = build_train(a.train, a.paste, a.paste_weight, a.tiny_px, a.confuser_boost, a.dumped,
                              return_meta=a.adele)
    n_pos = info["n_mask"]
    print(f"  학습 데이터(파일) {info['n_items']}장 중 마스크 파일 있음 {n_pos}장 ({n_pos / max(info['n_items'], 1):.1%})")
    if n_pos == 0 and not info.get("paste"):
        sys.exit("학습 데이터에 마스크가 하나도 없습니다. synth_train.tar 가 불완전하게 올라갔을 수 있습니다 "
                 "(정상이면 사진 30000장 / 마스크 약 15000장). 로컬에서 .tar 를 다시 묶어 올리세요")
    sampler = WeightedRandomSampler(w, num_samples=a.steps * a.bs, replacement=True)
    dl = DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers, pin_memory=device == "cuda",
                    drop_last=True, persistent_workers=a.workers > 0)
    vals = [(Path(v), list_items(v)) for v in a.val.split(",") if v]
    drops = [dumped_drop(its) if a.dumped else None for _, its in vals]
    for v, its in vals:
        print(f"  val {v}: {len(its)} 장, 양성 {sum(m is not None for _, m in its)}"
              + (f" (외딴 {a.tiny_px}px 미만" + (" 또는 투기물 의심" if a.dumped else "")
                 + " 쓰레기만 있는 패치는 평가에서 제외)" if a.tiny_px > 0 or a.dumped else ""))

    ema = EMA(model, a.ema)
    if a.adele:
        from knps.adele import AdeleCfg, IoUTracker, Ledger, correct_labels, multiscale_consistency, teacher_probs
        want = {g.strip() for g in a.adele_groups.split(",") if g.strip()}
        noisy = {i for i, n in enumerate(info["groups"]) if n in want and not n.startswith(("real_", "bg_"))}
        print("  [adele] 교정 대상 그룹:", [info["groups"][i] for i in sorted(noisy)],
              "| 교정 안 함:", [n for i, n in enumerate(info["groups"]) if i not in noisy])
        tracker = IoUTracker(info["groups"], every=a.adele_every, t_unit=a.steps, r=a.adele_r, total_steps=total,
                             fallback_frac=a.adele_fallback, force_step=a.adele_force_onset)
        acfg = AdeleCfg(noisy_groups=noisy, tau_rm=a.adele_tau,
                        tau_add=a.adele_tau if a.adele_tau_add is None else a.adele_tau_add, add_mode=a.adele_add_mode)
        ledger = Ledger()
        t_scales = tuple(float(v) for v in a.adele_teacher_scales.split(","))
        ms_scales = tuple(float(v) for v in a.adele_ms_scales.split(","))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    start_ep, step, best = 0, 0, -float("inf")   # 검증셋 없이 학습할 때도 첫 에폭은 best 로 저장
    if a.resume:
        start_ep, step, best = resume_state(out, spec, model, ema, opt, scaler, a.steps)
        if a.adele and (out / "resume.pt").exists():
            st = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
            if "adele" in st:
                if tracker.load_state_dict(st["adele"]["tracker"]):
                    ledger.d = st["adele"]["ledger"]
        if start_ep >= a.epochs:
            print(f"이미 {start_ep}/{a.epochs} 에폭까지 끝난 학습입니다. 더 하려면 --epochs 를 늘리세요")
            return
    uda_it = mon = ucfg = None
    if a.uda:
        from knps.uda import UDAMonitor, UnlabeledDataset, make_cfg, uda_step
        ucfg = make_cfg(json.loads(a.uda_cfg) if a.uda_cfg else {}, lam=a.uda_weight, start=a.uda_start,
                        src_cls_weight=a.cls_weight, tgt_bs=a.uda_bs or a.bs)
        uds = UnlabeledDataset(a.uda)
        for d, n, wt in uds.sources:
            print(f"  uda 비라벨 {d}: {n} 장, 가중치 {wt}")
        ubs = ucfg["tgt_bs"] * max(int(ucfg["oversample"]), 1)
        usampler = WeightedRandomSampler(uds.weights, num_samples=max(total - a.uda_start, 1) * ubs + ubs,
                                         replacement=True)
        udl = DataLoader(uds, batch_size=ubs, sampler=usampler, num_workers=a.workers,
                         pin_memory=device == "cuda", drop_last=True, persistent_workers=a.workers > 0)

        def _forever(loader):
            while True:
                yield from loader

        uda_it, mon = _forever(udl), UDAMonitor()
        if a.init and not a.resume:
            # 2단계: teacher 를 처음부터 α=0.999 로 (EMA 의 초반 가속 (1+t)/(10+t) 때문에 teacher 가 student 를
            # 바로 따라가 의사라벨 안정화 효과가 사라지는 것을 막음)
            ema.step = 10 ** 4
        print(f"  UDA 켬: 시작 스텝 {a.uda_start}, 대상 배치 {ucfg['tgt_bs']} (로더 {ubs}), 설정 "
              + json.dumps({k: v for k, v in ucfg.items() if k not in ("bn_update",)}, ensure_ascii=False))
    json.dump({"args": vars(a), "spec": spec}, open(out / "run.json", "w"), indent=2, ensure_ascii=False)
    log = open(out / "log.csv", "a", newline="")
    logw = csv.writer(log)
    for ep in range(start_ep, a.epochs):
        model.train()
        t0 = time.time()
        run = {"seg": 0.0, "cls": 0.0}
        ast = {"steps": 0, "n_samples": 0, "rm_px": 0, "add_px": 0, "patch_to0": 0, "patch_tounk": 0, "ms": 0.0}
        for batch in dl:
            if a.adele:
                x, y, wv, lab, gid, sid = batch
            else:
                x, y, wv, lab = batch
            lr_f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_f
            if soma:   # Annealed Weight Decay: 큰 wd 로 시작해 cosine 으로 0 근처까지 (SoMA 4.3)
                opt.param_groups[0]["weight_decay"] = a.awd_wd * awd_factor(step, total, a.awd_min)
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y, wv, lab = (y.to(device, non_blocking=True), wv.to(device, non_blocking=True),
                          lab.to(device, non_blocking=True))
            if uda_it is not None and step >= a.uda_start:
                if EXP_GT_DILATE:
                    for _ in range(EXP_GT_DILATE):
                        y = F.max_pool2d(y, 3, 1, 1)
                # 항목별로 따로 역전파 (MIC/VFM-UDA++ 와 같음): 최대 메모리 ≈ 순전파 1회분 + teacher(no_grad)
                opt.zero_grad(set_to_none=True)
                out_u = uda_step(model, ema.model, (x, y, wv, lab), next(uda_it), seg_loss, ucfg,
                                 backward=lambda l: scaler.scale(l).backward(), amp=device == "cuda", step=step)
                ls, lc = out_u["src"], out_u["src_cls"]
                mon.update(out_u)
                if step == a.uda_start or (step + 1) % 50 == 0:
                    print(f"  [uda] step {step + 1} lam {out_u['lam']:.2f} " + mon.summary(last=50), flush=True)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                run["seg"] += float(ls)
                run["cls"] += float(lc)
                step += 1
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                seg, cls = model(x)
            if a.adele:
                tracker.update(step, seg, y, wv, gid)          # 교정 전 원래 라벨 기준 학습 IoU (공식 npl_metrics=0)
                active = tracker.active_groups(noisy, step)
                if active & set(gid.tolist()):
                    prob_t = teacher_probs(ema.model, x, t_scales, amp=device == "cuda")
                    y, wv, lab, cst = correct_labels(prob_t, y, wv, sid, gid, acfg, active, lab, ledger)
                    ast["steps"] += 1
                    for k in ("n_samples", "rm_px", "add_px", "patch_to0", "patch_tounk"):
                        ast[k] += cst[k]
                    if a.adele_verbose:
                        print(f"    [adele] step {step}: 교정 샘플 {cst['n_samples']} 지움 {cst['rm_px']}px "
                              f"더함 {cst['add_px']}px 패치→음성 {cst['patch_to0']} 패치→모름 {cst['patch_tounk']}",
                              flush=True)
            if EXP_GT_DILATE:
                for _ in range(EXP_GT_DILATE):
                    y = F.max_pool2d(y, 3, 1, 1)
            ls = seg_loss(seg.float(), y, wv)
            valid = lab >= 0                       # -1 = 쓰레기 있는지 모름(채점 제외만 있는 패치) → 분류 손실에서 뺌
            lc = (F.binary_cross_entropy_with_logits(cls.float()[valid], lab[valid]) if valid.any()
                  else cls.float().sum() * 0)
            loss = ls + a.cls_weight * lc
            if a.adele and a.adele_ms > 0:
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    lms = multiscale_consistency(model, x, ms_scales, w=wv, rho=0.8, logits1=seg)
                loss = loss + a.adele_ms * lms
                ast["ms"] += lms.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            run["seg"] += ls.item()
            run["cls"] += lc.item()
            step += 1
        nb = max(a.steps, 1)
        msg = f"ep {ep + 1}/{a.epochs} seg {run['seg'] / nb:.4f} cls {run['cls'] / nb:.4f} {time.time() - t0:.0f}s"
        if mon is not None and mon.hist["mix"]:
            msg += " | " + mon.summary(last=nb)
            with open(out / "uda_log.csv", "a", newline="") as fu:   # log.csv 열 순서는 그대로 두고 따로 기록
                csv.writer(fu).writerow([ep + 1] + [mon.mean(k, nb) for k in mon.KEYS])
        scores = []
        for (v, its), drop in zip(vals, drops):
            probs, cls_p, gts = infer_items([ema.model], its, device, bs=int(os.environ.get("EXP_VAL_BS", "64")), tta=1)  # [exp] T4 16GB 에선 16
            keep, gts = contest_gts(gts, a.tiny_px, drop)      # 대회 기준 (tune.py 와 같음)
            probs, cls_p = probs[keep], cls_p[keep]
            nw = 1.0
            if a.test_pos_ratio > 0:   # 검증셋 양성 비율과 테스트 양성 비율의 차이를 음성 가중치로 보정 (tune.py 와 같은 방식)
                vp = max(float(gts.reshape(len(gts), -1).any(1).mean()), 1e-6)
                nw = ((1 - a.test_pos_ratio) / a.test_pos_ratio) / ((1 - vp) / vp)
            # [exp] 에폭 선택은 리더보드로 검증된 후처리(v4cfg+dil1)로 고정합니다.
            # 검증셋으로 후처리를 다시 고르면 리더보드가 세 번 떨어졌습니다 (STATE.md "평가 규칙 확정").
            d = fixed_score(probs, cls_p, gts, nw)
            scores.append(d["score"])
            msg += f" | {v.name}: {d['score']:.4f} (F1 {d['f1']:.3f} 형상 {d['shape']:.3f} 고정후처리)"
        print(msg, flush=True)
        if a.adele:
            print(f"  [adele] 교정한 스텝 {ast['steps']}/{nb}, 샘플 {ast['n_samples']}, 지운 화소 {ast['rm_px']}, "
                  f"더한 화소 {ast['add_px']}, 패치→음성 {ast['patch_to0']}, 패치→모름 {ast['patch_tounk']}"
                  + (f", 일관성 {ast['ms'] / nb:.4f}" if a.adele_ms > 0 else ""), flush=True)
            for line in tracker.summary():
                print("    " + line, flush=True)
            ledger.to_csv(out / "adele_ledger.csv", info["items"], info["group_of"], info["groups"])
        meta = {"epoch": ep + 1, "val": scores, "train": a.train}
        save_checkpoint(out / "last.pt", ema.model, spec, meta)
        sel = scores[0] if scores else -(run["seg"] / nb)
        if sel > best:
            best = sel
            save_checkpoint(out / "best.pt", ema.model, spec, meta)
            print(f"  ↳ best 갱신 {best:.4f}", flush=True)
        # 이어서 학습용 전체 상태 (런타임이 끊겨도 이 에폭까지는 보존). 쓰는 도중 끊겨도 깨지지 않게 임시 파일 후 교체
        tmp = out / "resume.pt.tmp"
        torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "ema_step": ema.step,
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "step": step, "best": best, "spec": spec,
                    **({"adele": {"tracker": tracker.state_dict(), "ledger": ledger.d}} if a.adele else {})}, tmp)
        os.replace(tmp, out / "resume.pt")
        logw.writerow([ep + 1, run["seg"] / nb, run["cls"] / nb] + scores)  # 저장이 끝난 에폭만 기록
        log.flush()
    log.close()
    print("완료. best:", best, "→", out / "best.pt")


if __name__ == "__main__":
    main()
