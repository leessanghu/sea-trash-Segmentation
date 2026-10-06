"""ADELE (Liu et al., CVPR 2022, arXiv 2110.03740) 를 우리 이진 분할 + 패치 분류 학습에 맞춘 구현.

논문 요지
  잡음 라벨로 분할을 학습하면 먼저 깨끗한 화소를 맞히다가(early learning) 나중에 틀린 라벨을 외운다(memorization).
  학습 IoU(모델 출력 vs 잡음 라벨) 곡선을 f(t) = a(1 - exp(-b t^c)) 로 맞추고, 기울기가 처음(t=1) 대비
  |f'(1) - f'(t)| / |f'(1)| > r (r=0.9) 로 꺾이면 그 클래스의 라벨 교정을 시작한다(클래스별로 따로).
  교정은 모델 확신도 > τ(0.7~0.8) 인 화소만 모델 예측으로 바꾼다. 선택적으로 다중 축척(0.7/1.0/1.5) 일관성 손실.

우리 데이터에 맞춘 차이 (자세한 근거는 보고서)
  - 그룹: 학습 소스 폴더별(합성 폴더만 교정, 실제 폴더는 절대 교정 안 함). 클래스: 0 배경 / 1 쓰레기.
  - 교정 시점 이후에는 매 스텝 같은 증강 배치에 EMA 교사의 확률로 즉석 교정 (저장 캐시 없음).
    3000 스텝 x bs32 = 9.6만 샘플인데 합성 파일은 수만 장이라 같은 파일을 1~3번밖에 안 보고,
    무작위 자르기·축척 지터·D4 를 되돌려 원본 좌표에 쌓아야 해서 캐시는 거의 재사용되지 않음.
    대신 파일별 교정 기록(ledger)을 남겨 어느 합성 파일이 자주 지워졌는지 학습 뒤 확인할 수 있게 함.
  - 쓰레기→배경(지우기)은 τ_rm, 배경→쓰레기(더하기)는 τ_add. 더하기는 기본적으로 원래 쓰레기가 있는 패치에서만
    (공식 VOC 코드의 class constraint 와 같은 뜻: 라벨에 없던 클래스를 새로 만들지 않음).
  - 채점 제외 화소(w=0)는 그대로. 교정으로 쓰레기 화소가 50 미만이 되면 패치 라벨을 다시 계산.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

MIN_PATCH_PX = 50   # 대회: 예측 마스크 50px 이상이면 양성 패치


# --------------------------------------------------------------------------------------------
# 곡선 맞춤 (논문 식 1, 2)
# --------------------------------------------------------------------------------------------
def curve(t, a, b, c):
    """논문 식 (1): f(t) = a (1 - exp(-b t^c)). (공식 코드는 a(1-exp(-(1/c) t^b)), b<=1 로 같은 모양)"""
    return a * (1.0 - np.exp(-b * np.power(t, c)))


def curve_deriv(t, a, b, c):
    """f'(t) = a b c exp(-b t^c) t^(c-1). t=0 근처 발산을 피하려고 공식 코드처럼 1e-6 더함."""
    t = np.asarray(t, dtype=np.float64) + 1e-6
    return a * b * c * np.exp(-b * np.power(t, c)) * np.power(t, c - 1.0)


def fit_curve(t, y):
    """최소제곱 맞춤. 공식 코드와 같이 trf + 뒤쪽 점에 큰 가중(sigma = geomspace(1, 0.1)) + 경계
    0<=a<=1, b>=0, 0<=c<=1 (c<=1 이면 오목 = 감속 곡선; 공식 코드의 지수 상한 1 과 같음).
    실패하면 None."""
    from scipy.optimize import curve_fit
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 4 or y.max() < 0.01:        # 점이 너무 적거나 IoU 가 사실상 0 (아직 아무것도 못 배움)
        return None
    a0 = float(np.clip(y.max() * 1.05, 1e-3, 1.0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(curve, t, y, p0=(a0, 1.0, 0.8), method="trf",
                                sigma=np.geomspace(1, 0.1, len(y)), absolute_sigma=True,
                                bounds=([0.0, 0.0, 0.0], [1.0, np.inf, 1.0]), maxfev=5000)
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(popt)) or popt[0] <= 0 or popt[1] <= 0 or popt[2] <= 0:
        return None
    return tuple(float(v) for v in popt)


def relative_slope_change(t_now, params):
    """식 (2) 의 |f'(1) - f'(t)| / |f'(1)|. 공식 코드처럼 1 을 넘으면(기울기가 오히려 커짐) 0 으로 봄."""
    d1 = abs(float(curve_deriv(1.0, *params)))
    if d1 <= 0 or not math.isfinite(d1):
        return 0.0
    rc = abs(abs(float(curve_deriv(t_now, *params))) - d1) / d1
    return 0.0 if rc > 1 else rc


# --------------------------------------------------------------------------------------------
# 그룹 x 클래스 학습 IoU 기록 + 교정 시작 감지
# --------------------------------------------------------------------------------------------
class IoUTracker:
    """(소스 그룹, 클래스) 마다 학습 IoU(학생 출력 > 0.5 vs 원래 잡음 라벨, 채점 제외 화소 뺌)를 모읍니다.

    every 스텝마다(그리고 그 구간 쓰레기 화소가 min_pix 이상 모였을 때) IoU 점 하나를 기록하고 곡선을 맞춥니다.
    시간 단위 t 는 t_unit 스텝(기본: 에폭 = steps). 논문·공식 코드는 에폭 단위이며 f'(1) 은 '첫 에폭 끝' 기울기.
    한 번 시작된 교정은 끝까지 유지(공식 코드와 같음).
    폴백: fallback_frac x 전체 스텝까지 곡선 맞춤이 한 번도 성립하지 않은 키는 그때 강제로 시작.
    force_step: 시험용. 그 스텝부터 모든 키를 강제로 시작.
    """

    def __init__(self, group_names, every=50, t_unit=500, r=0.9, total_steps=3000, min_points=5,
                 min_pix=500, fallback_frac=0.6, force_step=-1, classes=(0, 1)):
        self.names = list(group_names)
        self.every, self.t_unit, self.r = int(every), float(t_unit), float(r)
        self.total, self.min_points, self.min_pix = int(total_steps), int(min_points), int(min_pix)
        self.fallback_frac, self.force_step = float(fallback_frac), int(force_step)
        self.classes = tuple(classes)
        G, C = len(self.names), len(self.classes)
        self.acc = np.zeros((G, C, 3))           # TP, P, T (현재 구간)
        self.hist = {(g, c): [] for g in range(G) for c in self.classes}   # [(t, iou)]
        self.params = {}                          # 마지막으로 성공한 맞춤
        self.rel = {}                             # 마지막 상대 기울기 변화
        self.onset = {}                           # (g,c) -> 시작 스텝
        self.onset_how = {}                       # 'curve' | 'fallback' | 'forced'
        self.last_flush = 0

    # ---- 기록 ----
    @torch.no_grad()
    def update(self, step, logits, y, w, group):
        """logits/y/w: (B,1,H,W), group: (B,) int. y 는 교정 전 원래 라벨이어야 함."""
        pred = logits.detach().float() > 0
        yv = y > 0.5
        v = w > 0
        g = group.detach().long().view(-1)
        for ci, c in enumerate(self.classes):
            pc, tc = (pred, yv) if c == 1 else (~pred, ~yv)
            tp = (pc & tc & v).flatten(1).sum(1).long()
            p = (pc & v).flatten(1).sum(1).long()
            t = (tc & v).flatten(1).sum(1).long()
            st = torch.stack([tp, p, t], 1).cpu().numpy()          # (B,3)
            np.add.at(self.acc[:, ci], g.cpu().numpy(), st)
        if step + 1 - self.last_flush >= self.every:
            self._flush(step + 1)

    def _flush(self, step):
        self.last_flush = step
        t_now = step / self.t_unit
        for gi in range(len(self.names)):
            if self.acc[gi, self.classes.index(1) if 1 in self.classes else 0, 2] < self.min_pix:
                continue                         # 이 그룹 쓰레기 화소가 아직 적음 → 다음 구간과 합침
            for ci, c in enumerate(self.classes):
                tp, p, t = self.acc[gi, ci]
                if t + p > 0:
                    self.hist[(gi, c)].append((t_now, tp / max(t + p - tp, 1e-10)))
            self.acc[gi] = 0
        for key in self.hist:
            self._check(key, step)

    def _check(self, key, step):
        if key in self.onset:
            return
        if 0 <= self.force_step <= step:
            self.onset[key], self.onset_how[key] = step, "forced"
            return
        h = self.hist[key]
        t_now = step / self.t_unit
        if len(h) >= self.min_points and t_now > 1.0:
            ts, ys = zip(*h)
            prm = fit_curve(ts, ys)
            if prm is not None:
                self.params[key] = prm
                self.rel[key] = relative_slope_change(t_now, prm)
                if self.rel[key] > self.r:
                    self.onset[key], self.onset_how[key] = step, "curve"
                    return
        if key not in self.params and step >= self.fallback_frac * self.total:
            self.onset[key], self.onset_how[key] = step, "fallback"

    # ---- 조회 ----
    def onset_reached(self, group, cls=1, step=None):
        key = (int(group), int(cls))
        if step is not None and key not in self.onset and 0 <= self.force_step <= step:
            self.onset[key], self.onset_how[key] = step, "forced"
        return key in self.onset

    def active_groups(self, candidates, step=None, cls=1):
        """candidates 중 cls(기본 쓰레기) 교정이 시작된 그룹 id 집합."""
        return {g for g in candidates if self.onset_reached(g, cls, step)}

    def summary(self):
        out = []
        for (g, c), h in self.hist.items():
            if not h and (g, c) not in self.onset:
                continue
            s = f"{self.names[g]}/c{c}: n={len(h)} iou={h[-1][1]:.3f}" if h else f"{self.names[g]}/c{c}: n=0"
            if (g, c) in self.params:
                a, b, cc = self.params[(g, c)]
                s += f" fit(a={a:.2f},b={b:.2f},c={cc:.2f}) rel={self.rel[(g, c)]:.2f}"
            if (g, c) in self.onset:
                s += f" ONSET@{self.onset[(g, c)]}({self.onset_how[(g, c)]})"
            out.append(s)
        return out

    def state_dict(self):
        return {k: getattr(self, k) for k in ("names", "acc", "hist", "params", "rel", "onset", "onset_how",
                                              "last_flush")}

    def load_state_dict(self, st):
        """이어서 학습할 때. 소스 폴더 구성(--train)이 저장 때와 다르면 기록을 버리고 새로 시작."""
        if st.get("names") != self.names:
            print(f"  [adele] 저장된 그룹 {st.get('names')} 가 지금 {self.names} 와 달라 IoU 기록을 새로 시작합니다")
            return False
        for k, v in st.items():
            setattr(self, k, v)
        return True


# --------------------------------------------------------------------------------------------
# 교정 규칙
# --------------------------------------------------------------------------------------------
@dataclass
class AdeleCfg:
    noisy_groups: set = field(default_factory=set)   # 교정 대상 그룹 id (합성 폴더만)
    tau_rm: float = 0.8        # 쓰레기 라벨 → 배경: 교사 P(배경) > tau_rm
    tau_add: float = 0.8       # 배경 라벨 → 쓰레기: 교사 P(쓰레기) > tau_add
    add_mode: str = "pos_patch"  # 'none' | 'pos_patch'(원래 쓰레기가 있는 패치에서만, 기본) | 'all'
    temp: float = 1.0          # 교사 로짓 온도 (공식 VOC 코드는 3 으로 부드럽게 한 뒤 τ 적용)
    soft_ignore: bool = False  # 쓰레기 라벨인데 교사가 배경 쪽(p<0.5)이지만 τ 미달이면 채점 제외로 (공식 logic_255 와 유사)
    keep_min_frac: float = 0.0  # 교정 뒤 쓰레기 화소가 원래의 이 비율 미만이면 그 샘플 교정을 되돌림 (0 = 끔)
    min_patch_px: int = MIN_PATCH_PX


@torch.no_grad()
def teacher_probs(teacher, x, scales=(1.0,), temp=1.0, amp=False, round_to=32):
    """EMA 교사의 쓰레기 확률 (B,1,H,W). 여러 축척이면 원래 크기로 되돌려 평균(논문: 다중 축척 평균 출력으로 교정)."""
    h, w = x.shape[-2:]
    acc = 0.0
    for s in scales:
        xs = x if s == 1 else _rescale(x, s, round_to)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            seg, _ = teacher(xs)
        seg = seg.float()
        if seg.shape[-2:] != (h, w):
            seg = F.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)
        acc = acc + torch.sigmoid(seg / temp)
    return acc / len(scales)


@torch.no_grad()
def correct_labels(prob_ema, y, w, sample_ids, group, cfg: AdeleCfg, active_groups, lab=None, ledger=None):
    """ADELE 교정 규칙(이진판). 반환 (y', w', lab', 통계 dict). 입력 텐서는 바꾸지 않음.

    prob_ema: (B,1,H,W) 교사 쓰레기 확률, y/w: (B,1,H,W), sample_ids/group: (B,), lab: (B,) 패치 라벨(1/0/-1).
    교정 대상 = group ∈ cfg.noisy_groups ∩ active_groups (실제 폴더 그룹은 noisy_groups 에 넣지 말 것).
    """
    y2, w2 = y.clone(), w.clone()
    lab2 = lab.clone() if lab is not None else None
    stats = {"n_samples": 0, "rm_px": 0, "add_px": 0, "ign_px": 0, "patch_to0": 0, "patch_tounk": 0, "reverted": 0}
    elig = [int(g) in cfg.noisy_groups and int(g) in active_groups for g in group.view(-1).tolist()]
    if not any(elig):
        return y2, w2, lab2, stats
    p = prob_ema.float()
    valid = w > 0
    is_pos = y > 0.5
    for b in np.flatnonzero(elig):
        vb, pb, yb = valid[b], p[b], is_pos[b]
        n0 = int((yb & vb).sum())
        rm = yb & vb & ((1 - pb) > cfg.tau_rm)
        add = torch.zeros_like(rm)
        if cfg.add_mode == "all" or (cfg.add_mode == "pos_patch" and n0 > 0):
            add = (~yb) & vb & (pb > cfg.tau_add)
        ign = (yb & vb & (pb < 0.5) & ~rm) if cfg.soft_ignore else torch.zeros_like(rm)
        n_after = n0 - int(rm.sum()) - int(ign.sum()) + int(add.sum())
        if cfg.keep_min_frac > 0 and n0 > 0 and n_after < cfg.keep_min_frac * n0:
            stats["reverted"] += 1
            continue
        if not (rm.any() or add.any() or ign.any()):
            stats["n_samples"] += 1
            if ledger is not None:
                ledger.add(int(sample_ids[b]), n0, 0, 0)
            continue
        y2[b][rm] = 0.0
        y2[b][add] = 1.0
        w2[b][ign] = 0.0
        stats["n_samples"] += 1
        stats["rm_px"] += int(rm.sum())
        stats["add_px"] += int(add.sum())
        stats["ign_px"] += int(ign.sum())
        if lab2 is not None and float(lab[b]) == 1.0:
            n_pos = int(((y2[b] > 0.5) & (w2[b] > 0)).sum())
            if n_pos < cfg.min_patch_px:
                # 쓰레기가 전부 지워졌고 채점 제외 화소도 없으면 음성, 조금 남았거나 채점 제외가 있으면 '모름'(분류 손실 제외)
                if n_pos == 0 and bool((w2[b] > 0).all()):
                    lab2[b] = 0.0
                    stats["patch_to0"] += 1
                else:
                    lab2[b] = -1.0
                    stats["patch_tounk"] += 1
        if ledger is not None:
            ledger.add(int(sample_ids[b]), n0, int(rm.sum()), int(add.sum()))
    return y2, w2, lab2, stats


class Ledger:
    """파일(샘플 id)별 교정 기록: 본 횟수, 원래 쓰레기 화소 합, 지운 화소 합, 더한 화소 합."""

    def __init__(self):
        self.d = {}

    def add(self, sid, n0, rm, add):
        if sid < 0:
            return
        r = self.d.setdefault(sid, [0, 0, 0, 0])
        r[0] += 1
        r[1] += n0
        r[2] += rm
        r[3] += add

    def to_csv(self, path, items, group_of=None, group_names=None):
        import csv
        rows = sorted(self.d.items(), key=lambda kv: -kv[1][2] / max(kv[1][1], 1))
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["image", "group", "views", "orig_px", "removed_px", "added_px", "removed_frac"])
            for sid, (n, o, r, a) in rows:
                g = group_names[group_of[sid]] if group_of is not None and group_names is not None else ""
                wr.writerow([str(items[sid][0]), g, n, o, r, a, f"{r / max(o, 1):.3f}"])


# --------------------------------------------------------------------------------------------
# 다중 축척 일관성 (논문 식 3)
# --------------------------------------------------------------------------------------------
def _rescale(x, s, round_to=32):
    """smp U-Net 은 입력이 32 의 배수여야 하므로 축척 크기를 32 배수로 반올림 (256: 0.7→192(=0.75), 1.5→384)."""
    h, w = x.shape[-2:]
    nh = max(round_to, int(round(h * s / round_to)) * round_to)
    nw = max(round_to, int(round(w * s / round_to)) * round_to)
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)


def multiscale_consistency(student, x, scales=(0.7, 1.0, 1.5), w=None, rho=0.8, logits1=None, round_to=32):
    """L = (1/s) Σ_k KL(p_k || q), q = mean_k p_k (이진: [1-p, p] 분포). 공식 코드처럼 q 를 [1e-3, 1-1e-3] 으로 자르고
    이미지 단위 마스크(q 의 최대 클래스 확률의 최댓값 >= rho 인 이미지만)를 씀. q 로도 기울기가 흐름(공식 코드와 같음).
    logits1: 이미 계산한 1.0 축척 학생 로짓을 넘기면 다시 계산하지 않음. w: 채점 제외 화소를 손실에서 뺌(공식 기본은 안 뺌)."""
    h, wd = x.shape[-2:]
    ps = []
    for s in scales:
        if s == 1 and logits1 is not None:
            lg = logits1
        else:
            lg, _ = student(x if s == 1 else _rescale(x, s, round_to))
        lg = lg.float()
        if lg.shape[-2:] != (h, wd):
            lg = F.interpolate(lg, size=(h, wd), mode="bilinear", align_corners=False)
        ps.append(torch.sigmoid(lg).clamp(1e-6, 1 - 1e-6))
    q = torch.stack(ps).mean(0).clamp(1e-3, 1 - 1e-3)
    m = torch.ones_like(q) if w is None else (w > 0).float()
    conf = (torch.maximum(q, 1 - q) * m).flatten(1).amax(1)                  # (B,)
    img_mask = (conf >= rho).float().view(-1, 1, 1, 1)
    kl = 0.0
    for p in ps:
        kl = kl + p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
    kl = kl / len(ps) * img_mask * m
    return kl.sum() / m.sum().clamp(min=1)
