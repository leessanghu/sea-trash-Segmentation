"""검증 폴더 추론 + 평가 산식 기준 후처리 설정 탐색."""
import itertools

import numpy as np
import torch

from .data import center_crop, read_pair
from .metric import MIN_PX, patch_shape_score
from .model import predict_probs
from .postprocess import DEFAULT_CFG, mask_part

FULL_GRID = {
    "t_pix": [0.3, 0.4, 0.5, 0.6, 0.7],
    "close": [0, 1],
    "min_cc": [0, 10, 30],
    "rescue_px": [0, 50, 80],   # 50: 작은 쓰레기(<50px)는 최소 크기만 채워야 형상 점수가 덜 깎임
    "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
    "t_img": [round(float(v), 3) for v in np.linspace(0.05, 0.95, 19)],
    "min_area": [0, 30, 60, 100, 200],
    "dilate": [0],
}
QUICK_GRID = {"t_pix": [0.5], "close": [0], "min_cc": [0], "rescue_px": [80],
              "alpha": [0.5], "t_img": FULL_GRID["t_img"], "min_area": [0], "dilate": [0]}
# 절반 나눠 확인할 때 쓰는 중간 크기 탐색 범위 (반쪽 탐색을 여러 번 하므로 가볍게)
HALF_GRID = {"t_pix": [0.4, 0.5, 0.6], "close": [0, 1], "min_cc": [0, 10], "rescue_px": [50, 80],
             "alpha": [0.0, 0.25, 0.5, 1.0], "t_img": FULL_GRID["t_img"], "min_area": [0, 30, 100]}


@torch.no_grad()
def infer_items(models, items, device, bs=64, tta=1, weights=None, scales=(1.0,)):
    """반환: probs (N,H,W) float16, cls (N,) float32, gts (N,H,W) bool"""
    probs, clss, gts = [], [], []
    for s in range(0, len(items), bs):
        chunk = [center_crop(*read_pair(*it)) for it in items[s:s + bs]]   # 256 보다 크면 가운데를 잘라 평가
        x = torch.from_numpy(np.stack([c[0] for c in chunk]).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        p, c = predict_probs(models, x.to(device), tta=tta, weights=weights, scales=scales)
        probs.append(p.cpu().numpy().astype(np.float16))
        clss.append(c.cpu().numpy())
        gts += [c[1] for c in chunk]
    return np.concatenate(probs), np.concatenate(clss), np.stack(gts)


def search_cfg(probs, cls, gts, grid=None, neg_weight=1.0, verbose=False):
    """평가 산식을 최대화하는 후처리 설정을 찾습니다.

    1단계: 마스크 설정(t_pix, close, min_cc, rescue_px)마다 패치별 면적·형상 점수를 한 번씩 계산
    2단계: 판정 설정(alpha, t_img, min_area)은 벡터 연산으로 전수 탐색
    neg_weight: 음성 패치 가중치. 테스트의 양성 비율이 검증과 다를 때 F1 계산을 맞추는 용도
                (예: 검증 양성 50%, 테스트 양성 20% 추정 → neg_weight = (0.8/0.2)/(0.5/0.5) = 4)
    """
    g = {**FULL_GRID, **(grid or {})}
    n = len(probs)
    gpos = gts.reshape(n, -1).any(1)
    n_gpos = max(int(gpos.sum()), 1)
    flat = probs.reshape(n, -1).astype(np.float32)
    top = np.partition(flat, -MIN_PX, axis=1)[:, -MIN_PX:].mean(1)
    best = (-1.0, None, None)
    for t_pix, close, min_cc, rescue, dil in itertools.product(g["t_pix"], g["close"], g["min_cc"], g["rescue_px"],
                                                               g.get("dilate", [0])):
        c = {**DEFAULT_CFG, "t_pix": t_pix, "close": close, "min_cc": min_cc, "rescue_px": rescue, "dilate": dil}
        raw = np.zeros(n)
        fin = np.zeros(n)
        shp = np.zeros(n)
        for i in range(n):
            m, raw[i] = mask_part(flat[i].reshape(probs.shape[1:]), c)
            fin[i] = m.sum()
            if gpos[i] and fin[i] >= MIN_PX:
                shp[i] = patch_shape_score(m, gts[i])
        for alpha in g["alpha"]:
            s_img = alpha * cls + (1 - alpha) * top
            for min_area in g["min_area"]:
                base_ok = (fin >= MIN_PX) & (raw >= min_area)
                t = np.array(g["t_img"])[:, None]
                pp = (s_img[None] >= t) & base_ok[None]           # (T, N)
                tp = (pp & gpos).sum(1)
                fn = (~pp & gpos).sum(1)
                fp = (pp & ~gpos).sum(1) * neg_weight
                tn = (~pp & ~gpos).sum(1) * neg_weight
                f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9) + 2 * tn / np.maximum(2 * tn + fn + fp, 1e-9)) / 2
                shape = (pp * shp[None] * gpos[None]).sum(1) / n_gpos
                sc = 0.5 * f1 + 0.5 * shape
                j = int(np.argmax(sc))
                if sc[j] > best[0]:
                    cfg = {**c, "alpha": alpha, "t_img": float(g["t_img"][j]), "min_area": min_area}
                    best = (float(sc[j]), cfg, dict(score=float(sc[j]), f1=float(f1[j]), shape=float(shape[j]),
                                                    tp=int(tp[j]), fn=int(fn[j]),
                                                    fp=int(round(fp[j] / neg_weight)), tn=int(round(tn[j] / neg_weight))))
        if verbose:
            print(f"  t_pix={t_pix} close={close} min_cc={min_cc} rescue={rescue} dilate={dil} → best so far {best[0]:.4f}",
                  flush=True)
    return best[1], best[2]


def score_cfg(probs, cls, gts, cfg, neg_weight=1.0):
    """정해진 후처리 설정으로 채점 (반대쪽 절반 확인용). neg_weight 는 search_cfg 와 같은 뜻."""
    from .metric import patch_shape_score
    from .postprocess import postprocess
    tp = fn = fp = tn = 0
    shapes = []
    for p, c, g in zip(probs, cls, gts):
        m = postprocess(p.astype(np.float32), float(c), cfg)
        pp = int(m.sum()) >= MIN_PX
        if g.any():
            tp, fn = tp + pp, fn + (not pp)
            shapes.append(patch_shape_score(m, g) if pp else 0.0)
        else:
            fp, tn = fp + pp, tn + (not pp)
    fp, tn = fp * neg_weight, tn * neg_weight
    f1 = (2 * tp / max(2 * tp + fp + fn, 1e-9) + 2 * tn / max(2 * tn + fn + fp, 1e-9)) / 2
    return 0.5 * f1 + 0.5 * (float(np.mean(shapes)) if shapes else 0.0)


def half_split(items, val_dirs):
    """실제 검증셋을 촬영 장소(loc) 기준으로 둘로 나눕니다 (같은 장소의 다른 날짜가 양쪽에 섞이지 않게)."""
    import csv
    import zlib
    from pathlib import Path
    loc = {}
    for d in val_dirs:
        mp = Path(d) / "manifest.csv"
        if mp.exists():
            for r in csv.DictReader(open(mp, encoding="utf-8")):
                loc[r["id"]] = r.get("loc", r["id"])
    keys = [loc.get(Path(it[0]).stem, Path(it[0]).stem) for it in items]
    side = np.array([zlib.crc32(k.encode()) % 2 for k in keys])
    return np.flatnonzero(side == 0), np.flatnonzero(side == 1)
