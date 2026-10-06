"""실험 모델 평가 — 후처리를 리더보드 검증값(v4cfg+dil1)으로 고정하고 모델만 비교합니다.

STATE.md "평가 규칙 확정": 후처리를 로컬로 고르면 리더보드가 세 번 떨어졌고,
후처리를 고정하면 로컬@0.639 가 v4 < v5 순서를 맞게 매겼습니다.

python exp_eval.py ck1.pt [ck2.pt ...] [--tta 8] [--cache out.npz]
여러 개를 주면 앙상블(균등 평균)로 평가합니다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knps.data import TINY_PX, contest_gts, list_items           # noqa: E402
from knps.evaluate import infer_items                             # noqa: E402
from knps.metric import patch_shape_score                         # noqa: E402
from knps.model import load_checkpoint                            # noqa: E402
from knps.postprocess import DEFAULT_CFG, postprocess             # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "sea_trash_data"
FIXED = {**DEFAULT_CFG, "t_pix": 0.5, "t_img": 0.4, "alpha": 0.25, "min_cc": 10, "close": 0,
         "rescue_px": 80, "dilate": 1, "min_area": 0}
K = np.ones((3, 3), bool)


def score(masks, gts, r_pos):
    """양성비 r_pos 로 재가중한 대회 산식."""
    gpos = np.array([g.any() for g in gts])
    wp, wn = r_pos / max(gpos.sum(), 1), (1 - r_pos) / max((~gpos).sum(), 1)
    tp = fp = fn = tn = 0.0
    sh = []
    for m, g, gp in zip(masks, gts, gpos):
        pp = int(m.sum()) >= 50
        if gp:
            tp, fn = tp + wp * pp, fn + wp * (not pp)
            sh.append(patch_shape_score(m, g) if pp else 0.0)
        else:
            fp, tn = fp + wn * pp, tn + wn * (not pp)
    f1 = lambda a, b, c: 0.0 if 2 * a + b + c == 0 else 2 * a / (2 * a + b + c)   # noqa: E731
    mf1 = (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2
    s = float(np.mean(sh)) if sh else 0.0
    return {"score": 0.5 * mf1 + 0.5 * s, "f1": mf1, "shape": s, "npos": int(sum(int(m.sum()) >= 50 for m in masks))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--cache", default="")
    ap.add_argument("--val", default=str(DATA / "real_val"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    items = list_items(a.val)
    if a.cache and Path(a.cache).exists():
        z = np.load(a.cache)
        P, C, G = z["P"].astype(np.float32), z["C"], z["G"]
    else:
        models = [load_checkpoint(c, dev)[0] for c in a.ckpts]
        P, C, G = infer_items(models, items, dev, bs=16, tta=a.tta)
        P = P.astype(np.float32)
        if a.cache:
            np.savez_compressed(a.cache, P=P.astype(np.float16), C=C, G=G)
    keep, Gc = contest_gts(G, TINY_PX)
    P, C = P[keep], C[keep]
    masks = [postprocess(p, float(c), FIXED) for p, c in zip(P, C)]
    Gd = np.stack([ndimage.binary_dilation(g, K) if g.any() else g for g in Gc])
    out = {"ckpts": a.ckpts, "tta": a.tta, "n": int(len(keep)), "n_pos": int(Gc.reshape(len(Gc), -1).any(1).sum()),
           "local639": score(masks, Gc, 0.639), "gtdil639": score(masks, Gd, 0.639), "local127": score(masks, Gc, 0.127)}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
