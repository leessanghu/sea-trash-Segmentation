"""대회 평가 산식: 0.5 * macro F1(패치 존재) + 0.5 * 형상 점수(1화소 허용 F-score 평균)."""
import numpy as np
from scipy.ndimage import binary_dilation

MIN_PX = 50
_K = np.ones((3, 3), bool)


def patch_shape_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """정답 양성 패치 하나의 형상 점수. pred 는 이미 50px 규칙을 통과한 마스크여야 합니다."""
    p, g = pred.astype(bool), gt.astype(bool)
    if p.sum() == 0:
        return 0.0
    prec = (p & binary_dilation(g, _K)).sum() / p.sum()
    rec = (g & binary_dilation(p, _K)).sum() / g.sum()
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))


def score(preds, gts, return_detail: bool = False):
    """preds, gts: (H,W) 마스크 시퀀스. 반환 (최종, F1, 형상)."""
    tp = fp = fn = tn = 0
    shapes = []
    for p, g in zip(preds, gts):
        p = np.asarray(p, bool)
        g = np.asarray(g, bool)
        p_pos = int(p.sum()) >= MIN_PX
        g_pos = bool(g.any())
        if g_pos and p_pos:
            tp += 1
        elif g_pos:
            fn += 1
        elif p_pos:
            fp += 1
        else:
            tn += 1
        if g_pos:
            shapes.append(patch_shape_score(p, g) if p_pos else 0.0)
    f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    f1 = (f1_pos + f1_neg) / 2
    shape = float(np.mean(shapes)) if shapes else 0.0
    total = 0.5 * f1 + 0.5 * shape
    if return_detail:
        return dict(score=total, f1=f1, f1_pos=f1_pos, f1_neg=f1_neg, shape=shape,
                    tp=tp, fp=fp, fn=fn, tn=tn)
    return total, f1, shape
