"""확률맵 → 제출 마스크 후처리. 학습(tune.py)과 추론(predict.ipynb)이 같은 코드를 씁니다.

평가 산식 요약
- 패치 판정(macro F1): 마스크 1 화소가 50개 이상이면 양성 예측
- 형상 점수: 정답 양성 패치만 평균하며, 음성 예측을 낸 양성 패치는 0점
=> 양성 패치를 놓치는 비용이 오탐보다 3~5배 크므로 판정과 마스크 임계값을 분리해 튜닝합니다.
"""
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, label

MIN_PX = 50

DEFAULT_CFG = {
    "t_pix": 0.5,      # 화소 임계값 (마스크 모양)
    "t_img": 0.5,      # 패치 판정 임계값
    "alpha": 0.5,      # 판정 점수 = alpha*분류헤드 + (1-alpha)*상위50화소 평균확률
    "min_area": 0,     # 판정 양성이라도 prob>t_pix 면적이 이보다 작으면 음성 (0=끔)
    "min_cc": 0,       # 이보다 작은 연결요소 제거 (0=끔)
    "close": 0,        # 1 이면 3x3 closing 으로 좁은 틈 메움
    "rescue_px": 80,   # 판정 양성인데 마스크가 50px 미만이면 확률 상위 N 화소로 채움 (0=끔)
    "dilate": 0,       # 1 이면 마지막에 마스크를 1화소(3x3) 넓힘 (평가가 1화소 어긋남을 봐줌)
}


def image_score(prob: np.ndarray, cls_prob, alpha: float) -> float:
    flat = prob.reshape(-1)
    top = float(np.partition(flat, -MIN_PX)[-MIN_PX:].mean())
    if cls_prob is None:
        return top
    return float(alpha * cls_prob + (1 - alpha) * top)


def remove_small(mask: np.ndarray, min_cc: int) -> np.ndarray:
    if min_cc <= 1 or not mask.any():
        return mask
    lab, n = label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_cc
    keep[0] = False
    return keep[lab]


def mask_part(prob: np.ndarray, c: dict):
    """판정과 무관한 마스크 생성 단계. (마스크, 임계값 직후 면적) 을 반환합니다."""
    m = prob > c["t_pix"]
    raw_area = int(m.sum())
    if c["close"]:
        m = binary_closing(m, np.ones((3, 3), bool)) | m
    m = remove_small(m, int(c["min_cc"]))
    if m.sum() < MIN_PX:
        k = int(c["rescue_px"])
        if k >= MIN_PX:
            idx = np.argpartition(prob.reshape(-1), -k)[-k:]
            m = np.zeros(prob.size, bool)
            m[idx] = True
            m = m.reshape(prob.shape)
    if c.get("dilate", 0) and m.any():
        m = binary_dilation(m, np.ones((3, 3), bool))
    return m, raw_area


def postprocess(prob: np.ndarray, cls_prob=None, cfg: dict | None = None) -> np.ndarray:
    """prob: (H,W) 0~1 debris 확률, cls_prob: 패치에 debris 가 있을 확률(없으면 None)."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    empty = np.zeros(prob.shape, bool)
    if image_score(prob, cls_prob, c["alpha"]) < c["t_img"]:
        return empty
    m, raw_area = mask_part(prob, c)
    if c["min_area"] and raw_area < c["min_area"]:
        return empty
    return m if m.sum() >= MIN_PX else empty
