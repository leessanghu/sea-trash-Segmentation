"""학습 증강.

- 장면별 색 보정 흉내: 대회 영상은 장면마다 '보기 좋게' 따로 보정되어 색감 편차가 큼
  (누렇고 뿌연 장면 ~ 채도 높고 선명한 장면)
- 위성 센서 흉내: 항공영상(배경)을 0.3m 위성 팬샤프닝 영상처럼 만듦
  · 색(크로마)은 다중분광 해상도(~1.2m)라 번지고 밝기만 선명 → 크로마 저해상도화
  · PSF 블러 + 샤프닝 헤일로, 노이즈, JPEG
- D4(뒤집기·90도 회전): 위에서 내려다본 영상이라 방향 불변
- 약한 스케일 지터: 해상도는 0.3m 고정이므로 ±20% 이내

속도: 무료 코랩은 CPU 2코어라 증강이 학습 병목이 됩니다. OpenCV(cv2, 코랩 기본 설치)가 있으면
크기 변환·블러·JPEG 를 cv2 로 처리하고(수 배 빠름), 없으면 scipy/PIL 로 같은 연산을 합니다.
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    import cv2

    cv2.setNumThreads(0)  # DataLoader 워커마다 스레드를 또 만들지 않도록
except ImportError:  # pragma: no cover
    cv2 = None

_LUM = np.array([0.299, 0.587, 0.114], np.float32)


def _resize(img, h, w, nearest=False):
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    zy, zx = h / img.shape[0], w / img.shape[1]
    zoom = (zy, zx) + ((1,) if img.ndim == 3 else ())
    out = ndi.zoom(img, zoom, order=0 if nearest else 1)
    return out[:h, :w]


def _blur(img, sigma):
    if cv2 is not None:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    return ndi.gaussian_filter(img, (sigma, sigma) + ((0,) if img.ndim == 3 else ()))


def scene_color(img, rng, strength=1.0):
    """img: float32 [0,1] (H,W,3)"""
    s = strength
    # 화이트밸런스
    img = img * (1 + rng.uniform(-0.12, 0.12, 3).astype(np.float32) * s)
    # 백분위 스트레칭 (자동 대비 보정 흉내). 백분위는 1/16 표본으로 계산해도 충분
    if rng.random() < 0.7:
        sub = img[::4, ::4]
        lo = np.percentile(sub, rng.uniform(0, 3))
        hi = np.percentile(sub, rng.uniform(97, 100))
        img = (img - lo) / max(hi - lo, 1e-3)
    # 밝기·대비
    img = (img - 0.5) * (1 + rng.uniform(-0.3, 0.3) * s) + 0.5 + rng.uniform(-0.12, 0.12) * s
    img = np.clip(img, 0, 1)
    # 감마
    img = img ** np.float32(np.exp(rng.uniform(-0.35, 0.35) * s))
    # 채도
    lum = (img @ _LUM)[..., None]
    img = lum + (img - lum) * (1 + rng.uniform(-0.45, 0.5) * s)
    # 누런 연무
    if rng.random() < 0.3 * s:
        haze = np.array([0.80, 0.76, 0.62], np.float32) * rng.uniform(0.8, 1.1)
        a = rng.uniform(0.05, 0.3)
        img = img * (1 - a) + haze * a
    return np.clip(img, 0, 1).astype(np.float32)


import os as _os

# [exp] 증강 2판 (scratchpad/augres 조사): RHM(Yaras 2021, 도시 간 IoU 0.415→0.553), 0.3m 위성 팬:다중분광 4:1 크로마,
# 축소 후 확대(0.4~0.55m 제품을 0.3m 로 키운 경우). EXP_AUG2=1 일 때만 켜짐 (기존 학습 재현성 유지)
AUG2 = _os.environ.get("EXP_AUG2", "0") == "1"
# [DiWA] 증강 세기 배율 (런마다 조금씩 다르게 → 같은 init 의 K 런을 다양하게). 1 = 기존과 동일
AUG_STRENGTH = float(_os.environ.get("EXP_AUG_STRENGTH", "1"))
_RHM_REF = None


def _rhm_ref():
    """참조 영상들의 채널별 분위수 (K,3,Q). EXP_RHM_REF(.npy) 에서 한 번만 읽음 (make_rhm_ref.py 가 만듦)."""
    global _RHM_REF
    if _RHM_REF is None:
        p = _os.environ.get("EXP_RHM_REF", "")
        _RHM_REF = np.load(p).astype(np.float32) if p and _os.path.exists(p) else np.zeros((0, 3, 2), np.float32)
    return _RHM_REF


def _entropy(x, bins=64):
    h, _ = np.histogram(x, bins=bins, range=(0, 1))
    p = h[h > 0] / max(h.sum(), 1)
    return float(-(p * np.log2(p)).sum())


def rhm(img, rng, max_drop=0.6, tries=2):
    """Randomized Histogram Matching: 채널별 누적분포를 무작위 참조 영상의 것으로 맞춤.
    정보가 크게 깎이는 쌍(밝기 엔트로피가 max_drop 비트 넘게 감소)은 다시 뽑고, 끝내 안 되면 원본 유지."""
    ref = _rhm_ref()
    if len(ref) == 0:
        return img
    Q = ref.shape[2]
    qs = np.linspace(0, 1, Q, dtype=np.float32)
    sub = img[::4, ::4].reshape(-1, 3)
    src = np.quantile(sub, qs, axis=0).T.astype(np.float32)        # (3,Q)
    h0 = _entropy(img @ _LUM)
    dith = img + rng.uniform(-0.5 / 255, 0.5 / 255, img.shape).astype(np.float32)   # uint8 계단 때문에 생기는 색 띠(포스터화) 방지
    for _ in range(tries):
        r = ref[int(rng.integers(len(ref)))]
        out = np.empty_like(img)
        for c in range(3):
            s = np.maximum.accumulate(src[c] + np.arange(Q, dtype=np.float32) * 1e-6)   # 단조 증가 보장
            out[..., c] = np.interp(dith[..., c], s, r[c])
        if h0 - _entropy(out @ _LUM) <= max_drop:
            a = rng.uniform(0.5, 1.0)                                         # 원본과 섞어 강도 조절
            return np.clip(a * out + (1 - a) * img, 0, 1).astype(np.float32)
    return img


def sat_degrade(img, rng, p_chroma=0.6, p_blur=0.7, p_noise=0.5, p_jpeg=0.3):
    h, w = img.shape[:2]
    if AUG2 and rng.random() < 0.2:  # [exp] 축소 후 확대: 원래 0.4~0.55m 인 제품을 0.3m 로 키운 영상 흉내
        f = rng.uniform(1.2, 1.8)
        img = _resize(_resize(np.ascontiguousarray(img), max(int(h / f), 1), max(int(w / f), 1)), h, w)
    if rng.random() < p_chroma:  # 팬샤프닝: 색은 저해상도
        lum = (img @ _LUM)[..., None]
        chroma = np.ascontiguousarray(img - lum)
        f = rng.uniform(3.0, 5.0) if AUG2 else rng.uniform(1.5, 4.0)   # [exp] 0.3m급 위성은 대개 팬:다중분광 4:1
        small = _resize(chroma, max(int(h / f), 1), max(int(w / f), 1))
        img = lum + _resize(small, h, w)
    if rng.random() < p_blur:  # PSF + 샤프닝
        # 질감이 쓰레기의 가장 큰 단서라 σ 0.9 이하로 제한 (실제 데이터 분석: 쓰레기 5x5 표준편차 0.083, 포말·바위 0.04)
        blur = _blur(np.ascontiguousarray(img, dtype=np.float32), rng.uniform(0.3, 0.9))
        amt = rng.uniform(0, 1.2) if rng.random() < 0.5 else 0.0
        img = blur + amt * (blur - _blur(blur, 1.5)) if amt else blur
    if rng.random() < p_noise:
        img = img + rng.normal(0, rng.uniform(0.003, 0.02), img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    if rng.random() < p_jpeg:
        u8 = (img * 255 + 0.5).astype(np.uint8)
        q = int(rng.integers(55, 95))
        if cv2 is not None:
            ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
        else:
            buf = io.BytesIO()
            Image.fromarray(u8).save(buf, "JPEG", quality=q)
            img = np.asarray(Image.open(buf)).astype(np.float32) / 255.0
    return np.ascontiguousarray(img, dtype=np.float32)


def random_shadow(img, rng):
    """바위·건물 그림자 흉내: 불규칙하고 매끈한 어두운 영역을 드리웁니다 (라벨은 그대로).

    그림자 진 바위를 쓰레기로 착각하는 오탐을 줄이기 위한 증강입니다. 그림자는 질감 없이 매끈하게 어둡게 만들어
    '어둡다 = 쓰레기' 가 아니라 '거친 질감 = 쓰레기' 를 배우도록 합니다 (검은 고무 같은 어두운 쓰레기는 질감이 있음).
    """
    h, w = img.shape[:2]
    f = int(rng.choice([8, 16, 32]))                       # 그림자 덩어리 크기
    small = rng.random((h // f + 2, w // f + 2)).astype(np.float32)
    if rng.random() < 0.5:                                  # 한 방향으로 길쭉한 드리운 그림자
        small = np.repeat(small, 3, axis=int(rng.integers(2)))
    n = _blur(_resize(small, h, w), f / 3)
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    soft = _blur((n > rng.uniform(0.55, 0.8)).astype(np.float32), rng.uniform(0.8, 2.0))
    k = rng.uniform(0.3, 0.7)                              # 그림자 속 밝기 비율
    tint = np.array([0.94, 0.97, 1.06], np.float32)        # 그늘은 하늘빛이 섞여 살짝 푸름
    shade = 1 - soft[..., None] * (1 - k * tint)
    return np.clip(img * shade, 0, 1).astype(np.float32)


def d4(img, mask, rng):
    k = int(rng.integers(4))
    img, mask = np.rot90(img, k), np.rot90(mask, k)
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def scale_jitter(img, mask, rng, lo=0.85, hi=1.2):
    """mask: bool 또는 라벨맵 uint8(0 배경/1 쓰레기/2 채점 제외). 축소해서 생긴 테두리는 라벨맵이면 '채점 제외'로 채움
    (반사 패딩된 영상에 비친 쓰레기에 라벨이 없으므로)."""
    s = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    h, w = mask.shape
    nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
    im = _resize(img, nh, nw)
    is_bool = mask.dtype == bool
    mk = _resize(mask.astype(np.uint8), nh, nw, nearest=True)
    mk = mk.astype(bool) if is_bool else mk
    if s >= 1:
        y = int(rng.integers(0, im.shape[0] - h + 1))
        x = int(rng.integers(0, im.shape[1] - w + 1))
        return im[y:y + h, x:x + w], mk[y:y + h, x:x + w]
    py, px = h - im.shape[0], w - im.shape[1]
    pad = ((py // 2, py - py // 2), (px // 2, px - px // 2))
    return np.pad(im, pad + ((0, 0),), mode="reflect"), np.pad(mk, pad, mode="constant",
                                                              constant_values=False if is_bool else 2)


def train_augment(img_u8, mask, rng, p_scale=0.3, p_color=0.9, p_degrade=0.8, p_shadow=0.3):
    img = img_u8.astype(np.float32) / 255.0
    if AUG_STRENGTH != 1:
        p_scale, p_shadow, p_degrade = (min(1.0, p * AUG_STRENGTH) for p in (p_scale, p_shadow, p_degrade))
    img, mask = d4(img, mask, rng)
    if rng.random() < p_scale:
        img, mask = scale_jitter(img, mask, rng)
    if rng.random() < p_shadow:
        img = random_shadow(img, rng)
    if rng.random() < p_degrade:
        img = sat_degrade(img, rng)
    if AUG2 and rng.random() < 0.5:   # [exp] RHM 을 쓰면 장면 색 보정은 확률을 낮춤 (둘 다 색 분포를 바꿈)
        img = rhm(img, rng)
        p_color = min(p_color, 0.6)
    if rng.random() < p_color:
        img = scene_color(img, rng)
    return img, mask
