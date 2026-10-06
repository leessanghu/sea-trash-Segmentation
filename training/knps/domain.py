"""드론 영상(AIHub 실제 데이터) → 대회 위성 영상 쪽으로 맞추는 처리.

드론 영상의 바다는 청록색인데 대회 영상은 "아주 어두운 남색이거나 거의 검정"(대회 설명)입니다.
학습할 때 드론 영상의 바다를 확률적으로(70%) 어둡게 바꿔, 바다 색 차이를 모델이 단서로 쓰지 않게 합니다.
(검증셋은 바꾸지 않습니다. 실험끼리 같은 기준으로 비교하기 위해서)
"""
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

from . import synth as S

_LUM = np.array([0.299, 0.587, 0.114], np.float32)


def hsv(img):
    mx, mn = img.max(2), img.min(2)
    s = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    d = mx - mn + 1e-6
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, s, mx


def texture(img):
    """국소 밝기 표준편차(5x5)를 9x9 로 평균한 거칠기. 바다·모래는 0.01 이하, 숲·갯바위는 0.03 이상."""
    lum = img @ _LUM
    lstd = np.sqrt(np.maximum(ndi.uniform_filter(lum ** 2, 5) - ndi.uniform_filter(lum, 5) ** 2, 0))
    return ndi.uniform_filter(lstd, 9)


def drone_water(img, min_px=200):
    """드론 영상(NIR 없음)의 물 추정: 청록·파랑 계열 '이면서' 매끈한 면. img: float [0,1]

    색만 보면 연무 낀 숲(색상 약 185°)이, 매끈함만 보면 모래사장이 물로 잡히므로 두 조건을 모두 요구합니다.
    (예전에는 둘 중 하나만 맞아도 물로 봐서 숲·모래사장까지 남색으로 칠해졌음)
    """
    h, s, _ = hsv(img)
    water = (h > 150) & (h < 215) & (s > 0.08) & (texture(img) < 0.02)
    # 가장자리를 복제해 붙인 뒤 다듬어야 이미지 경계의 물이 깎이지 않음 (경계에 청록 줄이 남는 문제)
    water = ndi.binary_closing(np.pad(water, 3, mode="edge"), iterations=2)[3:-3, 3:-3]
    return S._remove_small(water, min_px)


def drone_veg(img):
    """드론 영상의 식생: 녹색~청록(연무) 계열이면서 거친 면 (물은 매끈, 갯바위는 무채색).
    색상 상한 200°: 그늘진 갯바위는 연무로 푸르스름하게(200° 이상) 찍혀 식생으로 잘못 잡히기 쉬움.
    실제 검증 영상 기준 숲 약 81%, 그늘진 갯바위 약 11% 가 식생으로 잡힘."""
    h, s, v = hsv(img)
    veg = (h > 55) & (h < 200) & (s > 0.07) & (v < 0.75) & (texture(img) > 0.02)
    veg = ndi.binary_closing(np.pad(veg, 3, mode="edge"), iterations=2)[3:-3, 3:-3]   # 나무 사이 틈 메움
    return ndi.binary_opening(veg, iterations=2)


def darken_drone_water(img, rng, min_px=3000):
    """드론 영상의 넓은 물(3000화소 이상: 청록 지붕 등 작은 면은 제외)을 위성 영상처럼 어두운 남색으로. img: float, 제자리 수정."""
    water = drone_water(img, min_px)
    if water.any():
        S.darken_water(img, SimpleNamespace(water=water), rng)
    return img
