"""해안쓰레기 집적대 절차적 합성기.

대회 설명을 그대로 옮긴 규칙
- 쓰레기 집적대: 물가를 따라 이어지는 띠, 한자리에 뭉친 덩어리, 흩어진 작은 무더기
- 질감: 흰색·회색·청색·녹색(+소량 주황·검정)이 섞인 1~3px 밝은 점과 작은 얼룩, 밀도가 들쭉날쭉하고 틈이 있음
- 조각은 0.1~0.3m 로 한 화소(0.3m)보다 작으므로 3배 해상도(0.1m)에서 그린 뒤 면적 평균으로 줄여
  화소 혼합(sub-pixel) 효과를 냅니다
- 정답 제외 대상(하드 네거티브, 라벨 0)도 같은 방식으로 합성합니다
  · 흰 자갈·조개껍질: 쓰레기와 같은 점 질감이지만 '단색(흰~크림)'  → 색 다양성이 핵심 단서임을 학습
  · 회갈색 자갈: 큰 점 + 그림자, 무채색 계열
  · 파도 거품: 물 쪽의 흰 선형 무늬
  · 해조류 퇴적·유목: 어두운 띠와 막대
  · 선박·차량: 큰 단색 인공물

배경(bg)은 실제 항공영상 패치를 넣는 것을 권장합니다. None 이면 절차적 배경을 만듭니다(파이프라인 점검용).
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

H = W = 256
S = 3  # 초해상 배율 (0.3m -> 0.1m)


# ----------------------------------------------------------------------------- 기본 도구
def _noise(shape, sigma, rng):
    """[0,1] 로 정규화한 가우시안 스무딩 노이즈. 큰 sigma 는 저해상도에서 만들어 확대합니다(속도)."""
    h, w = shape
    if sigma >= 4:
        f = int(sigma // 2)
        n = rng.standard_normal((h // f + 2, w // f + 2)).astype(np.float32)
        n = ndi.gaussian_filter(n, sigma / f, mode="reflect")
        n = ndi.zoom(n, f, order=1)[:h, :w]
    else:
        n = rng.standard_normal(shape).astype(np.float32)
        if sigma > 0:
            n = ndi.gaussian_filter(n, sigma, mode="reflect")
    n -= n.min()
    n /= n.max() + 1e-6
    return n


def _disk(r):
    if r <= 0:
        return np.ones((1, 1), bool)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r


def _otsu(v, bins=64):
    hist, edges = np.histogram(v, bins=bins, range=(0, 1))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mt = mu[-1]
    between = (mt * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return float(edges[int(np.argmax(between)) + 1])


def _remove_small(m, min_px):
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def estimate_water(bg):
    """아주 어두운 남색~검정 = 바다 (설명서 기준). 녹색 그늘(숲)은 제외."""
    f = bg.astype(np.float32) / 255.0
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)
    lum_s = ndi.gaussian_filter(lum, 2)
    thr = min(_otsu(lum_s), 0.28)
    dark = lum_s < thr
    green = (f[..., 1] > f[..., 0] * 1.08) & (f[..., 1] > f[..., 2] * 1.02) & (lum_s > 0.10)
    w = ndi.binary_opening(dark & ~green, iterations=2)
    return _remove_small(w, int(0.01 * w.size))


def estimate_veg(bg):
    """녹색이 우세한 화소 = 식생 (NIR 이 없을 때의 근사)."""
    f = bg.astype(np.float32) / 255.0
    g = ndi.gaussian_filter(f, (1.5, 1.5, 0))
    veg = (g[..., 1] > g[..., 0] * 1.05) & (g[..., 1] > g[..., 2] * 1.05) & (g.max(2) - g.min(2) > 0.04)
    return _remove_small(ndi.binary_opening(veg, iterations=1), 30)


AUX_LAND, AUX_WATER, AUX_VEG, AUX_NEAR = 0, 1, 2, 3
AUX_VERSION = "2"
SYNTH_VERSION = "4"  # 합성 방식이 바뀌면 올립니다 (3: 테트라포드·양식장·갯벌, 4: 그림자 진 갯바위 추가)
SHORE_MAX_PX = 70  # 쓰레기를 놓을 수 있는 물가로부터의 최대 거리 (21m)


def aux_from_nir(rgbn, min_water=15000, max_hole=3000, near_px=SHORE_MAX_PX):
    """NAIP 4밴드(R,G,B,NIR) → 보조 라벨. 패치가 아니라 넓은 블록(수백 m) 단위로 호출합니다.

    0 육지(쓰레기를 놓지 않는 곳: 물가에서 먼 땅)   1 물   2 식생   3 물가 땅(near_px 이내의 식생 아닌 땅)

    물: NIR 이 거의 0 이고 초록은 밝음 → (G-NIR) 절대 차이가 큼. 그림자(모든 밴드 어두움)와
    아스팔트·지붕(G≈NIR)은 이 조건에서 빠집니다 (NAIP 표본: 바다 G-N≈99, 그림자≈43, 도로≈28).
    - min_water: 이보다 작은 물 영역은 버림 (연못·풀장·어두운 지붕·그림자 오검출 제거, 15000px ≈ 37m×37m)
    - 쇄파·포말(밝고 G-N 이 중간)은 물에 붙어 있으면 물로 봄. 마른 모래는 G-N≈17 이라 걸리지 않음
    - max_hole: 물 안의 작은 구멍(포말·작은 바위)은 물로 채움
    """
    f = rgbn.astype(np.float32)
    r, g, nir = f[..., 0], f[..., 1], f[..., 3]
    d = ndi.gaussian_filter(g - nir, 1.5)
    ndwi = ndi.gaussian_filter((g - nir) / (g + nir + 1e-6), 1.5)
    core = _remove_small(ndi.binary_opening((d > 60) & (ndwi > 0.3), iterations=2), min_water)
    # 어둡게 찍힌 물(먼바다 촬영 구간 등): NIR 이 거의 0 인 넓은 영역. 건물 그림자는 이보다 작음
    dark = (ndi.gaussian_filter(nir, 2) < 25) & (ndwi > 0.4)
    core |= _remove_small(ndi.binary_opening(dark, iterations=2), 50000)
    water = core
    if core.any():
        # 탁한 바다·파도 포말 구역(오리건 등)은 G-N 이 44 전후로 낮아짐. 젖은·마른 모래는 22 이하.
        # 확실한 바다(core)에 이어진 영역만 물로 넓혀, 떨어진 그림자 등은 끌어들이지 않습니다.
        ext = (ndi.gaussian_filter(g - nir, 3) > 32) & (ndi.gaussian_filter((g - nir) / (g + nir + 1e-6), 3) > 0.1)
        lab, n = ndi.label(ext | core)
        keep = np.zeros(n + 1, bool)
        keep[np.unique(lab[core])] = True
        keep[0] = False
        water = keep[lab]
        foam = (ndi.gaussian_filter(g, 1.5) > 170) & (d > 30)
        water |= foam & (ndi.distance_transform_edt(~water) <= 60)
        holes = ndi.binary_fill_holes(water) & ~water
        water |= holes & ~_remove_small(holes, max_hole)
    ndvi = ndi.gaussian_filter((nir - r) / (nir + r + 1e-6), 1.5)
    veg = _remove_small(ndi.binary_opening((ndvi > 0.2) & ~water, iterations=1), 30)
    aux = np.zeros(rgbn.shape[:2], np.uint8)
    aux[water] = AUX_WATER
    aux[veg] = AUX_VEG
    if near_px and water.any():
        near = (ndi.distance_transform_edt(~water) <= near_px) & ~water & ~veg
        aux[near] = AUX_NEAR
    return aux


@dataclass
class Ctx:
    water: np.ndarray
    veg: np.ndarray
    ok: np.ndarray          # 쓰레기를 놓을 수 있는 곳 (물·식생 제외: 모래·자갈·바위·갯벌)
    land_dist: np.ndarray   # 육지 화소에서 가장 가까운 물까지 거리 (px)
    water_dist: np.ndarray  # 물 화소에서 가장 가까운 육지까지 거리 (px)
    has_shore: bool
    sun: tuple              # 그림자 방향 (초해상 화소 단위)


def make_ctx(bg, rng, water=None, veg=None, near=None):
    """near: 블록 단위로 미리 구한 '물가 땅' 마스크(aux==3). 주어지면 쓰레기는 거기에만 놓습니다.
    (해안선이 패치 밖에 있어도 정확함). 없으면 패치 안에서 추정합니다."""
    if water is None:
        water = estimate_water(bg)
    if veg is None:
        veg = estimate_veg(bg) & ~water
    frac = water.mean()
    has_shore = False
    if 0.02 < frac < 0.97:  # 연못·풀장이 아닌 넓은 물(바다·하구)이 있어야 해안으로 봄
        lab, n = ndi.label(water)
        has_shore = n > 0 and np.bincount(lab.ravel())[1:].max() >= 2000
    land_dist = ndi.distance_transform_edt(~water) if water.any() else np.full(water.shape, 1e3, np.float32)
    water_dist = ndi.distance_transform_edt(water) if water.any() else np.zeros(water.shape, np.float32)
    sun = (int(rng.choice([-1, 1]) * rng.integers(1, 4)), int(rng.choice([-1, 1]) * rng.integers(1, 4)))
    if near is not None:
        ok = near & ~water
    else:
        ok = ~water & ~veg
        if has_shore:
            ok &= land_dist <= SHORE_MAX_PX
    return Ctx(water, veg, ok, land_dist, water_dist, has_shore, sun)


# ----------------------------------------------------------------------------- 영역(모양) 생성
def _curve_dist(rng, shape=(H, W)):
    """패치를 가로지르는 구불구불한 곡선까지의 거리."""
    h, w = shape
    th = rng.uniform(0, np.pi)
    c = np.array([rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w])
    t = np.arange(-400, 400, 0.5)
    d = np.array([np.sin(th), np.cos(th)])
    n = np.array([-d[1], d[0]])
    off = rng.uniform(3, 25) * np.sin(2 * np.pi * t / rng.uniform(60, 300) + rng.uniform(0, 6.3)) \
        + rng.uniform(0, 8) * np.sin(2 * np.pi * t / rng.uniform(15, 60) + rng.uniform(0, 6.3))
    pts = c[None] + t[:, None] * d[None] + off[:, None] * n[None]
    pts = np.round(pts).astype(int)
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < h) & (pts[:, 1] >= 0) & (pts[:, 1] < w)
    canvas = np.ones(shape, bool)
    canvas[pts[ok, 0], pts[ok, 1]] = False
    return ndi.distance_transform_edt(canvas)


def _breaks(m, rng, p=0.7):
    if rng.random() > p:
        return m
    keep = _noise(m.shape, rng.uniform(5, 18), rng) > rng.uniform(0.15, 0.5)
    return m & keep


def region_band(ctx, rng):
    """물가를 따라 이어지는 띠 (표착선)."""
    wid = rng.uniform(2, 18) * (0.4 + 1.2 * _noise((H, W), 12, rng))
    if ctx.has_shore and rng.random() < 0.85:
        d0 = rng.uniform(0, 30) + 10 * (_noise((H, W), 20, rng) - 0.5)
        d0 = np.maximum(d0, 0)
        m = (ctx.land_dist >= d0) & (ctx.land_dist <= d0 + wid) & ctx.ok
    else:
        m = (_curve_dist(rng) <= wid / 2) & ctx.ok
    return _breaks(m, rng)


def _land_point(ctx, rng, near_shore=True):
    cand = np.argwhere(ctx.ok & (ctx.land_dist < 35)) if ctx.has_shore and near_shore else np.zeros((0, 2))
    if len(cand) == 0:
        cand = np.argwhere(ctx.ok)
    if len(cand) == 0:
        cand = np.argwhere(~ctx.water)
    if len(cand) == 0:
        cand = np.argwhere(np.ones((H, W), bool))
    return cand[rng.integers(len(cand))].astype(np.float32)


def _ellipse(center, rx, ry, rng, rough=0.5):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = rng.uniform(0, np.pi)
    u = (xx - center[1]) * np.cos(th) + (yy - center[0]) * np.sin(th)
    v = -(xx - center[1]) * np.sin(th) + (yy - center[0]) * np.cos(th)
    e = (u / rx) ** 2 + (v / ry) ** 2
    e = e + rough * 2 * (_noise((H, W), max(2.0, min(rx, ry) / 2.5), rng) - 0.5)
    return e < 1


def region_blob(ctx, rng):
    """한자리에 뭉친 덩어리. 오목하거나 여러 조각으로 나뉠 수 있음."""
    c = _land_point(ctx, rng, rng.random() < 0.7)
    m = np.zeros((H, W), bool)
    r0 = rng.uniform(4, 45)
    for i in range(int(rng.integers(1, 4))):
        rx = r0 * rng.uniform(0.5, 1.2)
        ry = rx * rng.uniform(0.3, 1.0)
        cc = c + (rng.uniform(-1.3, 1.3, 2) * r0 if i else 0)
        m |= _ellipse(cc, rx, ry, rng, rough=rng.uniform(0.2, 0.8))
    return _breaks(m, rng, 0.4)


def region_scatter(ctx, rng):
    """흩어진 작은 무더기 여러 개."""
    c = _land_point(ctx, rng)
    m = np.zeros((H, W), bool)
    spread = rng.uniform(10, 50)
    for _ in range(int(rng.integers(3, 10))):
        cc = c + rng.normal(0, spread, 2)
        r = rng.uniform(2, 8)
        m |= _ellipse(cc, r, r * rng.uniform(0.5, 1), rng, 0.3)
    return m


def make_region(ctx, rng, kind=None):
    kind = kind or rng.choice(["band", "blob", "scatter"], p=[0.45, 0.4, 0.15])
    for _ in range(6):
        m = {"band": region_band, "blob": region_blob, "scatter": region_scatter}[kind](ctx, rng)
        m &= ndi.binary_dilation(ctx.ok, iterations=1)
        if m.sum() >= 40:
            return m
    if not ctx.ok.any():
        return np.zeros((H, W), bool)
    return _ellipse(_land_point(ctx, rng, False), 12, 8, rng, 0.3) & ndi.binary_dilation(ctx.ok, iterations=1)


# ----------------------------------------------------------------------------- 팔레트
_DEBRIS = [  # (가중치, 기준 RGB, 지터)
    (0.28, (245, 245, 242), 12),   # 스티로폼·흰 플라스틱
    (0.14, (200, 202, 205), 20),   # 밝은 회색 / 투명 PET
    (0.10, (140, 142, 145), 20),   # 회색
    (0.15, (45, 105, 195), 35),    # 청색 플라스틱·로프
    (0.06, (130, 185, 225), 25),   # 하늘색
    (0.11, (55, 145, 90), 30),     # 녹색 그물·로프
    (0.06, (35, 35, 38), 12),      # 검정 고무·타이어
    (0.06, (225, 105, 40), 30),    # 주황·빨강 부표
    (0.04, (225, 205, 70), 30),    # 노랑
]
_SHELL = [(1.0, (238, 232, 215), 10)]                                       # 흰 자갈·조개: 단색
_GRAVEL = [(0.5, (170, 160, 145), 25), (0.3, (130, 120, 108), 20), (0.2, (200, 195, 185), 15)]  # 무채색 자갈


def _sample_colors(pal, n, rng, lum_jit=0.12):
    w = np.array([p[0] for p in pal])
    idx = rng.choice(len(pal), n, p=w / w.sum())
    base = np.array([p[1] for p in pal], np.float32)[idx]
    jit = np.array([p[2] for p in pal], np.float32)[idx]
    col = base + rng.normal(0, 1, (n, 3)).astype(np.float32) * jit[:, None] * 0.5
    col *= (1 + rng.normal(0, lum_jit, (n, 1))).astype(np.float32)
    return np.clip(col / 255.0, 0, 1)


# ----------------------------------------------------------------------------- 점 질감 렌더러
def render_speckles(img, region, rng, ctx, palette, *, density=(0.25, 0.85), sizes=None,
                    underlay=(0.0, 0.45), shadow=True, n_big_rate=1 / 300):
    """region 안을 초해상도에서 조각(점)으로 채운 뒤 면적 평균으로 줄여 img 에 합성합니다.

    반환: (H,W) 조각 피복률 (0~1)
    """
    cov_full = np.zeros(region.shape, np.float32)
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return cov_full
    y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, H)
    x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, W)
    reg = region[y0:y1, x0:x1]
    h, w = reg.shape
    hh, ww = h * S, w * S
    canvas = np.repeat(np.repeat(img[y0:y1, x0:x1], S, 0), S, 1)
    reg_up = np.repeat(np.repeat(reg, S, 0), S, 1)

    edge = ndi.zoom(ndi.distance_transform_edt(np.pad(reg, 1))[1:-1, 1:-1].astype(np.float32), S, order=1)
    edge = np.clip(edge / rng.uniform(0.7, 3), 0, 1) * reg_up
    dens = rng.uniform(*density) * (0.3 + 0.7 * _noise((hh, ww), S * rng.uniform(2, 8), rng)) * edge

    # 바닥 톤: 쓰레기 더미 사이의 그늘·섞인 흙 (어둡고 탁해짐)
    ua = rng.uniform(*underlay) * _noise((hh, ww), S * 2, rng) * reg_up
    if ua.max() > 0:
        tone = canvas[reg_up].mean(0) * rng.uniform(0.55, 0.95)
        tone = tone.mean() * 0.5 + tone * 0.5
        canvas = canvas * (1 - ua[..., None]) + tone[None, None] * ua[..., None]

    if sizes is None:
        sizes = ([1, 2, 3, 4, 6], [0.45, 0.30, 0.15, 0.07, 0.03])
    ks, kp = np.array(sizes[0]), np.array(sizes[1], np.float64)
    mean_area = float((ks ** 2 * kp).sum() / kp.sum())
    seed = rng.random((hh, ww)) < dens / mean_area * rng.uniform(0.6, 1.4)
    sy, sx = np.nonzero(seed)

    # 드물게 큰 물체 (스티로폼 부표 덩어리, 드럼통, 그물 뭉치) 0.6~1.2m
    n_big = rng.poisson(reg.sum() * n_big_rate * rng.uniform(0, 2))
    if n_big:
        cand = np.argwhere(reg_up)
        pick = cand[rng.integers(len(cand), size=n_big)]
        sy, sx = np.concatenate([sy, pick[:, 0]]), np.concatenate([sx, pick[:, 1]])
    n = len(sy)
    if n == 0:
        return cov_full
    kh = rng.choice(ks, n, p=kp / kp.sum())
    if n_big:
        kh[-n_big:] = rng.integers(6, 13, n_big)
    kw = kh.copy()
    elong = rng.random(n) < 0.25  # 병·로프 같은 길쭉한 조각
    kw[elong] = np.minimum(kh[elong] * rng.integers(2, 4, elong.sum()), 12)
    tr = rng.random(n) < 0.5
    kh, kw = np.where(tr, kw, kh), np.where(tr, kh, kw)
    colors = _sample_colors(palette, n, rng)

    # 모든 조각의 화소 좌표를 한 번에 펼쳐 칠합니다 (나중 조각이 앞 조각을 덮음)
    order = rng.permutation(n)
    sy, sx, kh, kw, colors = sy[order], sx[order], kh[order], kw[order], colors[order]
    area = kh * kw
    fid = np.repeat(np.arange(n), area)
    local = np.arange(fid.size) - np.repeat(np.cumsum(area) - area, area)
    yy = np.clip(sy[fid] + local // kw[fid] - kh[fid] // 2, 0, hh - 1)
    xx = np.clip(sx[fid] + local % kw[fid] - kw[fid] // 2, 0, ww - 1)
    frag = np.zeros((hh, ww), bool)
    shad = np.zeros((hh, ww), bool)
    canvas[yy, xx] = colors[fid]
    frag[yy, xx] = True
    if shadow:
        sdy, sdx = ctx.sun
        shad[np.clip(yy + sdy, 0, hh - 1), np.clip(xx + sdx, 0, ww - 1)] = True
    if shadow:
        sm = shad & ~frag
        canvas[sm] *= rng.uniform(0.5, 0.8)

    img[y0:y1, x0:x1] = canvas.reshape(h, S, w, S, 3).mean((1, 3))
    cov_full[y0:y1, x0:x1] = frag.reshape(h, S, w, S).mean((1, 3))
    return cov_full


def debris_label(cov, region, rng):
    """정답 규칙: 조각이 덮은 화소=1, 조각 사이 좁은 틈은 0 가능, 1화소 미만 조각은 미표시."""
    m = cov >= rng.uniform(0.08, 0.2)
    r = int(rng.choice([0, 1, 2], p=[0.2, 0.5, 0.3]))
    if r:
        m = ndi.binary_closing(m, _disk(r)) | m
    m &= ndi.binary_dilation(region, iterations=1)
    return _remove_small(m, 4)


# ----------------------------------------------------------------------------- 하드 네거티브
def add_foam(img, ctx, rng):
    """파도 거품: 물 쪽의 흰 선형 무늬 (정답 아님)."""
    if not (ctx.has_shore and ctx.water.any()):
        return
    alpha = np.zeros((H, W), np.float32)
    for _ in range(int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))):
        d = rng.uniform(0.5, 9) + 3 * (_noise((H, W), 15, rng) - 0.5)
        wd = rng.uniform(0.6, 3)
        line = np.clip(1 - np.abs(ctx.water_dist - d) / wd, 0, 1) * ctx.water
        alpha = np.maximum(alpha, line * rng.uniform(0.5, 0.95))
    if rng.random() < 0.3:  # 물가 포말 얼룩
        blot = _noise((H, W), 1.2, rng) ** 4 * (ctx.water_dist < rng.uniform(4, 12)) * ctx.water
        alpha = np.maximum(alpha, np.clip(blot * 1.5, 0, 0.6))
    alpha = _breaks(alpha > 0, rng, 0.6) * alpha
    alpha = ndi.gaussian_filter(alpha, 0.6)
    white = np.array([0.93, 0.95, 0.97], np.float32) * rng.uniform(0.85, 1.05)
    img[:] = img * (1 - alpha[..., None]) + white * alpha[..., None]


def add_wrack(img, ctx, rng):
    """해조류 퇴적: 물가의 어두운 띠."""
    if not ctx.has_shore:
        return
    d0 = rng.uniform(0, 12)
    wd = rng.uniform(2, 10) * (0.5 + _noise((H, W), 10, rng))
    m = (ctx.land_dist >= d0) & (ctx.land_dist <= d0 + wd) & ~ctx.water
    m = _breaks(m, rng, 0.8)
    col = np.array(rng.choice([[0.24, 0.20, 0.13], [0.16, 0.18, 0.11], [0.12, 0.10, 0.08]]), np.float32)
    col = col * rng.uniform(0.7, 1.3)
    a = m * rng.uniform(0.45, 0.9) * (0.5 + 0.5 * _noise((H, W), 1, rng))
    a = ndi.gaussian_filter(a.astype(np.float32), 0.5)
    img[:] = img * (1 - a[..., None]) + col * a[..., None]


def add_driftwood(img, ctx, rng):
    """유목: 어두운 갈색 막대."""
    if not (~ctx.water).any():
        return
    c = _land_point(ctx, rng)
    m = np.zeros((H, W), bool)
    for _ in range(int(rng.integers(1, 8))):
        p = c + rng.normal(0, 25, 2)
        th = rng.uniform(0, np.pi)
        L = rng.uniform(4, 30)
        t = np.linspace(-L / 2, L / 2, int(L * 2) + 2)
        yy = np.clip(np.round(p[0] + t * np.sin(th)).astype(int), 0, H - 1)
        xx = np.clip(np.round(p[1] + t * np.cos(th)).astype(int), 0, W - 1)
        m[yy, xx] = True
    if rng.random() < 0.5:
        m = ndi.binary_dilation(m)
    col = np.array([0.33, 0.25, 0.17], np.float32) * rng.uniform(0.6, 1.4)
    img[m] = img[m] * 0.25 + col * 0.75


def add_shells(img, ctx, rng):
    """흰 자갈·조개 무더기: 쓰레기와 같은 점 질감이지만 단색 (정답 아님)."""
    reg = make_region(ctx, rng, rng.choice(["band", "blob"]))
    render_speckles(img, reg, rng, ctx, _SHELL, density=(0.4, 0.95), underlay=(0.0, 0.2),
                    shadow=rng.random() < 0.5, n_big_rate=0)


def add_gravel(img, ctx, rng):
    """회갈색 자갈밭: 큰 점 + 그림자, 무채색 (정답 아님)."""
    reg = make_region(ctx, rng, rng.choice(["band", "blob"]))
    render_speckles(img, reg, rng, ctx, _GRAVEL, density=(0.5, 1.0), sizes=([2, 3, 4, 6, 8], [0.2, 0.3, 0.25, 0.15, 0.1]),
                    underlay=(0.1, 0.4), shadow=True, n_big_rate=0)


def add_rock_shadows(img, ctx, rng):
    """그림자 진 갯바위: 거친 회갈색 바위 면 + 틈·돌출부 뒤의 짙고 매끈한 그림자 (정답 아님).

    실제 검증에서 그림자로 어두워진 바위를 쓰레기로 잡는 오탐이 확인되어 추가했습니다.
    그림자는 질감 없이 매끈하게 어둡고, 바위 면은 무채색이라 '여러 색 점의 거친 군집'인 쓰레기와 구분됩니다.
    """
    # 갯바위는 넓으므로 쓰레기 배치 구역(ok)에 묶지 않고, 물가 근처에 큰 바위 덩어리를 놓습니다 (물가에 살짝 걸쳐도 됨)
    c = _land_point(ctx, rng, True)
    reg = np.zeros((H, W), bool)
    r0 = rng.uniform(25, 90)
    for i in range(int(rng.integers(1, 4))):
        cc = c + (rng.uniform(-1.0, 1.0, 2) * r0 if i else 0)
        rx = r0 * rng.uniform(0.6, 1.1)
        reg |= _ellipse(cc, rx, rx * rng.uniform(0.4, 1.0), rng, rough=rng.uniform(0.4, 0.8))
    if ctx.water.any():
        reg &= ~ctx.water | (ctx.water_dist <= rng.uniform(2, 10))
    reg &= ~ctx.veg
    if reg.sum() < 300:
        return
    a = ndi.gaussian_filter(reg.astype(np.float32), 1.5)[..., None]
    base = np.array([0.46, 0.43, 0.39], np.float32) * rng.uniform(0.7, 1.25) * rng.uniform(0.95, 1.05, 3).astype(np.float32)
    tex = 0.22 * (_noise((H, W), 1.0, rng) - 0.5) + 0.18 * (_noise((H, W), 3, rng) - 0.5)
    rock = base * (1 + tex[..., None])
    # 작은 바윗덩어리 사이 틈(가는 선)과, 틈에서 해를 등진 쪽으로 1~3화소 늘어진 그림자
    blocks = _noise((H, W), rng.uniform(1.5, 3.5), rng)
    edges = np.abs(blocks - 0.5) < rng.uniform(0.02, 0.05)
    edges &= rng.random((H, W)) < rng.uniform(0.6, 0.9)              # 틈이 군데군데 끊기게
    dark = edges | (_noise((H, W), rng.uniform(1.0, 2.0), rng) < rng.uniform(0.08, 0.18))
    sdy, sdx = np.sign(ctx.sun[0]) or 1, np.sign(ctx.sun[1]) or 1
    for k in range(1, int(rng.integers(2, 4))):
        dark |= np.roll(edges, (sdy * k, sdx * k), axis=(0, 1))
    lit = np.roll(edges, (-sdy, -sdx), axis=(0, 1)) & ~dark          # 해를 받는 면은 조금 밝게
    rock[lit & reg] = np.clip(rock[lit & reg] * rng.uniform(1.1, 1.25), 0, 1)
    sa = ndi.gaussian_filter(dark.astype(np.float32), 0.6)[..., None] * rng.uniform(0.75, 0.95)
    shadow = np.array([0.05, 0.06, 0.08], np.float32) * rng.uniform(0.6, 1.6)
    rock = rock * (1 - sa) + shadow * sa
    img[:] = img * (1 - a) + np.clip(rock, 0, 1) * a


# 실제 검증 오탐에서 확인된 '외딴 바위·밝은 바위·바위 주변 물보라' (v4 분석, 사용자 관찰)
# 회색·회갈색 바위. 흰색에 가까운 외딴 덩어리는 스티로폼 쓰레기와 헷갈리게 가르칠 수 있어 넣지 않음
_BOULDER = np.array([[0.56, 0.55, 0.53], [0.47, 0.42, 0.37], [0.64, 0.62, 0.58], [0.33, 0.31, 0.29]], np.float32)
_LUM3 = np.array([0.299, 0.587, 0.114], np.float32)


def _shift(m, dy, dx):
    """경계에서 되감기지 않는 평행 이동 (np.roll 은 반대편에 그림자가 생김)."""
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def _sun_dir(ctx):
    return int(np.sign(ctx.sun[0]) or 1), int(np.sign(ctx.sun[1]) or 1)


def _rock_shading(m, r, ctx):
    """해를 받는 쪽 +1, 반대쪽 -1 정도의 음영 (그림자는 ctx.sun 방향으로 드리움)."""
    g = ndi.gaussian_filter(m.astype(np.float32), max(r / 2.5, 0.8))
    gy, gx = np.gradient(g)
    sdy, sdx = _sun_dir(ctx)
    sh = gy * sdy + gx * sdx
    return sh / (np.abs(sh[m]).max() + 1e-6) if m.any() else sh


def _cast_shadow(img, m, ctx, length, strength):
    sdy, sdx = _sun_dir(ctx)
    sh = np.zeros_like(m)
    for k in range(1, length + 1):
        sh |= _shift(m, sdy * k, sdx * k)
    sh &= ~m
    img *= 1 - ndi.gaussian_filter(sh.astype(np.float32), 0.7)[..., None] * strength


def _paint_rock(img, m, r, ctx, rng, col, tex=0.08):
    """단색에 가까운 바위 면 + 음영. 쓰레기(여러 색 점의 군집)와 달리 한 덩어리로 매끈함."""
    shade = _rock_shading(m, r, ctx)
    rock = col * (1 + rng.uniform(0.2, 0.35) * shade[..., None]) * (1 + tex * (_noise((H, W), 0.8, rng) - 0.5))[..., None]
    a = np.clip(ndi.gaussian_filter(m.astype(np.float32), 0.5) * 1.2, 0, 1)[..., None]
    img[:] = img * (1 - a) + np.clip(rock, 0, 1) * a


def add_boulders(img, ctx, rng):
    """모래·자갈 위에 따로 떨어져 있는 바위 1~6개 (0.5~2m, 그림자 포함). 외딴 밝은 덩어리 = 쓰레기 아님."""
    for _ in range(int(rng.choice([1, 1, 2, 3, 4, 6]))):
        c = _land_point(ctx, rng, True)
        r = rng.uniform(1.8, 7)
        m = _ellipse(c, r, r * rng.uniform(0.55, 1.0), rng, rough=rng.uniform(0.15, 0.45))
        if m.sum() < 4 or (m & (ctx.water | ctx.veg)).mean() > 0.2:
            continue
        _cast_shadow(img, m, ctx, int(max(1, round(r * rng.uniform(0.3, 0.8)))), rng.uniform(0.35, 0.6))
        col = _BOULDER[rng.integers(len(_BOULDER))] * rng.uniform(0.85, 1.15) * rng.uniform(0.96, 1.04, 3).astype(np.float32)
        _paint_rock(img, m, r, ctx, rng, col)
        ctx.ok[ndi.binary_dilation(m, iterations=2)] = False      # 붙여넣는 쓰레기가 바위 위에 겹치지 않게


def add_bright_rocks(img, ctx, rng):
    """바위 무리 사이의 유난히 밝은 바위 면 (햇빛 받은 화강암 등): 주변 질감은 살리고 밝고 무채색으로.
    매끈한 모래 위의 흰 덩어리는 스티로폼 쓰레기와 비슷하므로, 표면이 거친 곳(갯바위·자갈)에만 넣습니다."""
    lum0 = img @ _LUM3
    rough = np.sqrt(np.maximum(ndi.uniform_filter(lum0 ** 2, 5) - ndi.uniform_filter(lum0, 5) ** 2, 0))
    cand = np.argwhere((rough > 0.05) & (lum0 < 0.6) & ~ctx.water & ~ctx.veg)   # 이미 밝은 모래·포말은 제외
    if len(cand) < 50:
        return
    c = cand[rng.integers(len(cand))].astype(np.float32)
    for i in range(int(rng.integers(1, 5))):
        cc = c + (rng.normal(0, 12, 2) if i else 0)
        r = rng.uniform(3, 12)
        m = _ellipse(cc, r, r * rng.uniform(0.4, 1.0), rng, rough=rng.uniform(0.3, 0.7))
        m &= ~ctx.water & ~ctx.veg
        if m.sum() < 8 or rough[m].mean() < 0.03:
            continue
        _cast_shadow(img, m, ctx, int(rng.integers(1, 3)), rng.uniform(0.3, 0.55))
        lum = img @ _LUM3
        k = min(rng.uniform(1.25, 1.7), 0.85 / max(float(lum[m].mean()), 0.05))  # 주변보다 밝게 (흰색까지는 안 감)
        pale = np.clip((0.55 * img + 0.45 * lum[..., None]) * k, 0, 1)        # 원래 질감 유지 + 약간 무채색
        shade = _rock_shading(m, r, ctx)
        pale = np.clip(pale * (1 + 0.15 * shade[..., None]), 0, 1)
        a = np.clip(ndi.gaussian_filter(m.astype(np.float32), 0.6) * 1.2, 0, 1)[..., None]
        img[:] = img * (1 - a) + pale * a
        ctx.ok[ndi.binary_dilation(m, iterations=2)] = False


def add_splash(img, ctx, rng):
    """물가·얕은 물의 바위와 그 둘레에 부서지는 흰 물보라 (끊긴 고리 + 흘러가는 거품)."""
    if not ctx.water.any():
        return
    cand = np.argwhere(ctx.water & (ctx.water_dist <= rng.uniform(4, 30)))
    if len(cand) == 0:
        return
    white = np.array([0.92, 0.94, 0.96], np.float32) * rng.uniform(0.85, 1.05)
    for _ in range(int(rng.integers(1, 5))):
        c = cand[rng.integers(len(cand))].astype(np.float32)
        r = rng.uniform(2, 9)
        m = _ellipse(c, r, r * rng.uniform(0.5, 1.0), rng, rough=rng.uniform(0.2, 0.5))
        if m.sum() < 4:
            continue
        col = _BOULDER[rng.integers(len(_BOULDER))] * rng.uniform(0.5, 0.9)      # 젖은 바위는 어두움
        _paint_rock(img, m, r, ctx, rng, col)
        wd = rng.uniform(1, 4)
        dist = ndi.distance_transform_edt(~m)
        ring = (dist > 0) & (dist <= wd + 2 * _noise((H, W), 2.0, rng))
        trail = (dist <= rng.uniform(4, 14)) & (_noise((H, W), rng.uniform(1.0, 2.5), rng) > rng.uniform(0.6, 0.75))
        foam = (ring | trail) & (ctx.water | (dist <= 2))
        foam &= _noise((H, W), 1.5, rng) > rng.uniform(0.1, 0.4)                # 군데군데 끊김
        alpha = np.clip(ndi.gaussian_filter(foam.astype(np.float32), 0.7), 0, 1) * rng.uniform(0.55, 0.95)
        img[:] = img * (1 - alpha[..., None]) + white * alpha[..., None]


ROCK_HARDNEG = [add_boulders, add_bright_rocks, add_splash]


def add_vessel(img, ctx, rng):
    """선박(물 위) 또는 차량(육지): 큰 단색 인공물 (정답 아님)."""
    on_water = ctx.water.any() and rng.random() < 0.7
    if on_water:
        cand = np.argwhere(ctx.water_dist > 4)
        L = rng.uniform(12, 60)
        wr = rng.uniform(0.25, 0.4)
    else:
        cand = np.argwhere(~ctx.water)
        L = rng.uniform(10, 16)
        wr = rng.uniform(0.4, 0.5)
    if len(cand) == 0:
        return
    c = cand[rng.integers(len(cand))].astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = rng.uniform(0, np.pi)
    u = (xx - c[1]) * np.cos(th) + (yy - c[0]) * np.sin(th)
    v = -(xx - c[1]) * np.sin(th) + (yy - c[0]) * np.cos(th)
    if on_water:
        body = (np.abs(v) <= L * wr / 2 * np.clip(1.4 - np.abs(u) / (L / 2), 0, 1)) & (np.abs(u) <= L / 2)
    else:
        body = (np.abs(u) <= L / 2) & (np.abs(v) <= L * wr / 2)
    inner = body & (np.abs(u) <= L * 0.25) & (np.abs(v) <= L * wr * 0.3)
    shadow = np.roll(body, (ctx.sun[0], ctx.sun[1]), axis=(0, 1)) & ~body
    img[shadow] *= 0.6
    col = np.array(rng.choice([[0.95, 0.95, 0.95], [0.85, 0.86, 0.88], [0.2, 0.35, 0.7], [0.75, 0.2, 0.15]]), np.float32)
    img[body] = col * rng.uniform(0.85, 1.05)
    img[inner] = img[inner] * rng.uniform(0.5, 0.9)


def _segments_mask(shape, p0, p1, radius):
    """선분들(p0→p1, (N,2) 행·열)을 radius 두께로 그린 마스크."""
    m = np.zeros(shape, bool)
    if len(p0) == 0:
        return m
    n = int(np.ceil(np.abs(p1 - p0).max())) * 2 + 2
    t = np.linspace(0, 1, n)[None, :, None]
    pts = np.round(p0[:, None] + (p1 - p0)[:, None] * t).reshape(-1, 2).astype(int)
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < shape[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < shape[1])
    m[pts[ok, 0], pts[ok, 1]] = True
    return ndi.binary_dilation(m, _disk(int(radius))) if radius >= 1 else m


def add_tetrapods(img, ctx, rng):
    """테트라포드·방파제 블록: 2~5m 회백색 콘크리트 블록이 촘촘히 얽힌 띠 + 사이의 짙은 그림자 (정답 아님).

    쓰레기처럼 '밝은 덩어리가 모인 띠'지만 단색이고 블록이 크고 규칙적입니다. 블록 위에는 쓰레기를 놓지 않습니다.
    """
    if ctx.has_shore and rng.random() < 0.65:   # 호안: 물가를 따라 물·뭍에 걸친 띠
        w = rng.uniform(6, 28)
        reg = (ctx.water & (ctx.water_dist <= w * 0.6)) | (~ctx.water & (ctx.land_dist <= w * 0.4))
        reg = _breaks(reg, rng, 0.3)
    else:                                        # 방파제: 곧게 뻗은 두꺼운 띠
        c = _land_point(ctx, rng, True) if ctx.has_shore else np.array([rng.uniform(40, 216), rng.uniform(40, 216)])
        th = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.abs(-(xx - c[1]) * np.sin(th) + (yy - c[0]) * np.cos(th))
        along = (xx - c[1]) * np.cos(th) + (yy - c[0]) * np.sin(th)
        reg = (d <= rng.uniform(5, 16)) & (along >= rng.uniform(-300, 0)) & (along <= rng.uniform(40, 300))
    if reg.sum() < 60:
        return
    ys, xs = np.nonzero(reg)
    y0, y1 = max(ys.min() - 3, 0), min(ys.max() + 4, H)
    x0, x1 = max(xs.min() - 3, 0), min(xs.max() + 4, W)
    r_ = reg[y0:y1, x0:x1]
    h, w_ = r_.shape
    hh, ww = h * S, w_ * S
    canvas = np.repeat(np.repeat(img[y0:y1, x0:x1], S, 0), S, 1)
    reg_up = np.repeat(np.repeat(r_, S, 0), S, 1)
    shade = np.array([0.22, 0.22, 0.21], np.float32) * rng.uniform(0.7, 1.3)   # 블록 사이 그늘
    canvas[reg_up] = canvas[reg_up] * 0.4 + shade * 0.6
    arm = rng.uniform(2.5, 4.5) * S                      # 다리 길이 (0.75~1.35m, 블록 폭 2~4m)
    rad = int(round(rng.uniform(1.3, 2.2)))              # 다리 두께 (초해상 화소, 약 0.3~0.45m)
    # 블록 중심: 다리 길이에 비례한 간격의 격자 + 흔들림 (서로 얽히도록 촘촘히)
    sp = arm * rng.uniform(1.2, 1.7)
    gy, gx = np.mgrid[0:hh:sp, 0:ww:sp]
    cen = np.stack([gy.ravel(), gx.ravel()], 1) + rng.uniform(-0.35, 0.35, (gy.size, 2)) * sp
    cen = cen[(cen[:, 0] >= 0) & (cen[:, 0] < hh) & (cen[:, 1] >= 0) & (cen[:, 1] < ww)]
    cen = cen[reg_up[cen[:, 0].astype(int), cen[:, 1].astype(int)]]
    if len(cen) == 0:
        return
    base_lum = rng.uniform(0.7, 0.95)
    tint = np.array([1.0, 0.99, 0.95], np.float32) * rng.uniform(0.97, 1.03, 3).astype(np.float32)
    blocks = np.zeros((hh, ww), bool)
    bins = rng.integers(0, 4, len(cen))
    for b in range(4):                                   # 블록마다 밝기 조금씩 다르게 (4단계)
        cb = cen[bins == b]
        if len(cb) == 0:
            continue
        rot = rng.uniform(0, 2 * np.pi, len(cb))[:, None]
        ang = rot + np.array([0, 2.094, 4.189])[None]    # 위에서 본 3갈래 다리 + 가운데 다리(점)
        p1 = np.stack([cb[:, None, 0] + arm * np.sin(ang), cb[:, None, 1] + arm * np.cos(ang)], -1).reshape(-1, 2)
        p0 = np.repeat(cb, 3, 0)
        m = _segments_mask((hh, ww), p0, p1, rad) & ndi.binary_dilation(reg_up, iterations=S)
        canvas[m] = tint * base_lum * (0.85 + 0.05 * b)
        blocks |= m
    sdy, sdx = ctx.sun                                   # 블록 그림자
    sh = np.roll(blocks, (sdy * 2, sdx * 2), axis=(0, 1)) & ~blocks & reg_up
    canvas[sh] *= 0.55
    img[y0:y1, x0:x1] = canvas.reshape(h, S, w_, S, 3).mean((1, 3))
    ctx.ok &= ~ndi.binary_dilation(reg, iterations=2)    # 블록 위에는 쓰레기를 놓지 않음


def add_aquaculture(img, ctx, rng):
    """양식장: 바다 위에 일정 간격으로 줄지은 흰 부표(스티로폼) + 가는 줄, 또는 가두리의 흰 사각·원형 테두리 (정답 아님).

    쓰레기와 같은 흰 점이지만 바다 위에 있고 격자처럼 규칙적입니다.
    """
    if not ctx.water.any() or ctx.water.mean() < 0.15:
        return
    cand = np.argwhere(ctx.water & (ctx.water_dist > 6))
    if len(cand) == 0:
        cand = np.argwhere(ctx.water)
    c = cand[rng.integers(len(cand))].astype(np.float32)
    th = rng.uniform(0, np.pi)
    dots = np.zeros((H, W), np.float32)
    lines = np.zeros((H, W), bool)
    if rng.random() < 0.7:                               # 부표 줄 (연승·뗏목식)
        L1, L2 = rng.uniform(60, 420), rng.uniform(30, 300)
        sr, sd = rng.uniform(6, 26), rng.uniform(2.5, 8)
        u = np.arange(-L1 / 2, L1 / 2, sd)
        for v in np.arange(-L2 / 2, L2 / 2, sr):
            uu = u + rng.normal(0, 0.3, len(u))
            keep = rng.random(len(u)) > rng.uniform(0.0, 0.3)  # 빠진 부표
            yy = c[0] + uu * np.sin(th) + v * np.cos(th)
            xx = c[1] + uu * np.cos(th) - v * np.sin(th)
            ok = keep & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            dots[yy[ok].astype(int), xx[ok].astype(int)] = 1.0
            if rng.random() < 0.6 and ok.any():
                p0 = np.array([[yy[ok][0], xx[ok][0]]])
                p1 = np.array([[yy[ok][-1], xx[ok][-1]]])
                lines |= _segments_mask((H, W), p0, p1, 0)
        size = rng.uniform(0.5, 1.0)                     # 부표 지름 0.5~1m
    else:                                                # 가두리: 흰 테두리 사각·원형
        size = rng.uniform(0.4, 0.7)
        for _ in range(int(rng.integers(2, 9))):
            cc = c + rng.normal(0, 25, 2)
            r = rng.uniform(5, 16)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            if rng.random() < 0.5:
                ring = np.abs(np.hypot(yy - cc[0], xx - cc[1]) - r) < 0.7
            else:
                dd = np.maximum(np.abs(yy - cc[0]), np.abs(xx - cc[1]))
                ring = np.abs(dd - r) < 0.7
            dots[ring] = 1.0
    if size > 0.75:   # 큰 부표는 1화소 키움
        dots = ndi.binary_dilation(dots > 0, _disk(1)).astype(np.float32)
    alpha = np.clip(ndi.gaussian_filter(dots, 0.5) * 2.2 * rng.uniform(0.8, 1.0), 0, 1)
    alpha = np.maximum(alpha, ndi.gaussian_filter(lines.astype(np.float32), 0.5) * rng.uniform(0.1, 0.3))
    alpha *= ndi.gaussian_filter(ctx.water.astype(np.float32), 0.7)
    col = np.array(rng.choice([[0.95, 0.95, 0.94], [0.95, 0.95, 0.94], [0.95, 0.95, 0.94],
                               [0.15, 0.15, 0.16], [0.9, 0.45, 0.15]]), np.float32)  # 대부분 흰 스티로폼
    img[:] = img * (1 - alpha[..., None]) + col * alpha[..., None]


def add_tidalflat(img, ctx, rng):
    """갯벌: 회갈색 펄 + 어둡거나 하늘이 비쳐 밝은 물고임 + 밝은 모래 줄무늬(사주) + 구불구불한 갯골 (정답 아님).

    얼룩덜룩해 쓰레기 구역과 헷갈리지만 무채색이고 무늬가 큽니다.
    """
    if ctx.has_shore:
        reg = (ctx.water & (ctx.water_dist <= rng.uniform(15, 90) * (0.5 + _noise((H, W), 20, rng)))) \
            | (~ctx.water & ~ctx.veg & (ctx.land_dist <= rng.uniform(0, 40)))
    elif rng.random() < 0.4:
        reg = ~ctx.veg
    else:
        reg = _ellipse(np.array([rng.uniform(0, 256), rng.uniform(0, 256)]), rng.uniform(40, 140),
                       rng.uniform(30, 120), rng, 0.6) & ~ctx.veg
    if reg.sum() < 200:
        return
    a = ndi.gaussian_filter(reg.astype(np.float32), 4)[..., None]
    mud = np.array([0.38, 0.35, 0.31], np.float32) * rng.uniform(0.7, 1.25) * rng.uniform(0.95, 1.05, 3).astype(np.float32)
    tex = 0.10 * (_noise((H, W), 12, rng) - 0.5) + rng.uniform(0.06, 0.16) * (_noise((H, W), 0.8, rng) - 0.5)
    flat = mud * (1 + tex[..., None])
    # 사주(밝은 모래 줄무늬): 한 방향으로 길게 늘인 노이즈
    streak = ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), (rng.uniform(1, 3), rng.uniform(8, 25)))
    streak = (streak - streak.min()) / (np.ptp(streak) + 1e-6)
    if rng.random() < 0.5:
        streak = streak.T
    sb = (np.clip((streak - rng.uniform(0.6, 0.75)) * 3, 0, 1) * rng.uniform(0.3, 0.7))[..., None]
    flat = flat * (1 - sb) + np.array([0.62, 0.58, 0.50], np.float32) * rng.uniform(0.85, 1.15) * sb
    # 물고임: 작은 불규칙 웅덩이, 어둡거나(물) 하늘이 비쳐 밝음
    pool = _noise((H, W), rng.uniform(0.8, 2), rng) > rng.uniform(0.68, 0.85)
    pool_col = np.array([0.12, 0.14, 0.17], np.float32) if rng.random() < 0.6 \
        else np.array([0.66, 0.70, 0.74], np.float32) * rng.uniform(0.85, 1.15)
    pa = ndi.gaussian_filter(pool.astype(np.float32), 0.6)[..., None] * rng.uniform(0.5, 0.9)
    flat = flat * (1 - pa) + pool_col * pa
    # 갯골: 구불구불한 어두운 물길
    ch = (np.abs(_noise((H, W), rng.uniform(8, 18), rng) - 0.5) < rng.uniform(0.008, 0.02)) & (rng.random() < 0.35)
    ca = ndi.gaussian_filter(ch.astype(np.float32), 1.0)[..., None] * rng.uniform(0.25, 0.45)
    flat = flat * (1 - ca) + mud * 0.45 * ca
    img[:] = img * (1 - a) + np.clip(flat, 0, 1) * a


def darken_water(img, ctx, rng):
    """배경의 물을 대회 영상처럼 '아주 어두운 남색~검정' 으로 바꿉니다 (항공영상 → 대회 도메인 정렬).

    물 내부 질감(쇄파 흰 선 등)은 평균과의 차이를 일부 보존해 남깁니다.
    """
    if not ctx.water.any():
        return
    a = ndi.gaussian_filter(ctx.water.astype(np.float32), 1.0)[..., None]
    mean = img[ctx.water].mean(0)
    navy = np.array([0.03, 0.06, 0.12], np.float32) * rng.uniform(0.5, 1.6) + rng.uniform(-0.01, 0.01, 3)
    k = rng.uniform(0.3, 0.8)
    new = np.clip(navy + (img - mean) * k, 0, 1)
    bright = np.clip(((img @ np.array([0.299, 0.587, 0.114], np.float32)) - mean.mean() - 0.15) * 3, 0, 1)[..., None]
    new = new * (1 - bright) + img * bright   # 밝은 쇄파·거품은 원래 밝기 유지
    img[:] = img * (1 - a) + new * a


# ----------------------------------------------------------------------------- 절차적 배경 (점검용)
def procedural_background(rng):
    """바다(어두운 남색) / 젖은 모래 / 모래·자갈·갯바위 / 숲 으로 된 단순 해안 배경."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    s = (xx - W / 2) * np.cos(th) + (yy - H / 2) * np.sin(th) + rng.uniform(-150, 150)
    s = s + 40 * (_noise((H, W), 30, rng) - 0.5) + 8 * (_noise((H, W), 6, rng) - 0.5)
    img = np.zeros((H, W, 3), np.float32)
    water = s < 0
    sea = np.array([0.04, 0.07, 0.14], np.float32) * rng.uniform(0.6, 1.5)
    img[:] = sea + 0.02 * (_noise((H, W), 3, rng)[..., None] - 0.5)
    ground = rng.choice(["sand", "rock", "flat"], p=[0.55, 0.25, 0.2])
    if ground == "sand":
        base = np.array([0.78, 0.70, 0.55], np.float32)
        tex = 0.06 * (_noise((H, W), 0.7, rng) - 0.5) + 0.08 * (_noise((H, W), 10, rng) - 0.5)
    elif ground == "rock":
        base = np.array([0.45, 0.42, 0.38], np.float32)
        tex = 0.25 * (_noise((H, W), 1.5, rng) - 0.5) + 0.2 * (_noise((H, W), 5, rng) - 0.5)
    else:
        base = np.array([0.35, 0.33, 0.30], np.float32)
        tex = 0.05 * (_noise((H, W), 1, rng) - 0.5) + 0.12 * (_noise((H, W), 15, rng) - 0.5)
    land = base * rng.uniform(0.93, 1.07, 3).astype(np.float32) * rng.uniform(0.8, 1.15) + tex[..., None]
    wet = (s >= 0) & (s < rng.uniform(3, 25))
    land[wet] *= 0.7
    forest_at = rng.uniform(60, 400)
    forest = s > forest_at
    canopy = _noise((H, W), 3, rng)
    fcol = np.array([0.25, 0.45, 0.2], np.float32) * rng.uniform(0.7, 1.3)
    land[forest] = fcol * (0.5 + canopy[forest][:, None])
    img[~water] = land[~water]
    return np.clip(img, 0, 1), water, forest & ~water


# ----------------------------------------------------------------------------- 메인 인터페이스
class DebrisSynth:
    """bg(uint8 HxWx3 또는 None) -> (img uint8, mask bool, info)

    하드 네거티브는 양성·음성 패치 모두에 넣어 화소 단위 구분을 학습시키고,
    음성 패치에는 hn_boost_neg 배 더 자주 넣어 오탐 유발 요소를 집중적으로 보여줍니다.
    """

    PRE = [("tidalflat", add_tidalflat, 0.2),      # 바탕을 바꾸는 것부터
           ("wrack", add_wrack, 0.35), ("driftwood", add_driftwood, 0.25),
           ("shells", add_shells, 0.45), ("gravel", add_gravel, 0.25),
           ("tetrapod", add_tetrapods, 0.2), ("aquaculture", add_aquaculture, 0.2),
           ("rockshadow", add_rock_shadows, 0.3)]
    POST = [("foam", add_foam, 0.5), ("vessel", add_vessel, 0.2)]  # 쓰레기 위에 그려질 수 있는 것

    def __init__(self, p_pos=0.5, hn_scale=1.0, hn_boost_neg=1.6, p_dark_water=0.6):
        self.p_pos = p_pos
        self.hn_scale = hn_scale
        self.hn_boost_neg = hn_boost_neg
        self.p_dark_water = p_dark_water

    @staticmethod
    def _ctx(bg, rng, aux, water=None, veg=None):
        if aux is not None:
            return make_ctx(bg, rng, aux == AUX_WATER, aux == AUX_VEG, aux == AUX_NEAR)
        return make_ctx(bg, rng, water, veg)

    @classmethod
    def placeable(cls, bg, rng, aux=None):
        """이 배경에 쓰레기를 자연스럽게 놓을 수 있는지.

        aux 가 있으면(NAIP): 물가 21m 이내의 식생 아닌 땅이 충분한가.
        aux 가 없으면: 해안선이 보이는 패치 위주로, 땅만 있는 패치는 30% 확률로만 허용.
        """
        ctx = cls._ctx(bg, rng, aux)
        if aux is not None:
            return ctx.ok.mean() > 0.02
        return ctx.ok.mean() > 0.03 and (ctx.has_shore or rng.random() < 0.3)

    def __call__(self, bg, rng, positive=None, aux=None):
        """aux: (H,W) uint8 보조 라벨 (0 육지, 1 물, 2 식생, 3 물가 땅). fetch_naip.py 가 aux/ 에 저장합니다."""
        water = veg = None
        if bg is None:
            img, water, veg = procedural_background(rng)
            bg_u8 = (img * 255).astype(np.uint8)
        else:
            bg_u8 = bg
            img = bg.astype(np.float32) / 255.0
        ctx = self._ctx(bg_u8, rng, aux, water, veg)
        if bg is not None and rng.random() < self.p_dark_water:
            darken_water(img, ctx, rng)
        if positive is None:
            if aux is not None:
                positive = rng.random() < self.p_pos and ctx.ok.mean() > 0.02
            else:  # 해안선이 없는 패치(시가지·숲일 수 있음)는 양성을 줄입니다
                p = self.p_pos if ctx.has_shore else self.p_pos * 0.3
                positive = rng.random() < p and ctx.ok.mean() > 0.03
        k = self.hn_scale * (1.0 if positive else self.hn_boost_neg)
        added = []
        for name, fn, p in self.PRE:
            if rng.random() < p * k:
                fn(img, ctx, rng)
                added.append(name)
        mask = np.zeros((H, W), bool)
        if positive:
            for _ in range(int(rng.choice([1, 2, 3], p=[0.65, 0.25, 0.1]))):
                reg = make_region(ctx, rng)
                cov = render_speckles(img, reg, rng, ctx, _DEBRIS)
                mask |= debris_label(cov, reg, rng)
            if mask.sum() < 20:  # 너무 옅게 그려진 경우 음성으로 취급
                mask[:] = False
        for name, fn, p in self.POST:
            if rng.random() < p * k:
                fn(img, ctx, rng)
                added.append(name)
        out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        return out, mask, {"positive": bool(mask.any()), "hardneg": added, "has_shore": ctx.has_shore}
