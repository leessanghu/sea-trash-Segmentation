"""유형별 증강 데이터 생성기 — "테스트엔 나오는데 학습엔 없는 유형"을 새 배경 위에 만듭니다.

근거 (STATE 10-05 테스트 분포 추정)
- 실제 양성은 태안 북부 한 곳(DR2001)뿐: 물가 10m 이내 0%, 띠형 1%. 규정은 "물가를 따라 띠"가 주요 형태
- 테스트 바다는 "아주 어두운 남색~검정", 지형은 모래·자갈·갯바위·갯벌·호안/방파제·포구
- 정답 제외(음성): 밝은 자갈·조개, 포말, 해조류·유목 어두운 띠, 갯벌 사주·물고임, 선박·어구 시설물

유형 (--mix 로 비율 지정)
  band_waterline  물가 0~3m 를 따라가는 쓰레기 띠 (가장 큰 구멍)
  band_wrackline  물가 3~12m 의 만조선 띠 (해변 안쪽)
  pile_beach      해변 안쪽·숲 가장자리 덩어리 (포켓비치형)
  pile_rock       거친 바닥(갯바위·자갈) 위 덩어리 (바위틈형)
  scatter         흩어진 작은 무더기 (갯벌·모래)
  neg_hard        쓰레기 없음 + 혼동물 2~4개 (해조류 띠·양식 부표·선박·테트라포드·갯벌·조개·포말)
모든 유형에서 바다는 확률 --dark 로 어두운 남색으로 바꾸고, 양성에도 혼동물을 일부 섞습니다(화소 단위 구분 학습).
쓰레기 질감은 팀 합성기(knps/synth.py render_speckles: 흰·회·청·녹·주황 점, 3배 해상도에서 그려 축소)를 씁니다.

배경 폴더: images/*.png (마스크 없는 깨끗한 음성) 또는 루트 *.png + aux/ (NAIP, 물·식생 보조 라벨).
출력: <out>/images, <out>/masks (255 쓰레기), manifest.csv, (선택) 유형별 미리보기.

python make_typed_aug.py --bg ../sea_trash_data/bg_ports_train ../sea_trash_data/real_landneg_train ../mudflat_nl \
    --out ../sea_trash_data/cp_typed_train --n 3000 --sheet-dir <dir>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knps import synth as S                                    # noqa: E402
from knps.data import _imread, pngs, random_crop                     # noqa: E402
from knps.domain import drone_veg, drone_water, texture        # noqa: E402

# 연무 낀 국내 드론 배경은 색만으로 물을 못 가려(숲 그늘 = 물, 청록 바다 놓침) 양성은 만들지 않고 음성 배경으로만 씀.
# 국내 암석 물가 띠는 이미 cp_band_train(731) 이 실제 조각으로 덮고 있음.
# 갯벌(mudflat_nl)은 물·갯벌 구분이 불안정해 음성 전용.
# 해외 포구(bg_ports_train)는 36장 육안 검수(10-05): 시가지 그늘·갯벌 얼룩을 물로, 얕은 해변 물을 땅으로 잘못 잡는 경우가
# 양성 후보의 33%. 단 "manifest kind=port 이고 물이 타일의 15% 이상" 이면 표본 9/9 정확 → 그 조건에서만 양성 허용.
NO_POS_SRC = {"real_landneg_train", "real_neg_train", "mudflat_nl"}
PORT_SRC = {"bg_ports_train"}
PORT_MIN_WATER = 0.15
_PORT_KIND = {}


def _port_kind(path):
    """bg_ports_train 의 manifest(.csv, _jp.csv) 에서 타일 종류(port/tidal/beach)."""
    if not _PORT_KIND:
        d = Path(path).parent.parent
        for m in d.glob("manifest*.csv"):
            for r in csv.DictReader(open(m, encoding="utf-8")):
                _PORT_KIND[r["id"].removesuffix(".png")] = r.get("kind", "")
    return _PORT_KIND.get(Path(path).stem, "")
SHORE_FX = {"wrack", "tetrapod", "aquaculture", "tidalflat", "foam", "vessel"}
TYPES = ["band_waterline", "band_wrackline", "pile_beach", "pile_rock", "scatter", "band_structure", "pile_styrofoam",
         "pile_large", "port_quay", "neg_hard", "neg_port"]
# 감사 결과(STATE 10-05): 같은 장면을 음성으로도 같은 비율 이상 넣어야 "장면 = 양성" 지름길을 막음 → 음성 40%
DEFAULT_MIX = ("band_waterline:0.22,band_wrackline:0.07,pile_beach:0.07,pile_rock:0.05,scatter:0.03,"
               "band_structure:0.07,pile_styrofoam:0.04,pile_large:0.05,port_quay:0.04,neg_hard:0.24,neg_port:0.12")
NEG_FX = [("wrack", S.add_wrack, 0.55), ("driftwood", S.add_driftwood, 0.3), ("shells", S.add_shells, 0.4),
          ("gravel", S.add_gravel, 0.3), ("tetrapod", S.add_tetrapods, 0.25), ("aquaculture", S.add_aquaculture, 0.3),
          ("tidalflat", S.add_tidalflat, 0.2), ("rockshadow", S.add_rock_shadows, 0.25),
          ("bright_rocks", S.add_bright_rocks, 0.3), ("foam", S.add_foam, 0.5), ("vessel", S.add_vessel, 0.3)]


def list_bgs(dirs):
    out = []
    for d in map(Path, dirs):
        bad = set()
        if (d / "qa_flags.csv").exists():                           # NAIP 확장분: 물 마스크(aux) 의심 타일은 뺌 (밝은 모래·지붕이 물로 번짐 등)
            bad = {r["file"] for r in csv.DictReader(open(d / "qa_flags.csv", encoding="utf-8")) if r.get("aux_suspect") == "1"}
        for p in pngs(d):                                          # NAIP 형식
            if p.name in bad:
                continue
            a = d / "aux" / p.name
            out.append((p, a if a.exists() else None, d.name))
        if (d / "images").is_dir():
            for p in pngs(d / "images"):
                m = d / "masks" / p.name
                if not m.exists() or _imread(m, gray=True).max() < 50:   # 채점 제외·쓰레기 화소가 있는 타일은 뺌
                    out.append((p, None, d.name))
    return out


def scene(bg_u8, aux, rng):
    """배경 → ctx (물·식생·놓을 수 있는 땅·거리맵). aux 없으면 색·질감으로 추정 (청록 드론/항공 바다 + 어두운 바다 둘 다)."""
    f = bg_u8.astype(np.float32) / 255.0
    if aux is not None:
        return _finish_ctx(bg_u8, f, rng, aux == S.AUX_WATER, aux == S.AUX_VEG)
    # 청록 바다(드론·항공)는 매끈함까지 요구, 어두운 바다는 파도로 거칠 수 있어 색만 봄 (estimate_water 가 숲 녹색은 제외)
    # 연무 낀 드론 숲은 어둡고 푸르스름해 '어두운 바다'로 잡히므로, 어두운 물도 매끈해야(숲·갯바위는 0.03 이상) 물로 봄
    tex = texture(f)
    water = (drone_water(f) & (tex < 0.025)) | (S.estimate_water(bg_u8) & (tex < 0.03) & ~drone_veg(f))
    # 테두리를 복제해 붙인 뒤 다듬어야 경계의 물이 깎이지 않음 (그냥 닫힘 연산을 하면 테두리 화소가 지워져 아래 '테두리에 닿은 물' 조건이 늘 실패)
    water = np.pad(water, 4, mode="edge")
    water = ndi.binary_closing(ndi.binary_opening(water, iterations=1), iterations=2)[4:-4, 4:-4]
    # 바다는 패치 밖으로 이어지므로 테두리에 닿은 큰 덩어리만 물로 봄 (숲 그늘·어두운 지붕·물고임 오검출 제거)
    lab, n = ndi.label(water)
    if n:
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool)
        keep[[b for b in border if b and sizes[b] >= 2000]] = True
        water = keep[lab]
    if water.any():   # 물 속 작은 구멍(햇빛 반사·항적·포말 덩어리·작은 암초)은 물로 메움 — 그 위에 쓰레기를 그리지 않게 (포구 검수 10-05)
        holes = ndi.binary_fill_holes(water) & ~water
        hl, hn = ndi.label(holes)
        if hn:
            hs = np.bincount(hl.ravel()); small = hs < 1500; small[0] = False
            water |= small[hl]
    veg = drone_veg(f) & ~water
    return _finish_ctx(bg_u8, f, rng, water, veg)


def _finish_ctx(bg_u8, f, rng, water, veg):
    """포말은 물 쪽으로 붙이고(띠는 포말선 위에서 시작), 매끈하고 밝은 인공물(선박 갑판·지붕)은 놓을 곳에서 뺌.
    포말·선박은 정답 제외 대상이라 그 위에 쓰레기를 그리면 '흰 것 = 쓰레기'를 가르치게 됨."""
    from knps.domain import hsv
    _, sat, val = hsv(f)
    if water.any():
        near = ndi.binary_dilation(water, iterations=15)
        foam = ndi.binary_dilation((sat < 0.16) & (val > 0.70) & near, iterations=2)
        water = water | (foam & near)
    deck = ndi.binary_dilation((texture(f) < 0.012) & (val > 0.60), iterations=3)   # 갑판·지붕·매끈한 콘크리트 상판
    white = ndi.binary_dilation((val > 0.72) & (sat < 0.20) & (texture(f) > 0.015), iterations=3)   # 흰 보트·밝은 장비 (원래 배경의 흰 물체)
    ctx = S.make_ctx(bg_u8, rng, water, veg)
    ctx.ok &= ~deck & ~white
    # 선박·잔교: 물에 둘러싸인 작은 땅 덩어리와 폭 2~3화소짜리 가는 구조물은 뺌 (테두리에 닿은 큰 땅만 남김)
    land = ndi.binary_opening(~water, iterations=5)      # 폭 3m 미만 부잔교·잔교 제거
    lab, n = ndi.label(land)
    if n:
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool)
        keep[[b for b in border if sizes[b] >= 2000]] = True
        ctx.ok &= ndi.binary_dilation(keep[lab], iterations=3)
    return ctx


def _band(ctx, rng, d_lo, d_hi, w_lo, w_hi):
    """물가에서 d0(px) 떨어진 곳부터 폭 wid 의 띠. 해안선 방향을 그대로 따라감."""
    d0 = rng.uniform(d_lo, d_hi) + 4 * (S._noise(ctx.ok.shape, 20, rng) - 0.5)
    wid = rng.uniform(w_lo, w_hi) * (0.5 + S._noise(ctx.ok.shape, 12, rng))
    m = (ctx.land_dist >= np.maximum(d0, 0.5)) & (ctx.land_dist <= d0 + wid) & ctx.ok
    return S._breaks(m, rng, 0.6)


def _blob_at(ctx, rng, cand, r_lo, r_hi):
    if len(cand) == 0:
        return np.zeros(ctx.ok.shape, bool)
    c = cand[rng.integers(len(cand))].astype(np.float32)
    m = np.zeros(ctx.ok.shape, bool)
    r0 = rng.uniform(r_lo, r_hi)
    for i in range(int(rng.integers(1, 4))):
        rx = r0 * rng.uniform(0.5, 1.2)
        cc = c + (rng.uniform(-1.3, 1.3, 2) * r0 if i else 0)
        m |= S._ellipse(cc, rx, rx * rng.uniform(0.3, 1.0), rng, rough=rng.uniform(0.2, 0.8))
    return S._breaks(m, rng, 0.4) & ctx.ok


_PIECES = None
_TINTS = np.array([(225, 105, 40), (55, 145, 90), (45, 105, 195), (35, 35, 38), (245, 245, 242)], np.float32) / 255.0


def load_pieces(d):
    global _PIECES
    from knps.paste import load_pieces as _lp
    _PIECES = _lp(d)
    return len(_PIECES)


def stamp_pieces(img, region, ctx, rng, white=False, scale=(0.6, 1.2), fill=(0.35, 0.7)):
    """region 안에 실제 쓰레기 조각을 촘촘히 찍음 (조각 = DR2001 실제 쓰레기 잘라낸 것).
    밝기는 비율 제한(1.4배) 대신 감마로 맞춰 밝은 모래에도 붙게 함 (감사: 모래 바닥 조각 6/5142 문제).
    25% 는 색을 바꾸고, 나머지는 드론 연무의 푸른 색조를 덜어냄. 1화소 그림자. 반환: 라벨 마스크."""
    from knps.augment import _resize
    H, W = region.shape
    lab = np.zeros((H, W), bool)
    area = int(region.sum())
    if area == 0 or not _PIECES:
        return lab
    target = rng.uniform(*fill) * area
    sy, sx = ctx.sun
    for _ in range(400):
        if lab.sum() >= target:
            break
        rgb, a, ring = _PIECES[int(rng.integers(len(_PIECES)))]
        k = int(rng.integers(4)); rgb, a = np.rot90(rgb, k), np.rot90(a, k)
        if rng.random() < 0.5:
            rgb, a = rgb[:, ::-1], a[:, ::-1]
        sc = rng.uniform(*scale)
        ph, pw = max(int(round(rgb.shape[0] * sc)), 2), max(int(round(rgb.shape[1] * sc)), 2)
        if ph >= H // 2 or pw >= W // 2:
            continue
        rgb = np.clip(_resize(np.ascontiguousarray(rgb), ph, pw), 0, 1)
        a = _resize(np.ascontiguousarray(a.astype(np.float32)), ph, pw) > 0.5
        if a.sum() < 3:
            continue
        cand = np.argwhere(region & ~lab)
        if len(cand) == 0:
            break
        cy, cx = cand[int(rng.integers(len(cand)))]
        y0, x0 = int(cy - ph // 2), int(cx - pw // 2)
        if y0 < 1 or x0 < 1 or y0 + ph >= H - 1 or x0 + pw >= W - 1:
            continue
        if region[y0:y0 + ph, x0:x0 + pw][a].mean() < 0.6:
            continue
        ringm = ndi.binary_dilation(a, iterations=3) & ~a
        new_ring = img[y0:y0 + ph, x0:x0 + pw][ringm].mean(0)
        g = rng.uniform(0.7, 1.0)
        piece = new_ring[None, None] * np.power(np.clip(rgb / np.maximum(ring, 1e-3), 0, 4), g)
        if rng.random() < 0.3:                                 # 젖은 쓰레기
            piece *= rng.uniform(0.7, 0.9)
        if white or rng.random() < 0.25:                       # 색 바꾸기 (스티로폼은 흰색)
            t = _TINTS[4] if white else _TINTS[int(rng.integers(len(_TINTS)))]
            lum = (rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
            tinted = t[None, None] * (0.6 + 0.8 * lum / max(float(lum.mean()), 1e-3)) * 0.85
            mix = 1.0 if white else rng.uniform(0.35, 0.65)    # 위성 0.3m 에선 색이 섞여 보이므로 원색 그대로 칠하지 않음
            piece = np.clip(mix * tinted + (1 - mix) * piece, 0, 1)
        else:                                                  # 푸른 연무 색조 덜어냄
            piece = piece - 0.5 * (ring - ring.mean())[None, None]
        piece = np.clip(piece, 0, 1)
        win = img[y0:y0 + ph, x0:x0 + pw]
        sh = np.roll(a, (int(np.sign(sy)), int(np.sign(sx))), axis=(0, 1))   # 해 반대쪽 1화소 그림자
        win[sh & ~a] *= 0.85
        soft = np.where(a, 0.92, ndi.gaussian_filter(a.astype(np.float32), 0.5))[..., None]
        img[y0:y0 + ph, x0:x0 + pw] = win * (1 - soft) + piece * soft
        lab[y0:y0 + ph, x0:x0 + pw] |= a
    if lab.any() and rng.random() < 0.6:                       # 조각 사이 좁은 틈은 메움 (2화소 이하)
        lab = ndi.binary_closing(lab, S._disk(int(rng.integers(1, 3)))) | lab
    return lab & ndi.binary_dilation(region, iterations=2)


def region_for(kind, ctx, img, rng):
    if kind == "band_waterline":
        return _band(ctx, rng, 0.5, 6, 3, 12)              # 물가 0.15~1.8m 부터 폭 1~4m
    if kind == "band_wrackline":
        return _band(ctx, rng, 10, 40, 4, 16)             # 물가 3~12m (만조선)
    if kind == "pile_beach":
        smooth = texture(img) < 0.03
        cand = np.argwhere(ctx.ok & smooth & (ctx.land_dist > 8) & (ctx.land_dist < 90))   # 물가 2.4~27m
        if len(cand) == 0:
            cand = np.argwhere(ctx.ok)
        return _blob_at(ctx, rng, cand, 5, 30)
    if kind == "pile_rock":
        rough = texture(img) > 0.03
        cand = np.argwhere(ctx.ok & rough & (ctx.land_dist < 90))
        if len(cand) == 0:
            cand = np.argwhere(ctx.ok)
        return _blob_at(ctx, rng, cand, 4, 22)
    if kind == "scatter":
        return S.region_scatter(ctx, rng) & ctx.ok
    if kind == "band_structure":                               # 호안·방파제·사석 발치: 매끈한 직선 구조물 옆 0~2m
        f = img
        straight = ndi.binary_opening((texture(f) < 0.015) & ~ctx.water & ~ctx.veg, iterations=2)
        if straight.mean() < 0.01:
            return _band(ctx, rng, 0.5, 6, 3, 10)
        d = ndi.distance_transform_edt(~straight)
        return S._breaks((d >= 1) & (d <= rng.uniform(3, 8)) & ctx.ok & (ctx.land_dist < 40), rng, 0.5)
    if kind == "pile_styrofoam":                               # 양식장 인접 해안의 흰 스티로폼 더미 (물가 2~15m)
        cand = np.argwhere(ctx.ok & (ctx.land_dist > 3) & (ctx.land_dist < 50))
        return _blob_at(ctx, rng, cand, 4, 18)
    if kind == "port_quay":                                    # 포구: 물양장·슬립웨이 하단·안벽 모서리 (물에서 0~3m)
        cand = np.argwhere(ctx.ok & (ctx.land_dist >= 1) & (ctx.land_dist <= 10))
        if rng.random() < 0.4:
            return _band(ctx, rng, 0.5, 4, 3, 10)              # 사석 발치를 따라 5~20m 띠
        return _blob_at(ctx, rng, cand, 4, 16)
    if kind == "pile_large":                                   # 길이 10~40m 대형 더미
        cand = np.argwhere(ctx.ok & (ctx.land_dist < 90))
        return _blob_at(ctx, rng, cand, 18, 50)
    return np.zeros(ctx.ok.shape, bool)


def add_gear(img, ctx, rng):
    """포구 물양장 위 어구·상자 적치 (정답 제외 = 인공 구조물): 건물과 같은 방향으로 정렬된 2~6m 직사각형, 채도 높은 단색."""
    cand = np.argwhere(ctx.ok & (ctx.land_dist > 15))
    if len(cand) == 0:
        return
    H, W = ctx.ok.shape
    c = cand[rng.integers(len(cand))]
    th = rng.uniform(0, np.pi)
    col = np.array(rng.choice([(40, 90, 200), (30, 140, 70), (220, 120, 40), (200, 40, 40), (230, 230, 230), (230, 200, 60)]),
                   np.float32) / 255.0
    nr, nc = int(rng.integers(1, 4)), int(rng.integers(2, 6))
    w, h = rng.uniform(6, 18), rng.uniform(5, 12)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(nr):
        for j in range(nc):
            off = np.array([i * (h + 2), j * (w + 2)], np.float32)
            cy = c[0] + off[0] * np.cos(th) - off[1] * np.sin(th)
            cx = c[1] + off[0] * np.sin(th) + off[1] * np.cos(th)
            u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
            v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
            m = (np.abs(u) < w / 2) & (np.abs(v) < h / 2) & ctx.ok
            shade = col * rng.uniform(0.75, 1.05)
            img[m] = shade + 0.03 * (rng.random((int(m.sum()), 3)).astype(np.float32) - 0.5)


def make_one(bgs, kind, rng, p_dark, tries=12):
    """성공하면 (img uint8, mask bool, info) 아니면 None."""
    need_shore = kind not in ("neg_hard", "neg_port")   # 해안쓰레기는 물가에 있어야 함 (테스트 패치는 "해안선을 담은" 패치)
    if kind == "neg_port":
        pool = [b for b in bgs if b[2] in PORT_SRC] or bgs
    elif kind.startswith("neg"):
        pool = bgs
    else:
        pool = [b for b in bgs if b[2] not in NO_POS_SRC] or bgs
    for _ in range(tries):
        p, ap, src = pool[int(rng.integers(len(pool)))]
        im = _imread(p)
        ax = _imread(ap, gray=True) if ap is not None else np.zeros(im.shape[:2], np.uint8)
        im, ax = random_crop(im, ax, rng, p_keep_pos=0.0)
        aux = ax if ap is not None else None
        ctx = scene(im, aux, rng)
        if need_shore and not ctx.has_shore:
            continue
        if not kind.startswith("neg") and src in PORT_SRC and (_port_kind(p) != "port" or ctx.water.mean() < PORT_MIN_WATER):
            continue
        if not kind.startswith("neg") and ctx.ok.mean() < 0.03:
            continue
        img = im.astype(np.float32) / 255.0
        if ctx.water.any() and rng.random() < p_dark:
            S.darken_water(img, ctx, rng)
        added = []
        k = 1.0 if kind in ("neg_hard", "neg_port") else 0.35  # 양성에도 혼동물을 조금
        fx = [x for x in NEG_FX if ctx.has_shore or x[0] not in SHORE_FX]   # 물이 없는 패치에 테트라포드·부표·선박 금지
        if kind == "neg_port":
            fx = [x for x in fx if x[0] in ("vessel", "aquaculture", "foam", "tetrapod", "shells")]
        n_fx = min(int(rng.integers(2, 5)), len(fx)) if kind in ("neg_hard", "neg_port") else 0
        for name, fn, pr in (fx if kind not in ("neg_hard", "neg_port") else
                              [fx[i] for i in rng.choice(len(fx), n_fx, replace=False)]):
            if name in ("foam", "vessel"):
                continue                                       # 쓰레기 위에 그려질 수 있는 것은 나중에
            if kind in ("neg_hard", "neg_port") or rng.random() < pr * k:
                try:
                    fn(img, ctx, rng); added.append(name)
                except Exception:                              # noqa: BLE001  (배경에 따라 실패하는 효과는 건너뜀)
                    pass
        if kind in ("port_quay", "neg_port"):
            for _g in range(int(rng.integers(1, 3)) if kind == "neg_port" else int(rng.random() < 0.6)):
                add_gear(img, ctx, rng); added.append("gear")
        mask = np.zeros(ctx.ok.shape, bool)
        if kind not in ("neg_hard", "neg_port"):
            for _r in range(int(rng.choice([1, 1, 2]))):
                reg = region_for(kind, ctx, img, rng)
                if reg.sum() < 30:
                    continue
                if kind == "pile_styrofoam":
                    mask |= stamp_pieces(img, reg, ctx, rng, white=True, scale=(0.5, 0.9), fill=(0.4, 0.8))
                    if rng.random() < 0.5:                     # 같은 타일 바다 위 줄 부표는 음성
                        try:
                            S.add_aquaculture(img, ctx, rng); added.append("aquaculture")
                        except Exception:                      # noqa: BLE001
                            pass
                elif kind == "pile_large":
                    mask |= stamp_pieces(img, reg, ctx, rng, fill=(0.45, 0.8))
                    cov = S.render_speckles(img, reg & ~mask, rng, ctx, S._DEBRIS, density=(0.15, 0.4))
                    mask |= S.debris_label(cov, reg, rng)
                elif _PIECES and rng.random() < 0.5:
                    mask |= stamp_pieces(img, reg, ctx, rng)
                    if kind.startswith("band") and rng.random() < 0.5:   # 같은 물가에 해조류 어두운 띠 (음성) 를 함께
                        try:
                            S.add_wrack(img, ctx, rng); added.append("wrack")
                        except Exception:                      # noqa: BLE001
                            pass
                else:
                    cov = S.render_speckles(img, reg, rng, ctx, S._DEBRIS)
                    mask |= S.debris_label(cov, reg, rng)
            if mask.sum() < 50:                                # 채점 양성 기준(50px) 미만이면 다시
                continue
            if _on_manmade(im, mask) > 0.10:                   # 선박·차량·아스팔트 위에 그려졌으면 버림 (그것들은 정답 제외 대상)
                continue
        for name, fn, pr in [x for x in NEG_FX if x[0] in ("foam", "vessel") and ctx.has_shore]:
            if rng.random() < pr * (1.0 if kind in ("neg_hard", "neg_port") else 0.5):
                try:
                    fn(img, ctx, rng); added.append(name)
                except Exception:                              # noqa: BLE001
                    pass
        out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        dist = ctx.land_dist[mask] if mask.any() else np.array([-1.0])
        return out, mask, {"src": src, "bg": p.name, "has_shore": ctx.has_shore, "water_pct": round(100 * float(ctx.water.mean()), 1),
                           "debris_px": int(mask.sum()), "dist_water_m": round(0.3 * float(np.nanmedian(dist)), 1),
                           "hardneg": "+".join(added)}
    return None


def _on_manmade(bg_u8, mask):
    """쓰레기 화소 중 원래 배경이 인공물(채도 높은 선체·차량, 어두운 아스팔트, 매끈하고 밝은 갑판·지붕)인 비율."""
    from knps.domain import hsv
    f = bg_u8.astype(np.float32) / 255.0
    _, sat, val = hsv(f)
    tex = texture(f)
    man = (sat > 0.38) | ((val < 0.24) & (tex < 0.02)) | ((val > 0.82) & (tex < 0.012))
    man = ndi.binary_dilation(ndi.binary_opening(man, iterations=1), iterations=2)
    return float(man[mask].mean()) if mask.any() else 0.0


def parse_mix(s):
    d = {}
    for part in s.split(","):
        k, v = part.split(":")
        assert k in TYPES, f"알 수 없는 유형 {k}. 가능: {TYPES}"
        d[k] = float(v)
    t = sum(d.values())
    return {k: v / t for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--dark", type=float, default=0.8, help="바다를 어두운 남색으로 바꿀 확률")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheet-dir", default="")
    ap.add_argument("--pieces", default="", help="실제 쓰레기 조각 폴더 (debris_pieces). 없으면 점묘만")
    a = ap.parse_args()
    if a.pieces:
        print("실제 조각", load_pieces(a.pieces), "개")
    bgs = list_bgs(a.bg)
    assert bgs, f"배경이 없습니다: {a.bg}"
    mix = parse_mix(a.mix)
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "masks").mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    from collections import Counter
    print(f"배경 {len(bgs)}장 ({dict(Counter(s for _, _, s in bgs))}), 목표 {a.n}장, 유형 비율 {mix}", flush=True)
    rows, sheets = [], {k: [] for k in mix}
    kinds = rng.choice(list(mix), a.n, p=list(mix.values()))
    fail = Counter()
    for i, kind in enumerate(kinds):
        r = make_one(bgs, kind, rng, a.dark)
        if r is None:
            fail[kind] += 1
            continue
        img, mask, info = r
        name = f"ty{a.seed}_{kind}_{i:05d}"
        Image.fromarray(img).save(out / "images" / f"{name}.png")
        if mask.any():                                         # 음성은 마스크 파일 없음 (로더 규칙: 없음 = 깨끗한 음성)
            Image.fromarray((mask * 255).astype(np.uint8)).save(out / "masks" / f"{name}.png")
        rows.append({"id": name, "type": kind, **info})
        if a.sheet_dir and len(sheets[kind]) < 24:
            vis = img.copy()
            edge = mask & ~ndi.binary_erosion(mask)
            vis[edge] = (255, 0, 0)
            sheets[kind].append(vis)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{a.n} 저장 {len(rows)} 실패 {dict(fail)}", flush=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    c = Counter(r["type"] for r in rows)
    pos = [r for r in rows if not r["type"].startswith("neg")]
    print(f"완료 {len(rows)}장 → {out} | 유형별 {dict(c)} | 실패 {dict(fail)}")
    if pos:
        print(f"  양성 쓰레기 화소 중앙값 {int(np.median([r['debris_px'] for r in pos]))}, "
              f"물가 거리 중앙값 {np.nanmedian([r['dist_water_m'] for r in pos]):.1f}m")
    if a.sheet_dir:
        sd = Path(a.sheet_dir); sd.mkdir(parents=True, exist_ok=True)
        for kind, th in sheets.items():
            if not th:
                continue
            cols = 6; rws = (len(th) + cols - 1) // cols
            sheet = np.zeros((rws * 258, cols * 258, 3), np.uint8)
            for k, t in enumerate(th):
                rr, cc = divmod(k, cols)
                sheet[rr * 258:rr * 258 + 256, cc * 258:cc * 258 + 256] = t
            Image.fromarray(sheet).save(sd / f"typed_{kind}.png")
        print("미리보기", sd)


if __name__ == "__main__":
    main()
