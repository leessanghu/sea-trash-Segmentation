"""실제 쓰레기 조각을 학습 중에 매번 새로 붙이는 데이터 소스 (online copy-paste).

미리 만든 붙여넣기 양성(cp_pos_train)은 같은 조각이 같은 자리에 반복되지만, 이 소스는 샘플마다
배경·위치·크기·회전·밝기가 달라 실제 생김새의 양성을 사실상 무한히 만듭니다.

- 조각: <pieces>/<id>.png (RGBA, 알파=마스크) + manifest.csv (ring_r/g/b = 원래 장면에서 조각 주변 평균색)
  학습 장소 조각만 씁니다 (tools/import_dataset_v2.py 가 split=train 만 내보냄)
- 배경: NAIP 해안(aux/ 물·식생·물가 마스크로 정확히 배치, 우리 하드 네거티브도 함께 합성)
         실제 드론 음성(색·질감으로 물·식생·포말을 추정해 배치)
- 조각 처리: D4, 축척 0.85~1.2, 주변 밝기 비율로 밝기 맞춤(1.4배 넘게 바꿔야 하면 그 자리는 피함),
  색조는 30%만 반영, 경계 1화소만 섞음(라벨은 원래 경계). 조각 화소 90% 이상이 땅 위일 때만 붙임
"""
import csv
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from . import synth as S
from .augment import _resize, train_augment
from .domain import darken_drone_water, drone_veg, drone_water, hsv
from .data import CROP, LAB_POS, P_DARK_DRONE, TINY_PX, _imread, ignore_tiny, patch_label, pngs, random_crop

_LUM = np.array([0.299, 0.587, 0.114], np.float32)
HARDNEG = [S.add_shells, S.add_gravel, S.add_tetrapods, S.add_aquaculture, S.add_tidalflat,
           S.add_rock_shadows, S.add_wrack, S.add_driftwood]


def load_pieces(pieces_dir):
    pieces = []
    for r in csv.DictReader(open(Path(pieces_dir) / "manifest.csv", encoding="utf-8")):
        rgba = _imread_rgba(Path(pieces_dir) / f"{r['id']}.png")
        ring = np.array([float(r["ring_r"]), float(r["ring_g"]), float(r["ring_b"])], np.float32) / 255.0
        pieces.append((rgba[..., :3].astype(np.float32) / 255.0, rgba[..., 3] > 127, ring))
    if not pieces:
        raise FileNotFoundError(f"쓰레기 조각이 없습니다: {pieces_dir}")
    return pieces


def _imread_rgba(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGBA"))


def drone_placeable(img):
    """드론 영상(NIR 없음)에서 쓰레기를 놓을 땅을 추정: 물(청록·매끈한 면)·식생·흰 파도를 빼고, 물에서 60m 이내."""
    h, s, v = hsv(img)
    water = drone_water(img)
    veg = drone_veg(img)
    surf = ndi.binary_dilation((s < 0.18) & (v > 0.68) & ndi.binary_dilation(water, iterations=12), iterations=3)
    land = ~water & ~veg & ~surf & (v > 0.25)
    land = ndi.binary_erosion(ndi.binary_opening(land, iterations=2), iterations=1)
    if water.any():
        land &= ndi.distance_transform_edt(~water) <= 200
    return land


class PasteSource(Dataset):
    """반환 형식은 SegDataset 과 같음: (x, y, w, 패치 라벨)"""

    def __init__(self, pieces_dir, bg_dirs, length=20000, p_hardneg=0.5, p_dark_water=0.6, p_dark_drone=P_DARK_DRONE,
                 tiny_px=TINY_PX, p_rockneg=0.35, group=0, return_meta=False):
        self.pieces = load_pieces(pieces_dir)
        self.bgs = []
        for d in bg_dirs:
            d = Path(d)
            for p in pngs(d):
                a = d / "aux" / p.name
                self.bgs.append((p, a if a.exists() else None))
            for p in (pngs(d / "images") if (d / "images").is_dir() else []):
                m = d / "masks" / p.name
                if not m.exists():           # 실제 음성 중 채점 제외 화소가 없는 깨끗한 타일만 배경으로
                    self.bgs.append((p, None))
        if not self.bgs:
            raise FileNotFoundError(f"붙여넣기 배경이 없습니다: {bg_dirs}")
        self.length = length
        self.p_hardneg = p_hardneg
        self.p_dark_water = p_dark_water
        self.p_dark_drone = p_dark_drone
        self.tiny_px = tiny_px
        self.p_rockneg = p_rockneg
        self.group, self.return_meta = group, return_meta   # ADELE: 소스 그룹 id (샘플 id 는 매번 새로 합성하므로 -1)

    def __len__(self):
        return self.length

    def _place(self, img, ok, lab, rng, near=None):
        rgb, alpha, ring = self.pieces[int(rng.integers(len(self.pieces)))]
        k = int(rng.integers(4))
        rgb, alpha = np.rot90(rgb, k), np.rot90(alpha, k)
        if rng.random() < 0.5:
            rgb, alpha = rgb[:, ::-1], alpha[:, ::-1]
        s = rng.uniform(0.85, 1.2)
        ph, pw = max(int(round(rgb.shape[0] * s)), 2), max(int(round(rgb.shape[1] * s)), 2)
        if ph >= CROP - 4 or pw >= CROP - 4:
            return False
        rgb = np.clip(_resize(np.ascontiguousarray(rgb), ph, pw), 0, 1)
        a = _resize(np.ascontiguousarray(alpha.astype(np.float32)), ph, pw) > 0.5
        if a.sum() < 4:
            return False
        H, W = ok.shape
        for _ in range(30):
            if near is not None and rng.random() < 0.7:
                cy, cx = (np.asarray(near) + rng.normal(0, rng.uniform(6, 25), 2)).astype(int)
            else:
                cand = np.argwhere(ok)
                if len(cand) == 0:
                    return False
                cy, cx = cand[int(rng.integers(len(cand)))]
            y0, x0 = int(cy - ph // 2), int(cx - pw // 2)
            if y0 < 0 or x0 < 0 or y0 + ph > H or x0 + pw > W:
                continue
            if ok[y0:y0 + ph, x0:x0 + pw][a].mean() < 0.9:
                continue
            ringm = ndi.binary_dilation(a, iterations=3) & ~a
            new_ring = img[y0:y0 + ph, x0:x0 + pw][ringm].mean(0)
            ratio = float(new_ring @ _LUM) / max(float(ring @ _LUM), 1e-3)
            if not (1 / 1.4 <= ratio <= 1.4):       # 밝기를 크게 바꿔야 하는 자리는 부자연스러우므로 피함
                continue
            piece = rgb * ratio * rng.uniform(0.92, 1.08) + 0.3 * (new_ring - ring * ratio)
            soft = ndi.gaussian_filter(a.astype(np.float32), 0.5)[..., None]
            soft = np.where(a[..., None], np.maximum(soft, 0.85), soft)   # 경계 1화소만 섞임
            win = img[y0:y0 + ph, x0:x0 + pw]
            img[y0:y0 + ph, x0:x0 + pw] = win * (1 - soft) + np.clip(piece, 0, 1) * soft
            lab[y0:y0 + ph, x0:x0 + pw][a] = LAB_POS
            ok[y0:y0 + ph, x0:x0 + pw][a] = False
            return (cy, cx)
        return False

    def _pick(self, rng, tries=8):
        """조각을 놓을 땅이 충분한 배경·자르기 위치를 고릅니다 (바다만 있는 타일·자르기가 많아서).
        끝내 못 찾으면 마지막 것을 그대로 씀(= 음성 샘플)."""
        for t in range(tries):
            p, ap = self.bgs[int(rng.integers(len(self.bgs)))]
            img_u8 = _imread(p)
            aux = _imread(ap, gray=True) if ap is not None else None
            lab0 = aux if aux is not None else np.zeros(img_u8.shape[:2], np.uint8)
            for _ in range(3 if img_u8.shape[0] > CROP else 1):   # 큰 타일은 자르기 위치도 몇 번 시도
                ci, cl = random_crop(img_u8, lab0, rng, p_keep_pos=0.0)
                if aux is not None:
                    ok_frac = (cl == S.AUX_NEAR).mean()
                else:
                    ok_frac = drone_placeable(ci.astype(np.float32) / 255.0).mean()
                if ok_frac > 0.03:
                    return ci, (cl if aux is not None else None)
        return ci, (cl if aux is not None else None)

    def make(self, rng):
        """(이미지 uint8 256x256x3, 라벨맵 uint8) — 증강 전."""
        img_u8, aux = self._pick(rng)
        img = img_u8.astype(np.float32) / 255.0
        if aux is not None:                               # NAIP: aux 로 정확히 배치 + 하드 네거티브 합성
            ctx = S.make_ctx(img_u8, rng, aux == S.AUX_WATER, aux == S.AUX_VEG, aux == S.AUX_NEAR)
            if rng.random() < self.p_dark_water:
                S.darken_water(img, ctx, rng)
            if rng.random() < self.p_hardneg:
                for fn in rng.choice(HARDNEG, size=int(rng.integers(1, 3)), replace=False):
                    fn(img, ctx, rng)
            if rng.random() < self.p_rockneg:
                rng.choice(S.ROCK_HARDNEG)(img, ctx, rng)
            ok = ctx.ok.copy()
        else:
            ok = drone_placeable(img)                     # 물 추정은 원래 색으로 한 뒤
            water = drone_water(img) if rng.random() < self.p_rockneg else None
            if rng.random() < self.p_dark_drone:          # 드론 바다를 위성처럼 어둡게
                darken_drone_water(img, rng)
            if water is not None:                         # 실제 드론 배경에도 외딴 바위·밝은 바위·물보라 (물보라는 어두워진 바다 위에)
                ctx = S.make_ctx(img_u8, rng, water, np.zeros_like(water), near=ok)
                rng.choice(S.ROCK_HARDNEG)(img, ctx, rng)
                ok = ctx.ok.copy()
        lab = np.zeros(img.shape[:2], np.uint8)
        near = None
        for _ in range(int(rng.choice([1, 1, 1, 2, 2, 3, 4, 6]))):
            got = self._place(img, ok, lab, rng, near)
            if got:
                near = got
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), lab

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,)).item()))
        img_u8, lab = self.make(rng)
        lab = ignore_tiny(lab, self.tiny_px)          # 붙인 조각이 외딴 작은 점이면 실제 데이터와 똑같이 채점 제외
        plab = patch_label(lab)
        img, lab = train_augment(img_u8, lab, rng)
        pos = lab == LAB_POS
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        y = torch.from_numpy(pos.astype(np.float32))[None]
        w = torch.from_numpy((lab != 2).astype(np.float32))[None]
        if self.return_meta:
            return x, y, w, torch.tensor(plab), torch.tensor(self.group), torch.tensor(-1)
        return x, y, w, torch.tensor(plab)
