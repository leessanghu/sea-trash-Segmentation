"""폴더 데이터셋.

폴더 규격 (합성 데이터·실제 데이터 공통)
  <root>/images/<id>.png   RGB. 256x256 이 기본이고, 더 크면(예: 실제 드론 데이터 342x342) 학습 때 256 으로 무작위 자름
  <root>/masks/<id>.png    0 = 배경, 255 = 쓰레기, 127 = 채점 제외(손실에서 뺌). 파일이 없으면 음성 패치
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from .augment import cv2, train_augment

MASK_POS, MASK_IGNORE = 255, 127
LAB_BG, LAB_POS, LAB_IGN = 0, 1, 2   # 내부 라벨맵 값
CROP = 256
# 대회 정답 단위는 '쓰레기 집적대'(수 m 이상). 서로 TINY_GAP 화소 이내로 모인 쓰레기 묶음의 화소 합이
# TINY_PX(30px ≈ 2.7m²) 미만이면 외딴 작은 점으로 보고 채점 제외로 바꿉니다 (학습·검증 공통)
TINY_PX, TINY_GAP = 30, 3
P_DARK_DRONE = 0.7   # 드론(AIHub) 영상의 청록 바다를 대회 영상처럼 어둡게 바꾸는 확률 (학습 때만)


def pngs(d):
    """폴더 안의 png 목록. 맥 tar 가 넣는 AppleDouble 숨김 파일(._이름.png)은 제외합니다
    (맥 tar 는 목록에서 숨기지만 리눅스에서 풀면 실제 파일로 나와 학습이 깨집니다)."""
    return sorted(p for p in Path(d).glob("*.png") if not p.name.startswith("._"))


def list_items(root):
    root = Path(root)
    items = []
    for p in pngs(root / "images"):
        m = root / "masks" / p.name
        items.append((p, m if m.exists() else None))
    return items


def _imread(path, gray=False):
    """cv2 가 있으면 cv2 로(PIL 보다 빠름), 없으면 PIL 로 읽습니다. 한글 경로도 되도록 바이트로 읽어 디코드."""
    if cv2 is not None:
        buf = np.fromfile(str(path), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if im is None:   # 깨진 파일·맥 AppleDouble(._x.png) 등: 원인을 바로 알 수 있게 경로를 알려 줌
            raise ValueError(f"이미지를 읽을 수 없습니다 ({len(buf)} 바이트): {path}")
        return im if gray else np.ascontiguousarray(im[..., ::-1])
    return np.asarray(Image.open(path).convert("L" if gray else "RGB"))


def read_label(img_path, mask_path):
    """(이미지 uint8 HxWx3, 라벨맵 uint8 HxW: 0 배경 / 1 쓰레기 / 2 채점 제외)"""
    img = _imread(img_path)
    lab = np.zeros(img.shape[:2], np.uint8)
    if mask_path is not None:
        m = _imread(mask_path, gray=True)
        lab[m >= 200] = LAB_POS
        lab[(m > 50) & (m < 200)] = LAB_IGN
    return img, lab


def read_pair(img_path, mask_path):
    """(이미지, 쓰레기 마스크 bool). 채점 제외 화소는 쓰레기가 아닌 것으로 봅니다."""
    img, lab = read_label(img_path, mask_path)
    return img, lab == LAB_POS


def ignore_tiny(lab, min_px=TINY_PX, gap=TINY_GAP, ring=2):
    """외딴 작은 쓰레기 묶음(+주변 ring 화소)을 채점 제외로 바꾼 라벨맵 사본을 반환합니다. min_px=0 이면 그대로."""
    pos = lab == LAB_POS
    if min_px <= 0 or not pos.any():
        return lab
    grp, n = ndi.label(ndi.binary_dilation(pos, iterations=gap), structure=np.ones((3, 3), bool))
    small = np.bincount(grp[pos], minlength=n + 1) < min_px
    small[0] = False
    if not small.any():
        return lab
    tiny = small[grp] & pos
    lab = lab.copy()
    lab[ndi.binary_dilation(tiny, iterations=ring) & ~(pos & ~tiny)] = LAB_IGN
    return lab


def contest_gts(gts, min_px=TINY_PX, drop=None):
    """대회 기준 검증: 외딴 작은 쓰레기 묶음(과 drop 화소)은 정답에서 빼고, 그것만 있던 양성 패치는 평가에서 제외.
    drop: (N,H,W) bool. 투기물 의심(masks_dumped) 등 정답에서 뺄 화소. 반환 (남길 패치 인덱스, 새 정답 마스크들)."""
    if min_px <= 0 and drop is None:
        return np.arange(len(gts)), gts
    src = gts if drop is None else gts & ~drop
    new = np.stack([ignore_tiny(g.astype(np.uint8), min_px) == LAB_POS for g in src])
    had, has = gts.reshape(len(gts), -1).any(1), new.reshape(len(gts), -1).any(1)
    keep = np.flatnonzero(~(had & ~has))
    return keep, new[keep]


def dumped_drop(items, size=CROP):
    """검증 폴더의 masks_dumped(투기물 의심 = 채점 제외 127)를 (N,H,W) bool 로. 폴더가 없으면 None."""
    if not items or not (Path(items[0][0]).parent.parent / "masks_dumped").is_dir():
        return None
    out = []
    for img_p, _ in items:
        dp = Path(img_p).parent.parent / "masks_dumped" / Path(img_p).name
        if not dp.exists():
            out.append(np.zeros((size, size), bool))
            continue
        m = _imread(dp, gray=True)
        h, w = m.shape
        y, x = max((h - size) // 2, 0), max((w - size) // 2, 0)
        m = m[y:y + size, x:x + size]
        out.append((m > 50) & (m < 200))
    return np.stack(out)


def center_crop(img, lab, size=CROP):
    h, w = lab.shape
    y, x = max((h - size) // 2, 0), max((w - size) // 2, 0)
    return img[y:y + size, x:x + size], lab[y:y + size, x:x + size]


def random_crop(img, lab, rng, size=CROP, p_keep_pos=0.8):
    """크면 무작위로 자릅니다 (고정 위치로 자르면 '쓰레기는 가운데'를 배움).

    쓰레기가 작아서 그냥 자르면 잘려 나가기 쉬우므로, 쓰레기가 있는 타일은 p_keep_pos 확률로
    무작위 쓰레기 화소 하나가 들어가는 범위 안에서 위치를 고릅니다 (위치는 여전히 무작위).
    """
    h, w = lab.shape
    if h == size and w == size:
        return img, lab
    ylo, yhi, xlo, xhi = 0, h - size, 0, w - size
    if rng.random() < p_keep_pos:
        pts = np.argwhere(lab == LAB_POS)
        if len(pts):
            py, px = pts[rng.integers(len(pts))]
            ylo, yhi = max(0, py - size + 1), min(h - size, py)
            xlo, xhi = max(0, px - size + 1), min(w - size, px)
    y = int(rng.integers(ylo, yhi + 1))
    x = int(rng.integers(xlo, xhi + 1))
    return img[y:y + size, x:x + size], lab[y:y + size, x:x + size]


def parse_sources(spec: str):
    """'data/synth:1.0,data/real:3' -> [(Path, weight)]"""
    out = []
    for part in [s for s in spec.split(",") if s.strip()]:
        path, _, w = part.strip().rpartition(":")
        try:
            weight = float(w)
        except ValueError:  # 가중치 생략 (윈도 드라이브 콜론 포함)
            path, weight = part.strip(), 1.0
        out.append((Path(path), weight))
    return out


def patch_label(lab):
    """분류 헤드 라벨: 쓰레기 있음 1 / 없음 0 / 모름 -1 (쓰레기는 없고 채점 제외만 있는 패치: 손실에서 뺌).
    증강(축소 시 테두리를 채점 제외로 채움) 전에 계산해야 합니다."""
    if (lab == LAB_POS).any():
        return 1.0
    return -1.0 if (lab == LAB_IGN).any() else 0.0


class SegDataset(Dataset):
    """반환: x (3,256,256) [0,1], y (1,256,256) 쓰레기, w (1,256,256) 손실에 쓰는 화소(채점 제외=0), 패치 라벨(1/0/-1)
    return_meta=True 이면 뒤에 (소스 그룹 id, 샘플 id) 를 덧붙임 (ADELE: 어느 폴더에서 온 샘플인지)."""

    def __init__(self, items, augment=False, drone=None, p_dark_drone=P_DARK_DRONE, tiny_px=TINY_PX,
                 groups=None, return_meta=False):
        self.items = items
        self.augment = augment
        self.drone = drone or [False] * len(items)   # 드론(AIHub) 영상인지: 학습 때 바다를 위성처럼 어둡게
        self.p_dark_drone = p_dark_drone
        self.tiny_px = tiny_px
        self.groups = groups if groups is not None else [0] * len(items)   # 항목별 소스 폴더 번호 (--train 순서)
        self.return_meta = return_meta

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, lab = read_label(*self.items[i])
        lab = ignore_tiny(lab, self.tiny_px)
        if self.augment:
            # 워커·에폭마다 다른 torch 시드에서 파생 (torch.manual_seed 로 재현 가능)
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,)).item()))
            img, lab = random_crop(img, lab, rng)
            plab = patch_label(lab)
            if self.drone[i] and rng.random() < self.p_dark_drone:
                from .domain import darken_drone_water
                f = img.astype(np.float32) / 255.0
                img = (np.clip(darken_drone_water(f, rng), 0, 1) * 255 + 0.5).astype(np.uint8)
            img, lab = train_augment(img, lab, rng)
        else:
            img, lab = center_crop(img, lab)
            img = img.astype(np.float32) / 255.0
            plab = patch_label(lab)
        pos = lab == LAB_POS
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        y = torch.from_numpy(pos.astype(np.float32))[None]
        w = torch.from_numpy((lab != LAB_IGN).astype(np.float32))[None]
        if self.return_meta:
            return x, y, w, torch.tensor(plab), torch.tensor(self.groups[i]), torch.tensor(i)
        return x, y, w, torch.tensor(plab)


CONFUSER_TAGS = ("bright_gravel", "foam", "seawall")


def build_train(spec: str, paste: str = "", paste_weight: float = 0.0, tiny_px: int = TINY_PX,
                confuser_boost: float = 2.0, dumped: bool = False, return_meta: bool = False):
    """소스별 가중치를 반영한 (dataset, 샘플 가중치, 정보) 를 만듭니다.

    paste: '조각폴더,배경폴더1,배경폴더2,...' 이면 실제 쓰레기 조각 온라인 붙여넣기 소스를 paste_weight 로 추가
    confuser_boost: 폴더에 confusers.csv(밝은 자갈·조개 / 포말 / 방파제 꼬리표)가 있으면 꼬리표 붙은 타일을 이 배수로
                    더 자주 뽑음 (폴더 전체 비중은 그대로)
    dumped: 폴더에 masks_dumped/ (도로·건물 20m 이내 쓰레기 = 채점 제외) 가 있으면 masks/ 대신 사용
    return_meta: 샘플마다 (소스 그룹 id, 샘플 id) 도 반환. 그룹 이름은 info["groups"] (폴더 이름, 붙여넣기는 "paste")
    """
    items, weights, drone, gids, gnames = [], [], [], [], []
    for gi, (root, w) in enumerate(parse_sources(spec)):
        its = list_items(root)
        if not its:
            raise FileNotFoundError(f"비어 있는 데이터 폴더: {root}")
        note = ""
        dd = Path(root) / "masks_dumped"
        if dumped and dd.is_dir():
            n0 = sum((dd / p.name).exists() for p, _ in its)
            its = [(p, dd / p.name if (dd / p.name).exists() else m) for p, m in its]
            note += f" (투기물 의심 대체 마스크 {n0}장)"
        iw = np.ones(len(its))
        cf = Path(root) / "confusers.csv"
        if confuser_boost != 1 and cf.exists():
            tagged = {r["id"] for r in csv.DictReader(open(cf, encoding="utf-8"))
                      if any(float(r.get(k) or 0) > 0 for k in CONFUSER_TAGS)}
            iw = np.array([confuser_boost if p.stem in tagged else 1.0 for p, _ in its])
            note += f" (혼동 대상 꼬리표 {int((iw > 1).sum())}장 x{confuser_boost:g})"
        items += its
        weights += list(w * iw / iw.sum())
        is_drone = Path(root).name.startswith(("real_", "cp_"))   # AIHub 드론 영상 폴더
        drone += [is_drone] * len(its)
        gids += [gi] * len(its)
        gnames.append(Path(root).name)
        print(f"  train source {root}: {len(its)} 장, 가중치 {w}" + (f" (드론: 바다 어둡게 {P_DARK_DRONE:.0%})" if is_drone else "")
              + note)
    ds = SegDataset(items, augment=True, drone=drone, tiny_px=tiny_px, groups=gids, return_meta=return_meta)
    info = {"n_items": len(items), "n_mask": sum(m is not None for _, m in items),
            "groups": gnames, "group_of": gids, "items": items}
    if paste and paste_weight > 0:
        from torch.utils.data import ConcatDataset

        from .paste import PasteSource
        parts = [s for s in paste.split(",") if s]
        ps = PasteSource(parts[0], parts[1:], tiny_px=tiny_px, group=len(gnames), return_meta=return_meta)
        info["groups"] = gnames + ["paste"]
        print(f"  train source 붙여넣기: 조각 {len(ps.pieces)}개 × 배경 {len(ps.bgs)}장 (매번 새로 합성), 가중치 {paste_weight}")
        weights += [paste_weight / len(ps)] * len(ps)
        ds = ConcatDataset([ds, ps])
        info["paste"] = True
    return ds, torch.tensor(weights, dtype=torch.double), info
