"""대회 제출 규격 RLE: 행 우선으로 편 뒤 0-based '시작 길이' 쌍."""
import numpy as np

H, W = 256, 256


def rle_encode(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return " ".join(f"{a} {b - a}" for a, b in zip(st.tolist(), en.tolist()))


def rle_decode(rle: str, shape=(H, W)) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    vals = [int(v) for v in str(rle or "").split()]
    if len(vals) % 2:
        raise ValueError("RLE 값 개수가 홀수입니다")
    for s, n in zip(vals[0::2], vals[1::2]):
        if n < 1 or s < 0 or s + n > flat.size:
            raise ValueError(f"잘못된 구간: {s} {n}")
        flat[s:s + n] = True
    return flat.reshape(shape)
