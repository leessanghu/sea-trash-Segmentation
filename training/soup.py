r"""모델 수프 (가중치 평균, Model soups ICML'22 / DiWA NeurIPS'22) — 같은 spec 체크포인트 K개를 하나의 일반 체크포인트로.

예측(확률) 앙상블은 멤버 수만큼 추론 시간이 들지만, 수프는 가중치를 평균해 모델 1개 시간으로 돌립니다.
조건 (두 논문 공통): 같은 사전학습 가중치 + 같은 헤드 초기값(공유 초기화)에서 '순한' 하이퍼파라미터 차이로 학습한 런끼리만.
다른 초기화(예: 따로 무작위 초기화한 디코더)나 학습률이 크게 다른 런을 섞으면 평균 모델이 멤버보다 나빠집니다 (DiWA Fig.5/Tab.5).

  python soup.py --out soup.pt a.pt b.pt c.pt                     # 균등 평균 (기본, 권장)
  python soup.py --out soup.pt a.pt b.pt c.pt --select greedy --held-out <검증폴더>   # 탐욕 선택 (검증이 믿을 만할 때만)
  python soup.py --out soup.pt a.pt b.pt --bn-recal 200 --bn-data "<폴더>:0.5,<폴더>:0.5"  # BN 통계 다시 계산

평균 규칙
  - 부동소수 텐서(가중치, BN running_mean/var 포함)는 float64 로 (가중)평균 후 원래 dtype 으로.
    BN 통계를 평균하는 것은 train.py 의 EMA 가 이미 하는 방식과 같습니다. 정확히 맞추려면 --bn-recal.
  - 정수 버퍼(num_batches_tracked 등)는 첫 번째 체크포인트 값.
  - LoRA/DoRA 멤버는 저장 때 이미 W0+ΔW 로 합쳐져 있으므로 합쳐진 가중치를 그대로 평균합니다.
    LoRA: mean_k(W0 + s·B_k A_k) = W0 + s·mean_k(B_k A_k) 로 '보정을 평균한 것'과 정확히 같음 (A·B 를 따로 평균하면 틀림).
    DoRA: 합쳐진 행렬(m_k·방향_k)의 평균. (m, A, B) 를 평균해 다시 합친 것과는 다르지만, 추론에 쓰는 것은 합쳐진 행렬뿐.
결과는 save_checkpoint 와 같은 형식 {"spec", "state_dict", "meta"} 이고 knps.model.load_checkpoint 로 바로 읽힙니다.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knps.model import build_model, load_checkpoint  # noqa: E402


def _dev():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def _spec_key(spec):
    return json.dumps(spec, sort_keys=True)


def load_cks(paths):
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


def check_compatible(cks, names):
    """spec·키·모양·dtype 이 모두 같아야 평균할 수 있습니다. 다르면 ValueError. 경고 문자열 목록을 반환."""
    s0 = _spec_key(cks[0]["spec"])
    sd0 = cks[0]["state_dict"]
    warns = []
    for ck, n in zip(cks[1:], names[1:]):
        if _spec_key(ck["spec"]) != s0:
            raise ValueError(f"spec 이 다릅니다: {names[0]} {cks[0]['spec']} / {n} {ck['spec']}")
        sd = ck["state_dict"]
        if set(sd) != set(sd0):
            a, b = sorted(set(sd0) - set(sd))[:5], sorted(set(sd) - set(sd0))[:5]
            raise ValueError(f"state_dict 키가 다릅니다 ({names[0]} 에만 {a}, {n} 에만 {b})")
        for k, v in sd0.items():
            if sd[k].shape != v.shape or sd[k].dtype != v.dtype:
                raise ValueError(f"{k}: 모양/dtype 불일치 {tuple(v.shape)} {v.dtype} vs {tuple(sd[k].shape)} {sd[k].dtype}")
    tags = [json.dumps(ck.get("meta", {}).get("lora"), sort_keys=True) for ck in cks]
    if len(set(tags)) > 1:
        warns.append(f"학습 방식(LoRA/DoRA/전체)이 섞여 있습니다: {dict(zip(names, tags))}. "
                     "spec 은 같지만 같은 초기화에서 출발했는지 확인하세요 (평균 가능성은 검증으로 확인)")
    return warns


def average_state_dicts(sds, weights=None):
    """부동소수 텐서는 가중평균, 정수 텐서는 첫 번째 것."""
    w = np.ones(len(sds)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    for k, v0 in sds[0].items():
        if v0.is_floating_point():
            acc = torch.zeros(v0.shape, dtype=torch.float64)
            for wi, sd in zip(w, sds):
                acc += float(wi) * sd[k].double()
            out[k] = acc.to(v0.dtype)
        else:
            out[k] = v0.clone()
    return out


def locality(sds):
    """멤버별 ||θ_k - θ̄|| / ||θ̄|| (BN 통계 제외 부동소수 파라미터). DiWA 의 '국소성' 진단용 (작을수록 평균이 안전)."""
    keys = [k for k, v in sds[0].items() if v.is_floating_point() and "running_" not in k]
    mean = average_state_dicts([{k: sd[k] for k in keys} for sd in sds])
    nm = math.sqrt(sum(float(mean[k].double().pow(2).sum()) for k in keys))
    res = []
    for sd in sds:
        d = math.sqrt(sum(float((sd[k].double() - mean[k].double()).pow(2).sum()) for k in keys))
        res.append(d / max(nm, 1e-12))
    return res


# ------------------------------------------------------------------ 검증 점수 (탐욕 선택용)
def held_out_score(model, items, device, tta=1, bs=16):
    """exp_eval.py 와 같은 고정 후처리(v4cfg+dil1)로 local639 / gtdil639 를 계산."""
    from scipy import ndimage
    from exp_eval import FIXED, score
    from knps.data import TINY_PX, contest_gts
    from knps.evaluate import infer_items
    from knps.postprocess import postprocess
    P, C, G = infer_items([model], items, device, bs=bs, tta=tta)
    keep, Gc = contest_gts(G, TINY_PX)
    P, C = P[keep].astype(np.float32), C[keep]
    masks = [postprocess(p, float(c), FIXED) for p, c in zip(P, C)]
    Gd = np.stack([ndimage.binary_dilation(g, np.ones((3, 3), bool)) if g.any() else g for g in Gc])
    l, g = score(masks, Gc, 0.639)["score"], score(masks, Gd, 0.639)["score"]
    return {"local639": l, "gtdil639": g, "mean": 0.5 * (l + g)}


def greedy_select(cks, names, items, device, tta=1, metric="mean", margin=0.0, bs=16):
    """Model soups Recipe 1 / DiWA-restricted: 검증 점수 내림차순으로 하나씩 넣어 보고 점수가 (margin 이상) 안 떨어지면 유지.
    논문 의사코드는 '≥', 공식 model-soups 코드는 '>' 입니다. margin>0 이면 검증 잡음보다 큰 개선만 받아들입니다."""
    model = build_model(cks[0]["spec"], pretrained=False).to(device).eval()

    def ev(sd):
        model.load_state_dict(sd)
        return held_out_score(model, items, device, tta, bs)

    indiv = []
    for ck, n in zip(cks, names):
        r = ev(ck["state_dict"])
        indiv.append(r)
        print(f"  단독 {n}: local639 {r['local639']:.4f} gtdil639 {r['gtdil639']:.4f}", flush=True)
    order = sorted(range(len(cks)), key=lambda i: -indiv[i][metric])
    chosen = [order[0]]
    best = indiv[order[0]][metric]
    log = [{"try": names[order[0]], "score": indiv[order[0]], "accept": True}]
    for i in order[1:]:
        cand = chosen + [i]
        r = ev(average_state_dicts([cks[j]["state_dict"] for j in cand]))
        ok = r[metric] >= best + margin
        print(f"  +{names[i]} → {metric} {r[metric]:.4f} (현재 {best:.4f}) {'채택' if ok else '제외'}", flush=True)
        log.append({"try": names[i], "score": r, "accept": bool(ok)})
        if ok:
            chosen, best = cand, r[metric]
    del model
    return chosen, {"individual": dict(zip(names, indiv)), "steps": log}


# ------------------------------------------------------------------ BN 통계 재계산
def has_batchnorm(model):
    return any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches, device):
    """torch.optim.swa_utils.update_bn 과 같은 방식: BN 통계를 비우고 학습 데이터 N 배치의 누적 평균으로 다시 채움.
    BN 이 있는 층만 바뀝니다 (ViT 의 LayerNorm 은 통계가 없어 영향 없음, DinoUNet 디코더·스템의 BN 은 바뀜)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    mom = {}
    for m in bns:
        m.reset_running_stats()
        mom[m] = m.momentum
        m.momentum = None                  # 누적 평균
    was = model.training
    model.train()                          # BN 통계 갱신 (드롭아웃은 분류 헤드에만 있어 BN 입력엔 영향 없음)
    n = 0
    for batch in loader:
        x = batch[0].to(device)
        model(x)
        n += 1
        if n >= n_batches:
            break
    for m in bns:
        m.momentum = mom[m]
    model.train(was)
    return n


def bn_loader(spec, paste, paste_weight, bs, n_batches, workers, seed=0):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from knps.data import build_train
    torch.manual_seed(seed)
    ds, w, _ = build_train(spec, paste, paste_weight)
    sampler = WeightedRandomSampler(w, num_samples=bs * n_batches, replacement=True)
    return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers, drop_last=True)


# ------------------------------------------------------------------
def make_soup(paths, out, weights=None, select="uniform", held_out="", tta=1, metric="mean", margin=0.0,
              bn_recal=0, bn_data="", bn_paste="", bn_paste_weight=0.25, bn_bs=16, workers=2, device=None, verbose=True):
    """paths 의 체크포인트를 평균해 out 에 저장하고 meta 를 반환합니다 (run_retrain.py 에서도 이 함수를 씀)."""
    device = device or _dev()
    names = [str(p) for p in paths]
    cks = load_cks(paths)
    for wmsg in check_compatible(cks, names):
        print("  [경고] " + wmsg, flush=True)
    sds = [ck["state_dict"] for ck in cks]
    loc = locality(sds)
    if verbose:
        print("  국소성 ||θk-θ̄||/||θ̄||: " + ", ".join(f"{Path(n).parent.name}/{Path(n).name} {d:.4f}"
                                                for n, d in zip(names, loc)), flush=True)
    idx = list(range(len(cks)))
    sel_info = {}
    if select == "greedy":
        if not held_out:
            raise ValueError("--select greedy 에는 --held-out 검증 폴더가 필요합니다")
        from knps.data import list_items
        if weights is not None:
            print("  [경고] greedy 선택에서는 --weights 를 무시하고 고른 멤버를 균등 평균합니다", flush=True)
        idx, sel_info = greedy_select(cks, names, list_items(held_out), device, tta, metric, margin)
        weights = None
    elif select != "uniform":
        raise ValueError(select)
    w = None if weights is None else [weights[i] for i in idx]
    sd = average_state_dicts([sds[i] for i in idx], w)
    spec = cks[0]["spec"]
    meta = {"soup": {"members": [names[i] for i in idx], "candidates": names, "weights": w, "select": select,
                     "locality": dict(zip(names, loc)), "member_meta": [cks[i].get("meta", {}) for i in idx]}}
    lora = {json.dumps(cks[i].get("meta", {}).get("lora"), sort_keys=True) for i in idx}
    if len(lora) == 1 and cks[idx[0]].get("meta", {}).get("lora"):
        meta["lora"] = cks[idx[0]]["meta"]["lora"]
    if sel_info:
        meta["soup"]["greedy"] = sel_info
    if bn_recal > 0:
        model = build_model(spec, pretrained=False)
        model.load_state_dict(sd)
        model = model.to(device)
        if not has_batchnorm(model):
            print("  BN 층 없음 → 재계산 생략", flush=True)
        elif not bn_data:
            raise ValueError("--bn-recal 에는 --bn-data (train.py --train 과 같은 '폴더:가중치,...') 가 필요합니다")
        else:
            n = recalibrate_bn(model, bn_loader(bn_data, bn_paste, bn_paste_weight, bn_bs, bn_recal, workers),
                               bn_recal, device)
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            meta["soup"]["bn_recal"] = {"batches": n, "bs": bn_bs, "data": bn_data}
            print(f"  BN 통계 재계산: {n} 배치 × {bn_bs}", flush=True)
        del model
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": spec, "state_dict": sd, "meta": meta}, out)
    load_checkpoint(out)                     # 엄격 로드 확인 (키·모양)
    if verbose:
        print(f"수프 저장: {out} (멤버 {len(idx)}/{len(cks)}: {[Path(names[i]).name for i in idx]})", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="", help="쉼표로 멤버별 가중치 (기본 균등)")
    ap.add_argument("--select", default="uniform", choices=["uniform", "greedy"])
    ap.add_argument("--held-out", default="", help="greedy 용 검증 폴더 (images/, masks/). 학습·후처리 선택에 안 쓴 것")
    ap.add_argument("--metric", default="mean", choices=["local639", "gtdil639", "mean"])
    ap.add_argument("--margin", type=float, default=0.0, help="greedy: 이만큼 이상 좋아질 때만 채택 (검증 잡음 대비)")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--bn-recal", type=int, default=0, help="BN 통계를 학습 데이터 N 배치로 다시 계산 (0 = 평균 그대로)")
    ap.add_argument("--bn-data", default="", help="train.py --train 형식 '폴더:가중치,...'")
    ap.add_argument("--bn-paste", default="", help="train.py --paste 형식 (선택)")
    ap.add_argument("--bn-paste-weight", type=float, default=0.25)
    ap.add_argument("--bn-bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    w = [float(x) for x in a.weights.split(",")] if a.weights else None
    if w is not None and len(w) != len(a.ckpts):
        sys.exit("--weights 개수가 체크포인트 개수와 다릅니다")
    make_soup(a.ckpts, a.out, w, a.select, a.held_out, a.tta, a.metric, a.margin,
              a.bn_recal, a.bn_data, a.bn_paste, a.bn_paste_weight, a.bn_bs, a.workers)


if __name__ == "__main__":
    main()
