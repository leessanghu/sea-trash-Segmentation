"""② 위치 틀림 / ③ 확신 오탐 집중 처리: 하드 예제 채굴 → v5 6개 멤버 짧은 이어학습 → v5 와 비교 → 제출 폴더.

단계 (각 단계 결과를 --out 에 저장, 다시 실행하면 끝난 단계는 건너뜀)
  1 mine   v5 6개(TTA1)로 학습용 음성(real_neg_train)·양성(real_pos_train)을 추론.
           - 음성인데 50px 이상 예측(=오탐) 또는 판정점수 0.3 이상 → real_hardneg_train  (③)
           - 양성인데 놓침 또는 형상 < 0.5                         → real_hardpos_train  (②)
           검증셋(real_val)은 절대 채굴에 쓰지 않음 (누수 방지)
  2 base   v5 멤버별 검증 확률 (TTA8) 저장
  3 ft     멤버별로 v5 가중치에서 이어학습 (하드 예제 비중 높임) → 검증 확률 (TTA8) 저장
           실패한 멤버는 v5 가중치 그대로 사용
  4 gate   후처리 고정(v4cfg+dil1)으로 v5 / 새 앙상블 / 멤버 교체(좋아진 것만) 비교 → 고른 구성으로 제출 폴더

python hnm_pipeline.py --repo /content/sea_trash --data /content/data --out <드라이브> --v5 <v5 폴더(model0~5.pt)> --tpl <제출 틀>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MEMBERS = [  # (이름, lr, bs)  — v5 원래 lr 의 약 1/3~1/4
    ("unet_r18_base_x2", 1.5e-4, 16),
    ("unet_cnxt_t_x2", 1e-4, 8),
    ("dino_s", 7e-5, 12),
    ("dino3_s", 7e-5, 12),
    ("dino3_s_lora", 7e-5, 12),   # 체크포인트는 합쳐진 일반 dino 로 저장되어 있어 일반 이어학습
    ("dino_s_dora", 7e-5, 12),
]
OOM = ("out of memory", "OutOfMemoryError", "MPS backend out of memory")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dev():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def infer(ckpts, items, tta, bs):
    import torch
    from knps.evaluate import infer_items
    from knps.model import load_checkpoint
    d = dev()
    models = [load_checkpoint(c, d)[0] for c in ckpts]
    P, C, G = infer_items(models, items, d, bs=bs, tta=tta)
    del models
    if d == "cuda":
        torch.cuda.empty_cache()
    return P, C, G


# ---------------------------------------------------------------- 1. 채굴
def mine(a, out):
    from knps.data import TINY_PX, contest_gts, list_items
    from knps.metric import patch_shape_score
    from knps.postprocess import postprocess
    from exp_eval import FIXED
    res_p = out / "mine.json"
    if res_p.exists():
        log("1 mine: 저장된 결과 사용")
        r = json.load(open(res_p))
    else:
        v5 = [str(Path(a.v5) / f"model{i}.pt") for i in range(6)]
        r = {"neg": [], "pos": []}
        for kind, folder in (("neg", "real_neg_train"), ("pos", "real_pos_train")):
            items = list_items(f"{a.data}/{folder}")
            if a.limit:
                items = items[:a.limit]
            log(f"1 mine: {folder} {len(items)}장 추론 (v5 6개, TTA1)")
            P, C, G = infer(v5, items, 1, a.infer_bs)
            P = P.astype(np.float32)
            keep, Gc = contest_gts(G, TINY_PX)
            gmap = {int(k): g for k, g in zip(keep, Gc)}
            for i, (p, c) in enumerate(zip(P, C)):
                top = float(np.partition(p.ravel(), -50)[-50:].mean())
                ds = FIXED["alpha"] * float(c) + (1 - FIXED["alpha"]) * top
                m = postprocess(p, float(c), FIXED)
                npx = int(m.sum())
                g = gmap.get(i)
                gp = g is not None and bool(g.any())
                row = {"name": items[i][0].name, "dscore": round(ds, 4), "ppx": npx}
                if kind == "neg":
                    if i in gmap and not gp and (npx >= 50 or ds >= 0.3):
                        r["neg"].append(row)
                elif gp:
                    sh = patch_shape_score(m, g) if npx >= 50 else 0.0
                    if sh < 0.5:
                        r["pos"].append({**row, "shape": round(float(sh), 4)})
        r["neg"] = sorted(r["neg"], key=lambda x: -x["dscore"])[:400]
        json.dump(r, open(res_p, "w"), indent=1)
    for kind, src, dst in (("neg", "real_neg_train", "real_hardneg_train"), ("pos", "real_pos_train", "real_hardpos_train")):
        d = Path(a.data) / dst
        if d.exists() and len(list((d / "images").glob("[!._]*.png"))) == len(r[kind]):
            continue   # 이미 같은 내용으로 만들어져 있음 (GPU 여러 장에서 동시에 실행할 때 서로 지우지 않게)
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        (d / "masks").mkdir()
        for row in r[kind]:
            n = row["name"]
            os.symlink(Path(a.data, src, "images", n).resolve(), d / "images" / n)
            mk = Path(a.data, src, "masks", n)
            if mk.exists():
                os.symlink(mk.resolve(), d / "masks" / n)
    log(f"1 mine: 하드 음성(③ 오탐 후보) {len(r['neg'])}장, 하드 양성(② 위치 틀림/놓침) {len(r['pos'])}장")
    return r


def train_spec(a, r):
    L = a.data
    s = [f"{L}/real_pos_train:0.12", f"{L}/cp_pos_train:0.08", f"{L}/cp_band_train:0.08",
         f"{L}/real_neg_train:0.22", f"{L}/synth_train:0.1"]
    s.append(f"{L}/real_hardneg_train:0.15" if len(r["neg"]) >= 5 else f"{L}/real_neg_train:0.15")
    s.append(f"{L}/real_hardpos_train:0.05" if len(r["pos"]) >= 3 else f"{L}/real_pos_train:0.05")
    return ",".join(s)


# ---------------------------------------------------------------- 2·3. 확률 저장, 이어학습
def val_probs(a, ckpt, dst):
    from knps.data import list_items
    if dst.exists():
        return
    items = list_items(f"{a.data}/real_val")
    P, C, G = infer([ckpt], items, a.tta, a.infer_bs)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, P=P.astype(np.float16), C=C, G=G)
    os.replace(tmp, dst)


def run_logged(cmd, env, logfile, cwd):
    with open(logfile, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%F %T')} $ {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in p.stdout:
            lf.write(line); lf.flush()
            if line.startswith(("ep ", "  ↳", "완료", "init from", "Traceback")) or "Error" in line:
                print("    " + line.rstrip(), flush=True)
            tail = (tail + [line])[-60:]
        p.wait()
    return p.returncode, "".join(tail)


def finetune(a, i, name, lr, bs, spec, d):
    if (d / "best.pt").exists() and (d / "done").exists():
        return True
    env = {**os.environ, "EXP_LOSS": "dice", "EXP_GT_DILATE": "0", "EXP_VAL_BS": str(a.val_bs), "PYTHONUNBUFFERED": "1"}
    L = a.data
    for attempt in range(2):
        cmd = [sys.executable, "train.py", "--model", "unet_r18_base_x2", "--init", str(Path(a.v5) / f"model{i}.pt"),
               "--train", spec, "--val", f"{L}/real_val",
               "--paste", f"{L}/debris_pieces,{L}/bg_naip_train,{L}/real_neg_train", "--paste-weight", "0.25",
               "--test-pos-ratio", "0.64", "--lr", str(lr * a.lr_mult), "--epochs", str(a.epochs), "--steps", str(a.steps),
               "--bs", str(bs), "--workers", str(a.workers), "--seed", "0", "--out", str(d)]
        if (d / "resume.pt").exists():
            cmd.append("--resume")
        log(f"3 ft[{i}] {name}: 학습 (bs {bs}, lr {lr * a.lr_mult:g}, {a.epochs}x{a.steps})")
        rc, tail = run_logged(cmd, env, d / "train.log", a.repo)
        if rc == 0 and (d / "best.pt").exists():
            (d / "done").write_text("ok")
            return True
        if any(m in tail for m in OOM) and attempt == 0 and bs > 2:
            bs //= 2
            for f in ("resume.pt", "last.pt", "best.pt"):
                (d / f).unlink(missing_ok=True)
            log(f"3 ft[{i}]: 메모리 부족 → bs {bs} 로 재시도")
            continue
        log(f"3 ft[{i}] 실패 (rc={rc}) — v5 가중치 유지. 로그 {d}/train.log\n{tail[-600:]}")
        return False
    return False


# ---------------------------------------------------------------- 4. 비교, 제출 폴더
def gate(a, out, ok):
    from knps.data import TINY_PX, contest_gts
    from knps.postprocess import postprocess
    from scipy import ndimage
    from exp_eval import FIXED, score
    K = np.ones((3, 3), bool)
    pr = out / "probs"
    base = [np.load(pr / f"v5_{i}.npz") for i in range(6)]
    G = base[0]["G"]
    keep, Gc = contest_gts(G, TINY_PX)
    Gd = np.stack([ndimage.binary_dilation(g, K) if g.any() else g for g in Gc])
    PB = [z["P"][keep].astype(np.float32) for z in base]
    CB = [z["C"][keep] for z in base]
    PF, CF = {}, {}
    for i in range(6):
        f = pr / f"ft_{i}.npz"
        if ok[i] and f.exists():
            z = np.load(f)
            PF[i], CF[i] = z["P"][keep].astype(np.float32), z["C"][keep]

    def ev(use_ft):
        P = sum(PF[i] if i in use_ft else PB[i] for i in range(6)) / 6
        C = sum(CF[i] if i in use_ft else CB[i] for i in range(6)) / 6
        masks = [postprocess(p, float(c), FIXED) for p, c in zip(P, C)]
        gn = [not g.any() for g in Gc]
        fp = sum(int(m.sum()) >= 50 and n for m, n in zip(masks, gn))
        return {"local639": score(masks, Gc, 0.639), "gtdil639": score(masks, Gd, 0.639), "fp_neg": int(fp)}

    res = {"v5": ev(set()), "ft_all": ev(set(PF))}
    # 멤버 교체: 하나씩 바꿔서 두 지표 모두 안 떨어진 멤버만 교체
    good = []
    b = res["v5"]
    for i in PF:
        r = ev({i})
        res[f"swap_{i}"] = r
        if r["local639"]["score"] >= b["local639"]["score"] and r["gtdil639"]["score"] >= b["gtdil639"]["score"]:
            good.append(i)
    res["swap_good"] = ev(set(good)) if good else res["v5"]
    res["swap_good_members"] = good
    rows = ["| 구성 | local639 | gtdil639 | F1 | 형상 | 음성오탐 |", "|---|---|---|---|---|---|"]
    for k, r in res.items():
        if isinstance(r, dict) and "local639" in r:
            rows.append(f"| {k} | {r['local639']['score']:.4f} | {r['gtdil639']['score']:.4f} | "
                        f"{r['local639']['f1']:.3f} | {r['local639']['shape']:.3f} | {r['fp_neg']} |")
    # 선택: v5 대비 두 지표 모두 +0.005 이상인 후보 중 local639 최고. 없으면 v5 그대로 (제출 비추천)
    cands = []
    for k, use in (("ft_all", set(PF)), ("swap_good", set(good))):
        r = res[k]
        d1 = r["local639"]["score"] - b["local639"]["score"]
        d2 = r["gtdil639"]["score"] - b["gtdil639"]["score"]
        if use and d1 > 0.005 and d2 > 0.005:
            cands.append((r["local639"]["score"], k, sorted(use)))
    if cands:
        _, pick, use = max(cands)
        verdict = f"GO — {pick} (이어학습 멤버 {use}), v5 대비 local639 {res[pick]['local639']['score'] - b['local639']['score']:+.4f} / gtdil639 {res[pick]['gtdil639']['score'] - b['gtdil639']['score']:+.4f}"
    else:
        pick, use = "v5", []
        verdict = "NO-GO — 두 지표 모두 +0.005 넘는 구성이 없음. 본 제출 비추천 (v5 그대로면 이미 낸 0.7079 와 같음)"
    md = "\n".join(rows) + f"\n\n**판정: {verdict}**\n\n후처리 고정(v4cfg+dil1), 같은 T4·같은 TTA8 로 v5 를 다시 계산해 비교했습니다.\n"
    res.update({"pick": pick, "use_ft": use, "verdict": verdict})
    json.dump(res, open(out / "gate.json", "w"), indent=1, ensure_ascii=False)
    open(out / "gate.md", "w", encoding="utf-8").write(md)
    print(md, flush=True)
    return pick, use


def package(a, out, use):
    sub = out / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    shutil.copytree(a.tpl, sub)
    conf = json.load(open(Path(a.tpl) / "assets/model/config.json"))
    srcs = []
    for i in range(6):
        src = out / "ft" / f"m{i}" / "best.pt" if i in use else Path(a.v5) / f"model{i}.pt"
        shutil.copy(src, sub / "assets/model" / f"model{i}.pt")
        srcs.append(str(src))
    conf["sources"] = srcs
    conf["hnm"] = {"use_ft": use}
    json.dump(conf, open(sub / "assets/model/config.json", "w"), indent=2, ensure_ascii=False)
    size = sum(p.stat().st_size for p in sub.rglob("*") if p.is_file())
    log(f"제출 폴더 {sub} ({size / 1e6:.0f} MB), 후처리 {conf['cfg']}")
    for p in sorted(sub.rglob("*")):
        if p.is_file():
            print("   ", p.relative_to(sub))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v5", required=True)
    ap.add_argument("--tpl", required=True, help="검증된 제출 틀 폴더 (predict.ipynb, assets/code, config.json)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--bs-mult", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--infer-bs", type=int, default=16)
    ap.add_argument("--val-bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="테스트용: 채굴 장수 제한")
    ap.add_argument("--only-members", default="", help="테스트용: 예 0,2")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "probs").mkdir(exist_ok=True)
    os.chdir(a.repo)
    t0 = time.time()

    r = mine(a, out)
    spec = train_spec(a, r)
    log(f"학습 구성: {spec}")

    members = [int(x) for x in a.only_members.split(",")] if a.only_members else list(range(6))
    for i in range(6):
        log(f"2 base[{i}]: v5 멤버 검증 확률 (TTA{a.tta})")
        val_probs(a, str(Path(a.v5) / f"model{i}.pt"), out / "probs" / f"v5_{i}.npz")

    ok = [False] * 6
    for i in members:
        name, lr, bs = MEMBERS[i]
        d = out / "ft" / f"m{i}"; d.mkdir(parents=True, exist_ok=True)
        try:
            ok[i] = finetune(a, i, name, lr, max(2, int(bs * a.bs_mult)), spec, d)
            if ok[i]:
                log(f"3 ft[{i}]: 검증 확률 (TTA{a.tta})")
                val_probs(a, str(d / "best.pt"), out / "probs" / f"ft_{i}.npz")
        except Exception as e:  # noqa: BLE001
            ok[i] = False
            log(f"3 ft[{i}] 예외 — v5 유지: {type(e).__name__}: {e}")
        log(f"   경과 {(time.time() - t0) / 60:.0f}분")

    pick, use = gate(a, out, ok)
    package(a, out, use)
    log(f"전체 끝 ({(time.time() - t0) / 60:.0f}분). 판정: {pick}")


if __name__ == "__main__":
    main()
