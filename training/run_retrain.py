"""전체 재학습 (Plan A: 모델 키우기 / Plan B: 데이터로 승부) — 멤버 하나씩 처음부터 학습 → 앙상블을 v5 와 비교 → 제출 폴더.

단계 (모두 --out 에 저장, 다시 실행하면 끝난 단계는 건너뜀 → 세션이 끊겨도 이어짐)
  1 mine    v5 6개(TTA1)로 학습용 음성/양성을 추론해 어려운 예제 폴더 생성 (hnm_pipeline.mine 재사용, 검증셋 미사용)
  2 train   멤버별 학습 (--init 없음 = 사전학습 가중치에서 처음부터). resume.pt 있으면 이어서, OOM 이면 배치 절반.
            끝난 멤버는 probs/<멤버>.npz (검증 TTA8 확률) 저장
  3 gate    끝난 멤버 전부의 균등 앙상블 vs v5 앙상블(--v5probs 의 v5_*.npz, 없으면 --v5 가중치로 계산), 후처리 고정
  4 package 끝난 멤버로 제출 폴더 (--tpl 틀 + config.json 갱신)

Plan D (DiWA, arXiv 2205.09739 / Model soups ICML'22): 계열(family)마다
  2a init   백본을 얼리고(--backbone-lr-mult 0) 디코더·헤드만 S0 스텝 학습 → members/<계열>/init/init.pt (공유 초기화, LP 에 해당)
  2b runs   init.pt 에서 K 번 학습, 런마다 순한 흔들기: 시드, lr x{1, 0.7, 1.4}, 데이터 비중 x[0.8,1.25], 붙여넣기 비중,
            혼동 타일 배수, 증강 세기(EXP_AUG_STRENGTH). LoRA/DoRA 계열은 init 위에 다시 LoRA 를 붙여 학습(--reapply-lora)
  2c soup   K 런의 last.pt(기본, 검증으로 고르지 않음)를 soup.py 로 균등 평균 → members/<계열>/best.pt (일반 체크포인트 1개)
  → gate·package 는 기존과 같음 (계열 하나 = 제출 멤버 하나). --diag 1 이면 런별·런 앙상블 점수도 gate 표에 추가

python run_retrain.py --repo /content/sea_trash --data /content/data --out <드라이브>/retrain_A --plan A \
    --v5 <v5 폴더> --v5probs <드라이브>/hnm/probs --tpl /content/sea_trash/sub_tpl [--members dino3_b,dino_b_dora]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hnm_pipeline as H  # noqa: E402  (mine, val_probs, run_logged, infer, log 재사용)

# 멤버: (preset, lr, bs). T4 16GB 기준 배치. OOM 이면 자동 절반.
PLANS = {
    # 근거: v5 멤버 단독 점수 dv3-LoRA .787 > dv2-DoRA .764 > dv3 .744 > dv2 .725 > r18 .713 > cnxt .686 (STATE 10-05)
    #       → DINO 계열을 Base 로 키우고 가장 약한 cnxt·dv2 제외. r18 은 싸고 HNM 효과를 LB 에서 보여 유지.
    "A": [("dino3_s_lora", 3e-4, 16), ("dino_s_dora", 3e-4, 16), ("dino3_b", 3e-4, 12),
          ("dino_b_dora", 3e-4, 8), ("dino3_s_560", 3e-4, 16), ("unet_r18_base_x2", 5e-4, 32)],
    # v5 구조 그대로, 데이터만 바꿈 (어려운 음성·육지 음성·갯벌 처음부터 포함)
    "B": [("unet_r18_base_x2", 5e-4, 32), ("unet_cnxt_t_x2", 5e-4, 16), ("dino_s", 3e-4, 16),
          ("dino3_s", 3e-4, 16), ("dino3_s_lora", 3e-4, 16), ("dino_s_dora", 3e-4, 16)],
    # [DiWA] 계열 3개 x K 런 수프 = 제출 멤버 3개 (v5 단독 상위 2개 DINO 계열 + 값싸고 구조가 다른 r18).
    # 추론 시간은 v5 6멤버의 약 절반 → 남는 시간은 TTA8 유지/큰 백본에 씀. 약한 cnxt·dv2 전체학습은 뺌 (STATE 10-05)
    "D": [("dino3_s_lora", 3e-4, 16), ("dino_s_dora", 3e-4, 16), ("unet_r18_base_x2", 5e-4, 32)],
    # Plan E: 논문 6편 정독 종합 (STATE 10-05, scratchpad/deep/)
    #   - 약한 멤버 제거(cnxt, DINOv2 전체 FT): 빼면 로컬 +0.002/+0.008, DiWA/Abe: 앙상블 이득은 멤버 정확도에서 옴
    #   - DINOv3 논문 + 우리 고정특징 탐침: S@512 가 S@448 보다 일관되게 나음(+0.009/+0.014), S+ 는 못함 → S@512 추가
    #   - SoMA(CVPR'25): LoRA 를 소성분 SVD 초기화 + 앞 3블록 동결 + AWD 로. LoRA 대비 이득은 작을 수 있어 LoRA 와 나란히 (다양성)
    #   - r18: 싸고, HNM 교체가 LB 에서 효과(0.7079→0.7195) 본 멤버
    "E": [("dino3_s_lora", 3e-4, 16), ("dino3_s_512_lora", 3e-4, 12), ("dino3_s_soma", 3e-4, 16),
          ("dino_s_dora", 3e-4, 16), ("dino_s_soma", 3e-4, 16), ("unet_r18_base_x2", 5e-4, 32)],
    # Plan S: 형상(②너무 넓게 잡음) 집중. --loss tolf 와 함께 쓰는 용도.
    #   - 실패 분석: 로컬 양성 100장 중 '너무 넓음' 44 > '좋음' 38. 형상 0.638, 오라클 탐지여도 0.826 상한.
    #   - tol_f 손실은 대회 형상점수와 Pearson 1.000 (STATE 10-05) 인데 아직 한 번도 학습에 안 써봄.
    #   - 비용이 비슷한 4멤버만 골라 T4 2장에 2개씩 → 에폭을 더 줄 수 있음 (512 는 혼자 2.4h 라 제외).
    "S": [("dino3_s_lora", 3e-4, 16), ("dino3_s_soma", 3e-4, 16),
          ("dino_s_dora", 3e-4, 16), ("unet_r18_base_x2", 5e-4, 32)],
    # Plan C: CNN 판정 멤버 강화 (STATE 10-06 02:50). LB 5점이 판정 앙상블의 CNN 비중과 같이 움직임,
    #   PANGAEA 단일클래스·MADOS(해양쓰레기)에서 UNet > ViT. 레인 배분: [0::2]=무거운 cnxt_s + 가벼운 r18, [1::2]=cnxt_t + effv2_s
    #   레인: [0::2] = cnxt_s, effv2_s, r34 / [1::2] = cnxt_t, r18, effb3
    "C": [("unet_cnxt_s_x2", 5e-4, 8), ("unet_cnxt_t_x2", 5e-4, 16), ("unet_effv2_s_x2", 5e-4, 12),
          ("unet_r18_base_x2", 5e-4, 32), ("unet_r34_x2", 5e-4, 24), ("unet_effb3_x2", 5e-4, 16)],
    # Plan C2: 형상용 CNN (UNet++ 촘촘한 스킵 연결로 경계 정밀) — 판정은 C, 마스크 보강은 C2.
    #   MiT(SegFormer) 는 ImageNet 가중치가 NVIDIA SegFormer 라이선스(비상업)라 제외.
    #   레인: [0::2] = unetpp_r50, unetpp_effb3 / [1::2] = unetpp_r34, unet_r50 (ConvNeXt 는 UNet++ 와 구조가 안 맞음)
    "C2": [("unetpp_r50_x2", 5e-4, 12), ("unetpp_r34_x2", 5e-4, 16),
           ("unetpp_effb3_x2", 5e-4, 12), ("unet_r50_x2", 5e-4, 16)],
}
# 플랜별 멤버 시작점·시드 (없으면 기본: r18 은 v5 model0, 나머지는 ImageNet)
PLAN_INIT = {("C", "unet_cnxt_t_x2"): "model1.pt"}     # v5 cnxt(0.7195 구성 멤버)에서 이어 학습 — hnm-b r18 과 같은 방식
PLAN_SEED = {"C": 1, "C2": 2}                                     # Plan E 의 r18(시드 0)과 다른 멤버가 되도록
DIWA_PLANS = {"D"}
LR_MULTS = (1.0, 0.7, 1.4)          # DiWA 의 순한 범위({1,3,5}e-5, 5배) 보다 더 좁게 (2배)


def train_spec(a, r):
    """v5 비중(합 0.75) 위에 어려운 음성·육지 음성·갯벌을 끼워 넣고 합을 0.75 로 다시 맞춤 (붙여넣기 0.25 비중 유지)."""
    L = Path(a.data)
    w = {"real_pos_train": 0.15, "cp_pos_train": 0.08, "cp_band_train": 0.08, "real_neg_train": 0.22, "synth_train": 0.08}
    w["real_hardneg_train" if len(r["neg"]) >= 5 else "real_neg_train"] = w.get("real_hardneg_train", 0) + 0.12
    if len(r["pos"]) >= 3:
        w["real_hardpos_train"] = 0.04
    # 선택 데이터: 육지·마을 음성, 갯벌 음성, 해외 포구·갯벌·해변 배경(음성), 그 배경 위 붙여넣기 양성
    # cp_typed_train = 유형별 증강 (make_typed_aug.py: 물가 띠·구조물 발치·스티로폼·대형 더미 + 같은 장면 음성 40%)
    # cp_typed_train_b = 물가 띠·구조물·포구 위주 2차 세트
    # cp_typed_train_c = 확장 배경(NAIP 175곳 0.3m, 포구 8.5k) 위 유형별 증강, bg_coast2/bg_hawaii = 새 음성 배경
    for name, wt, fallback in (("real_landneg_train", 0.06, "real_neg_train"), ("mudflat_nl", 0.02, "real_neg_train"),
                               ("bg_ports_train", 0.04, "real_neg_train"), ("bg_coast2_train", 0.02, "real_neg_train"),
                               ("bg_hawaii_train", 0.02, "real_neg_train"),
                               ("cp_typed_train", 0.05, "cp_pos_train"), ("cp_typed_train_b", 0.04, "cp_band_train"),
                               ("cp_typed_train_c", 0.07, "cp_pos_train")):
        if (L / name / "images").is_dir():
            w[name] = wt
        else:
            H.log(f"  {name} 없음 → 비중을 {fallback} 에 합침")
            w[fallback] += wt
    s = sum(w.values())
    return ",".join(f"{L / k}:{v * 0.75 / s:.4f}" for k, v in w.items())


def paste_spec(a):
    L = Path(a.data)
    bgs = [L / "bg_naip_train", L / "real_neg_train"]
    for name in ("real_landneg_train", "bg_ports_train"):
        if (L / name / "images").is_dir():
            bgs.append(L / name)
    return ",".join(str(p) for p in [L / "debris_pieces"] + bgs)


def jitter_spec(spec, rng, amp=1.25):
    """'폴더:비중,...' 의 비중을 소스마다 x[1/amp, amp] (로그 균등) 로 흔들고 합을 원래대로 맞춤."""
    parts = [p.rpartition(":") for p in spec.split(",") if p]
    w = np.array([float(x) for _, _, x in parts])
    w2 = w * np.exp(rng.uniform(-np.log(amp), np.log(amp), len(w)))
    w2 *= w.sum() / w2.sum()
    return ",".join(f"{p}:{v:.4f}" for (p, _, _), v in zip(parts, w2))


def train_member(a, preset, lr, bs, spec, d, init=None, seed=None, paste_weight=0.25, epochs=None, steps=None,
                 extra=(), env_extra=None):
    if (d / "done").exists() and (d / "best.pt").exists():
        return True
    env = {**os.environ, "EXP_LOSS": a.loss, "EXP_GT_DILATE": "0", "EXP_VAL_BS": str(a.val_bs), "PYTHONUNBUFFERED": "1",
           "EXP_AUG2": "1" if a.aug2 else "0", "EXP_RHM_REF": str(Path(a.repo) / "assets/rhm_ref.npy"), **(env_extra or {})}
    L = a.data
    for attempt in range(3):
        cmd = [sys.executable, "train.py", "--model", preset, "--train", spec, "--val", f"{L}/real_val",
               "--paste", paste_spec(a), "--paste-weight", str(paste_weight), "--test-pos-ratio", "0.64",
               "--lr", str(lr), "--epochs", str(epochs or a.epochs), "--steps", str(steps or a.steps), "--bs", str(bs),
               "--workers", str(a.workers), "--seed", str(a.seed if seed is None else seed), "--out", str(d)] + list(extra)
        if a.no_pretrained:
            cmd.append("--no-pretrained")
        if init:
            cmd += ["--init", str(init)]
        elif preset.startswith("unet_r18_base") and not (Path(a.repo) / "assets/model/unet_r18_debris_lite.pt").exists():
            cmd += ["--init", str(Path(a.v5) / "model0.pt")]   # 베이스라인 가중치가 없으면 v5 r18 에서 시작
            H.log(f"  {preset}: 베이스라인 없음 → v5 model0.pt 에서 시작")
        if (d / "resume.pt").exists():
            cmd.append("--resume")
            H.log(f"  {preset}: resume.pt 발견 → 이어서")
        H.log(f"2 train {preset}: bs {bs}, lr {lr:g}, {a.epochs}x{a.steps}")
        rc, tail = H.run_logged(cmd, env, d / "train.log", a.repo)
        if rc == 0 and (d / "best.pt").exists():
            (d / "done").write_text("ok")
            return True
        if any(m in tail for m in H.OOM) and bs > 2:
            bs //= 2
            for f in ("resume.pt", "last.pt", "best.pt"):
                (d / f).unlink(missing_ok=True)
            H.log(f"  {preset}: 메모리 부족 → bs {bs} 로 재시도")
            continue
        H.log(f"  {preset} 실패 (rc={rc}). 로그 {d}/train.log\n{tail[-800:]}")
        return False
    return False


def run_jitter(k, fam_seed):
    """K 런의 순한 흔들기. 0번 런은 기본값 그대로(=Plan A/B 멤버와 같은 설정)."""
    rng = np.random.default_rng(fam_seed * 1000 + k)
    if k == 0:
        return {"seed": fam_seed * 100, "lr_mult": 1.0, "paste_weight": 0.25, "confuser": 2.0, "aug": 1.0, "rng": rng, "mix": False}
    return {"seed": fam_seed * 100 + k, "lr_mult": LR_MULTS[k % 3] if k < 3 else float(rng.choice(LR_MULTS)),
            "paste_weight": float(rng.choice([0.2, 0.25, 0.3])), "confuser": float(rng.choice([1.5, 2.0, 2.5])),
            "aug": float(rng.choice([0.8, 1.0, 1.2])), "rng": rng, "mix": True}


def train_family(a, fi, preset, lr, bs, spec, d):
    """DiWA 계열 하나: 공유 초기화 → K 런 → 균등 수프 → d/best.pt. 성공하면 True."""
    from soup import make_soup
    if (d / "done").exists() and (d / "best.pt").exists():
        return True
    fam_seed = a.seed + fi + 1
    # 2a 공유 초기화: 백본 고정, 디코더·헤드만. 검증으로 고르지 않고 마지막 가중치(last.pt) 사용
    d0 = d / "init"; d0.mkdir(parents=True, exist_ok=True)
    init = d0 / "init.pt"
    if not init.exists():
        H.log(f"2a init {preset}: 백본 고정, {a.s0_epochs}x{a.s0_steps or a.steps} 스텝")
        if not train_member(a, preset, lr, bs, spec, d0, seed=fam_seed * 100, epochs=a.s0_epochs, steps=a.s0_steps or a.steps,
                            extra=["--backbone-lr-mult", "0"]):
            return False
        shutil.copy(d0 / "last.pt", init)
    # 2b K 런
    is_lora = preset in ("dino3_s_lora", "dino_s_dora", "dino_b_dora")
    ok = []
    for k in range(a.K):
        j = run_jitter(k, fam_seed)
        rs = jitter_spec(spec, j["rng"]) if j["mix"] else spec
        dk = d / f"run{k}"; dk.mkdir(exist_ok=True)
        json.dump({kk: v for kk, v in j.items() if kk != "rng"} | {"train": rs, "lr": lr * j["lr_mult"]},
                  open(dk / "jitter.json", "w"), indent=1)
        H.log(f"2b {preset} run{k}/{a.K}: seed {j['seed']} lr x{j['lr_mult']} paste {j['paste_weight']} "
              f"confuser {j['confuser']} aug x{j['aug']} 비중흔들기 {j['mix']}")
        extra = ["--confuser-boost", str(j["confuser"])] + (["--reapply-lora"] if is_lora else [])
        if train_member(a, preset, lr * j["lr_mult"], bs, rs, dk, init=init, seed=j["seed"], paste_weight=j["paste_weight"],
                        extra=extra, env_extra={"EXP_AUG_STRENGTH": str(j["aug"])}):
            ok.append(dk)
            if a.diag:
                H.val_probs(a, str(dk / f"{a.soup_ckpt}.pt"), d.parent.parent / "probs" / f"{preset}__run{k}.npz")
    if len(ok) < 2:
        H.log(f"  {preset}: 끝난 런 {len(ok)}개 → 수프 불가 (2개 이상 필요)")
        return False
    # 2c 균등 수프 (검증이 LB 를 못 맞추므로 greedy 를 쓰지 않음)
    H.log(f"2c soup {preset}: {len(ok)}개 런의 {a.soup_ckpt}.pt 균등 평균")
    make_soup([dk / f"{a.soup_ckpt}.pt" for dk in ok], d / "best.pt")
    (d / "done").write_text("ok")
    return True


def gate(a, out, done):
    from scipy import ndimage
    from exp_eval import FIXED, score
    from knps.data import TINY_PX, contest_gts
    from knps.postprocess import postprocess
    pr = out / "probs"
    v5p = Path(a.v5probs) if a.v5probs else pr
    v5 = []
    for i in range(6):
        f = v5p / f"v5_{i}.npz"
        if not f.exists():
            H.log(f"3 gate: v5 멤버 {i} 확률 없음 → 계산")
            H.val_probs(a, str(Path(a.v5) / f"model{i}.pt"), f)
        v5.append(np.load(f))
    G = v5[0]["G"]
    keep, Gc = contest_gts(G, TINY_PX)
    Gd = np.stack([ndimage.binary_dilation(g, np.ones((3, 3), bool)) if g.any() else g for g in Gc])
    gneg = ~Gc.reshape(len(Gc), -1).any(1)

    def ev(P, C):
        m = [postprocess(p, float(c), FIXED) for p, c in zip(P, C)]
        pp = np.array([int(x.sum()) >= 50 for x in m])
        return {"local639": score(m, Gc, 0.639), "gtdil639": score(m, Gd, 0.639), "fp_neg": int((pp & gneg).sum())}

    res = {"v5": ev(sum(z["P"][keep].astype(np.float32) for z in v5) / 6, sum(z["C"][keep] for z in v5) / 6)}
    new = {n: np.load(pr / f"{n}.npz") for n in done}
    for n, z in new.items():
        assert (z["G"] == G).all(), f"{n}: 검증 정렬 불일치"
        res[f"single:{n}"] = ev(z["P"][keep].astype(np.float32), z["C"][keep])
        runs = [np.load(f) for f in sorted(pr.glob(f"{n}__run*.npz"))]   # [DiWA] 런별 / 런 예측앙상블(ENS) vs 수프
        for k, r in enumerate(runs):
            res[f"  run{k}:{n}"] = ev(r["P"][keep].astype(np.float32), r["C"][keep])
        if len(runs) > 1:
            res[f"  ENS{len(runs)}:{n}"] = ev(sum(r["P"][keep].astype(np.float32) for r in runs) / len(runs),
                                            sum(r["C"][keep] for r in runs) / len(runs))
    if new:
        res[f"plan{a.plan}:{len(new)}멤버"] = ev(sum(z["P"][keep].astype(np.float32) for z in new.values()) / len(new),
                                                sum(z["C"][keep] for z in new.values()) / len(new))
        # 새 멤버 전부 + v5 전부 (혼합) 도 참고로
        allz = list(new.values()) + v5
        res[f"plan{a.plan}+v5:{len(allz)}멤버"] = ev(sum(z["P"][keep].astype(np.float32) for z in allz) / len(allz),
                                                   sum(z["C"][keep] for z in allz) / len(allz))
    b = res["v5"]
    rows = ["| 구성 | local639 | gtdil639 | F1 | 형상 | 음성오탐 | v5 대비 |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        l, g = r["local639"], r["gtdil639"]
        d = "" if k == "v5" else f"{l['score'] - b['local639']['score']:+.4f} / {g['score'] - b['gtdil639']['score']:+.4f}"
        rows.append(f"| {k} | {l['score']:.4f} | {g['score']:.4f} | {l['f1']:.3f} | {l['shape']:.3f} | {r['fp_neg']} | {d} |")
    md = "\n".join(rows) + (f"\n\n끝난 멤버 {len(done)}/{len(PLANS[a.plan])}: {', '.join(done)}\n"
                            "후처리 고정(v4cfg+dil1), TTA8. 로컬(드론 검증셋)은 큰 변화의 방향만 봅니다 — 작은 차이(±0.01)는 LB 를 예측 못 합니다 (STATE 10-05).\n")
    json.dump(res, open(out / "gate.json", "w"), indent=1, ensure_ascii=False)
    open(out / "gate.md", "w", encoding="utf-8").write(md)
    print(md, flush=True)


def package(a, out, done):
    sub = out / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    shutil.copytree(a.tpl, sub)
    for f in (sub / "assets/model").glob("model*.pt"):
        f.unlink()
    # 제출 폴더의 추론 모듈을 지금 코드로 교체 (틀의 model.py 는 v5 시절 판 — tap_norm·AnyUp 등 새 spec 을 못 읽음)
    from package import INFER_MODULES
    for m in INFER_MODULES:
        src_m = Path(a.repo) / "knps" / m
        if src_m.exists():
            shutil.copy(src_m, sub / "assets/code/knps" / m)
    conf = json.load(open(Path(a.tpl) / "assets/model/config.json"))
    import torch
    models, specs, srcs = [], [], []
    for k, n in enumerate(done):
        src = out / "members" / n / "best.pt"
        shutil.copy(src, sub / "assets/model" / f"model{k}.pt")
        models.append(f"assets/model/model{k}.pt")
        specs.append(torch.load(src, map_location="cpu", weights_only=False)["spec"])
        srcs.append(str(src))
    conf.update({"models": models, "specs": specs, "sources": srcs, "weights": [1.0] * len(models),
                 "retrain": {"plan": a.plan, "members": done}})
    if any("dinov3" in json.dumps(s) for s in specs):
        conf.setdefault("license_notes", [])
        if not any("DINOv3" in n for n in conf["license_notes"]):
            conf["license_notes"].append("DINOv3 License (Meta): 재배포 시 라이선스 사본 동봉 + 'Built with DINOv3' 표기 의무")
    if any(s.get("up") == "anyup" for s in specs):   # AnyUp 코드·가중치 CC-BY-4.0: 출처 표기 의무
        note = ("\n\n[참가자 추가] AnyUp feature upsampler (Wimmer et al., ICLR 2026), https://github.com/wimmerth/anyup, "
                "code and weights licensed under CC BY 4.0. knps/anyup_up.py is a modified port.\n")
        for nf in (sub / "NOTICE", sub / "assets" / "NOTICE"):
            if nf.exists():
                with open(nf, "a", encoding="utf-8") as fh:
                    fh.write(note)
    json.dump(conf, open(sub / "assets/model/config.json", "w"), indent=2, ensure_ascii=False)
    size = sum(p.stat().st_size for p in sub.rglob("*") if p.is_file())
    H.log(f"4 package: {sub} ({size / 1e6:.0f} MB, 멤버 {len(models)}개, 후처리 {conf['cfg']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", required=True, choices=list(PLANS))
    ap.add_argument("--v5", required=True, help="v5 model0~5.pt 폴더 (채굴용)")
    ap.add_argument("--v5probs", default="", help="hnm/probs 처럼 v5_i.npz 가 있는 폴더 (있으면 재계산 생략)")
    ap.add_argument("--tpl", required=True)
    ap.add_argument("--members", default="", help="쉼표로 preset 이름 일부만 (계정 나눠 돌릴 때)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--infer-bs", type=int, default=16)
    ap.add_argument("--val-bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="테스트용: 채굴 장수 제한")
    ap.add_argument("--no-pretrained", action="store_true", help="테스트용")
    ap.add_argument("--skip-train", action="store_true", help="학습 없이 gate·package 만 (끝난 멤버로)")
    ap.add_argument("--loss", default="dice", choices=["dice", "tolf"],
                    help="tolf: 대회 형상점수(1화소 허용 F)를 그대로 손실로. 형상이 병목일 때")
    ap.add_argument("--aug2", type=int, default=1, help="1: 증강 2판 (RHM + 0.3m 위성 센서 모사, knps/augment.py)")
    ap.add_argument("--no-gate", action="store_true", help="학습만 하고 비교·패키징은 건너뜀 (GPU 여러 장 동시 실행용)")
    ap.add_argument("--bs-scale", type=float, default=1.0,
                    help="PLANS 배치에 곱함 (T4 16GB 기준값. 4090 24GB 는 1.5, 테스트·작은 GPU 는 0.25 등)")
    ap.add_argument("--K", type=int, default=3, help="[Plan D] 계열당 런 수 (T4: 2~3, RTX 4090: 4~5 권장)")
    ap.add_argument("--s0-epochs", type=int, default=1, help="[Plan D] 공유 초기화(백본 고정) 에폭")
    ap.add_argument("--s0-steps", type=int, default=0, help="[Plan D] 공유 초기화 에폭당 스텝 (0 = --steps)")
    ap.add_argument("--soup-ckpt", default="last", choices=["last", "best"],
                    help="[Plan D] 런마다 수프에 넣을 가중치. last = 검증으로 고르지 않음(기본), best = 검증 최고 에폭")
    ap.add_argument("--diag", type=int, default=1, help="[Plan D] 1: 런별 검증 확률도 저장해 gate 에 런·ENS 행 추가")
    a = ap.parse_args()
    out = Path(a.out); (out / "probs").mkdir(parents=True, exist_ok=True); (out / "members").mkdir(exist_ok=True)
    os.chdir(a.repo)
    t0 = time.time()

    r = H.mine(a, out)
    spec = train_spec(a, r)
    H.log(f"학습 구성: {spec}\n붙여넣기: {paste_spec(a)} (0.25)")

    members = PLANS[a.plan]
    only = set(filter(None, a.members.split(",")))
    for preset, lr, bs0 in members:
        if a.skip_train or (only and preset not in only):
            continue
        bs = max(2, int(bs0 * a.bs_scale))
        d = out / "members" / preset; d.mkdir(exist_ok=True)
        try:
            fam = a.plan in DIWA_PLANS
            ini = PLAN_INIT.get((a.plan, preset))
            ini = str(Path(a.v5) / ini) if ini else None
            sd = PLAN_SEED.get(a.plan)
            if (train_family(a, members.index((preset, lr, bs0)), preset, lr, bs, spec, d) if fam
                    else train_member(a, preset, lr, bs, spec, d, init=ini, seed=sd)):
                H.log(f"  {preset}: 검증 확률 (TTA{a.tta})")
                H.val_probs(a, str(d / "best.pt"), out / "probs" / f"{preset}.npz")
        except Exception as e:  # noqa: BLE001
            H.log(f"  {preset} 예외: {type(e).__name__}: {e}")
        H.log(f"   경과 {(time.time() - t0) / 60:.0f}분")

    done = [p for p, _, _ in members if (out / "members" / p / "done").exists() and (out / "probs" / f"{p}.npz").exists()]
    H.log(f"끝난 멤버 {len(done)}/{len(members)}: {done}")
    if a.no_gate:
        H.log("--no-gate: 비교·패키징은 마지막에 한 번만 (다른 GPU 작업과 겹치지 않게)")
    elif done:
        gate(a, out, done)
        package(a, out, done)
    else:
        H.log("끝난 멤버가 없어 비교·패키징을 건너뜁니다")
    H.log(f"전체 끝 ({(time.time() - t0) / 60:.0f}분)")


if __name__ == "__main__":
    main()
