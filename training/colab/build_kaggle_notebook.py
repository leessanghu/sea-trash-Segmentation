"""Kaggle 용 전체 재학습 노트북 (GPU T4×2 → 멤버를 두 장에 나눠 동시 학습).

python build_kaggle_notebook.py   ->  ../../retrain_kaggle.ipynb

Kaggle 차이점
- 구글 드라이브 없음: 팀 공개 tar·v5 가중치는 gdown 으로 받고(인터넷 켜기 필요),
  우리가 만든 추가 데이터 tar 는 Kaggle Dataset 으로 올려 노트북 입력(/kaggle/input/...)으로 붙임
- 결과는 /kaggle/working (최대 20GB) → "Save Version → Save & Run All" 로 돌리면 세션을 꺼도 끝까지 돌고 출력이 보존됨
- 세션 최대 12시간, 주 30시간
"""
import base64
import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V5_IDS = ["1PgscENu9Pxr-zRL7i131rXIuwT7lJnga", "1IxW-ZUyyLqBKNH5QPjJxaPpgE-0SFHLe", "1G23IXeXRDenigAB4aQ44jNESjIT1xsWh",
          "1mWtj2n0YXrThiviFCnFGcMKV3le7m3JW", "1LkVeGi6swbQb7G4GcpYflJHsEJpNNfy1", "1DbILk9np83-bUhSs-D5-ZRS3wGZRfpXW"]
EXPECT_CODE = "20261006-0330-planC-C2"   # Dataset sea-trash-code 에 올라간 코드 버전 (다르면 노트북이 바로 멈춤)
TARS = {
    "real_pos_train": "1cXXCzcczO0hfT2Ibl0628QACOQfwZ0Lg", "real_val": "1J-4vhyy4qs1zCGNlZZonV-cPByjALBkt",
    "real_neg_train": "1wPLE3zdvrknGvOzoypj0pnE43qIpEZOm", "cp_pos_train": "1aT-4JPgrlA7iCZ5f8GRXMJQ8XT3qY0NN",
    "cp_band_train": "1cN5GZAvh4KJALsMMy3tDcYhd_BbiianT", "debris_pieces": "1Da6pYlb3F8gp4nBk2F3CBoQ_C14rv4qS",
    "bg_naip_train": "12a1Y0xD2lxfhubkD92dpxTST_rywypZ5", "synth_train": "1qz-ewLEtflpo08x9qOtoy2-xtP-Z1TB7",
}


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in ["train.py", "exp_eval.py", "hnm_pipeline.py", "run_retrain.py", "soup.py", "find_noisy_synth.py", "package.py"]:
            t.add(REPO / f, arcname=f)
        for p in sorted((REPO / "knps").glob("*.py")):
            t.add(p, arcname=f"knps/{p.name}")
        t.add(HERE / "sub_tpl", arcname="sub_tpl")
        t.add(REPO / "assets/rhm_ref.npy", arcname="assets/rhm_ref.npy")
        t.add(REPO / "assets/anyup.pth", arcname="assets/anyup.pth")
    return base64.b64encode(buf.getvalue()).decode()


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {"trusted": True}, "execution_count": None, "outputs": [], "source": s}


def main():
    global A
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="E", help="E: 전체 재학습 / S: 형상 집중")
    ap.add_argument("--loss", default="dice", choices=["dice", "tolf"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    def fill(c):
        if c["cell_type"] == "code":
            c["source"] = (c["source"].replace("__TARS__", json.dumps(TARS)).replace("__V5_IDS__", json.dumps(V5_IDS))
                           .replace("__EXPECT_CODE__", EXPECT_CODE))
        return c

    cells = [
        md("""# 해안쓰레기 전체 재학습 — Kaggle (GPU T4 ×2)

**처음 한 번만 설정** (오른쪽 패널 Settings)
1. **Accelerator → GPU T4 x2**
2. **Internet → On** (휴대폰 인증 필요. 패키지·사전학습 가중치·팀 데이터 다운로드에 필요)
3. **Add Input → Datasets**: `sea-trash-code`, `sea-trash-team`, `sea-trash-extra`, `sea-trash-extra2` (이미 연결돼 있으면 그대로)

**실행:** 오른쪽 위 **Save Version → Save & Run All (Commit)**. 브라우저를 닫아도 끝까지 돌고, 결과가 버전 출력으로 저장됩니다.
(편집 화면에서 Run All 로 돌리면 창을 닫는 순간 멈춥니다.)

**이어 돌리기:** 12시간 안에 다 못 끝내면, 다음 버전에서 **Add Input → 이 노트북의 이전 버전 출력**을 붙이고 다시 Save & Run All.
끝난 멤버는 이전 출력에서 가져와 건너뜁니다.

**결과:** Output 탭의 `retrain_<PLAN>/gate.md`(v5 비교표)와 `retrain_<PLAN>/submission/`(제출 폴더). 제출은 하지 않습니다.
"""),
        md("## 0. 설정"),
        code(f"""PLAN = "{A.plan}"          # E(전체 재학습) / S(형상 집중) / A / B / D
MEMBERS = ""        # 비우면 Plan 전부. 일부만 쉼표로
EPOCHS, STEPS = {A.epochs}, {A.steps}
AUG2 = 1            # 증강 2판 (히스토그램 무작위 매칭 + 0.3m 위성 센서 모사)
LOSS = "{A.loss}"       # dice(기본) / tolf(대회 형상점수 1화소 허용 F 를 그대로 손실로)"""),
        md("## 1. 경로, 패키지, GPU"),
        code("""import os, sys, json, shutil, time, subprocess, glob
T_START = time.time()
def log(*a):
    # 모든 줄 앞에 한국시간 + 시작 후 경과(분). Kaggle 로그에서 어디서 멈췄는지 바로 보이게
    kst = time.strftime("%H:%M:%S", time.gmtime(time.time() + 9 * 3600))
    print(f"[{kst} +{(time.time() - T_START) / 60:5.1f}분]", *a, flush=True)

OUT = f"/kaggle/working/retrain_{PLAN}"      # 결과 (버전 출력으로 보존, 20GB 한도)
L = "/tmp/data"                               # 학습 데이터 (심볼릭 링크, 출력에 안 남음)
V5 = "/tmp/v5_models"
REPO = "/tmp/sea_trash"
for d in (OUT, L, V5): os.makedirs(d, exist_ok=True)
log("1단계 시작: 패키지 설치")
!pip install -q "segmentation-models-pytorch>=0.5" "timm>=1.0.30" gdown
import torch
assert torch.cuda.is_available(), "GPU 가 없습니다 — Settings → Accelerator 를 GPU T4 x2 로"
NGPU = torch.cuda.device_count()
log("GPU", NGPU, "장:", [torch.cuda.get_device_name(i) for i in range(NGPU)])
!df -h /tmp /kaggle/working | tail -2
log("1단계 끝")"""),
        md("## 2. 입력 확인 + 코드 풀기 (Dataset `sea-trash-code`)"),
        code("""import tarfile as _tf
# 데이터 위치를 정확한 경로로 바로 확인합니다.
# (예전 판은 /kaggle/input 전체를 재귀로 훑어서 그림 수십만 장 때문에 폴더 하나 찾는 데 23분씩 걸렸음 — 이제 안 함)
# Kaggle 은 tar 를 자동으로 풀어 /kaggle/input/<데이터셋>/<이름>/<이름>/ 으로 둡니다.
ROOTS = ["/kaggle/input", "/kaggle/input/datasets/leessssang"]
DATASETS = ["sea-trash-code", "sea-trash-team", "sea-trash-extra", "sea-trash-extra2"]

def locate(name, need=None):
    # 붙은 데이터셋 4개 × 경로 3가지만 직접 확인 (재귀 탐색 안 함 → 즉시 끝남)
    for r in ROOTS:
        for ds in DATASETS:
            for c in (f"{r}/{ds}/{name}/{name}", f"{r}/{ds}/{name}"):
                if os.path.isdir(c) and (need is None or os.path.exists(os.path.join(c, need))):
                    return c
            c = f"{r}/{ds}/{name}.tar"
            if os.path.isfile(c):
                return c
    return None

log("2단계 시작: 붙은 입력 데이터셋 확인")
for ds in DATASETS:
    roots = [f"{r}/{ds}" for r in ROOTS if os.path.isdir(f"{r}/{ds}")]
    if roots:
        log(f"  ✅ {ds:18s} {roots[0]}  →  {sorted(os.listdir(roots[0]))}")
    else:
        log(f"  ❌ {ds:18s} 안 붙어 있음")
missing = [ds for ds in DATASETS if not any(os.path.isdir(f"{r}/{ds}") for r in ROOTS)]
assert not missing, f"입력 데이터셋이 빠졌습니다: {missing} — 오른쪽 Input → Add Input 에서 붙이세요"

EXPECT_CODE = "__EXPECT_CODE__"
tgz = next((c for r in ROOTS for c in (f"{r}/sea-trash-code/sea_trash_code.tgz",) if os.path.isfile(c)), None)
assert tgz, "sea_trash_code.tgz 가 없습니다 — Dataset 'sea-trash-code' 확인"
if os.path.isdir(REPO): shutil.rmtree(REPO)
os.makedirs(REPO)
with _tf.open(tgz, "r:gz") as t:
    try: t.extractall(REPO, filter="data")
    except TypeError: t.extractall(REPO)
for f in ("train.py", "run_retrain.py", "hnm_pipeline.py", "sub_tpl/predict.ipynb", "assets/anyup.pth", "assets/rhm_ref.npy"):
    assert os.path.exists(f"{REPO}/{f}"), f"코드 묶음에 {f} 가 없습니다"
ver = open(f"{REPO}/CODE_VERSION").read().strip() if os.path.exists(f"{REPO}/CODE_VERSION") else "표시 없음"
log("코드 버전:", ver)
assert ver == EXPECT_CODE, f"코드 버전이 다릅니다 (기대 {EXPECT_CODE}, 실제 {ver}) — 데이터셋이 아직 갱신 중이면 잠시 뒤 다시"
log("2단계 끝: 코드 준비 완료")"""),
        md("## 3. 데이터 붙이기 (정확한 경로로 바로 연결, 없을 때만 구글 드라이브) + 이전 버전 결과"),
        code("""import gdown, tarfile

def fetch(fid, dst, tries=4):
    for k in range(tries):
        try:
            gdown.download(id=fid, output=dst, quiet=True)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                return
        except Exception as e:
            log(f"  다운로드 실패({k + 1}/{tries}): {type(e).__name__} — 30초 뒤 재시도")
        time.sleep(30)
    raise RuntimeError(f"다운로드 실패: {dst}")

def untar(tar, dst):
    with tarfile.open(tar) as tf:
        try: tf.extractall(dst, filter="data")
        except TypeError: tf.extractall(dst)

def n_png(d):
    # 그림 장수만 빠르게 셈 (scandir 한 번)
    d = os.path.join(d, "images") if os.path.isdir(os.path.join(d, "images")) else d
    with os.scandir(d) as it:
        return sum(1 for e in it if e.name.endswith(".png") and not e.name.startswith("._"))

TEAM = __TARS__          # 이름 → 구글 드라이브 id (Dataset 에 없을 때만 사용)
NEED = {"bg_naip_train": "aux", "debris_pieces": "manifest.csv"}
EXTRA = ["cp_typed_train", "cp_typed_train_b", "cp_typed_train_c", "real_landneg_train",
         "bg_ports_train", "bg_hawaii_train", "bg_coast2_train", "mudflat_nl"]
rows = []

def attach(name, need="images", fid=None):
    t = time.time()
    dst = f"{L}/{name}"
    if os.path.lexists(dst):
        src = "이미 있음"
    else:
        c = locate(name, need)
        if c and c.endswith(".tar"):
            untar(c, L); src = c
        elif c:
            os.symlink(c, dst); src = c
        elif fid:
            tar = f"{L}/{name}.tar"
            fetch(fid, tar); untar(tar, L); os.remove(tar); src = "구글 드라이브"
        else:
            log(f"  ⚠️ {name:20s} 없음 → 제외"); rows.append((name, 0, "없음")); return
    n = n_png(dst) if os.path.isdir(dst) else 0
    log(f"  {name:20s} {n:7d}장  {time.time() - t:5.1f}초  ← {src}")
    rows.append((name, n, src))

log("3단계 시작: 데이터 연결")
# v5 가중치
c = locate("v5_models")
if c and glob.glob(os.path.join(c, "model*.pt")):
    V5 = c
else:
    for i, fid in enumerate(__V5_IDS__):
        p = f"{V5}/model{i}.pt"
        if not os.path.exists(p) or os.path.getsize(p) < 10_000_000:
            fetch(fid, p)
log(f"  v5 가중치 {len(glob.glob(os.path.join(V5, 'model*.pt')))}개 ← {V5}")

log("팀 데이터:")
for name, fid in TEAM.items():
    attach(name, NEED.get(name, "images"), fid)
log("추가 데이터:")
for name in EXTRA:
    attach(name)

REQUIRED = ["real_pos_train", "real_val", "real_neg_train", "debris_pieces", "bg_naip_train"]
lack = [r for r, n, _ in rows if r in REQUIRED and (n == 0 and r != "debris_pieces")]
lack += [r for r in REQUIRED if not os.path.lexists(f"{L}/{r}")]
assert not lack, f"꼭 필요한 데이터가 없습니다: {sorted(set(lack))}"

# 이전 버전 출력(이어 돌리기): 끝난 멤버를 가져옴 (Add Input 으로 이전 버전 출력을 붙였을 때만)
for prev in sorted(glob.glob(f"/kaggle/input/*/retrain_{PLAN}") + glob.glob(f"/kaggle/input/*/*/retrain_{PLAN}")):
    for sub in ("members", "probs"):
        src = os.path.join(prev, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(OUT, sub), dirs_exist_ok=True)
    if os.path.exists(os.path.join(prev, "mine.json")):
        shutil.copy(os.path.join(prev, "mine.json"), OUT)
    log("이전 버전 결과 가져옴:", prev)
log(f"3단계 끝: 데이터 {sum(1 for _, n, _ in rows if n)}개 폴더, 그림 합계 {sum(n for _, n, _ in rows):,}장")"""),
        md("## 3-1. 학습 전 데이터 점검 (깨진 그림이 있으면 GPU 쓰기 전에 멈춤)"),
        code("""sys.path.insert(0, REPO)
from knps.data import pngs, _imread
import random
log("3-1단계 시작: 폴더마다 그림 30장씩 실제로 열어 봄")
bad = []
for d in sorted(glob.glob(f"{L}/*")):
    if not os.path.isdir(d): continue
    t = time.time()
    files = pngs(d) + (pngs(os.path.join(d, "images")) if os.path.isdir(os.path.join(d, "images")) else [])
    sample = random.Random(0).sample(files, min(30, len(files)))
    nb = 0
    for f in sample:
        try:
            _imread(f)
        except Exception as e:
            bad.append(str(e)); nb += 1
    log(f"  {os.path.basename(d):22s} 그림 {len(files):7d}장  깨짐 {nb}/{len(sample)}  {time.time() - t:4.1f}초")
assert not bad, "읽을 수 없는 그림:\\n" + "\\n".join(bad[:10])
log("3-1단계 끝: 데이터 점검 통과 → 이제 GPU 작업 시작")"""),
        md("## 4. 채굴 (한 번만) → 멤버를 GPU 장수만큼 나눠 동시에 학습 → 비교 → 제출 폴더"),
        code("""%cd {REPO}
sys.path.insert(0, REPO)
import run_retrain as RR
import re, signal
members = [m for m, _, _ in RR.PLANS[PLAN]]
if MEMBERS:
    members = [m for m in members if m in set(MEMBERS.split(","))]
base = [sys.executable, "-u", "run_retrain.py", "--repo", REPO, "--data", L, "--out", OUT, "--plan", PLAN, "--v5", V5,
        "--tpl", f"{REPO}/sub_tpl", "--epochs", str(EPOCHS), "--steps", str(STEPS), "--workers", "2",
        "--tta", "8", "--infer-bs", "16", "--val-bs", "16", "--aug2", str(AUG2), "--loss", LOSS]
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
STOP_AT_H = 10.4      # 시작 후 10.4시간이 되면 아직 학습 중인 멤버를 멈추고 끝난 멤버로 비교표를 만듦 (12시간 제한 전에 결과 확보)

def last_line(path):
    # 로그 끝에서 의미 있는 마지막 한 줄 (진행 막대의 \\r 도 줄로 취급)
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 4000))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    parts = [x.strip() for x in re.split(r"[\\r\\n]", txt) if x.strip()]
    return parts[-1][:160] if parts else ""

def run_logged(cmd, path, tag):
    # 채굴·비교처럼 혼자 도는 단계: 출력을 노트북 로그에 그대로 흘리고 파일에도 남김
    with open(path, "a") as lf:
        p = subprocess.Popen(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            lf.write(line); lf.flush()
            print(f"  [{tag}] {line}", end="", flush=True)
        return p.wait()

log(f"4단계 시작: 플랜 {PLAN}, 손실 {LOSS}, 멤버 {len(members)}개 {members}, 에폭 {EPOCHS} × 스텝 {STEPS}")
# 4-1 채굴 (멤버 없이: 어려운 예제 폴더만 만듦)
log("4-1 채굴 시작")
rc = run_logged(base + ["--skip-train"], f"{OUT}/mine.log", "채굴")
assert rc == 0, f"채굴 실패 (종료코드 {rc}) — 위 [채굴] 줄 확인"
log("4-1 채굴 끝")

# 4-2 멤버를 GPU 에 번갈아 배정해 동시에 학습 (각 GPU 는 자기 몫을 순서대로)
lanes = [members[g::NGPU] for g in range(NGPU)]
procs = []
for g, lane in enumerate(lanes):
    if not lane: continue
    logf = open(f"{OUT}/lane{g}.log", "a")
    env = {**ENV, "CUDA_VISIBLE_DEVICES": str(g)}
    procs.append((g, lane, subprocess.Popen(base + ["--members", ",".join(lane), "--no-gate"], env=env, stdout=logf,
                                            stderr=subprocess.STDOUT, start_new_session=True)))
    log(f"4-2 GPU{g} 학습 시작: {lane}")
t0, stopped = time.time(), False
while any(p.poll() is None for _, _, p in procs):
    time.sleep(120 if time.time() - t0 < 1800 else 300)      # 처음 30분은 2분마다, 이후 5분마다
    fin = [m for m in members if os.path.exists(f"{OUT}/members/{m}/done")]
    log(f"학습 {(time.time() - t0) / 60:.0f}분째 — 끝난 멤버 {len(fin)}/{len(members)} {fin}")
    for g, lane, p in procs:
        st = "끝" if p.poll() is not None else "도는 중"
        log(f"   GPU{g} ({st}): {last_line(f'{OUT}/lane{g}.log')}")
    if not stopped and time.time() - T_START > STOP_AT_H * 3600:
        log(f"⏰ 시작 후 {STOP_AT_H}시간 — 남은 학습을 멈추고 끝난 멤버로 마무리합니다")
        for _, _, p in procs:
            if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)    # 자식 train.py 까지 같이 멈춤
        stopped = True
for g, lane, p in procs:
    log(f"GPU{g} 종료코드 {p.returncode} — 로그 {OUT}/lane{g}.log")
    !tail -5 {OUT}/lane{g}.log

# 4-3 끝난 멤버 전부로 v5 비교표 + 제출 폴더
log("4-3 비교표·제출 폴더 만들기 시작")
run_logged(base + ["--skip-train"], f"{OUT}/gate.log", "비교")
log("4단계 끝")"""),
        md("## 5. 비교표"),
        code("""from IPython.display import Markdown, display
p = f"{OUT}/gate.md"
if os.path.exists(p):
    display(Markdown(open(p, encoding="utf-8").read()))
    !du -sh {OUT}/submission
else:
    print("비교표가 없습니다 (끝난 멤버 없음). lane 로그를 확인하세요:", OUT)
# 출력 용량 한도(20GB) 관리: 학습 중간 파일(resume.pt, last.pt)은 지움 — 이어 돌리기는 best.pt + done 으로 충분
for f in glob.glob(f"{OUT}/members/*/resume.pt") + glob.glob(f"{OUT}/members/*/last.pt"):
    if os.path.exists(os.path.join(os.path.dirname(f), "done")):
        os.remove(f)
!du -sh {OUT}"""),
    ]
    cells = [fill(c) for c in cells]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                       "language_info": {"name": "python"},
                                       "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
          "nbformat": 4, "nbformat_minor": 4}
    out = Path(A.out) if A.out else REPO.parent / "retrain_kaggle.ipynb"
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(out, len(cells), "셀", out.stat().st_size // 1000, "KB")


if __name__ == "__main__":
    main()
