r"""제출 폴더를 만듭니다. (aifactory submit 은 '노트북이 들어 있는 폴더 전체'를 올리므로
학습 데이터·코드가 섞인 작업 폴더에서 바로 제출하면 안 됩니다)

python package.py          # → C:\sea_trash_data\submission 폴더 생성 (OneDrive 밖)
python package.py --zip    # submission.zip 도 생성

진단용 변형 제출 (재학습 없이 후처리 설정만 덮어씀, tune.py 결과 폴더는 그대로):
python package.py --out C:/sea_trash_data/sub_loose --set t_img=0.3          # 판정을 느슨하게 (양성 더 많이)
python package.py --out C:/sea_trash_data/sub_strict --set t_img=0.7         # 판정을 엄격하게
python package.py --out C:/sea_trash_data/sub_empty --empty                  # 빈 예측 → 테스트 양성 비율 추정

구성: predict.ipynb, requirements.txt, NOTICE, LICENSE,
      assets/model/ ← <데이터폴더>/model/ 의 config.json + 가중치 (없으면 베이스라인), assets/code/knps/(추론 모듈)
"""
import argparse
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knps.paths import data_path, ensure_outside_onedrive  # noqa: E402

INFER_MODULES = ["__init__.py", "model.py", "postprocess.py", "rle.py", "anyup_up.py"]   # anyup_up: spec up="anyup" 모델용


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="기본: <데이터폴더>/submission")
    ap.add_argument("--model-dir", default=None, help="tune.py 결과 폴더. 기본: <데이터폴더>/model")
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="후처리 설정 덮어쓰기 (여러 번 가능). 예: --set t_img=0.3 --set t_pix=0.5")
    ap.add_argument("--empty", action="store_true",
                    help="모든 패치를 빈 예측으로 내는 제출 (점수 s → 테스트 양성 비율 r = (1-4s)/(1-2s))")
    a = ap.parse_args()
    a.out = a.out or str(data_path("submission"))
    model_dir = Path(a.model_dir or data_path("model"))
    ensure_outside_onedrive(a.out)
    root = Path(__file__).resolve().parent
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "assets/model").mkdir(parents=True)
    (out / "assets/code/knps").mkdir(parents=True)
    for f in ["predict.ipynb", "NOTICE", "LICENSE"]:
        if (root / f).exists():
            shutil.copy(root / f, out / f)
    # requirements.txt: 줄 끝 주석은 채점 서버 설치 단계에서 패키지 이름으로 잘못 읽혀 설치 전체가 실패할 수 있어 지웁니다
    req = [l.split(" #")[0].rstrip() if not l.lstrip().startswith("#") else l.rstrip()
           for l in open(root / "requirements.txt", encoding="utf-8")]
    with open(out / "requirements.txt", "w", encoding="utf-8", newline="\n") as f:  # 리눅스 서버용 LF
        f.writelines(l + "\n" for l in req if l.strip())
    conf_path = model_dir / "config.json"
    if conf_path.exists():
        conf = json.load(open(conf_path, encoding="utf-8"))
        overrides = {}
        from knps.postprocess import DEFAULT_CFG
        known = {**DEFAULT_CFG, **conf["cfg"]}          # 예전 config 에 없는 새 항목(dilate 등)도 허용
        for kv in a.set:
            k, _, v = kv.partition("=")
            if k not in known:
                raise SystemExit(f"--set {kv}: 알 수 없는 설정 '{k}'. 가능한 키: {sorted(known)}")
            overrides[k] = type(known[k])(float(v)) if isinstance(known[k], (int, float)) else v
        if a.empty:
            overrides["t_img"] = 99.0   # 판정 점수는 0~1 이므로 어떤 패치도 양성이 되지 않음
        if overrides:
            print(f"후처리 설정 덮어쓰기: {overrides} (원래 {({k: known.get(k) for k in overrides})})")
            conf["cfg"].update(overrides)
            conf["overrides"] = overrides
        json.dump(conf, open(out / "assets/model/config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        for m in conf["models"]:          # m = "assets/model/modelK.pt"
            shutil.copy(model_dir / Path(m).name, out / m)
        print(f"튜닝 결과 {conf_path} (모델 {len(conf['models'])}개, 검증 점수 {conf['val']['score']:.4f})")
        for n in conf.get("license_notes", []):
            print("[!] 라이선스 확인:", n)
        if any("DINOv3" in n for n in conf.get("license_notes", [])):
            with open(out / "NOTICE", "a", encoding="utf-8") as f:
                f.write("\n\n[참가자 추가] Built with DINOv3.\n"
                        "이 제출물의 일부 모델은 Meta DINOv3 사전학습 가중치(timm/vit_small_patch16_dinov3.lvd1689m)로\n"
                        "초기화했으며 DINOv3 License 를 따릅니다: "
                        "https://ai.meta.com/resources/models-and-libraries/dinov3-license/\n")
    else:
        print(f"{conf_path} 가 없어 주최 측 베이스라인 가중치로 패키징합니다")
        shutil.copy(root / "assets/model/unet_r18_debris_lite.pt", out / "assets/model/")
    # 제출 도구는 predict.ipynb·requirements.txt·assets/ 만 올리므로, 라이선스 표기(Built with DINOv3 등)가
    # 실제 제출물에 포함되도록 NOTICE·LICENSE 를 assets/ 안에도 둡니다
    for f in ["NOTICE", "LICENSE"]:
        if (out / f).exists():
            shutil.copy(out / f, out / "assets" / f)
    for m in INFER_MODULES:
        shutil.copy(root / "knps" / m, out / "assets/code/knps" / m)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"제출 폴더 {out} ({size / 1e6:.1f} MB)")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(out).as_posix())
    if a.zip:
        z = shutil.make_archive(str(out), "zip", out)
        print("zip:", z)


if __name__ == "__main__":
    main()
