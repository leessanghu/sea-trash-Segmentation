"""노트북에서 스크립트를 실행하는 도구 (로컬 VSCode 노트북용).

VSCode/Jupyter 의 `!python` 은 커널과 다른 파이썬을 부를 수 있으므로,
커널과 같은 파이썬(sys.executable)으로 스크립트를 실행하고 출력을 실시간으로 보여 줍니다.
"""
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(script, *args):
    """python <script> <args...> 를 저장소 폴더에서 실행. 실패하면 예외."""
    cmd = [sys.executable, "-u", str(REPO / script), *map(str, args)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"실패 (exit {p.returncode}): {' '.join(cmd[2:])}")
    print(f"[{script} 완료, {time.time() - t0:.0f}s]")


def has_images(folder) -> bool:
    folder = Path(folder)
    return folder.is_dir() and any(folder.glob("*.png")) or (folder / "images").is_dir() and any((folder / "images").glob("*.png"))


def pack(folder, out_dir) -> Path:
    """폴더를 압축 없이 .tar 하나로 묶습니다 (PNG 는 이미 압축돼 있어 재압축 이득이 거의 없음).

    Drive 에 작은 파일 수만 개를 올리면 매우 느리므로 파일 하나로 올립니다.
    """
    folder, out_dir = Path(folder), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{folder.name}.tar"
    tmp = out_dir / f"{folder.name}.tar.partial"
    n_img = len(list((folder / "images").glob("*.png"))) or len(list(folder.glob("*.png")))  # 배경·조각 폴더는 바로 아래
    n_mask = len(list((folder / "masks").glob("*.png"))) or len(list((folder / "aux").glob("*.png")))
    t0 = time.time()
    # 임시 파일에 다 쓴 뒤에만 이름을 바꿉니다. 중간에 멈춰도 불완전한 .tar 가 완성본처럼 남지 않음
    try:
        with tarfile.open(tmp, "w") as tf:
            tf.add(folder, arcname=folder.name)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    print(f"{out} ({out.stat().st_size / 1e9:.2f} GB, 사진 {n_img}장 / 마스크 {n_mask}장, {time.time() - t0:.0f}s)")
    return out
