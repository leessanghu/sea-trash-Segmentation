"""데이터 저장 위치.

작업 폴더가 OneDrive 안에 있으므로, 수만 개의 데이터 파일이 동기화되지 않도록
모든 데이터(배경, 합성 데이터, 체크포인트, 제출 폴더)는 OneDrive 밖에 둡니다.

- 로컬(Windows) 기본값: C:\\sea_trash_data
- 코랩: /content/data
- 환경변수 SEA_TRASH_DATA 로 바꿀 수 있습니다
"""
import os
import sys
from pathlib import Path


def _default_root() -> Path:
    if os.environ.get("SEA_TRASH_DATA"):
        return Path(os.environ["SEA_TRASH_DATA"])
    if "google.colab" in sys.modules or Path("/content").is_dir() and os.name != "nt":
        return Path("/content/data")
    return Path(r"C:\sea_trash_data")


DATA_ROOT = _default_root()


def data_path(*parts) -> Path:
    return DATA_ROOT.joinpath(*parts)


def ensure_outside_onedrive(path) -> Path:
    """데이터 출력 경로가 OneDrive 안이면 멈춥니다."""
    p = Path(path).resolve()
    if "onedrive" in str(p).lower():
        raise SystemExit(f"데이터를 OneDrive 안에 만들지 않습니다: {p}\n"
                         f"경로를 지정하지 않으면 기본 위치({DATA_ROOT})에 저장됩니다.")
    return p
