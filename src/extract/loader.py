"""IEEE-CIS 원본 CSV 로더.

이 단계의 책임은 읽기까지다. 타입 최적화나 파티셔닝은 하지 않는다.
스키마 확정 전이므로 dtype을 지정하지 않고 pandas 추론에 맡긴다.
"""

from pathlib import Path

import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

# 원본 파일명 -> 논리적 이름
SOURCE_FILES = {
    "train_transaction": "train_transaction.csv",
    "train_identity": "train_identity.csv",
    "test_transaction": "test_transaction.csv",
    "test_identity": "test_identity.csv",
}


def raw_path(source: str) -> Path:
    """원본 CSV 경로를 돌려준다."""
    if source not in SOURCE_FILES:
        raise KeyError(f"알 수 없는 소스: {source!r} (가능: {sorted(SOURCE_FILES)})")

    path = get_settings().data_raw_dir / SOURCE_FILES[source]
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. kaggle 데이터를 먼저 내려받아야 한다.")
    return path


def load_raw(source: str, nrows: int | None = None) -> pd.DataFrame:
    """원본 CSV를 그대로 읽는다.

    Args:
        source: SOURCE_FILES 의 키.
        nrows: 앞 N 행만 읽는다. 스키마 탐색용.
    """
    path = raw_path(source)
    logger.info("읽는 중: %s (nrows=%s)", path.name, nrows or "전체")

    df = pd.read_csv(path, nrows=nrows)

    logger.info("완료: %s - %d 행 x %d 열", source, len(df), df.shape[1])
    return df
