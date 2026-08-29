"""IEEE-CIS 원본 CSV 로더.

이 단계의 책임은 읽기까지다. 파티셔닝이나 변환은 하지 않는다.
dtype 은 schema 모듈이 정한 것을 쓴다.
메모리 3437MB -> 2007MB (-42%)
"""

from pathlib import Path

import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger
from src.extract.schema import (
    identity_dtypes,
    normalize_columns,
    transaction_dtypes,
)

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


def dtypes_for(source: str) -> dict[str, str]:
    """소스별 dtype 매핑. test 에는 라벨 컬럼이 없다."""
    if "identity" in source:
        return identity_dtypes()
    return transaction_dtypes(with_label=source.startswith("train"))


def load_raw(source: str, nrows: int | None = None) -> pd.DataFrame:
    """원본 CSV 를 schema 가 정한 dtype 으로 읽는다.

    identity 파일은 test 쪽 컬럼명이 id-01 처럼 하이픈을 쓰므로 먼저
    언더스코어로 정규화한 뒤 dtype 을 적용한다.

    Args:
        source: SOURCE_FILES 의 키.
        nrows: 앞 N 행만 읽는다. 탐색용.
    """
    path = raw_path(source)
    logger.info("읽는 중: %s (nrows=%s)", path.name, nrows or "전체")

    # 컬럼명이 어긋난 상태로는 dtype 을 못 넘긴다. 헤더만 먼저 보고 매핑을 만든다.
    header = pd.read_csv(path, nrows=0).columns
    renames = normalize_columns(header)

    dtypes = dtypes_for(source)

    # boolean 은 read_csv 로 직접 못 받는다. 원본이 T/F 인데 pandas 는
    # True/False 나 1/0 만 인식한다. 문자열로 읽어서 뒤에 변환한다.
    bool_cols = [c for c, d in dtypes.items() if d == "boolean"]
    read_dtypes = {c: ("str" if d == "boolean" else d) for c, d in dtypes.items()}

    # 정규화 전 이름으로 dtype 을 걸어야 read_csv 를 사용 가능하다.
    inverse = {v: k for k, v in renames.items()}
    read_dtypes = {inverse.get(c, c): d for c, d in read_dtypes.items()}

    df = pd.read_csv(path, nrows=nrows, dtype=read_dtypes)
    if renames:
        df = df.rename(columns=renames)
        logger.info(
            "컬럼명 정규화: %d 개 (예: %s)", len(renames), next(iter(renames.items()))
        )

    for c in bool_cols:
        df[c] = df[c].map({"T": True, "F": False}).astype("boolean")

    mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("완료: %s - %d 행 x %d 열, %.0f MB", source, len(df), df.shape[1], mem)
    return df
