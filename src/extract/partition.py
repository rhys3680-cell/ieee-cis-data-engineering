"""원본 CSV를 날짜 파티션 Parquet으로 변환한다.

원본은 6개월치 데이터가 한 파일에 뭉쳐있고 절대시각이 없다. TransactionDT
(특정 시점 기준 초 오프셋)를 날짜로 바꿔 일자별로 쪼갠다. 이 구조가 있어야
증분 적재, Airflow 백필, 파티션 프루닝이 성립한다.

train/test 는 날짜가 겹치지 않으므로 (train 2017-12-02~2018-06-01,
test 2018-07-02~2018-12-31) 한 디렉토리에 함께 쓰고 source_split으로
구분한다.

    data/processed/transactions/dt=2017-12-02/part.parquet
    data/processed/identity/dt=2017-12-02/part.parquet

실행:
    uv run python -m src.extract.partition            # 전체
    uv run python -m src.extract.partition 2018-01-15 # 특정 날짜만
"""

import hashlib
import json
import os
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger
from src.extract.loader import load_raw
from src.extract.schema import LABEL, PARTITION_SOURCE, PRIMARY_KEY, to_transaction_date

logger = get_logger(__name__)

PARTITION_KEY = "transaction_date"
SPLIT_COLUMN = "source_split"
INGESTED_AT = "ingested_at"

DATASETS = ("transactions", "identity")

# train 마지막 날. 이 날짜 이후는 test 다. 두 구간은 겹치지 않고
# 사이에 30일 공백이 있다 (2018-06-02 ~ 07-01).
TRAIN_LAST_DATE = date(2018, 6, 1)


def split_for(dt: date) -> str:
    """날짜가 속한 split. 하루치만 처리할 때 필요없는 파일을 읽지 않으려고 쓴다."""
    return "train" if dt <= TRAIN_LAST_DATE else "test"


def partition_root(dataset: str) -> Path:
    return get_settings().data_processed_dir / dataset


def partition_path(dataset: str, dt: date) -> Path:
    """Hive 스타일 파티션 경로. 실제 시스템에 적용되는지는 확인 필요."""
    return partition_root(dataset) / f"dt={dt.isoformat()}" / "part.parquet"


def _prepare(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 split의 transaction/identity 를 읽고 파티션 키를 붙인다.

    identity 에는 TransactionDT가 없다. transaction에서 날짜를 가져와
    조인으로 붙인다. (TransactionID의 관계는 1:1이다.)
    """
    tx = load_raw(f"{split}_transaction")
    idn = load_raw(f"{split}_identity")

    tx[PARTITION_KEY] = to_transaction_date(tx[PARTITION_SOURCE])
    tx[SPLIT_COLUMN] = split

    # test에는 라벨이 없다. 스키마를 맞추기 위해 NULL 컬럼을 만든다.
    if LABEL not in tx.columns:
        tx[LABEL] = pd.Series([pd.NA] * len(tx), dtype="Int32")

    idn = idn.merge(
        tx[[PRIMARY_KEY, PARTITION_KEY]], on=PRIMARY_KEY, how="left", validate="1:1"
    )
    idn[SPLIT_COLUMN] = split

    return tx, idn


def manifest_path(dataset: str) -> Path:
    """파티션 기록 파일. 한 줄에 파티션 하나."""
    return get_settings().data_processed_dir / "_manifest" / f"{dataset}.jsonl"


def _content_hash(df: pd.DataFrame) -> str:
    """데이터 내용의 해시.

    ingested_at 은 매번 달라지므로 제외한다. 그래야 '다시 돌렸는데 실제로
    내용이 바뀌었는가'를 판단할 수 있다.
    """
    payload = df.drop(columns=[INGESTED_AT], errors="ignore")
    return hashlib.sha256(
        pd.util.hash_pandas_object(payload, index=False).to_numpy().tobytes()
    ).hexdigest()[:16]


def _load_manifest(dataset: str) -> dict[str, dict]:
    """기록된 파티션 상태를 날짜별로 읽는다. 같은 날짜는 마지막 것이 유효하다."""
    path = manifest_path(dataset)
    if not path.exists():
        return {}

    state: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                state[rec["dt"]] = rec
    return state


def _append_manifest(dataset: str, record: dict) -> None:
    """파티션 기록을 덧붙인다. 덮어쓰지 않으므로 변경 이력이 남는다."""
    path = manifest_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_atomic(df: pd.DataFrame, out: Path) -> None:
    """임시 파일에 쓴 뒤 교체한다.

    to_parquet 도중 프로세스가 죽으면 잘린 파일이 남고, 다음 실행이 그것을
    정상 파티션으로 착각한다. os.replace 는 같은 볼륨에서 원자적이므로
    파일은 '이전 내용' 아니면 '새 내용' 둘 중 하나만 된다.
    """
    tmp = out.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False, compression="snappy")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def _write_partitions(df: pd.DataFrame, dataset: str, only: date | None) -> dict:
    """날짜별로 쪼개 Parquet 으로 쓴다.

    멱등성은 세 가지로 확보한다.
      - 파티션 단위 덮어쓰기: 다시 돌려도 행이 늘지 않는다
      - 원자적 교체: 중단되어도 잘린 파일이 남지 않는다
      - 내용 해시 비교: 실제로 바뀐 파티션만 다시 쓴다

    manifest 에 행 수와 해시를 남겨 무엇이 언제 바뀌었는지 추적한다.
    """
    known = _load_manifest(dataset)
    stats = {"written": 0, "skipped": 0}

    for dt, group in df.groupby(PARTITION_KEY, sort=True):
        if only is not None and dt != only:
            continue

        group = group.drop(columns=[PARTITION_KEY])
        digest = _content_hash(group)
        key = dt.isoformat()

        out = partition_path(dataset, dt)
        prev = known.get(key)

        # 내용이 같고 파일도 멀쩡하면 다시 쓸 이유가 없다.
        if prev and prev["checksum"] == digest and out.exists():
            stats["skipped"] += 1
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        group = group.assign(**{INGESTED_AT: pd.Timestamp.now(tz="UTC")})
        _write_atomic(group, out)

        _append_manifest(
            dataset,
            {
                "dt": key,
                "rows": len(group),
                "bytes": out.stat().st_size,
                "checksum": digest,
                "ingested_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "previous_checksum": prev["checksum"] if prev else None,
            },
        )
        stats["written"] += 1

    return stats


def build(only: date | None = None) -> None:
    """원본을 읽어 날짜 파티션으로 사용한다.

    Args:
        only: 이 날짜의 파티션만 쓴다. None이면 전체.
            Airflow가 logical_date로 하루치씩 부를 예정이다. (확인 필요)
    """
    if only is None:
        # 전체 재생성이므로 이전 결과를 지운다. 남겨두면 원본에서 사라진
        # 날짜의 파티션이 그대로 남는다. 덮어쓰기는 존재하는 것만 갱신한다.
        for ds in DATASETS:
            root = partition_root(ds)
            if root.exists():
                logger.info("기존 파티션 삭제: %s", root)
                shutil.rmtree(root)
            manifest_path(ds).unlink(missing_ok=True)

    # 하루치만 만들 때 네 파일을 다 읽을 이유가 없다. 날짜로 split 이 정해진다.
    splits = ("train", "test") if only is None else (split_for(only),)

    total = {ds: {"written": 0, "skipped": 0} for ds in DATASETS}
    for split in splits:
        tx, idn = _prepare(split)

        for ds, frame in (("transactions", tx), ("identity", idn)):
            s = _write_partitions(frame, ds, only)
            total[ds]["written"] += s["written"]
            total[ds]["skipped"] += s["skipped"]

        logger.info("[%s] 파티션 기록 완료", split)

    for ds in DATASETS:
        root = partition_root(ds)
        size = sum(f.stat().st_size for f in root.rglob("*.parquet")) / 1024**2
        count = len(list(root.glob("dt=*"))) if root.exists() else 0
        logger.info(
            "%s: %d 개 파티션, %.0f MB (기록 %d / 변경없음 %d)",
            ds,
            count,
            size,
            total[ds]["written"],
            total[ds]["skipped"],
        )


def read_partition(dataset: str, dt: date) -> pd.DataFrame:
    """파티션 하나를 읽는다. 없으면 빈 DataFrame을 돌려준다.

    train 과 test 사이에 30일 공백이 있어 데이터가 없는 날짜가 존재한다.
    """
    path = partition_path(dataset, dt)
    if not path.exists():
        logger.info("파티션 없음: %s %s", dataset, dt)
        return pd.DataFrame()
    return pd.read_parquet(path)


if __name__ == "__main__":
    target = None
    if len(sys.argv) > 1:
        # 파티션 키는 날짜뿐이라 시각/시간대가 필요없다.
        target = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()  # noqa: DTZ007
        logger.info("대상 날짜: %s", target)

    build(only=target)
