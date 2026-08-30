"""원본 CSV 를 날짜 파티션 Parquet 으로 GCS 에 올린다.

원본은 6개월치가 한 파일에 뭉쳐 있고 절대시각이 없다. TransactionDT
(특정 시점 기준 초 오프셋)를 날짜로 바꿔 일자별로 쪼갠다.

한 번만 돌린다. 원본이 정적이므로 다시 돌릴 이유가 없고, 이후 파이프라인은
CSV 가 아니라 GCS 의 파티션에서 시작한다. DAG 이 CSV 를 직접 읽으면
하루치를 만들려고 매번 1.3GB 를 파싱하게 된다.

로컬 파일을 만들지 않는다. DataFrame 을 메모리에서 직렬화해 바로 올린다.
GCS 는 객체 단위로 원자적이라 업로드가 끊기면 객체가 만들어지지 않는다.

train/test 는 날짜가 겹치지 않으므로 같은 데이터셋에 함께 올리고
source_split 으로 구분한다. 사이에 공백이 있어 파티션이 없는 날짜가 있다.

    gs://<bucket>/transactions/dt=2017-12-02/part.parquet
    gs://<bucket>/identity/dt=2017-12-02/part.parquet

실행:
    uv run python -m src.extract.ingest
"""

import pandas as pd

from src.common.logging import get_logger
from src.extract.loader import load_raw
from src.extract.schema import LABEL, PARTITION_SOURCE, PRIMARY_KEY, to_transaction_date
from src.load.gcs import upload_partition

logger = get_logger(__name__)

PARTITION_KEY = "transaction_date"
SPLIT_COLUMN = "source_split"
INGESTED_AT = "ingested_at"

DATASETS = ("transactions", "identity")


def _prepare(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 split 의 transaction/identity 를 읽고 파티션 키를 붙인다.

    identity 에는 TransactionDT 가 없다. transaction 에서 날짜를 가져와
    조인으로 붙인다. TransactionID 가 1:1 이고 고아 레코드가 없다는 것은
    profile 에서 확인했고, 조인 무결성은 dbt relationships 테스트가 본다.
    """
    tx = load_raw(f"{split}_transaction")
    idn = load_raw(f"{split}_identity")

    now = pd.Timestamp.now(tz="UTC")

    tx[PARTITION_KEY] = to_transaction_date(tx[PARTITION_SOURCE])
    tx[SPLIT_COLUMN] = split
    tx[INGESTED_AT] = now

    # test 에는 라벨이 없다. 스키마를 맞추기 위해 NULL 컬럼을 만든다.
    if LABEL not in tx.columns:
        tx[LABEL] = pd.Series([pd.NA] * len(tx), dtype="Int32")

    idn = idn.merge(
        tx[[PRIMARY_KEY, PARTITION_KEY]], on=PRIMARY_KEY, how="left", validate="1:1"
    )
    idn[SPLIT_COLUMN] = split
    idn[INGESTED_AT] = now

    return tx, idn


def _upload_all(df: pd.DataFrame, dataset: str) -> tuple[int, int]:
    """날짜별로 쪼개 GCS 에 올린다. (파티션 수, 총 바이트).

    순차로 올린다. 부트스트랩은 한 번 돌리고 결과를 확인하는 작업이라
    속도보다 관찰 가능성이 중요하다. 병렬로 올려도 파티션끼리 겹치지
    않아 정합성은 같지만, 실패했을 때 어디까지 갔는지 알기 어렵다.
    """
    groups = list(df.groupby(PARTITION_KEY, sort=True))
    total = 0

    for i, (dt, group) in enumerate(groups, 1):
        total += upload_partition(group, dataset, dt)
        if i % 30 == 0 or i == len(groups):
            logger.info("  %s %d/%d (%s)", dataset, i, len(groups), dt)

    return len(groups), total


def ingest() -> None:
    """원본을 읽어 GCS 파티션으로 올린다."""
    for split in ("train", "test"):
        tx, idn = _prepare(split)

        for ds, frame in (("transactions", tx), ("identity", idn)):
            parts, size = _upload_all(frame, ds)
            logger.info(
                "[%s] %s: %d 개 파티션, %.0f MB", split, ds, parts, size / 1024**2
            )


if __name__ == "__main__":
    ingest()
