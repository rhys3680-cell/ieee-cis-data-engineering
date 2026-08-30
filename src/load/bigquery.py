"""GCS Parquet를 BigQuery 파티션 테이블로 적재한다.

파티션 데코레이터($YYYYMMDD)를 반드시 사용한다. 데코레이터 없이
WRITE_TRUNCATE로 로드하면 BigQuery가 테이블 전체를 교체한다.

멱등성 -> 같은 날짜를 몇 번 로드해도 그 파티션만 교체되므로 중복이 불가능하다.
"""

from datetime import date
from functools import lru_cache

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from src.common.config import get_settings
from src.common.logging import get_logger
from src.extract.schema import bigquery_schema
from src.load.gcs import gcs_uri, list_partitions

logger = get_logger(__name__)

PARTITION_FIELD = "transaction_date"

# 자주 필터하는 컬럼을 클러스터링 키로 둔다. 파티션 안에서 다시 정렬되어
# 스캔량이 줄어든다. 앞의 것을 우선하기 때문에 순서가 중요하다.
CLUSTERING = {
    "transactions": ["source_split", "ProductCD", "card1"],
    "identity": ["source_split", "DeviceType"],
}


def table_id(dataset: str, partition: date | None = None) -> str:
    """정규화된 테이블 이름. partition을 주면 데코레이터를 붙인다."""
    s = get_settings()
    base = f"{s.gcp_project_id}.{s.bq_dataset_raw}.{dataset}"
    if partition is None:
        return base
    return f"{base}${partition:%Y%m%d}"


@lru_cache
def ensure_table(dataset: str) -> bigquery.Table:
    """테이블이 없으면 만든다. 있으면 그대로 돌려준다.

    load_partition 이 매번 부르므로 캐시한다. 365개를 로드할 때 테이블
    존재 확인만 365번 왕복하게 된다.

    미리 만들어 두는 이유는 로드 경로를 하나로 통일하기 위해서다.
    테이블이 없는 상태에서는 파티션 데코레이터를 쓸 수 없어, 첫 로드만
    '테이블 전체 교체'로 분기해야 한다. 그 분기가 있으면 언젠가
    데코레이터 없는 로드가 실수로 남는다.
    """
    client = _client()
    tid = table_id(dataset)

    try:
        return client.get_table(tid)
    except NotFound:
        pass

    table = bigquery.Table(
        tid,
        schema=[bigquery.SchemaField(c, t) for c, t in bigquery_schema(dataset)],
    )
    table.time_partitioning = bigquery.TimePartitioning(field=PARTITION_FIELD)
    table.clustering_fields = CLUSTERING.get(dataset)

    created = client.create_table(table)
    logger.info(
        "테이블 생성: %s (%d 열, 파티션=%s, 클러스터=%s)",
        tid,
        len(created.schema),
        PARTITION_FIELD,
        created.clustering_fields,
    )
    return created


def load_partition(dataset: str, dt: date) -> int:
    """GCS 의 하루치를 BigQuery 파티션 하나로 로드한다. 반환값은 적재된 행 수.

    데코레이터 없이 WRITE_TRUNCATE 를 쓰면 테이블 전체가 교체되므로
    항상 붙인다. 같은 날짜를 몇 번 로드해도 그 파티션만 바뀐다.
    """
    ensure_table(dataset)

    job = _client().load_table_from_uri(
        gcs_uri(dataset, dt),
        table_id(dataset, dt),
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            # 스키마는 테이블이 이미 갖고 있다. 파일이 스키마와 어긋나면
            # 여기서 실패해야 한다.
            schema_update_options=None,
        ),
    )
    job.result()

    logger.info("적재 %s %s: %d 행", dataset, dt, job.output_rows)
    return job.output_rows


def load_all(dataset: str) -> tuple[int, int]:
    """GCS에 올라가 있는 파티션을 전부 로드한다. (파티션 수, 행 수).

    파티션마다 개별 로드 잡을 낸다. 와일드카드로 한 번에 읽을 수도 있지만
    그러면 데코레이터를 쓸 수 없어 테이블 전체가 교체된다. 느려도 파티션
    단위로 넣어야 재실행과 부분 복구가 가능하다.
    """
    ensure_table(dataset)

    partitions = list_partitions(dataset)
    if not partitions:
        logger.warning("%s: GCS 에 파티션이 없다. ingest 를 먼저 돌려야 한다.", dataset)
        return 0, 0

    total = 0
    for i, dt in enumerate(partitions, 1):
        total += load_partition(dataset, dt)
        if i % 30 == 0 or i == len(partitions):
            logger.info("  %s %d/%d (%s)", dataset, i, len(partitions), dt)

    return len(partitions), total


def describe(dataset: str) -> None:
    """테이블 상태를 찍는다. 콘솔에 들어가지 않고 확인하기 위한 것이다."""
    try:
        table = _client().get_table(table_id(dataset))
    except NotFound:
        logger.info("%s: 테이블 없음", dataset)
        return

    # rows 는 BigQuery 예약어라 컬럼명으로 쓸 수 없다.
    q = f"""
        SELECT
          COUNT(*) AS row_count,
          COUNT(DISTINCT {PARTITION_FIELD}) AS partition_count,
          MIN({PARTITION_FIELD}) AS first_date,
          MAX({PARTITION_FIELD}) AS last_date,
          MAX(ingested_at) AS last_ingested
        FROM `{table_id(dataset)}`
    """
    r = _client().query(q).to_dataframe().iloc[0]

    logger.info(
        "%s: %s 행, 파티션 %s개 (%s ~ %s), %.1f MB, 최근 적재 %s",
        dataset,
        f"{r.row_count:,}",
        r.partition_count,
        r.first_date,
        r.last_date,
        table.num_bytes / 1024**2,
        r.last_ingested,
    )


@lru_cache
def _client() -> bigquery.Client:
    return bigquery.Client(project=get_settings().gcp_project_id)


if __name__ == "__main__":
    import sys

    if "--describe" in sys.argv:
        for ds in ("transactions", "identity"):
            describe(ds)
    else:
        for ds in ("transactions", "identity"):
            parts, rows = load_all(ds)
            logger.info("%s 완료: %d 개 파티션, %d 행", ds, parts, rows)
