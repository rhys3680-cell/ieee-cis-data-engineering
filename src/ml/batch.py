"""하루치를 채점해 BigQuery 에 쓴다.

    uv run python -m src.ml.batch 2018-07-02

거래가 들어오는 즉시 판단하는 실시간 추론이 아니라 배치다. TransactionDT 가
임의 기준의 초 오프셋이라 "지금"이라는 개념이 없고 새 거래도 들어오지 않는다.
하루치를 모아 채점하고 테이블에 쌓는 형태가 이 데이터에 맞는다.

**점수만 저장하고 판정은 하지 않는다.** 임계값은 비용 가정에 따라 4배 넘게
움직이는데, 판정을 구워 넣으면 가정이 바뀔 때마다 50만 건을 다시 채점해야
한다. 소비처가 점수에 임계값을 적용하면 재추론이 필요 없고, 소비처마다 다른
값을 쓸 수도 있다.

멱등하다. 파티션 데코레이터로 그 파티션만 교체하므로 같은 날짜를 몇 번
돌려도 중복이 생기지 않는다.
"""

from datetime import date, datetime

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from src.common.config import get_settings
from src.common.logging import get_logger
from src.ml.dataset import DROP_COLUMNS
from src.ml.model_store import Bundle, load_from_gcs
from src.ml.predict import score

logger = get_logger(__name__)

MODEL = "lgbm-all"
TABLE = "predictions"
PARTITION_FIELD = "transaction_date"

SCHEMA = [
    bigquery.SchemaField("transaction_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField(PARTITION_FIELD, "DATE", mode="REQUIRED"),
    bigquery.SchemaField("score", "FLOAT", mode="REQUIRED"),
    # 어느 모델이 매긴 점수인지. 없으면 점수 분포가 바뀌었을 때 모델이
    # 바뀐 것인지 데이터가 바뀐 것인지 구분할 수 없다.
    bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_trained_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("scored_at", "TIMESTAMP", mode="REQUIRED"),
]


def table_id(partition: date | None = None) -> str:
    s = get_settings()
    base = f"{s.gcp_project_id}.{s.bq_dataset_raw}.{TABLE}"
    if partition is None:
        return base
    # $YYYYMMDD 가 없으면 WRITE_TRUNCATE 가 테이블 전체를 교체한다.
    return f"{base}${partition:%Y%m%d}"


def ensure_table() -> bigquery.Table:
    """없으면 만든다. 미리 만들어야 첫 로드도 데코레이터를 쓸 수 있다."""
    client = _client()
    try:
        return client.get_table(table_id())
    except NotFound:
        pass

    table = bigquery.Table(table_id(), schema=SCHEMA)
    table.time_partitioning = bigquery.TimePartitioning(field=PARTITION_FIELD)
    created = client.create_table(table)
    logger.info("테이블 생성: %s", table_id())
    return created


def read_day(dt: date) -> pd.DataFrame:
    """그날 거래를 읽는다.

    transaction_id 와 transaction_date 는 결과에 필요하므로 남기고, 피처가
    될 수 없는 것은 학습 경로와 같은 목록(DROP_COLUMNS)으로 걷어낸다. 두
    곳에서 따로 관리하면 학습에 없던 컬럼이 추론 입력에 섞인다.
    """
    s = get_settings()
    q = f"""
        select t.*,
               mod(div(t.transaction_dt, 3600), 24) as transaction_hour,
               i.transaction_id is not null         as has_identity
        from `{s.gcp_project_id}.dev_staging.stg_transactions` t
        left join `{s.gcp_project_id}.dev_staging.stg_identity` i
               using(transaction_id)
        where t.transaction_date = @dt
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("dt", "DATE", dt)]
    )
    return _client().query(q, job_config=job_config).to_dataframe()


def run(dt: date, bundle: Bundle | None = None) -> int:
    """하루치를 채점해 파티션 하나로 쓴다. 반환값은 행 수."""
    bundle = bundle or load_from_gcs(MODEL)
    df = read_day(dt)

    if df.empty:
        # train/test 사이 30일 공백처럼 데이터가 없는 날짜가 정상적으로
        # 존재한다. 부르는 쪽이 skip 할지 정한다.
        logger.info("%s: 거래 없음", dt)
        return 0

    # 식별자와 날짜는 결과에 쓰고 피처에서는 뺀다.
    keys = df[["transaction_id", "transaction_date"]]
    features = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    scores = score(features, bundle)
    out = pd.DataFrame(
        {
            "transaction_id": keys["transaction_id"],
            PARTITION_FIELD: keys["transaction_date"],
            "score": scores,
            "model": MODEL,
            "model_trained_at": bundle.trained_at,
            "scored_at": datetime.now(tz=None).astimezone(),
        }
    )

    ensure_table()
    job = _client().load_table_from_dataframe(
        out,
        table_id(dt),
        job_config=bigquery.LoadJobConfig(
            schema=SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()

    logger.info("%s: %s 행 채점 (평균 %.3f)", dt, f"{len(out):,}", float(scores.mean()))
    return len(out)


def _client() -> bigquery.Client:
    return bigquery.Client(project=get_settings().gcp_project_id)


if __name__ == "__main__":
    import sys

    run(date.fromisoformat(sys.argv[1]))