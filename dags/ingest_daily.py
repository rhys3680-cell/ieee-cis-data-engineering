"""GCS 파티션을 BigQuery로 하루치씩 적재한다.

원본 CSV는 부트스트랩(src/extract/bootstrap.py)에서 이미 GCS로 올렸다.
이 DAG는 거기서 시작한다. CSV를 직접 읽으면 하루치를 만들려고 매번 데이터가 큰
CSV를 파싱하게 된다.

멱등하다. BigQuery 파티션 데코레이터로 그 파티션만 교체하므로 같은 날짜를
몇 번 돌려도 중복이 생기지 않는다. 실패한 태스크를 재시도해도 안전하다.

train과 test 사이에 30일 공백이 있어(2018-06-02~07-01) 데이터가 없는
날짜가 존재한다. 실패가 아니라 정상이므로 skip으로 표시한다.
"""

import pendulum
from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowSkipException
from assets import RAW_LOADED

from src.load.bigquery import load_partition
from src.load.gcs import blob_exists

DATASETS = ("transactions", "identity")


@dag(
    dag_id="ingest_daily",
    schedule="@daily",
    start_date=pendulum.datetime(2017, 12, 2, tz="UTC"),
    end_date=pendulum.datetime(2018, 12, 31, tz="UTC"),
    catchup=True,
    # 백필 365회가 BigQuery 로드 잡을 한꺼번에 몰지 않도록 제한한다.
    max_active_runs=4,
    default_args={"retries": 2},
    tags=["ingest"],
)
def ingest_daily():
    @task(outlets=[RAW_LOADED])
    def load(dataset: str, **context) -> int:
        """하루치 파티션을 BigQuery 로 로드한다."""
        dt = context["logical_date"].date()

        if not blob_exists(dataset, dt):
            raise AirflowSkipException(f"{dataset} {dt}: GCS 에 파티션 없음")

        return load_partition(dataset, dt)

    for ds in DATASETS:
        load.override(task_id=f"load_{ds}")(ds)


ingest_daily()
