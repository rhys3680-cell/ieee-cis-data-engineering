"""하루치를 채점해 predictions 파티션에 쓴다.

mart 가 갱신되어야 의미가 있으므로 transform 이 만든 Asset 을 기다린다.
ingest_daily 와 같은 구조다 — 하루치, 파티션 데코레이터, 재시도 안전.

라벨이 없는 구간(test)을 채점하는 것이 정상 동작이다. 실무에서도 최근 거래는
조사 중이라 라벨이 비어 있고, 사기 점수는 그 상태에서 매긴다.
"""

import pendulum
from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowSkipException
from assets import MART_READY

from src.ml.batch import run


@dag(
    dag_id="predict_daily",
    schedule=[MART_READY],
    start_date=pendulum.datetime(2018, 7, 2, tz="UTC"),
    catchup=False,
    # 모델을 한 번 내려받아 여러 태스크가 쓰도록 동시 실행을 낮게 둔다.
    max_active_runs=2,
    default_args={"retries": 2},
    tags=["ml"],
)
def predict_daily():
    @task
    def score_day(**context) -> int:
        """그날 거래에 점수를 매긴다.

        거래가 없는 날짜가 정상적으로 존재한다(train/test 사이 30일 공백).
        실패로 두면 백필에서 빨간색이 늘어 진짜 실패를 가린다.
        """
        dt = context["logical_date"].date()
        rows = run(dt)

        if rows == 0:
            raise AirflowSkipException(f"{dt}: 채점할 거래 없음")
        return rows

    score_day()


predict_daily()