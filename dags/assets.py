"""DAG 간 의존을 표현하는 Asset 정의.

Asset 은 '이 데이터가 갱신되었다'는 신호다. 생산하는 DAG 이 outlets 로
표시하고, 소비하는 DAG 이 schedule 로 기다린다. TriggerDagRunOperator 로
직접 부르는 것과 달리 양쪽이 서로를 몰라도 된다.
"""

from airflow.sdk import Asset

# ieee_raw 의 파티션이 갱신되었다는 신호. ingest_daily 가 생산하고
# transform 이 소비한다.
RAW_LOADED = Asset("bq://ieee_raw")
