"""dbt 모델을 빌드하고 테스트한다.

dbt 는 별도 venv 에 있다. dbt-bigquery 가 google-cloud-bigquery <3.3.3 을
요구하는데 Airflow 의 GCP 프로바이더는 더 높은 버전을 쓰기 때문이다.

적재가 끝나야 의미가 있으므로 ingest_daily 가 만든 Asset 을 기다린다.
"""

from airflow.sdk import dag, task
from assets import RAW_LOADED

DBT = "/opt/dbt-venv/bin/dbt"
DBT_DIR = "/opt/project/dbt"

# target/ 의 파싱 캐시를 호스트(Windows)와 컨테이너가 공유한다. 한쪽에서 만든
# 캐시를 다른 쪽이 읽으면 어댑터 매크로를 못 찾고 KeyError 로 죽는다.
# 모델이 다섯 개라 전체 파싱 비용이 작으므로 끈다.
DBT_FLAGS = "--no-partial-parse --profiles-dir ."


@dag(
    dag_id="transform",
    schedule=[RAW_LOADED],
    catchup=False,
    tags=["dbt"],
)
def transform():
    @task.bash
    def dbt_run() -> str:
        return f"cd {DBT_DIR} && {DBT} run {DBT_FLAGS}"

    @task.bash
    def dbt_test() -> str:
        return f"cd {DBT_DIR} && {DBT} test {DBT_FLAGS}"

    dbt_run() >> dbt_test()


transform()
