FROM apache/airflow:3.3.1-python3.12

# dbt는 별도 venv에 넣는다. dbt-bigquery가 google-cloud-bigquery <3.3.3을
# 요구하는데 Airflow의 GCP 프로바이더는 상위 버전을 사용한다. 환경을 분리해야 한다.
USER root
RUN python -m venv /opt/dbt-venv && \
    /opt/dbt-venv/bin/pip install --no-cache-dir \
        dbt-core==1.12.3 \
        dbt-bigquery==1.12.0 && \
    chown -R airflow:root /opt/dbt-venv

USER airflow

# GCS/BigQuery를 다루는 프로바이더와 프로젝트 의존성
RUN pip install --no-cache-dir \
        apache-airflow-providers-google \
        pandas \
        pyarrow

ENV DBT_BIN=/opt/dbt-venv/bin/dbt