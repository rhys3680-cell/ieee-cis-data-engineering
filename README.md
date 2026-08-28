# IEEE-CIS Fraud Detection Pipeline

IEEE-CIS 사기 탐지 데이터로 구축한 배치 데이터 파이프라인.
CSV → GCS(Parquet) → BigQuery → dbt → 모델 학습/추론.

## 스택

Python 3.12 · pandas · BigQuery · dbt · Airflow · MLflow

## 시작하기

전제: Python 3.12, [Kaggle API 토큰](https://www.kaggle.com/docs/api), GCP 서비스 계정

​```bash
uv sync
cp .env.example .env          # GCP 프로젝트, 버킷 설정
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw
​```

dbt는 `google-cloud-*` 버전 충돌 때문에 격리 설치한다.

​```bash
uv tool install dbt-core --with dbt-bigquery
​```

## 구조

​```
src/common/   설정, 로깅
src/extract/  CSV 파싱, 스키마
src/load/     GCS·BigQuery 적재
src/ml/       피처, 학습, 추론
dags/         Airflow DAG
dbt/          staging · marts
​```

## 설계 노트

`TransactionDT`는 기준 시점으로부터의 초 오프셋이다. 이를 날짜로 변환해
일자별 파티션으로 재구성하고, 그 위에 증분 적재와 백필을 올린다.
원본이 정적 데이터이므로 실시간 수집은 시뮬레이션이다.
