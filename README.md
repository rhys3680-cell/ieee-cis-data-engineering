# IEEE-CIS Fraud Detection Pipeline

IEEE-CIS 사기 탐지 데이터로 구축한 배치 데이터 파이프라인.
CSV → GCS(Parquet) → BigQuery → dbt → 모델 학습/추론.

## 스택

Python 3.12 · pandas · GCS · BigQuery · dbt · Airflow · MLflow

## 시작하기

전제: Python 3.12, [uv](https://docs.astral.sh/uv/), [Kaggle API 토큰](https://www.kaggle.com/docs/api), GCP 프로젝트

### 1. 의존성과 데이터

```bash
uv sync
cp .env.example .env          # GCP 프로젝트, 버킷 이름 입력
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw
```

### 2. GCP 인증

서비스 계정 키 대신 ADC(Application Default Credentials)를 쓴다.
2024년 5월 이후 만들어진 조직은 `iam.disableServiceAccountKeyCreation`
정책이 기본값이라 키 발급이 막혀 있고, 키 파일을 주고받지 않는 편이 안전하다.

```bash
gcloud auth login
gcloud config set project <프로젝트-ID>
gcloud auth application-default login
```

버킷과 데이터셋을 만든다. **리전이 서로 같아야 한다** — 다르면 로드가 실패한다.

```bash
gcloud storage buckets create gs://<버킷> --location=asia-northeast3
bq --location=asia-northeast3 mk --dataset <프로젝트-ID>:ieee_raw
```

### 3. 적재

```bash
uv run python -m src.extract.bootstrap      # CSV → GCS (365 × 2 파티션)
uv run python -m src.load.bigquery       # GCS → BigQuery
uv run python -m src.load.bigquery --describe
```

### 4. dbt

`google-cloud-*` 버전이 프로젝트 의존성과 충돌하므로 격리 설치한다.

```bash
uv tool install dbt-core --with dbt-bigquery
cd dbt
cp profiles.yml.example profiles.yml     # 프로젝트 ID 입력
dbt run --profiles-dir .
dbt test --profiles-dir .
```

## 구조

```
src/common/    설정, 로깅
src/extract/   CSV 파싱, 스키마 정의, GCS 적재
src/load/      GCS 업로드, BigQuery 로드
src/ml/        피처, 학습, 추론
dbt/models/    staging · marts
analysis/      일회성 조사 (파이프라인에 포함되지 않음)
dags/          Airflow DAG
```

## 데이터 흐름

```
Kaggle CSV  →  GCS                    BigQuery
               dt=YYYY-MM-DD/         ├ ieee_raw       원본 그대로
               part.parquet           │   ↓ dbt
                                      ├ dev_staging    이름·타입 표준화
                                      │   ↓ dbt
                                      └ dev_mart       팩트 · 집계
```

| 계층 | 책임 | 컬럼 처리 |
| --- | --- | --- |
| raw | 원본 보존 | 손대지 않음 |
| staging | 표준화 | `SELECT * EXCEPT` |
| mart | 소비처와의 계약 | 명시 |

### 마트

| 테이블 | 입자 | 행 |
| --- | --- | --- |
| `fct_transactions` | 거래 한 건 | 1,097,231 |
| `agg_transactions_daily` | 날짜 × 시각 × split × 제품 × 기기 | 61,685 |
| `agg_pipeline_daily` | 날짜 × split | 365 |

## 대시보드

[거래 현황](https://datastudio.google.com/reporting/b8b98f79-97c6-4447-91f7-59f285d9f162) — Looker Studio

집계 마트를 읽는다. 거래량·사기율 추이, 시간대 패턴, 제품·기기별 비교.

`fraud_rate` 를 그대로 평균하면 그룹 가중치가 무시된다.
`SUM(fraud_count) / SUM(labeled_count)` 로 계산해야 한다.
test 구간은 라벨이 없어 사기율이 NULL 이므로 `source_split` 필터의
기본값을 train 으로 둔다.

## 설계 노트

### 시간축

`TransactionDT`는 기준 시점으로부터의 초 오프셋이고, 대회는 그 기준 시점을
공개하지 않았다. 공휴일 패턴으로 역산을 시도했으나 판별력이 없어 실패했다
(`analysis/verify_origin.py`). 기준일은 임의값이며 절대 날짜에 의미가 없다.
날짜 타입을 쓰는 이유는 BigQuery 파티셔닝과 Airflow catchup 을 그대로
쓰기 위해서다.

train(2017-12-02~2018-06-01)과 test(2018-07-02~2018-12-31)는 날짜가 겹치지
않고 사이에 30일 공백이 있다. 파티션이 없는 날짜가 정상적으로 존재한다.

### 멱등성

BigQuery 파티션 데코레이터(`table$YYYYMMDD`)와 `WRITE_TRUNCATE` 로 파티션을
통째로 교체한다. 같은 날짜를 몇 번 로드해도 그 파티션만 바뀐다.

데코레이터 없이 `WRITE_TRUNCATE` 를 쓰면 **테이블 전체가 교체된다.**
하루치를 넣었을 뿐인데 기존 파티션이 전부 사라진다.

### 스키마 검증

BigQuery 는 컬럼이 추가되거나 타입이 다르면 로드를 거부하지만, 컬럼이
빠진 것은 통과시키고 NULL 로 채운다. 소스에서 컬럼이 사라져도 적재가
성공해 며칠 뒤에야 발견하게 되므로, 업로드 전에 컬럼 집합을 대조한다.

### 라벨

test 구간은 `is_fraud` 가 NULL 이다. 실무에서도 최근 거래는 조사 중이라
라벨이 비어 있으므로 예외가 아니라 정상 상태로 다룬다.

## 진행 상황

- [x] 적재 — CSV → GCS → BigQuery, 파티션 단위 멱등
- [x] staging — 이름·타입 표준화, 소스 테스트
- [x] mart — 거래 팩트, 집계 (테스트 24개)
- [x] 대시보드 — 거래 현황 (Looker Studio)
- [ ] Airflow — DAG, 백필
- [ ] ML — 피처 레이어, 시간 분할 학습, MLflow
- [ ] 모델 운영 — 배치 추론, 성능 모니터링, 손실 비용