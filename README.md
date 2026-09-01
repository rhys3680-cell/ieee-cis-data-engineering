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

### 5. Airflow

```bash
docker compose up -d          # 첫 실행은 이미지 빌드로 몇 분 걸린다
docker compose ps             # 네 서비스가 running 인지 확인
```

http://localhost:8080 · `admin` / `admin`

DAG는 기본적으로 정지 상태로 만들어진다. UI 에서 `ingest_daily` 를 켜면
2017-12-02 부터 백필이 시작되고, 끝나면 Asset 신호를 받아 `transform` 이
dbt 를 돌린다.

| 서비스                  | 역할                                  |
| ----------------------- | ------------------------------------- |
| `airflow-apiserver`     | UI · API (8080)                       |
| `airflow-scheduler`     | 태스크 스케줄링                       |
| `airflow-dag-processor` | DAG 파일 파싱 (Airflow 3 에서 분리됨) |
| `postgres`              | 메타DB                                |

`.env` 를 compose 가 그대로 읽으므로 1번에서 만든 파일이 있어야 한다.
인증은 호스트의 `%APPDATA%\gcloud` 를 읽기 전용으로 마운트해서 쓴다 —
컨테이너 안에서 다시 로그인할 필요가 없다.

Celery 관련 구성(redis, worker, flower)은 두지 않았다. LocalExecutor 로
충분한 규모다.

```bash
docker compose logs -f airflow-scheduler   # 로그
docker compose down                        # 정지 (메타DB 는 남는다)
docker compose down -v                     # 볼륨까지 삭제
```

### 6. 학습

```bash
uv run python -m src.ml.train                # curated 피처 (9 개)
uv run python -m src.ml.train --all-columns  # 익명 컬럼까지 (394 개)
```

세 모델을 valid 에서 비교하고 지표를 `mlflow.db` 에 남긴다. lgbm 은
`models/*.joblib` 로 저장되어 임계값 탐색과 배치 추론이 재학습 없이 읽는다.

실험 기록을 UI 로 보려면 서버를 띄운다. 프로젝트 의존성에는 트래킹
클라이언트(`mlflow-skinny`)만 있으므로 UI 는 dbt 처럼 격리 설치한다.

```bash
uv tool install mlflow                       # 1회
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

http://localhost:5000 · 실험 이름 `ieee-cis-baseline`

`mlflow` 풀 패키지를 프로젝트 의존성에 넣으면 안 된다. `pandas<3` 을
요구하는데 프로젝트는 pandas 3 을 쓴다. dbt 와 같은 이유의 격리다.

## 구조

```
src/common/    설정, 로깅
src/extract/   CSV 파싱, 스키마 정의, GCS 적재
src/load/      GCS 업로드, BigQuery 로드
src/ml/        조회 · 피처 · 학습 · 모델 저장 · 임계값
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
                                      └ dev_mart       팩트 · 집계 · dim_split
                                            │
                                            ↓  dataset.load(split)
                                        src/ml/
                                          features.build   파생 (학습·추론 공용)
                                          train            더미 / 로지스틱 / LightGBM
                                          model_store      모델 + 범주 목록 + 컬럼
                                          threshold        비용으로 임계값 결정
                                            │
                                            ├→ mlflow.db   실험 기록 (SQLite)
                                            └→ models/     추론용 번들 (joblib)
```

적재와 변환은 Airflow 가 돌리고, 학습은 손으로 돌린다. 배치 추론이 붙으면
(Phase 8) 예측도 DAG 으로 들어간다.

| 계층    | 책임            | 컬럼 처리         |
| ------- | --------------- | ----------------- |
| raw     | 원본 보존       | 손대지 않음       |
| staging | 표준화          | `SELECT * EXCEPT` |
| mart    | 소비처와의 계약 | 명시              |

### 마트

| 테이블                   | 입자                              | 행        |
| ------------------------ | --------------------------------- | --------- |
| `fct_transactions`       | 거래 한 건                        | 1,097,231 |
| `agg_transactions_daily` | 날짜 × 시각 × split × 제품 × 기기 | 61,685    |
| `agg_pipeline_daily`     | 날짜 × split                      | 365       |
| `dim_split`              | 날짜                              | 365       |

## 베이스라인

valid(2018-05-04~06-01, 82,325 행, 사기 2,868 건)에서 잰 값이다.

| 모델             | 피처 | PR-AUC | ROC-AUC |
| ---------------- | ---- | ------ | ------- |
| dummy            | -    | 0.0348 | 0.500   |
| logreg (curated) | 9    | 0.1426 | 0.756   |
| lgbm (curated)   | 9    | 0.1842 | 0.786   |
| lgbm (all)       | 394  | 0.5313 | 0.915   |

더미의 PR-AUC 가 기저 사기율(0.0348)과 일치하고 ROC-AUC 가 0.500 이다.
이론값 그대로라 평가 경로가 정상이라는 뜻이다.

사기율이 3.5% 라 ROC-AUC 는 음성을 맞히는 것만으로도 오른다. 판단은
PR-AUC 로 한다 — 바닥값이 기저 사기율과 같아 "무엇이든 배웠는가"를 바로
읽을 수 있다.

익명 컬럼을 넣으면 PR-AUC 가 2.9배가 된다. 다만 logreg 는 curated 에서만
돌린다. 선형 모델을 두는 이유가 계수를 읽어 관계를 설명하는 것인데, 의미를
모르는 컬럼 378 개를 넣으면 설명할 것이 없어진다.

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
- [x] mart — 거래 팩트, 집계 (테스트 29개)
- [x] 대시보드 — 거래 현황 (Looker Studio)
- [x] Airflow — `ingest_daily`, `transform`, Asset 연결
- [x] 시간 분할 — `dim_split`, `dataset.load` (누수 통제)
- [x] 베이스라인 — 더미 / 로지스틱 / LightGBM, MLflow, 모델 저장
- [ ] 모델 운영 — 배치 추론, 성능 모니터링, 손실 비용
- [ ] 웹 — 임계값 화면 (FastAPI + Cloud Run)
