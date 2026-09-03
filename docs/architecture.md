# 아키텍처

## 목적과 범위

IEEE-CIS Fraud Detection 데이터를 대상으로 하는 배치 사기 탐지 파이프라인이다.
원본 CSV를 보존 가능한 저장소와 분석용 웨어하우스로 적재하고, 변환·학습·배치
추론·임계값 화면까지 연결한다. 이 문서는 **현재 구현**의 책임과 의존성, 데이터
계약, 운영 흐름을 기록한다.

## 시스템 흐름

```text
Kaggle CSV
  │  src.extract.bootstrap
  ▼
GCS Parquet (dataset / dt=YYYY-MM-DD / part.parquet)
  │  src.load.bigquery, Airflow ingest_daily
  ▼
BigQuery ieee_raw
  │  dbt staging
  ▼
BigQuery dev_staging
  ├─────────────────────► src.ml.dataset / features / train
  │                         │
  │                         ▼
  │                       models/*.joblib → GCS models/
  │                         │
  │                         ▼
  │                       src.ml.batch → ieee_raw.predictions
  │
  └──► dbt marts ◄─────────┘
         │
         ├── Looker Studio 등 분석 소비처
         └── src.ml.dataset (시간 분할 참조)

valid 점수 + 저장 모델
  │  api.build_data (수동 실행)
  ▼
api/static/curve.json → FastAPI 정적 서빙 → Cloud Run 임계값 UI
```

## 구성 요소와 책임

| 구성 요소 | 책임 | 외부 시스템 |
| --- | --- | --- |
| `src/common` | 설정 로딩, 표준 출력 로깅 | `.env` |
| `src/extract` | CSV 읽기, 원본 스키마·파티션 규칙, 초기 GCS 업로드 | Kaggle 데이터, GCS |
| `src/load` | GCS Parquet 입출력, BigQuery 원본 테이블 생성·파티션 적재 | GCS, BigQuery |
| `dbt/models/staging` | raw 컬럼명·타입 표준화 | BigQuery raw |
| `dbt/models/marts` | 소비자용 팩트·집계·시간 분할·예측 분석 테이블 | BigQuery staging/raw |
| `src/ml` | 데이터 조회, 피처 구성, 학습, 모델 저장, 점수 계산, 비용 기반 임계값 | BigQuery, GCS, MLflow |
| `dags` | Airflow schedule과 Asset 신호로 적재·변환·배치 추론 실행 | Airflow |
| `api` | 미리 생성된 임계값 곡선 JSON 서빙 | Cloud Run |
| `analysis` | 일회성 검증·설명 분석. 운영 경로에는 포함하지 않음 | 로컬/BigQuery |

`api/`는 독립 배포 단위다. 런타임에는 `api/static/`만 필요하므로 모델,
BigQuery 클라이언트, ML 라이브러리를 포함하지 않는다.

## 데이터 계층과 계약

| 계층 | 위치 | 계약 | 주요 검증 |
| --- | --- | --- | --- |
| 원본 | GCS Parquet | dataset별 날짜 파티션, 원본 컬럼과 파티션 메타데이터 | 업로드 전 `_check_schema` |
| raw | `ieee_raw` | 원본 값 보존, `transaction_date`, `source_split`, `ingested_at` 추가 | BigQuery 스키마, source tests |
| staging | `dev_staging` | 이름 표준화, 값·의미는 변경하지 않음 | dbt 모델·source tests |
| mart | `dev_mart` | 분석 소비처용 명시적 컬럼과 집계 단위 | dbt schema·관계·행 수 tests |
| ML 입력 | `Dataset.X`, `Dataset.y`, `Dataset.dates` | 시간 분할, 누수 컬럼 제외, test는 라벨 없음 | pytest, `load_training` 방어 |
| 모델 | `Bundle` | 예측기, 도메인 목록, 학습 컬럼 순서, 피처 집합, 지표, 학습 시각 | `format_version`, `Bundle.align` |
| 예측 | `ieee_raw.predictions` | 거래·날짜별 점수와 모델 메타데이터, 판정은 저장하지 않음 | 파티션 단위 `WRITE_TRUNCATE` |
| UI 산출물 | `api/static/curve.json` | valid 구간의 점수·금액 기반 임계값 곡선 | `api.build_data` 재생성 |

현재 mart는 여덟 개다. 거래 영역은 `fct_transactions`,
`agg_transactions_daily`, `agg_pipeline_daily`, `dim_split`이고, 모델 영역은
`agg_predictions_daily`, `agg_score_amount`, `agg_feature_buckets`,
`agg_feature_values`다.

## Python 의존성 방향

현재의 주된 의존 방향은 다음과 같다.

```text
common ← extract ← load
common ← ml
extract.schema ← load
load.gcs ← extract.bootstrap
ml.features / model_store ← ml.predict ← ml.batch
src.* ← dags
src.ml.* ← api.build_data
```

`dags`는 실행 어댑터로서 `src`의 유스케이스를 호출하고, ML의 점수 계산은
`predict.score()` 한 곳에 둔다. 학습과 추론은 모두 `features.build()`와
`Bundle.align()`을 사용해 피처 구성과 열 순서를 맞춘다.

현재 `extract.bootstrap → load.gcs`와 `load → extract.schema` 의존은
추출·적재 경계를 교차한다. 이는 현재 동작하는 구조이나, 공통 스키마 계약을
별도 모듈로 분리할 후보로 기록한다.

## 실행 흐름

### 초기 적재와 변환

1. `src.extract.bootstrap`이 train/test CSV를 읽고 날짜별 Parquet으로 GCS에 올린다.
2. `src.load.bigquery` 또는 `ingest_daily`가 GCS 파티션을 `ieee_raw`로 적재한다.
3. `transform` DAG가 dbt run과 dbt test를 실행한다.
4. dbt test가 통과한 뒤 `MART_READY` Asset을 발행한다.

BigQuery 적재와 predictions 쓰기는 파티션 데코레이터와 `WRITE_TRUNCATE`를
사용한다. 같은 날짜를 재실행해도 해당 날짜 파티션만 교체하는 멱등 계약이다.

### 학습과 배치 추론

1. `src.ml.train`이 `dim_split` 기준 train/valid를 조회한다.
2. train에서 도메인 목록을 맞추고, 모델·입력 컬럼·지표를 `Bundle`로 저장한다.
3. 운영 모델 파일을 GCS에 올린다.
4. `predict_daily` 또는 `src.ml.batch`가 모델을 내려받아 하루치를 채점한다.
5. 점수와 모델 메타데이터를 `ieee_raw.predictions`에 기록하고 dbt 모델이 이를 집계한다.

임계값은 점수 생성과 분리한다. predictions에는 이진 차단 판정을 저장하지 않고,
소비자가 비용 가정에 맞춰 점수에 임계값을 적용한다.

### 임계값 UI 배포

1. `api.build_data`가 valid 구간과 저장 모델로 `curve.json`을 만든다.
2. Docker 이미지는 `api/`만 포함해 FastAPI로 정적 파일을 제공한다.
3. Cloud Run은 모델을 로드하거나 BigQuery를 조회하지 않는다.

따라서 모델을 재학습하면 모델 업로드, 예측 재생성, dbt 갱신, `curve.json` 재생성,
UI 재배포 순서가 필요하다.

## 오케스트레이션

```text
ingest_daily
  ├── load_transactions ─┐
  └── load_identity ─────┴── RAW_LOADED
                                 │
                                 ▼
                            transform
                            dbt run → dbt test → MART_READY
                                                     │
                                                     ▼
                                               predict_daily
```

현재 두 적재 태스크가 동일한 `RAW_LOADED` Asset을 발행한다. 그러므로 transform이
두 입력 데이터셋의 같은 날짜 적재를 모두 기다린다는 계약은 코드에서 명시되지
않는다. 향후 `TRANSACTIONS_LOADED`, `IDENTITY_LOADED`처럼 Asset을 분리하고,
transform의 입력으로 둘 다 선언하는 것이 목표 구조다.

## 검증과 품질 게이트

| 단계 | 현재 검증 |
| --- | --- |
| GCS 업로드 전 | 컬럼 추가·누락·타입 검사 |
| dbt 실행 후 | source, 범위, 관계, 행 수, 시간 분할 테스트 |
| Python 단위 계약 | 피처, 모델 번들, 점수, 임계값, 테이블 스키마 pytest |
| 정적 검사 | Ruff, mypy를 개발 환경에서 실행 가능 |
| 운영 실행 | Airflow 재시도, 날짜 파티션 멱등성, 빈 날짜 skip |

BigQuery를 실제로 읽는 테스트는 네트워크와 ADC에 의존하므로, unit test와
`bq` integration test를 분리해 CI에서 각각 다른 단계로 실행하는 것을 목표로 한다.

## 현재 제약과 후속 결정

- predictions에는 `model_trained_at`이 있으나 일별 집계는 `model`만 기준으로
  그룹화한다. 모델 재학습 전후 분포를 명확히 분리하려면 불변 `model_version`을
  predictions와 mart의 키로 도입해야 한다.
- 예측 마트는 현재 raw predictions source를 직접 읽는다. `stg_predictions`를
  두면 raw → staging → mart 규칙을 모든 소비 데이터에 일관되게 적용할 수 있다.
- 재학습 이후의 모델 업로드·예측·마트·UI 갱신은 수동 절차다. 산출물의 버전과
  갱신 순서를 자동 검증하는 release workflow는 아직 없다.
- `src` 패키지 경계와 import 방향은 관례로만 유지된다. 공통 계약 모듈 분리와
  import 규칙 검사를 도입하면 구조 위반을 CI에서 차단할 수 있다.

## 관련 문서

- [README](../README.md): 실행 방법, 배포, 실험 결과
- [CLAUDE.md](../CLAUDE.md): 작업 규약과 운영상 주의점
- `dbt/models/`: 데이터 모델과 dbt 테스트
- `tests/`: Python 코드 계약 테스트
