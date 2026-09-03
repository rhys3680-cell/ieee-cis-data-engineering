# IEEE-CIS Fraud Detection Pipeline

IEEE-CIS 사기 탐지 데이터로 구축한 배치 데이터 파이프라인.
CSV → GCS(Parquet) → BigQuery → dbt → 모델 학습/추론.

Python 3.12 · pandas · GCS · BigQuery · dbt · Airflow · MLflow · LightGBM

## 데이터 흐름

```
Kaggle CSV  →  GCS                    BigQuery
               dt=YYYY-MM-DD/         ├ ieee_raw       원본 그대로
               part.parquet           │   ↓ dbt
                                      ├ dev_staging    이름·타입 표준화
                                      │   ↓ dbt
                                      └ dev_mart       팩트 · 집계 · 분할 · 예측
                                            │
                                            ↓  dataset.load(split)
                                        src/ml/
                                          features    파생 (학습·추론 공용)
                                          train       더미 / 로지스틱 / LightGBM
                                          model_store 모델 + 범주 목록 + 컬럼
                                          predict     배치·API 공용 진입점
                                          threshold   비용으로 임계값 결정
                                            │
                                            ↓  build_data (배포 전 1회)
                                        api/static/curve.json
                                            │
                                        Cloud Run  임계값 화면
```

적재와 변환은 Airflow 가 돌리고 학습은 손으로 돌린다.

| 계층 | 책임 | 컬럼 처리 |
| --- | --- | --- |
| raw | 원본 보존 | 손대지 않음 |
| staging | 표준화 | `SELECT * EXCEPT` |
| mart | 소비처와의 계약 | 명시 |

마트는 일곱이다. 거래 쪽은 `fct_transactions`(1,097,231행),
`agg_transactions_daily`(61,685행), `agg_pipeline_daily`(365행),
`dim_split`(365행). 모델 쪽은 `agg_predictions_daily`(일별 차단율과 점수
분포), `agg_score_amount`(점수 × 금액 사분면), `agg_feature_buckets` ·
`agg_feature_values`(SHAP 상위 피처의 값별 사기율)다.

## 시작하기

전제: Python 3.12, [uv](https://docs.astral.sh/uv/),
[Kaggle API 토큰](https://www.kaggle.com/docs/api), GCP 프로젝트

### 1. 의존성과 데이터

```bash
uv sync
cp .env.example .env          # GCP 프로젝트, 버킷 이름 입력
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw
```

### 2. GCP 인증

서비스 계정 키 대신 ADC 를 쓴다. 2024년 5월 이후 만들어진 조직은
`iam.disableServiceAccountKeyCreation` 정책이 기본값이라 키 발급이 막혀 있다.

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

### 3. 적재와 변환

```bash
uv run python -m src.extract.bootstrap    # CSV → GCS (365 × 2 파티션)
uv run python -m src.load.bigquery        # GCS → BigQuery

uv tool install dbt-core --with dbt-bigquery   # 버전 충돌로 격리 설치
cd dbt && cp profiles.yml.example profiles.yml
dbt run --profiles-dir . && dbt test --profiles-dir .
```

### 4. Airflow

```bash
docker compose up -d          # 첫 실행은 이미지 빌드로 몇 분
```

http://localhost:8080 · `admin` / `admin`

DAG 은 정지 상태로 생성된다. `ingest_daily` 를 켜면 2017-12-02 부터 백필이
시작되고, 끝나면 Asset 신호를 받아 `transform` 이 dbt 를 돌린다. 서비스는
api-server, scheduler, dag-processor, postgres 넷이다 — dag-processor 는
Airflow 3 에서 scheduler 와 분리된 별도 컴포넌트다.

`.env` 를 compose 가 읽고, 인증은 호스트의 `%APPDATA%\gcloud` 를 읽기 전용으로
마운트한다. LocalExecutor 라 Celery 구성(redis, worker, flower)은 두지 않았다.

### 5. 학습

```bash
uv run python -m src.ml.train                # curated 피처 (9 개)
uv run python -m src.ml.train --all-columns  # 익명 컬럼까지 (394 개)
uv run python -m src.ml.threshold            # 비용별 임계값

uv tool install mlflow                       # UI 도 격리 설치
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 6. 임계값 화면

[threshold-ui.run.app](https://threshold-ui-59694923737.asia-northeast3.run.app)
— 비용 가정을 슬라이더로 바꿔가며 임계값이 어떻게 움직이는지 본다.

```bash
uv run python -m api.build_data              # 곡선을 JSON 으로 굽는다
uv run uvicorn api.main:app --reload         # localhost:8000
```

**서버가 모델을 들고 있지 않다.** 임계값 곡선은 재학습해야 바뀌는 값이라
요청마다 계산할 이유가 없다. `build_data` 가 99행 12 KB 를 굽고 API 는 그것을
정적 파일로 내준다. 그래서 컨테이너에 모델도 BigQuery 클라이언트도 lightgbm 도
필요 없어 211 MB 에 그치고, 콜드 스타트가 1초 미만이다.

슬라이더도 서버에 묻지 않는다. 곡선이 `fp_cost` 에 의존하지 않는 원자료라
총비용은 화면이 `missed_amount + false_alarms * fp_cost` 로 계산한다.

**배포**

```bash
docker build -t threshold-ui .
gcloud auth configure-docker asia-northeast3-docker.pkg.dev
docker tag threshold-ui asia-northeast3-docker.pkg.dev/<프로젝트-ID>/web/threshold-ui
docker push asia-northeast3-docker.pkg.dev/<프로젝트-ID>/web/threshold-ui

gcloud run deploy threshold-ui \
  --image asia-northeast3-docker.pkg.dev/<프로젝트-ID>/web/threshold-ui \
  --region asia-northeast3 --allow-unauthenticated --min-instances 0
```

Cloud Build 대신 로컬에서 빌드해 푸시한다. 새 프로젝트에서는 Compute 기본
서비스 계정에 역할이 붙지 않아 `gcloud builds submit` 이 자기 소스 tarball
조차 읽지 못한다(`storage.objects.get denied`). 화면 하나를 올리는 데
IAM 역할 셋을 붙이는 것보다 로컬 빌드가 간단하다 — 자동 빌드가 필요해지면
그때 정리한다.

### 7. 모델을 다시 학습했을 때

파생 산출물이 자동으로 따라오지 않는다. 순서대로 돌려야 화면과 테이블이
같은 모델을 말한다.

```bash
uv run python -m src.ml.train --all-columns   # 1. 학습 → models/*.joblib
uv run python -m src.ml.model_store lgbm-all  # 2. GCS 로 올린다

# 3. 예측을 다시 만든다. Airflow 에서 predict_daily 를 재실행하거나,
#    로컬이면 날짜별로 src.ml.batch 를 돌린다.
uv run python -m src.ml.batch 2018-07-02

cd dbt && dbt run --profiles-dir .            # 4. 예측을 읽는 마트 갱신
cd .. && uv run python -m api.build_data      # 5. 곡선을 다시 굽는다
docker build -t threshold-ui . && docker push ...   # 6. 재배포
```

**빠뜨리면 어긋난다.** 2번 없이 3번을 돌리면 옛 모델로 채점하고, 5번을
빠뜨리면 화면만 옛 곡선을 보여준다. 그러면 임계값 화면과 사분면 마트가
서로 다른 모델을 말하게 되는데, 에러가 나지 않아 눈치채기 어렵다.

`predictions` 테이블에 `model_trained_at` 을 남겨 두었으므로 어느 모델이
매긴 점수인지는 확인할 수 있다. 자동 감지는 아직 없다 — 재학습이 드물어
절차를 지키는 편이 단순하다.

자동으로 갱신되지 않는 것이 나쁜 것만은 아니다. 화면의 숫자가 바뀌는
시점이 명시적이고, 재학습 결과가 나쁘면 배포하지 않으면 된다.

## 결과

valid(2018-05-04~06-01, 82,325행, 사기 2,868건)에서 잰 값이다.

| 모델 | 피처 | PR-AUC | ROC-AUC |
| --- | --- | --- | --- |
| dummy | - | 0.0348 | 0.500 |
| logreg (curated) | 9 | 0.1426 | 0.756 |
| lgbm (curated) | 9 | 0.1842 | 0.786 |
| lgbm (all) | 394 | 0.5313 | 0.915 |

더미의 PR-AUC 가 기저 사기율(0.0348)과 일치하고 ROC-AUC 가 0.500 이다.
이론값 그대로라 평가 경로가 정상이라는 뜻이다. 사기율이 3.5% 라 ROC-AUC 는
음성을 맞히는 것만으로도 오르므로 판단은 PR-AUC 로 한다.

**임계값은 모델이 아니라 비용이 정한다.** 놓친 사기는 거래액만큼 잃고, 막은
정상은 처리·이탈 비용이 든다. 뒤의 값을 얼마로 보느냐에 따라 최적 임계값이
0.21 에서 0.87 까지 **4.1배** 움직인다. 그래서 값 하나를 고르는 대신 가정별
대응을 남긴다 — `uv run python -m src.ml.threshold` 로 낸다.

어떤 피처가 기여하는지는 SHAP 으로 확인했다
(`uv run --with shap python -m analysis.shap_importance`). 접두사별 개당
기여가 `V*` 0.10%, `C*` 1.41%, `D*` 0.84% 로 갈려, 익명 컬럼의 힘은 339개의
`V*` 가 아니라 `C*`/`D*` 29개에서 상당 부분 나온다.

## 설계 노트

**시간축.** `TransactionDT` 는 기준 시점으로부터의 초 오프셋이고 대회는 그
시점을 공개하지 않았다. 공휴일 패턴으로 역산을 시도했으나 판별력이 없어
실패했다(`analysis/verify_origin.py`). 절대 날짜에 의미가 없고 상대 순서만
쓴다. 날짜 타입을 쓰는 이유는 BigQuery 파티셔닝과 Airflow catchup 을 그대로
쓰기 위해서다.

**멱등성.** 파티션 데코레이터(`table$YYYYMMDD`)와 `WRITE_TRUNCATE` 로 파티션을
통째로 교체한다. 데코레이터 없이 쓰면 테이블 전체가 교체되어, 하루치를
넣었을 뿐인데 기존 파티션이 전부 사라진다.

**스키마 검증.** BigQuery 는 컬럼이 추가되거나 타입이 다르면 로드를 거부하지만
빠진 것은 NULL 로 채운다. 소스에서 컬럼이 사라져도 적재가 성공해 며칠 뒤에야
발견하게 되므로, 업로드 전에 컬럼 집합을 대조한다.

**라벨.** test 구간은 `is_fraud` 가 NULL 이다. 실무에서도 최근 거래는 조사
중이라 라벨이 비어 있으므로 예외가 아니라 정상 상태로 다룬다. 평가는 train
안에서 시간 순으로 자른 valid 로만 한다.

**학습·추론 일치.** 조회(`dataset.load`), 파생(`features.build`),
저장(`model_store.Bundle`)을 나눠 학습과 추론이 같은 함수를 부르게 한다.
규칙이 두 곳에 있으면 갈라지는데, 에러 없이 점수만 조용히 틀린다.

## 진행 상황

- [x] 적재 — CSV → GCS → BigQuery, 파티션 단위 멱등
- [x] dbt — staging, mart 7개, 테스트 55개
- [x] 대시보드 — 거래 현황 ([Looker Studio](https://datastudio.google.com/reporting/b8b98f79-97c6-4447-91f7-59f285d9f162))
- [x] Airflow — `ingest_daily`, `transform`, Asset 연결
- [x] 시간 분할 — `dim_split`, `dataset.load` (누수 통제)
- [x] 베이스라인 — 더미 / 로지스틱 / LightGBM, MLflow, 모델 저장
- [x] 임계값 — 비용 기반 결정, 민감도 분석
- [x] 웹 — 임계값 화면 (FastAPI + Cloud Run)
- [ ] 배치 추론 — `predict_daily`, 예측 테이블, 모니터링
- [ ] 운영 화면 — 일별 추이 (`/ops` 탭)
