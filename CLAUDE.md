# 작업 규약

IEEE-CIS 사기 탐지 데이터로 만드는 배치 파이프라인.
`CSV → GCS(Parquet) → BigQuery → dbt → 모델`

프로젝트 개요와 실행 결과는 README 를 본다. 여기에는 모르면 사고가 나는
것만 적는다.

## 실행

```bash
uv sync
uv run python -m src.extract.bootstrap     # CSV → GCS (1회)
uv run python -m src.load.bigquery         # GCS → BQ
cd dbt && dbt run --profiles-dir .
docker compose up -d                       # Airflow (localhost:8080, admin/admin)
uv run python -m src.ml.train              # --all-columns 로 익명 컬럼까지
uv run python -m src.ml.threshold          # 비용별 임계값 (재학습 없음)
```

**dbt 와 mlflow 는 프로젝트 의존성에 넣지 않는다.** 둘 다 해결 불가능한
버전 충돌이 있다.

- `dbt-bigquery` 가 `google-cloud-bigquery <3.3.3` 을 요구하는데 프로젝트는
  최신을 쓴다. `uv tool install dbt-core --with dbt-bigquery` 로 격리하고
  Airflow 이미지에서는 별도 venv(`/opt/dbt-venv`)에 넣는다.
- `mlflow` 풀 패키지가 `pandas<3` 을 요구하는데 프로젝트는 pandas 3 을 쓴다.
  의존성에는 `mlflow-skinny`(pandas 제약 없음)만 넣고 UI 는
  `uv tool install mlflow` 로 따로 둔다. 파일 백엔드(`mlruns/`)는 MLflow 3
  에서 유지보수 모드라 SQLite 를 쓴다.

## BigQuery 함정

**파티션 데코레이터를 반드시 쓴다.** 없이 `WRITE_TRUNCATE` 로 로드하면
테이블 전체가 교체된다. 하루치를 넣었을 뿐인데 기존 파티션이 전부 사라진다.
실험으로 확인했다.

```python
table_id = f"{project}.{dataset}.transactions${dt:%Y%m%d}"   # $YYYYMMDD 필수
```

**컬럼 누락은 조용히 통과한다.** 컬럼이 추가되거나 타입이 다르면 로드를
거부하지만, 빠진 것은 NULL 로 채우고 성공한다. 소스에서 컬럼이 사라져도
며칠 뒤에야 발견하게 되므로 `gcs.upload_partition` 이 업로드 전에 컬럼
집합을 대조한다.

**`rows` 는 예약어다.** `COUNT(*) AS rows` 가 문법 오류를 낸다.

**`percentile_cont` 는 분석 함수로 쓸 때 `ORDER BY` 를 허용하지 않는다.**
이동 중앙값을 낼 수 없어 `avg` 로 대체했다.

**32비트 타입이 없다.** `Int32`/`float32` 가 모두 64비트로 올라간다.
dtype 최적화는 로컬 메모리와 Parquet 크기에만 효과가 있다.

## 계층별 책임

| 계층 | 책임 | 컬럼 처리 |
| --- | --- | --- |
| raw (`ieee_raw`) | 원본 보존 | 손대지 않음 |
| staging (`dev_staging`) | 이름·타입 표준화 | `SELECT * EXCEPT` |
| mart (`dev_mart`) | 소비처와의 계약 | 명시 |

staging 에서 `SELECT *` 를 쓰는 이유는 익명 컬럼이 387개라 이름이 바뀔 것이
없기 때문이다. 소스에 컬럼이 추가되면 여기를 통과하지만, 적재 단계의 스키마
대조가 막고 mart 가 관문 역할을 한다.

데이터셋 이름은 dbt 관례를 따른다. `profiles.yml` 의 `dataset: dev` 에
`dbt_project.yml` 의 `+schema` 가 붙어 `dev_staging`, `dev_mart` 가 된다.
접두사를 없애는 매크로를 쓰지 않는다 — 한 웨어하우스를 공유할 때 서로의
스키마를 침범하지 않기 위한 관례다.

## ML 규약

**누수 통제가 우선이다.** 계층을 미루더라도 이것은 미루지 않는다.

- 파생에 `is_fraud` 를 쓰지 않는다.
- 집계 파생은 윈도우를 반드시 그 시점까지로 자른다
  (`rows between unbounded preceding and 1 preceding`).
- `transaction_dt` 와 `source_split` 은 피처에 넣지 않는다. train/valid
  경계를 그대로 담고 있어 valid 점수만 좋아지고 test 에서 무너진다.
- 분할 경계는 `dim_split` 하나가 정한다. 날짜를 코드에 박지 않는다.

**학습과 추론은 같은 함수를 부른다.** 조회는 `dataset.load`, 파생은
`features.build`, 저장은 `model_store.Bundle`. 규칙이 두 곳에 있으면 언젠가
갈라지는데 에러가 나지 않고 점수만 조용히 틀린다. 라벨 없는 구간을 채점하는
파이프라인이라 나중에도 알아채기 어렵다.

**피처 레이어는 미뤘다.** 필요한 피처가 무엇인지 모르는 상태에서 계층부터
만들면 학습해보고 다시 고치게 된다. `dataset.load` 가 staging 을 직접 읽고,
값어치가 확인된 것만 나중에 dbt 로 승격한다.

**임계값은 비용이 정한다.** 놓친 사기는 거래액만큼 잃고(데이터에 있다), 막은
정상은 처리·이탈 비용이 든다(데이터에 없다). `fp_cost` 하나가 유일한 가정이다
— 늘리면 정교해 보이지만 각각을 검증할 방법이 없다. 값 하나를 고르는 대신
가정별 대응을 남긴다(`src/ml/threshold.py`).

## 인증

서비스 계정 키 대신 ADC 를 쓴다. 조직 정책
`iam.disableServiceAccountKeyCreation` 으로 키 발급이 막혀 있고, 키 파일을
주고받지 않는 편이 안전하다.

```bash
gcloud auth application-default login
```

컨테이너는 호스트의 `%APPDATA%\gcloud` 를 마운트해서 쓴다.

## 검증 층위

| | 대상 | 언제 |
| --- | --- | --- |
| `gcs._check_schema` | 컬럼 집합 | 업로드 전 |
| dbt tests | 데이터 | `dbt test` |
| pytest | 코드 계약 | 커밋 시 |

테스트는 구현이 아니라 계약을 본다. 해시 방식이나 임시 파일 이름 같은
세부는 바뀔 수 있으므로 검증 대상이 아니다. "무엇이 조용히 잘못될 수
있는가"에서 출발해 거기에만 붙인다.

`bq` 마커가 붙은 pytest 는 BigQuery 를 실제로 조회해 느리다.
`-m "not bq"` 로 뺀다.

dbt 의 `relationships` 와 `equal_rowcount` 가 조인 무결성을 고정한다.
identity 는 1:1 이므로 LEFT JOIN 해도 행이 늘면 안 된다.

## 데이터 성질

- `TransactionDT` 는 기준 시점 기준 초 오프셋. **기준일은 임의값이다.**
  역산을 시도했으나 실패했다(`analysis/verify_origin.py`). 절대 날짜에
  의미가 없고 상대 순서만 쓴다.
- train(2017-12-02~2018-06-01)과 test(2018-07-02~2018-12-31)는 겹치지 않고
  **사이에 30일 공백**이 있다. 파티션이 없는 날짜가 정상적으로 존재하며
  DAG 은 이를 skip 으로 처리한다.
- test 구간은 `is_fraud` 가 NULL 이다. 실무에서도 최근 거래는 조사 중이라
  라벨이 비어 있으므로 예외가 아니라 정상 상태다.
- 조인은 1:1, 고아 레코드 0, identity 보유율 26%.
- 사기율 3.5%. 거래가 적은 시각(h=7~9)에 3배.
- 익명 컬럼의 접두사 의미(`C` 는 카운팅, `D` 는 시간 델타)는 커뮤니티
  통설이며 데이터로 확인되지 않았다. `analysis/shap_importance.py` 참고.

## 커밋

`feat:`, `fix:`, `refactor:`, `chore:` prefix 에 한글 본문.
무엇을 했는지보다 **왜 그렇게 했는지**를 남긴다. 특히 실험으로 확인한
사실은 근거와 함께 적는다.

`.gitignore`: `data/`, `.env`, `config/gcp-key.json`, `dbt/profiles.yml`,
`dbt/target/`, `dbt/dbt_packages/`, `logs/`, `models/`, `mlflow.db`,
`mlruns/`