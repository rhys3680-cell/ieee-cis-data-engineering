# 작업 규약

IEEE-CIS 사기 탐지 데이터로 만드는 배치 파이프라인.
`CSV → GCS(Parquet) → BigQuery → dbt → 모델`

## 실행

```bash
uv sync                                    # 의존성
uv run python -m src.extract.bootstrap     # CSV → GCS (1회)
uv run python -m src.load.bigquery         # GCS → BQ
uv run python -m src.load.bigquery --describe

cd dbt && dbt run --profiles-dir .         # dbt 는 uv tool 로 격리 설치
docker compose up -d                       # Airflow (localhost:8080, admin/admin)
```

dbt 를 프로젝트 의존성에 넣으면 안 된다. `dbt-bigquery` 가
`google-cloud-bigquery <3.3.3` 을 요구하는데 프로젝트는 최신을 쓴다.
해결 불가능한 충돌이므로 `uv tool install dbt-core --with dbt-bigquery` 로
격리하고, Airflow 이미지에서는 별도 venv(`/opt/dbt-venv`)에 넣는다.

## 계층별 책임

| 계층 | 책임 | 컬럼 처리 |
| --- | --- | --- |
| raw (`ieee_raw`) | 원본 보존 | 손대지 않음 |
| staging (`dev_staging`) | 이름·타입 표준화 | `SELECT * EXCEPT` |
| mart (`dev_mart`) | 소비처와의 계약 | 명시 |

staging 에서 `SELECT *` 를 쓰는 이유는 익명 컬럼(`V1`~`V339`, `C*`, `D*`, `M*`,
`id_*`)이 387개라 이름이 바뀔 것이 없기 때문이다. 소스에 컬럼이 추가되면
여기를 통과하지만, 적재 단계의 스키마 대조가 막고 mart 가 관문 역할을 한다.

데이터셋 이름은 dbt 관례를 따른다. `profiles.yml` 의 `dataset: dev` 에
`dbt_project.yml` 의 `+schema` 가 접미사로 붙어 `dev_staging`, `dev_mart` 가
된다. 접두사를 없애는 매크로를 쓰지 않는다 — 여러 사람이 한 웨어하우스를
공유할 때 서로의 스키마를 침범하지 않기 위한 관례다.

## BigQuery 함정

**파티션 데코레이터를 반드시 쓴다.** 없이 `WRITE_TRUNCATE` 로 로드하면
테이블 전체가 교체된다. 하루치를 넣었을 뿐인데 기존 파티션이 전부 사라진다.
실험으로 확인했다.

```python
table_id = f"{project}.{dataset}.transactions${dt:%Y%m%d}"   # $YYYYMMDD 필수
```

**컬럼 누락은 조용히 통과한다.** BigQuery 는 컬럼이 추가되거나 타입이 다르면
로드를 거부하지만, 빠진 것은 NULL 로 채우고 성공한다. 소스에서 컬럼이
사라져도 며칠 뒤에야 발견하게 되므로 `gcs.upload_partition` 이 업로드 전에
컬럼 집합을 대조한다.

**`rows` 는 예약어다.** `COUNT(*) AS rows` 가 문법 오류를 낸다.

**`percentile_cont` 는 분석 함수로 쓸 때 `ORDER BY` 를 허용하지 않는다.**
이동 중앙값을 낼 수 없어 `avg` 로 대체했다.

**32비트 타입이 없다.** `Int32`/`float32` 가 모두 64비트로 올라간다.
dtype 최적화는 로컬 메모리와 Parquet 크기에만 효과가 있다.

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
| dbt tests (24개) | 데이터 | `dbt test` |
| pytest | 코드 계약 | 커밋 시 |

테스트는 구현이 아니라 계약을 본다. 해시 방식이나 임시 파일 이름 같은
세부는 바뀔 수 있으므로 검증 대상이 아니다. "무엇이 조용히 잘못될 수
있는가"에서 출발해 거기에만 붙인다.

dbt 의 `relationships` 와 `equal_rowcount` 가 조인 무결성을 고정한다.
identity 는 1:1 이므로 LEFT JOIN 해도 행이 늘면 안 된다.

## 데이터 성질

- `TransactionDT` 는 기준 시점 기준 초 오프셋. **기준일은 임의값이다.**
  공휴일 패턴으로 역산을 시도했으나 판별력이 없어 실패했다
  (`analysis/verify_origin.py`). 절대 날짜에 의미가 없고 상대 순서만 쓴다.
- train(2017-12-02~2018-06-01)과 test(2018-07-02~2018-12-31)는 날짜가 겹치지
  않고 **사이에 30일 공백**이 있다. 파티션이 없는 날짜가 정상적으로 존재하며
  DAG 은 이를 skip 으로 처리한다.
- test 구간은 `is_fraud` 가 NULL 이다. 실무에서도 최근 거래는 조사 중이라
  라벨이 비어 있으므로 예외가 아니라 정상 상태다.
- 조인은 1:1, 고아 레코드 0, identity 보유율 26%.
- 사기율 3.5%. 거래가 적은 시각(h=7~9)에 9~10.6% 로 3배, mobile 이
  desktop 의 1.6배.

## 커밋

`feat:`, `fix:`, `refactor:`, `chore:` prefix 에 한글 본문.
무엇을 했는지보다 **왜 그렇게 했는지**를 남긴다. 특히 실험으로 확인한
사실(파티션 데코레이터, 컬럼 누락 등)은 근거와 함께 적는다.

`.gitignore` 에 있는 것: `data/`, `.env`, `config/gcp-key.json`,
`dbt/profiles.yml`, `dbt/target/`, `dbt/dbt_packages/`, `logs/`

## 진행 상황

- [x] 적재 — CSV → GCS → BigQuery, 파티션 단위 멱등
- [x] dbt — staging, mart(팩트 + 집계 2), 테스트 24개
- [x] 대시보드 — Looker Studio (거래 현황)
- [x] Airflow — `ingest_daily`, `transform`, Asset 연결
- [ ] ML — 피처 레이어(누수 통제), 시간 분할, MLflow
- [ ] 모델 운영 — 배치 추론, 손실 비용, Metabase