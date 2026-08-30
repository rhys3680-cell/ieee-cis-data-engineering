"""학습·추론용 데이터를 BigQuery에서 읽는다.

조회만. 파생 피처는 features.py에서 진행한다. 나누는 이유는 학습과 추론이 같은 파생 피처 함수를 불러야 하기 때문이다. 규칙을 한 곳에 모아두기 위함이다.

학습과 추론이 같은 조회 경로를 쓴다. DROP_COLUMNS 가 한 곳에만 있어야
양쪽의 X 가 같은 형태로 보장된다. 다만 진입점은 나눈다 — 학습은
load_training 을 쓰고, 여기서 test 는 거부된다.

분할은 dim_split을 조인해서 가져온다. 날짜를 코드에 적어두면 dbt의
valid_start와 어긋날 가능성이 있다.
"""

from dataclasses import dataclass

import pandas as pd
from google.cloud import bigquery

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "valid", "test")

# transaction_dt 는 기준 시점 기준 초 오프셋이라 train/valid 경계를
# 그대로 담고 있다. 모델이 이 값으로 시간을 외우면 valid 점수는 좋아
# 보이지만 test 는 본 적 없는 범위라 무너진다. 시각·요일 같은 파생은
# 안전하지만 원본 오프셋은 아니다.
DROP_COLUMNS = (
    "transaction_id",
    "transaction_dt",
    "transaction_date",
    "is_fraud",
    "ml_split",
    "source_split",
    "ingested_at",
)


@dataclass(frozen=True)
class Dataset:
    """X와 y를 가지고 있는 클래스.

    한 DataFrame 으로 주고 라벨을 누군가가 실수로 넣고 학습을 시킬 수도 있다.
    넣은 채 학습하면 정확도 100% 가 나오고, 그게 버그라는 걸 알아채기까지
    시간이 걸린다.
    """

    X: pd.DataFrame
    y: pd.Series | None  # test는 라벨이 없어서 None
    dates: pd.Series  # 시간 순서 확인, 시각대별 성능 분석용

    def __len__(self) -> int:
        return len(self.X)


def load(split: str, columns: list[str] | None = None) -> Dataset:
    """split 하나를 읽는다.

    columns를 주면 그것만 읽는다. 메모리 절약 목적이다.
    """
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나여야 한다.: {split!r}")

    s = get_settings()

    # columns 는 피처를 고르는 인자다. 날짜와 라벨은 반환값에 항상 필요하므로
    # 고르는 대상이 아니고, 빠지면 조회 뒤에야 KeyError 로 드러난다.
    if columns:
        wanted = list(dict.fromkeys([*columns, "transaction_date", "is_fraud"]))
        select = ", ".join(f"t.{c}" for c in wanted)
    else:
        select = "t.*"

    # 파라미터 바인딩을 쓴다.
    q = f"""
        select {select}, s.ml_split
        from `{s.gcp_project_id}.dev_staging.stg_transactions` t
        join `{s.gcp_project_id}.dev_mart.dim_split` s using(transaction_date)
        where s.ml_split = @split
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("split", "STRING", split)]
    )
    df = _client().query(q, job_config=job_config).to_dataframe()

    if df.empty:
        raise ValueError(f"{split}: 행이 없다. dim_split을 먼저 빌드해야 한다.")

    dates = df["transaction_date"]

    if split == "test":
        # 라벨이 NULL 이다. 예외를 던지지 않는 이유는 배치 추론이 이 구간을
        # 읽어야 하기 때문이다 — 라벨 없는 데이터에 점수를 매기는 것이 추론이다.
        # 학습 경로는 load_training 이 먼저 막고, y is None 은 그 뒤의 방어선이다.
        y = None
    else:
        y = df["is_fraud"].astype("int8")
        # train/valid 에 NULL 이 있으면 분할이 잘못된 것이다.
        assert df["is_fraud"].notna().all(), f"{split}: is_fraud 에 NULL 이 있다"

    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    logger.info(
        "%s: %s 행, 피처 %d 개%s",
        split,
        f"{len(df):,}",
        X.shape[1],
        f", 사기 {y.sum():,} 건" if y is not None else " (라벨 없음)",
    )
    return Dataset(X=X, y=y, dates=dates)


def load_training(split: str, columns: list[str] | None = None) -> Dataset:
    """학습·평가용. train 과 valid 만 받는다.

    test 는 is_fraud 가 NULL 이라 학습도 평가도 할 수 없다. load 가 세
    split 을 다 받으므로, 의도를 이름으로 갈라 실수를 조회 전에 막는다.
    """
    if split == "test":
        raise ValueError(
            "test 는 라벨이 없어 학습에 쓸 수 없다. 추론이라면 load 를 쓴다."
        )
    return load(split, columns)


def _client() -> bigquery.Client:
    return bigquery.Client(project=get_settings().gcp_project_id)
