"""IEEE-CIS 원본 스키마 정의.

dtype 은 src/extract/profile.py 의 판정 결과다. 재생성하려면:

    uv run python -m src.extract.profile
    # -> data/processed/column_profile_*.csv

여기에는 그 결과를 리터럴로 고정한다. 스키마는 계약이므로 원본 데이터가
없어도 읽을 수 있어야 하고, 데이터가 바뀌었을 때 조용히 따라 변하면 안 된다.

판정 기준:
  Int32     정수만 담긴 수치. nullable 이라 결측이 있어도 유지된다.
  float32   소수를 담은 수치. 값 범위가 float32 에 안전히 들어간다.
  boolean   원본이 T/F 문자열. nullable.
            read_csv 에 직접 넘길 수 없다 (pandas 는 True/False, 1/0 만
            인식한다). loader 가 문자열로 읽은 뒤 변환한다.
  category  고유값이 30개 이하인 문자열.
  str       그 외 문자열.
"""

from datetime import date

import pandas as pd

# TransactionDT 는 이 시각 기준 초 오프셋으로 해석한다.
# 임의값이다. 근거는 analysis/verify_origin.py 참고.
TRANSACTION_DT_ORIGIN = date(2017, 12, 1)

SECONDS_PER_DAY = 86_400

# --- 역할이 정해진 컬럼 ---
PRIMARY_KEY = "TransactionID"
LABEL = "isFraud"  # test 에는 없다
PARTITION_SOURCE = "TransactionDT"  # 파티션 키 계산에만 쓴다

# --- transaction ---

# V126-137, V159-166, V202-216, V263-278, V306-321, V331-339 가 소수를 담는다.
# 나머지 263 개는 정수다.
_V_FLOAT_RANGES = [
    (126, 137),
    (159, 166),
    (202, 216),
    (263, 278),
    (306, 321),
    (331, 339),
]
V_FLOAT_COLUMNS = frozenset(
    f"V{i}" for lo, hi in _V_FLOAT_RANGES for i in range(lo, hi + 1)
)
V_COLUMNS = tuple(f"V{i}" for i in range(1, 340))

C_COLUMNS = tuple(f"C{i}" for i in range(1, 15))  # 전부 Int32
D_COLUMNS = tuple(f"D{i}" for i in range(1, 16))  # D8, D9 만 float32
D_FLOAT_COLUMNS = frozenset({"D8", "D9"})

# M1-M9 는 T/F 인데 M4 만 M0/M1/M2 카테고리다.
M_COLUMNS = tuple(f"M{i}" for i in range(1, 10))
M_CATEGORY_COLUMNS = frozenset({"M4"})

TRANSACTION_CATEGORY_COLUMNS = frozenset({"ProductCD", "card4", "card6", "M4"})
TRANSACTION_STR_COLUMNS = frozenset({"P_emaildomain", "R_emaildomain"})

# --- identity ---

ID_COLUMNS = tuple(f"id_{i:02d}" for i in range(1, 39))

IDENTITY_CATEGORY_COLUMNS = frozenset(
    {
        "id_12",
        "id_15",
        "id_16",
        "id_23",  # IP_PROXY:ANONYMOUS / HIDDEN / TRANSPARENT
        "id_27",
        "id_28",
        "id_29",
        "id_34",  # match_status:-1 / 0 / 1 / 2
        "DeviceType",
    }
)
IDENTITY_BOOLEAN_COLUMNS = frozenset({"id_35", "id_36", "id_37", "id_38"})
IDENTITY_STR_COLUMNS = frozenset({"id_30", "id_31", "id_33", "DeviceInfo"})
IDENTITY_FLOAT_COLUMNS = frozenset({"id_11"})


def transaction_dtypes(*, with_label: bool = True) -> dict[str, str]:
    """train/test transaction 의 컬럼별 dtype.

    Args:
        with_label: False 면 isFraud 를 뺀다 (test 용).
    """
    dtypes: dict[str, str] = {
        PRIMARY_KEY: "Int32",
        PARTITION_SOURCE: "Int32",
        "TransactionAmt": "float32",
        "ProductCD": "category",
        "P_emaildomain": "str",
        "R_emaildomain": "str",
    }
    if with_label:
        dtypes[LABEL] = "Int32"

    for c in ("card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"):
        dtypes[c] = "Int32"
    for c in ("card4", "card6"):
        dtypes[c] = "category"

    for c in C_COLUMNS:
        dtypes[c] = "Int32"
    for c in D_COLUMNS:
        dtypes[c] = "float32" if c in D_FLOAT_COLUMNS else "Int32"
    for c in M_COLUMNS:
        dtypes[c] = "category" if c in M_CATEGORY_COLUMNS else "boolean"
    for c in V_COLUMNS:
        dtypes[c] = "float32" if c in V_FLOAT_COLUMNS else "Int32"

    return dtypes


def identity_dtypes() -> dict[str, str]:
    """train/test identity 의 컬럼별 dtype."""
    dtypes: dict[str, str] = {PRIMARY_KEY: "Int32"}

    for c in (*ID_COLUMNS, "DeviceType", "DeviceInfo"):
        if c in IDENTITY_CATEGORY_COLUMNS:
            dtypes[c] = "category"
        elif c in IDENTITY_BOOLEAN_COLUMNS:
            dtypes[c] = "boolean"
        elif c in IDENTITY_STR_COLUMNS:
            dtypes[c] = "str"
        elif c in IDENTITY_FLOAT_COLUMNS:
            dtypes[c] = "float32"
        else:
            dtypes[c] = "Int32"

    return dtypes


def normalize_columns(columns: pd.Index) -> dict[str, str]:
    """BigQuery 가 받을 수 있는 컬럼명으로 바꾸는 매핑을 만든다.

    test_identity 는 id-01 처럼 하이픈을 쓰는데 train_identity 는 id_01 이다.
    BigQuery 는 컬럼명에 하이픈을 허용하지 않으므로 언더스코어로 통일한다.
    """
    return {c: c.replace("-", "_") for c in columns if "-" in c}


def to_transaction_date(dt_seconds: pd.Series) -> pd.Series:
    """TransactionDT(초 오프셋)를 파티션 키가 될 날짜로 바꾼다."""
    origin = pd.Timestamp(TRANSACTION_DT_ORIGIN)
    return (origin + pd.to_timedelta(dt_seconds, unit="s")).dt.date


# pandas dtype -> BigQuery 타입.
# BigQuery에는 32비트 타입이 없어
# Int32/float32는 모두 64비트로 올라간다.
_BQ_TYPES = {
    "Int32": "INT64",
    "Int64": "INT64",
    "float32": "FLOAT64",
    "float64": "FLOAT64",
    "boolean": "BOOL",
    "category": "STRING",
    "str": "STRING",
}


def bigquery_schema(dataset: str) -> list[tuple[str, str]]:
    """BigQuery 테이블 스키마. (컬럼명, 타입) 목록.

    원본 컬럼에 파이프라인이 붙이는 세 개를 더한다.
      transaction_date  파티션 키
      source_split      train/test 구분
      ingested_at       적재 시각
    """
    if dataset == "transactions":
        dtypes = transaction_dtypes(with_label=True)
    elif dataset == "identity":
        dtypes = identity_dtypes()
    else:
        raise KeyError(f"알 수 없는 데이터셋: {dataset!r}")

    fields = [(col, _BQ_TYPES[dt]) for col, dt in dtypes.items()]
    fields += [
        ("transaction_date", "DATE"),
        ("source_split", "STRING"),
        ("ingested_at", "TIMESTAMP"),
    ]
    return fields
