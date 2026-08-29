"""원본 데이터 프로파일링.

파티션 설계에 필요한 사실을 확인한다.
 - TransactionDT 범위 -> 실제 기간
 - 일별 거래 건수 -> 파티션 크기 균등성
 - 일별 사기율 -> 학습/검증 분할 지점 근거

실행: uv run python -m src.extract.profile
"""

import pandas as pd

from src.common.config import get_settings
from src.common.logging import get_logger
from src.extract.loader import raw_path
from src.extract.schema import SECONDS_PER_DAY, to_transaction_date

logger = get_logger(__name__)


def profile_transactions() -> pd.DataFrame:
    """거래 데이터의 시간축 분포를 집계한다."""
    # 시간축 분석에 필요한 것만 읽는다.
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
    df = pd.read_csv(raw_path("train_transaction"), usecols=cols)

    # 날짜 변환은 schema 가 정한 기준일을 그대로 쓴다.
    df["dt"] = to_transaction_date(df["TransactionDT"])

    logger.info("전체 %d 건", len(df))
    logger.info(
        "TransactionDT: %d ~ %d (%.1f 일)",
        df["TransactionDT"].min(),
        df["TransactionDT"].max(),
        (df["TransactionDT"].max() - df["TransactionDT"].min()) / SECONDS_PER_DAY,
    )
    logger.info("기간: %s ~ %s", df["dt"].min(), df["dt"].max())
    logger.info("전체 사기율: %.4f", df["isFraud"].mean())

    daily = (
        df.groupby("dt")
        .agg(
            tx_count=("TransactionID", "count"),
            fraud_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            amt_sum=("TransactionAmt", "sum"),
        )
        .reset_index()
    )

    logger.info("일자 수: %d", len(daily))
    logger.info(
        "일별 건수 - 최소 %d / 중앙 %d / 최대 %d",
        daily["tx_count"].min(),
        int(daily["tx_count"].median()),
        daily["tx_count"].max(),
    )
    logger.info(
        "일별 사기율 - 최소 %.4f / 중앙 %.4f / 최대 %.4f",
        daily["fraud_rate"].min(),
        daily["fraud_rate"].median(),
        daily["fraud_rate"].max(),
    )

    return daily


def profile_schema() -> None:
    """컬럼 구성과 결측 현황을 확인한다."""
    df = pd.read_csv(raw_path("train_transaction"), nrows=50_000)

    logger.info("컬럼 %d 개", df.shape[1])
    logger.info("dtype 분포:\n%s", df.dtypes.value_counts().to_string())

    null_ratio = df.isna().mean().sort_values(ascending=False)
    logger.info("결측 90%% 이상 컬럼: %d 개", (null_ratio > 0.9).sum())
    logger.info("결측 없는 컬럼: %d 개", (null_ratio == 0).sum())


# float32 가 안전하게 표현할 수 있는 범위.
# 유효숫자 7자리를 넘으면 값이 뭉개지므로 여유를 두고 자른다.
FLOAT32_SAFE_MAX = 3.0e38

# 이 개수 이하의 고유값을 가진 문자열은 카테고리 후보로 본다.
# 이 데이터는 고유값이 2~5 다음 59 로 건너뛰어 그 사이가 비어 있다.
# 임계값을 이 구간 어디에 두든 결과가 같다.
CATEGORY_MAX_UNIQUE = 30

# 불리언으로 볼 값 쌍. 원본이 T/F 문자열로 저장되어 있다.
BOOLEAN_VALUES = {"T", "F"}


def profile_columns(source: str) -> pd.DataFrame:
    """컬럼별 권장 dtype 을 판정한다.

    익명 컬럼이라 이름으로는 타입을 알 수 없다. 실제 값을 보고
    아래 순서로 판정한다.

      전량 결측        -> drop      (담긴 정보가 없음)
      T/F 문자열       -> boolean   (nullable. 결측률이 높은 컬럼이 많다)
      저카디널리티 문자 -> category  (후보. 최종 판단은 schema 에서)
      그 외 문자열     -> str
      정수만 담김      -> Int*      (nullable 정수. 결측이 있어도 유지된다)
      float32 범위     -> float32   (기본. 메모리 절반)
      그 외 수치       -> float64
    """
    df = pd.read_csv(raw_path(source))
    rows = []

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        null_ratio = 1.0 - len(non_null) / len(s)
        values = None

        # pandas 3 에서 문자열은 object 가 아니라 str dtype 이다.
        if len(non_null) == 0:
            rec = ("drop", None, None)
        elif s.dtype == "str":
            uniq = set(non_null.unique())
            if uniq <= BOOLEAN_VALUES:
                dtype = "boolean"
            elif len(uniq) <= CATEGORY_MAX_UNIQUE:
                dtype = "category"
            else:
                dtype = "str"
            # 판정 근거를 실행을 통해 남긴다. 많으면 앞부분만.
            if len(uniq) <= CATEGORY_MAX_UNIQUE:
                values = "|".join(sorted(map(str, uniq)))
            rec = (dtype, None, None)
        else:
            lo, hi = float(non_null.min()), float(non_null.max())
            is_int = bool((non_null % 1 == 0).all())

            if is_int:
                # 결측이 있으면 nullable 정수여야 한다. numpy int 는 NaN 을 못 담는다.
                if lo >= -2_147_483_648 and hi <= 2_147_483_647:
                    dtype = "Int32"
                else:
                    dtype = "Int64"
            elif abs(lo) < FLOAT32_SAFE_MAX and abs(hi) < FLOAT32_SAFE_MAX:
                dtype = "float32"
            else:
                dtype = "float64"
            rec = (dtype, lo, hi)

        rows.append(
            {
                "column": col,
                "current_dtype": str(s.dtype),
                "recommended": rec[0],
                "null_ratio": round(null_ratio, 4),
                "nunique": int(non_null.nunique()) if len(non_null) else 0,
                "min": rec[1],
                "max": rec[2],
                "values": values,
            }
        )

    out = pd.DataFrame(rows)
    logger.info(
        "[%s] 권장 dtype 분포:\n%s",
        source,
        out["recommended"].value_counts().to_string(),
    )

    mem_before = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("[%s] 현재 메모리 %.0f MB", source, mem_before)

    return out


def compare_schemas() -> None:
    """train/test 간 컬럼 구성이 일치하는지 확인한다.

    불일치가 있으면 staging 에서 정규화해야 한다.
    identity 파일은 컬럼명이 서로 다르다.
    이외에도 컬럼 구성이 다른 게 있는지 확인한다.
    """
    for kind in ("transaction", "identity"):
        tr = pd.read_csv(raw_path(f"train_{kind}"), nrows=5).columns
        te = pd.read_csv(raw_path(f"test_{kind}"), nrows=5).columns

        only_train = sorted(set(tr) - set(te))
        only_test = sorted(set(te) - set(tr))

        logger.info("[%s] train %d 열 / test %d 열", kind, len(tr), len(te))
        if only_train:
            logger.info("  train 에만: %s", only_train[:10])
        if only_test:
            logger.info("  test 에만: %s", only_test[:10])
        if not only_train and not only_test:
            logger.info("  컬럼 구성 일치")

        # 하이픈/언더스코어만 다른 쌍을 찾는다. BigQuery 는 컬럼명에 하이픈을 못 쓴다.(확인 필요)
        def norm(c: str) -> str:
            return c.replace("-", "_")

        renamed = [(a, b) for a in only_train for b in only_test if norm(a) == norm(b)]
        if renamed:
            logger.info(
                "  정규화하면 같아지는 쌍 %d 개 (예: %s)", len(renamed), renamed[:3]
            )


def check_join_integrity() -> None:
    """transaction 과 identity 의 조인 관계를 확인한다.

    조인 후 행 수가 늘면 1:N 이므로 집계 로직이 달라진다.
    dbt 에서 이 성질을 테스트로 고정해야 한다.(확인 필요)
    """
    for split in ("train", "test"):
        tx = pd.read_csv(raw_path(f"{split}_transaction"), usecols=["TransactionID"])
        idn = pd.read_csv(raw_path(f"{split}_identity"), usecols=["TransactionID"])

        tx_ids, idn_ids = tx["TransactionID"], idn["TransactionID"]

        logger.info("[%s] transaction %d 행 / identity %d 행", split, len(tx), len(idn))
        logger.info(
            "  TransactionID 유일성 — transaction %s / identity %s",
            "OK" if tx_ids.is_unique else "중복 있음",
            "OK" if idn_ids.is_unique else "중복 있음",
        )

        matched = idn_ids.isin(set(tx_ids)).sum()
        logger.info(
            "  identity 매칭 %d / %d (고아 %d)",
            matched,
            len(idn),
            len(idn) - matched,
        )
        logger.info(
            "  transaction 중 identity 보유 %.1f%%",
            100 * tx_ids.isin(set(idn_ids)).mean(),
        )


if __name__ == "__main__":
    out_dir = get_settings().data_processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_schema()

    daily = profile_transactions()
    daily.to_csv(out_dir / "daily_profile.csv", index=False)
    logger.info("일별 집계 저장: %s", out_dir / "daily_profile.csv")

    logger.info("=== 파일 간 스키마 비교 ===")
    compare_schemas()

    logger.info("=== 조인 무결성 ===")
    check_join_integrity()

    logger.info("=== 컬럼별 권장 dtype ===")
    for source in ("train_transaction", "train_identity"):
        cols = profile_columns(source)
        path = out_dir / f"column_profile_{source}.csv"
        cols.to_csv(path, index=False)
        logger.info("저장: %s", path)
