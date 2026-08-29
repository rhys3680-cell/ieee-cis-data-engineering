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

logger = get_logger(__name__)

SECONDS_PER_DAY = 86_400


def profile_transactions() -> pd.DataFrame:
    """거래 데이터의 시간축 분포를 집계한다."""
    settings = get_settings()

    # 시간축 분석에 필요한 것만 읽는다.
    cols = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
    df = pd.read_csv(raw_path("train_transaction"), usecols=cols)

    origin = pd.Timestamp(settings.transaction_dt_origin)
    df["ts"] = origin + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["dt"] = df["ts"].dt.date

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


if __name__ == "__main__":
    profile_schema()
    daily = profile_transactions()

    out = get_settings().data_processed_dir / "daily_profile.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out, index=False)
    logger.info("일별 집계 저장: %s", out)
