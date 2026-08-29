"""TransactionDT 기준일 검증 (일회성 조사 스크립트).

파이프라인에 포함되지 않는다. config.transaction_dt_origin 을 임의값으로
둔 근거를 남기기 위해 보관한다.

대회는 기준 시점을 공개하지 않았다. 미국 소비 데이터라면 공휴일에 거래량이
특징적으로 변한다는 가정으로 기준일 후보를 전수 탐색해 채점했다.

결론: 특정 실패. 상위 후보가 여러 해에 흩어지고 점수 차이도 작아 판별력이
없다. 기준일이 필요한 이유는 날짜 타입 파티션 키를 얻기 위해서일 뿐이고
절대 날짜의 정확성은 어떤 계산에도 영향을 주지 않으므로, 커뮤니티 추정값을
임의로 채택했다.

실행: uv run python -m analysis.verify_origin
"""

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.extract.loader import raw_path

logger = get_logger(__name__)

SECONDS_PER_DAY = 86_400

# 검증 대상 공휴일과 '예상되는 거래량 방향'.
# -1 = 평소보다 급감 / +1 = 평소보다 급증
FIXED_HOLIDAYS = {
    (12, 24): ("크리스마스이브", -1),
    (12, 25): ("크리스마스", -1),
    (12, 26): ("크리스마스 다음날", -1),
    (1, 1): ("새해", -1),
    (7, 4): ("독립기념일", -1),
}


def thanksgiving(year: int) -> pd.Timestamp:
    """11월 넷째 목요일."""
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    thursdays = nov[nov.dayofweek == 3]
    return thursdays[3]


def build_holidays(years: range) -> dict[pd.Timestamp, tuple[str, int]]:
    out: dict[pd.Timestamp, tuple[str, int]] = {}
    for y in years:
        for (m, d), spec in FIXED_HOLIDAYS.items():
            out[pd.Timestamp(year=y, month=m, day=d)] = spec
        tg = thanksgiving(y)
        out[tg] = ("추수감사절", -1)
        out[tg + pd.Timedelta(days=1)] = ("블랙프라이데이", +1)
        out[tg + pd.Timedelta(days=4)] = ("사이버먼데이", +1)
    return out


def daily_counts() -> pd.Series:
    """day_index 별 거래 건수. 기준일과 무관하게 계산된다."""
    df = pd.read_csv(raw_path("train_transaction"), usecols=["TransactionDT"])
    day = df["TransactionDT"] // SECONDS_PER_DAY
    return day.value_counts().sort_index()


def deseasonalized_z(s: pd.Series) -> pd.Series:
    """요일 효과와 추세를 제거한 뒤 z-score 로 표준화한다.

    1) 28일 중심이동중앙값으로 나눠 장기 추세를 제거
    2) 요일별 중앙값으로 다시 나눠 주간 주기를 제거
    3) 남은 잔차를 z-score 로 변환

    반환값의 부호는 거래량 방향과 같다. 음수면 평소보다 적고, 양수면 많다.
    """
    trend = s.rolling(28, center=True, min_periods=7).median()
    detrended = s / trend

    dow_median = detrended.groupby(detrended.index.dayofweek).transform("median")
    resid = detrended / dow_median

    return (resid - resid.mean()) / resid.std()


def score_origin(counts: pd.Series, origin: pd.Timestamp) -> dict:
    """기준일 후보 하나를 확인한다.

    공휴일마다 '예상 방향(급증/급감)'이 정해져 있으므로, z-score 에 그 방향을
    곱해 더한다. 방향까지 맞으면 점수가 오르고 반대로 나오면 깎인다.

    후보마다 범위 안에 들어오는 공휴일 수가 다르므로 합이 아니라 평균을 쓴다.
    """
    dates = origin + pd.to_timedelta(counts.index, unit="D")
    s = pd.Series(counts.to_numpy(), index=dates)
    z = deseasonalized_z(s)

    holidays = build_holidays(range(dates.min().year, dates.max().year + 1))
    detail: list[tuple] = []
    scores: list[float] = []

    for h, (name, direction) in sorted(holidays.items()):
        if h not in z.index or pd.isna(z[h]):
            continue
        signed = float(z[h]) * direction  # 방향이 맞으면 양수
        scores.append(signed)
        detail.append((h.date(), name, direction, round(float(z[h]), 2)))

    return {
        "origin": origin.date(),
        "n_holidays": len(scores),
        "score": float(np.mean(scores)) if scores else 0.0,
        "detail": detail,
    }


if __name__ == "__main__":
    counts = daily_counts()
    logger.info(
        "day_index %d ~ %d (%d 일)", counts.index.min(), counts.index.max(), len(counts)
    )

    # 3년치를 전부 훑는다. 특정 연도를 가정하지 않기 위해서다.
    candidates = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    results = [score_origin(counts, c) for c in candidates]

    # 공휴일이 너무 적게 걸리는 후보는 비교 대상에서 뺀다.
    valid = [r for r in results if r["n_holidays"] >= 5]
    ranked = sorted(valid, key=lambda r: r["score"], reverse=True)

    all_scores = np.array([r["score"] for r in valid])
    logger.info(
        "후보 %d 개 - 점수 평균 %.3f, 표준편차 %.3f",
        len(valid),
        all_scores.mean(),
        all_scores.std(),
    )

    logger.info("=== 상위 10 후보 ===")
    for r in ranked[:10]:
        z_vs_others = (r["score"] - all_scores.mean()) / all_scores.std()
        logger.info(
            "%s  점수 %+.3f  (대조군 대비 %+.2f σ, 공휴일 %d 개)",
            r["origin"],
            r["score"],
            z_vs_others,
            r["n_holidays"],
        )

    best = ranked[0]
    logger.info("=== 최적 후보 %s 상세 ===", best["origin"])
    logger.info("  (방향: -1=급감 예상, +1=급증 예상 / z: 실제 관측값)")
    for d, name, direction, z in best["detail"]:
        mark = "O" if z * direction > 0 else "X"
        logger.info("  %s %s  %-16s 방향 %+d  z %+.2f", mark, d, name, direction, z)

    # 커뮤니티 통용값과 비교
    community = next(r for r in results if str(r["origin"]) == "2017-12-01")
    c_z = (community["score"] - all_scores.mean()) / all_scores.std()
    c_rank = sum(1 for r in valid if r["score"] > community["score"]) + 1
    logger.info(
        "커뮤니티 통용값 2017-12-01: 점수 %+.3f (%+.2f σ), %d 위 / %d",
        community["score"],
        c_z,
        c_rank,
        len(valid),
    )
