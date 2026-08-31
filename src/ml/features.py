"""파생 피처를 만든다.

학습과 추론이 이 함수를 같이 부른다. 규칙이 두 곳에 있으면 실수로 한 곳만 변경할 수 있는데,
그때 에러가 나지 않고 점수만 조용히 틀린다. 라벨이 없는 구간을 채점하는
파이프라인이라 틀린 것을 나중에도 알아채기 어렵다.

넣은 것은 train 에서 판별력을 확인한 것뿐이다. 사기율이 몇 배 갈리는지 재고
그 값을 주석에 남긴다.

    transaction_hour   3.00배   (dataset.py 가 만든다)
    has_identity       3.76배   (dataset.py 가 만든다)
    email_domain       4.08배   9.46% ~ 2.32%
    amount_band        2.43배   6.97% ~ 2.87%

요일은 제외한다. mod(div(transaction_dt, 86400), 7) 로 재보니 3.15~3.72% 로
1.18배에 그친다. 기준일이 임의값이라 7일 주기가 실제 요일과 정확히 일치되어 있지
않고, 공휴일로 기준일을 역산하려던 시도도 실패했다(analysis/verify_origin.py).
"""

import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

# 금액 구간. 사기율이 U 자를 그린다 — 소액 6.97%, 중간 2.87%, 고액 4.72%.
#
# log 변환을 쓰지 않는 이유가 여기 있다. 로그는 단조 관계를 펴는 변환이라
# U 자에는 듣지 않고, 트리 모델은 애초에 단조 변환에 불변이라 분기점만
# 바뀐다. 구간을 범주로 주면 선형 모델도 U 자를 잡을 수 있다.
AMOUNT_BINS = [0, 25, 50, 100, 250, 500, float("inf")]
AMOUNT_LABELS = ["<25", "25-50", "50-100", "100-250", "250-500", "500+"]

# 표기가 갈린 도메인을 합친다. gmail 과 gmail.com 이 따로 있고 yahoo 는
# 국가별로 일곱 개다. 같은 서비스를 다른 범주로 두면 각각의 표본이 줄어
# 사기율 추정이 흔들린다.
DOMAIN_ALIASES = {
    "gmail": "gmail.com",
    "yahoo.co.jp": "yahoo.com",
    "yahoo.co.uk": "yahoo.com",
    "yahoo.com.mx": "yahoo.com",
    "yahoo.de": "yahoo.com",
    "yahoo.es": "yahoo.com",
    "yahoo.fr": "yahoo.com",
    "hotmail.co.uk": "hotmail.com",
    "hotmail.de": "hotmail.com",
    "hotmail.es": "hotmail.com",
    "hotmail.fr": "hotmail.com",
    "live.com.mx": "live.com",
    "outlook.es": "outlook.com",
}

# 이 개수 미만인 도메인은 other 로 묶는다. 59 종 중 대부분이 수백 건이라
# 그대로 두면 범주마다 사기 사례가 몇 건씩밖에 없다.
DOMAIN_MIN_COUNT = 1000

# 결측을 버리지 않고 범주로 남긴다. 도메인은 17.9% 가 비어 있는데, 비어
# 있다는 사실 자체가 신호일 수 있다 — 실제로 (null) 구간 사기율이 2.95% 로
# 전체 평균보다 낮다.
MISSING = "(missing)"

# 명시적으로 범주로 다룰 컬럼. 문자열 컬럼은 dtype 으로 따로 잡으므로
# 여기에는 amount_band 처럼 만들어낸 것과 이름이 확실한 것만 둔다.
CATEGORICAL = (
    "product_cd",
    "card4",
    "card6",
    "purchaser_email_domain",
    "recipient_email_domain",
    "amount_band",
)


def build(
    df: pd.DataFrame, domains: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """dataset.load 가 준 X 에 파생 컬럼을 추가한다.

    원본을 바꾸지 않고 복사본을 돌려준다. 학습 코드가 같은 DataFrame 을
    두 번 넣어도 결과가 같아야 한다.

    domains 는 도메인 범주 목록이다. 주지 않으면 이 데이터에서 빈도로
    정하고, 주면 그것을 쓴다. 학습 때 fit_domains 로 뽑아 두고 평가·추론에
    넘겨야 한다 — 아래 _normalize_domain 의 설명을 볼 것.
    """
    out = df.copy()

    if "transaction_amt" in out.columns:
        out["amount_band"] = pd.cut(
            out["transaction_amt"],
            bins=AMOUNT_BINS,
            labels=AMOUNT_LABELS,
            right=False,
        )

    for col in ("purchaser_email_domain", "recipient_email_domain"):
        if col in out.columns:
            keep = None if domains is None else domains.get(col)
            out[col] = _normalize_domain(out[col], keep)

    # 범주형은 category dtype 으로 넘긴다. LightGBM 이 직접 처리하므로
    # 원핫이 필요 없다 — 도메인만 59 종이라 원핫으로 펼치면 열이 급증한다.
    #
    # 받을 수 있는 것을 명시하고 나머지를 전부 바꾼다. 문자열 dtype 을
    # 열거하는 방식은 두 번 샜다 — 이름 목록일 때 M4 가 빠졌고, dtype 을
    # 열거해도 pandas 3 의 str/string/object 셋을 다 적어야 했다. 어느
    # 쪽이든 새로운 것이 들어오면 조용히 통과해 학습 직전에 LightGBM 이
    # "pandas dtypes must be int, float or bool" 로 죽는다.
    MODEL_READY = ["number", "bool", "boolean", "category"]
    other = list(out.select_dtypes(exclude=MODEL_READY).columns)

    unexpected = [c for c in other if str(out[c].dtype) not in ("object", "string", "str")]
    if unexpected:
        # 날짜 같은 것이 걸리면 category 로 바꿔 죽는 것은 막지만, 값마다
        # 범주가 생겨 카디널리티가 커지고 순서 정보를 잃는다. 제대로 쓰려면
        # 파생을 따로 만들어야 하므로 조용히 넘기지 않는다.
        logger.warning(
            "문자열이 아닌 컬럼을 범주로 바꾼다 — 파생이 필요할 수 있다: %s",
            {c: str(out[c].dtype) for c in unexpected},
        )

    for col in [*CATEGORICAL, *other]:
        if col in out.columns:
            out[col] = out[col].astype("object").fillna(MISSING).astype("category")

    added = [c for c in out.columns if c not in df.columns]
    logger.info("파생 %d 개 추가: %s", len(added), ", ".join(added) or "없음")
    return out


def fit_domains(df: pd.DataFrame) -> dict[str, list[str]]:
    """학습 데이터에서 도메인 범주 목록을 뽑는다.

    build 에 넘겨 평가와 추론이 같은 범주를 쓰게 한다. 학습 때 한 번 부르고
    모델과 함께 보관한다.
    """
    out = {}
    for col in ("purchaser_email_domain", "recipient_email_domain"):
        if col in df.columns:
            counts = df[col].astype("object").replace(DOMAIN_ALIASES).value_counts()
            out[col] = sorted(counts[counts >= DOMAIN_MIN_COUNT].index.dropna())
    return out


def _normalize_domain(s: pd.Series, keep: list[str] | None = None) -> pd.Series:
    """표기를 합치고 목록에 없는 것은 other 로 묶는다.

    keep 을 주지 않으면 이 데이터의 빈도로 정하는데, 그러면 데이터마다 범주가
    달라진다. 실제로 train 은 19 종, valid 는 7 종이 나왔다 — valid 가 6배
    작아 기준을 못 넘는 것이 많다. 그러면 사기율 9.46% 로 가장 높은
    outlook.com 이 valid 에서 other 에 섞여, 모델이 학습한 것을 평가에서
    쓰지 못한다. 성능이 실제보다 낮게 나오고 배치 추론에서는 더 심해진다.

    빈도는 라벨이 아니라 입력 분포에서 오는 값이라 누수는 아니지만, 학습과
    추론의 입력 형태가 갈리는 것은 막아야 한다. 그래서 학습 때 fit_domains 로
    목록을 고정해 넘긴다.
    """
    normalized = s.astype("object").replace(DOMAIN_ALIASES)

    if keep is None:
        counts = normalized.value_counts()
        keep = counts[counts >= DOMAIN_MIN_COUNT].index

    return normalized.where(normalized.isin(keep) | normalized.isna(), "other")
