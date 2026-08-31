"""features.build 의 계약.

핵심은 학습과 추론의 입력 형태가 하나의 형태로 일치하는 것이다. 하나로 일치하지 않으면 에러가 나지 않고 점수만 조용히 틀리므로, 눈으로는 발견되지 않는다.

대부분 BigQuery 없이 돈다. 실제 데이터로 확인해야 하는 것만 bq 마커를 단다.
"""

import pandas as pd
import pytest

from src.ml.dataset import load
from src.ml.features import DOMAIN_MIN_COUNT, MISSING, build, fit_domains


def _frame(domains: list[str], amounts: list[float] | None = None) -> pd.DataFrame:
    n = len(domains)
    return pd.DataFrame(
        {
            "transaction_amt": amounts or [30.0] * n,
            "product_cd": ["W"] * n,
            "purchaser_email_domain": domains,
        }
    )


def test_같은_domains_면_입력이_달라도_범주가_같다():
    """학습과 추론이 같은 범주를 써야 모델이 학습한 것을 그대로 쓴다.

    domains 를 넘기지 않으면 각 데이터의 빈도로 범주가 정해져, 작은 쪽에서
    희소 도메인이 other 로 뭉갠다.
    """
    dom = {"purchaser_email_domain": ["gmail.com", "outlook.com"]}

    big = build(_frame(["gmail.com"] * 50 + ["outlook.com"] * 50), domains=dom)
    small = build(_frame(["gmail.com", "outlook.com"]), domains=dom)

    assert set(big["purchaser_email_domain"].cat.categories) == set(
        small["purchaser_email_domain"].cat.categories
    )


def test_domains_를_빠뜨리면_범주가_갈린다():
    """고정하지 않으면 무슨 일이 생기는지 고정한다.

    이것이 참이기 때문에 fit_domains 가 필요하다. 이 테스트가 깨졌다면
    빈도 기준이 사라진 것이므로 domains 인자의 존재 이유를 다시 봐야 한다.
    """
    big = build(_frame(["gmail.com"] * DOMAIN_MIN_COUNT + ["rare.com"] * 5))
    assert "rare.com" not in set(big["purchaser_email_domain"].cat.categories)


def test_목록에_없는_도메인은_other_가_된다():
    out = build(
        _frame(["gmail.com", "unseen.com"]),
        domains={"purchaser_email_domain": ["gmail.com"]},
    )
    assert out["purchaser_email_domain"].tolist() == ["gmail.com", "other"]


def test_결측은_버리지_않고_범주로_남는다():
    """비어 있다는 사실 자체가 신호다. 도메인은 17.9% 가 결측이다."""
    out = build(
        _frame(["gmail.com", None]), domains={"purchaser_email_domain": ["gmail.com"]}
    )
    assert MISSING in out["purchaser_email_domain"].tolist()


def test_금액_구간은_경계를_왼쪽에_포함한다():
    """25 는 25-50 에 들어간다. 경계가 흔들리면 사기율 측정과 어긋난다."""
    out = build(_frame(["gmail.com"] * 3, amounts=[24.99, 25.0, 25.01]))
    assert out["amount_band"].tolist() == ["<25", "25-50", "25-50"]


def test_원본을_바꾸지_않는다():
    """같은 DataFrame 을 두 번 넣어도 결과가 같아야 한다."""
    df = _frame(["gmail.com", "unseen.com"])
    before = df.copy()
    build(df)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.bq
def test_실제_split_에서_범주가_일치한다():
    """train 19 종 / valid 7 종으로 갈렸던 실제 사례를 고정한다.

    valid 가 6배 작아 빈도 기준을 못 넘는 도메인이 많았다. 사기율이 가장
    높은 outlook.com 이 valid 에서 other 에 섞이면 평가가 실제보다 낮게 나온다.
    """
    cols = ["transaction_amt", "product_cd", "purchaser_email_domain"]
    train_x = load("train", columns=cols).X
    valid_x = load("valid", columns=cols).X

    dom = fit_domains(train_x)
    a = build(train_x, domains=dom)
    b = build(valid_x, domains=dom)

    cats_a = set(a["purchaser_email_domain"].cat.categories)
    cats_b = set(b["purchaser_email_domain"].cat.categories)
    assert cats_a == cats_b
    assert "outlook.com" in cats_b
