"""score 의 계약.

학습과 추론이 같은 함수를 부르게 하려고 만든 모듈이라, 여기서 볼 것은
"입력이 학습 때의 형태로 맞춰지는가" 뿐이다. 점수가 얼마인지는 모델의
성질이지 이 코드의 계약이 아니다.

BigQuery 없이 돈다.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.ml.features import FeatureSet
from src.ml.model_store import Bundle
from src.ml.predict import prepare, score

# build 가 만들어낼 컬럼까지 포함한 학습 때의 구성.
TRAINED_COLUMNS = ["transaction_amt", "product_cd", "amount_band"]


class _EchoModel:
    """첫 컬럼 값을 그대로 점수로 돌려준다.

    열 순서가 어긋나면 다른 값이 나오므로 정렬이 실제로 되는지 보인다.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = X.iloc[:, 0].to_numpy(dtype=float)
        return np.column_stack([1 - positive, positive])


def _bundle() -> Bundle:
    return Bundle(
        model=_EchoModel(),
        domains={"purchaser_email_domain": ["gmail.com"]},
        columns=TRAINED_COLUMNS,
        feature_set=FeatureSet.CURATED,
        metrics={},
        trained_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


def _frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {"transaction_amt": [10.0, 30.0, 300.0][:n], "product_cd": ["W"] * n}
    )


def test_파생을_얹어_학습_때_구성으로_맞춘다():
    """입력에 없던 amount_band 가 build 를 거쳐 생긴다."""
    out = prepare(_frame(), _bundle())
    assert list(out.columns) == TRAINED_COLUMNS


def test_컬럼_순서가_어긋나도_되돌린다():
    """BigQuery 가 다른 순서로 주면 모델이 다른 자리의 값을 읽는다."""
    df = _frame()[["product_cd", "transaction_amt"]]
    assert list(prepare(df, _bundle()).columns) == TRAINED_COLUMNS


def test_학습_때_있던_컬럼이_없으면_실패한다():
    """조용히 채우면 그 열이 결측인 채로 점수가 나온다."""
    with pytest.raises(ValueError, match="컬럼"):
        prepare(pd.DataFrame({"transaction_amt": [10.0]}), _bundle())


def test_행_순서와_개수가_보존된다():
    """점수를 원본 행에 되붙일 수 있어야 한다."""
    df = _frame()
    scores = score(df, _bundle())

    assert len(scores) == len(df)
    # _EchoModel 이 첫 컬럼(transaction_amt)을 그대로 돌려준다.
    assert scores.tolist() == [10.0, 30.0, 300.0]


def test_원본을_바꾸지_않는다():
    """같은 DataFrame 을 두 번 채점해도 결과가 같아야 한다."""
    df = _frame()
    before = df.copy()

    first = score(df, _bundle())
    second = score(df, _bundle())

    pd.testing.assert_frame_equal(df, before)
    assert first.tolist() == second.tolist()


def test_임계값을_적용하지_않는다():
    """점수와 판정을 나눠야 임계값이 바뀌어도 재추론이 필요 없다."""
    scores = score(_frame(), _bundle())
    assert scores.dtype.kind == "f"
    assert not set(np.unique(scores)) <= {0.0, 1.0}