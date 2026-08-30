"""dataset.load 의 계약.

에러를 발생시키지 않고 성공할 수 있는 경우의 수를 테스트 한다.
데이터 누수 방지를 확인한다.

BigQuery 를 실제로 조회하므로 느리다. -m "not bq" 로 제외할 수 있다.
"""

import pytest

from src.ml.dataset import load, load_training

pytestmark = pytest.mark.bq

# 계약을 보는 데 컬럼 전체가 필요하지 않다. 397 개를 다 읽으면 조회가
# 느려서 커밋 때마다 돌리기 어려워진다.
COLUMNS = ["transaction_amt", "product_cd", "card1"]


@pytest.fixture(scope="module")
def train():
    return load("train", columns=COLUMNS)


@pytest.fixture(scope="module")
def valid():
    return load("valid", columns=COLUMNS)


def test_시간_순서가_지켜진다(train, valid):
    """train 이 valid 보다 먼저 끝나야 한다. 겹치면 미래를 학습한다."""
    assert train.dates.max() < valid.dates.min()


@pytest.mark.parametrize("col", ["is_fraud", "transaction_dt", "source_split"])
def test_누수_컬럼이_피처에_없다(train, col):
    assert col not in train.X.columns


def test_test_split_은_라벨이_없다():
    """추론 전용이다. 학습에 쓰면 y is None 에서 드러난다."""
    assert load("test", columns=["transaction_amt"]).y is None


def test_라벨에_결측이_없다(train, valid):
    for ds in (train, valid):
        assert ds.y.notna().all()


def test_알_수_없는_split_은_거부한다():
    with pytest.raises(ValueError):
        load("training")


def test_학습_경로는_test_를_거부한다():
    """라벨이 없는 구간으로 학습하는 실수를 조회 전에 막는다.

    load 는 추론 때문에 test 를 받아야 하므로, 막는 것은 진입점의 몫이다.
    """
    with pytest.raises(ValueError):
        load_training("test", columns=COLUMNS)
