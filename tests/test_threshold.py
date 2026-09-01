"""비용 계산과 임계값 선택의 계약.

BigQuery 없이 돈다. 손으로 셀 수 있는 작은 예제를 쓴다 — 비용 함수가
틀리면 임계값이 틀리고, 임계값이 틀리면 운영에서 돈이 샌다. 그런데 결과가
그럴듯한 숫자로 나오기 때문에 눈으로는 검증되지 않는다.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.threshold import cost, pick, sensitivity, sweep

# 사기 둘(100, 200달러), 정상 둘. 점수는 사기 쪽이 높다.
Y = pd.Series([1, 1, 0, 0])
SCORES = np.array([0.9, 0.6, 0.4, 0.1])
AMOUNTS = pd.Series([100.0, 200.0, 50.0, 30.0])


def test_손으로_센_비용과_같다():
    """임계값 0.7 이면 첫 사기(0.9)만 잡는다.

    놓친 사기 200달러 + 막은 정상 0건 = 200. 정상 거래는 둘 다 0.7 미만이라
    걸리지 않는다.
    """
    out = cost(Y, SCORES, AMOUNTS, threshold=0.7, fp_cost=10.0)

    assert (out.caught, out.missed, out.false_alarms) == (1, 1, 0)
    assert out.missed_amount == 200.0
    assert out.total_cost == 200.0


def test_임계값과_같은_점수는_막는다():
    """blocked = scores >= threshold 다.

    부등호가 뒤집히면 경계에 걸린 거래의 처리가 바뀐다. 점수가 임계값에
    정확히 걸리는 경우는 드물지만, 격자를 0.01 단위로 훑으므로 실제로 생긴다.
    """
    out = cost(Y, SCORES, AMOUNTS, threshold=0.6, fp_cost=10.0)

    assert out.caught == 2  # 0.9 와 0.6 둘 다
    assert out.missed == 0


def test_정상을_막으면_건수로_센다():
    """임계값 0.3 이면 정상 하나(0.4)가 걸린다.

    놓친 사기 0 + 막은 정상 1건 x 10달러 = 10. 거래액 50달러가 아니라
    처리 비용 10달러다 — 정상을 막는 비용은 금액에 비례하지 않는다.
    """
    out = cost(Y, SCORES, AMOUNTS, threshold=0.3, fp_cost=10.0)

    assert (out.caught, out.missed, out.false_alarms) == (2, 0, 1)
    assert out.total_cost == 10.0


def test_전부_통과시키면_사기_거래액_전부를_잃는다():
    """모델을 쓰지 않는 것과 같은 상태. 비교의 기준선이다."""
    out = cost(Y, SCORES, AMOUNTS, threshold=1.0, fp_cost=10.0)

    assert out.caught == 0
    assert out.total_cost == 300.0  # 100 + 200


def test_fp_cost_가_0_이면_전부_막는_쪽으로_간다():
    """정상을 막는 비용이 없으면 놓치지 않는 것만 이득이다.

    이것이 성립하지 않으면 비용 함수의 부호나 방향이 틀린 것이다.
    """
    best = pick(Y, SCORES, AMOUNTS, fp_cost=0.0)

    assert best.missed == 0
    assert best.total_cost == 0.0


def test_fp_cost_가_커지면_임계값이_올라간다():
    """정상을 막는 것이 비싸질수록 조심스러워진다.

    민감도 표가 이 방향으로 정렬되지 않으면 해석이 뒤집힌다.
    """
    table = sensitivity(Y, SCORES, AMOUNTS, fp_costs=(0.0, 1.0, 1000.0))
    thresholds = table["threshold"].tolist()

    assert thresholds == sorted(thresholds)


def test_비용은_음수가_될_수_없다():
    for t in (0.05, 0.5, 0.95):
        assert cost(Y, SCORES, AMOUNTS, t, fp_cost=10.0).total_cost >= 0


def test_같은_비용이면_낮은_임계값을_고른다():
    """낮은 쪽이 사기를 더 잡는다. 놓치는 것보다 막아 두는 편이 낫다."""
    y = pd.Series([1, 0])
    scores = np.array([0.9, 0.1])
    amounts = pd.Series([100.0, 100.0])

    # 0.2~0.9 어디서 잘라도 결과가 같다. 가장 낮은 값이 나와야 한다.
    best = pick(y, scores, amounts, fp_cost=1.0, grid=np.array([0.2, 0.5, 0.9]))
    assert best.threshold == 0.2


def test_sweep_은_격자를_모두_돌려준다():
    grid = np.array([0.1, 0.5, 0.9])
    table = sweep(Y, SCORES, AMOUNTS, fp_cost=10.0, grid=grid)

    assert len(table) == 3
    assert table["threshold"].tolist() == [0.1, 0.5, 0.9]


@pytest.mark.parametrize(
    ("threshold", "expected_recall"),
    [(0.05, 1.0), (0.7, 0.5), (0.95, 0.0)],
)
def test_탐지율은_사기_중_잡은_비율이다(threshold, expected_recall):
    assert cost(Y, SCORES, AMOUNTS, threshold, 10.0).recall == expected_recall