"""점수를 어디서 구간별로 나눌지 비용으로 정한다.

    uv run python -m src.ml.threshold                  # lgbm-all
    uv run python -m src.ml.threshold lgbm-curated

모델은 0~1 점수를 주지만 실무에서 필요한 것은 "거래를 차단할지"를 결정해야 한다. 그 경계가
임계값이고, 통계가 아니라 비용이 정한다. PR-AUC 가 높아도 임계값이 틀리면
손실이 난다.

두 종류의 실수가 값이 다르다.

    놓친 사기(FN)   그 거래액만큼 잃는다.        데이터에 있다.
    막은 정상(FP)   고객 이탈, 재승인 처리 비용.  데이터에 없다.

FN은 transaction_amt 로 계산한다. FP는 회사마다 다르고 IEEE-CIS 에 없으므로
가정해야 한다 — fp_cost 하나가 이 모듈의 유일한 가정이다.

그래서 결론은 임계값 하나가 아니라 "fp_cost 가 이만큼이면 임계값은 이만큼"
이라는 대응이다. sensitivity 가 그 대응을 낸다. 가정에 둔감하면 안심하고
고르면 되고, 민감하면 fp_cost 를 진지하게 추정해야 한다는 뜻이다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.logging import get_logger

logger = get_logger(__name__)

# 훑을 임계값. 0 과 1 은 넣지 않는다 — 전부 막거나 전부 통과시키는 것은
# 모델을 쓰지 않는 것과 같아 비교 대상이 아니다.
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

# 민감도를 볼 fp_cost 후보(달러). 정상 거래 한 건을 잘못 막았을 때의 비용을
# 얼마로 보느냐다. 재승인 안내 한 번이면 몇 달러지만, 고객이 떠나면 그
# 고객의 생애가치를 잃는다. 업계 조사에서 오탐을 겪은 소비자의 38% 가
# 거래처를 바꾸고, 단골은 이후 주문량이 65% 줄어든다고 한다.
#
# $100 까지 보는 이유는 거기서 경계가 드러나기 때문이다. 그 위로 가면
# 총비용이 아무것도 막지 않을 때($441,297)에 수렴해 모델을 쓰지 않는 편이
# 나은 영역이 된다 — $400 에서 $417,389 로 5% 차이밖에 나지 않는다.
FP_COSTS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class Outcome:
    """임계값 하나를 적용했을 때 무슨 일이 일어나는가."""

    threshold: float
    # 막은 사기 거래 / 놓친 사기 거래 / 막은 정상 거래
    caught: int
    missed: int
    false_alarms: int
    # 놓친 사기의 거래액 합계. 이것이 실제 손실이다.
    missed_amount: float
    total_cost: float

    @property
    def recall(self) -> float:
        """사기 중 몇 %를 잡았는가."""
        total = self.caught + self.missed
        return self.caught / total if total else 0.0

    @property
    def precision(self) -> float:
        """막은 것 중 몇 %가 실제 사기였는가."""
        blocked = self.caught + self.false_alarms
        return self.caught / blocked if blocked else 0.0


def cost(
    y_true: pd.Series,
    scores: np.ndarray,
    amounts: pd.Series,
    threshold: float,
    fp_cost: float,
) -> Outcome:
    """임계값 하나의 결과를 낸다.

    금액 기준과 건수 기준을 섞는다. 놓친 사기는 금액으로, 막은 정상은
    건수로 센다 — 정상 거래를 막는 비용은 거래액이 아니라 처리 비용이라
    금액에 비례하지 않는다.
    """
    blocked = scores >= threshold
    is_fraud = y_true.to_numpy().astype(bool)
    amt = amounts.to_numpy()

    caught = blocked & is_fraud
    missed = ~blocked & is_fraud
    false_alarms = blocked & ~is_fraud

    missed_amount = float(amt[missed].sum())
    return Outcome(
        threshold=float(threshold),
        caught=int(caught.sum()),
        missed=int(missed.sum()),
        false_alarms=int(false_alarms.sum()),
        missed_amount=missed_amount,
        total_cost=missed_amount + float(false_alarms.sum()) * fp_cost,
    )


def sweep(
    y_true: pd.Series,
    scores: np.ndarray,
    amounts: pd.Series,
    fp_cost: float,
    grid: np.ndarray = GRID,
) -> pd.DataFrame:
    """임계값을 훑어 비용 곡선을 낸다. threshold 순으로 정렬된 표."""
    rows = [cost(y_true, scores, amounts, t, fp_cost) for t in grid]
    return pd.DataFrame(
        [
            {
                "threshold": r.threshold,
                "caught": r.caught,
                "missed": r.missed,
                "false_alarms": r.false_alarms,
                "missed_amount": r.missed_amount,
                "total_cost": r.total_cost,
                "recall": r.recall,
                "precision": r.precision,
            }
            for r in rows
        ]
    )


def pick(
    y_true: pd.Series,
    scores: np.ndarray,
    amounts: pd.Series,
    fp_cost: float,
    grid: np.ndarray = GRID,
) -> Outcome:
    """비용이 가장 작은 임계값. 같은 비용이면 낮은 쪽을 고른다.

    낮은 쪽이 사기를 더 잡는다. 비용이 같다면 놓치는 것보다 막아 두는 편이
    회수 가능성이 있다.
    """
    outcomes = [cost(y_true, scores, amounts, t, fp_cost) for t in grid]
    return min(outcomes, key=lambda o: (o.total_cost, o.threshold))


def sensitivity(
    y_true: pd.Series,
    scores: np.ndarray,
    amounts: pd.Series,
    fp_costs: tuple[float, ...] = FP_COSTS,
    grid: np.ndarray = GRID,
) -> pd.DataFrame:
    """fp_cost 를 바꿔가며 최적 임계값이 어떻게 움직이는지 본다.

    이 표가 Phase 9 화면의 내용이다. 가정에 따른 변화량을 확인한다.
    """
    rows = []
    for c in fp_costs:
        best = pick(y_true, scores, amounts, c, grid)
        rows.append(
            {
                "fp_cost": c,
                "threshold": best.threshold,
                "recall": best.recall,
                "precision": best.precision,
                "caught": best.caught,
                "missed": best.missed,
                "false_alarms": best.false_alarms,
                "missed_amount": best.missed_amount,
                "total_cost": best.total_cost,
            }
        )
    return pd.DataFrame(rows)


def main(model_name: str = "lgbm-all") -> pd.DataFrame:
    """저장한 모델로 valid 에서 민감도 표를 낸다.

    재학습하지 않는다. model_store 가 모델과 범주 목록을 함께 들고 있어
    학습 때와 같은 입력으로 점수를 다시 낼 수 있다.
    """
    # 지연 임포트. 이 모듈의 계산 함수들은 BigQuery 를 모르는데, 모듈을
    # 임포트하는 것만으로 조회 경로가 딸려오면 테스트가 느려진다.
    from src.ml.dataset import load_training
    from src.ml.features import FeatureSet, build
    from src.ml.model_store import load
    from src.ml.train import COLUMNS

    bundle = load(model_name)
    # 학습 때 읽은 것과 같은 범위를 읽어야 align 이 성립한다.
    columns = None if bundle.feature_set is FeatureSet.ALL else COLUMNS
    valid = load_training("valid", columns=columns)

    X = bundle.align(build(valid.X, domains=bundle.domains))
    scores = bundle.model.predict_proba(X)[:, 1]
    amounts = valid.X["transaction_amt"]

    table = sensitivity(valid.y, scores, amounts, grid=GRID)

    no_model = float(amounts[valid.y == 1].sum())
    logger.info("모델: %s (%s)", model_name, bundle.feature_set)
    logger.info("아무것도 막지 않으면 손실 $%s", f"{no_model:,.0f}")
    for row in table.itertuples():
        logger.info(
            "fp_cost $%-5.0f 임계값 %.2f  탐지 %.1f%%  정밀도 %.1f%%  비용 $%s",
            row.fp_cost,
            row.threshold,
            row.recall * 100,
            row.precision * 100,
            f"{row.total_cost:,.0f}",
        )
    return table


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "lgbm-all")
