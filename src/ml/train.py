"""베이스라인을 학습하고 valid 에서 비교한다.

    uv run python -m src.ml.train                # curated 피처
    uv run python -m src.ml.train --all-columns  # 익명 컬럼까지

목적은 성능이 아니라 파이프라인이 누수 없이 도는지 확인하는 것이다. 그래서
피처를 판별력이 확인된 것으로 제한하고, 익명 컬럼(V*, C*, D*, M*)은 넣지
않는다. 처음부터 387 개를 넣으면 점수가 좋아도 나빠도 이유를 알 수 없다.

세 모델은 각각 다음의 역할을 가진다.

    dummy    baselien 파악. 항상 다수 클래스를 찍는 모델의 점수다.
    logreg   해석 가능한 선형 관계만으로 어디까지 가는가.
    lgbm     비선형과 상호작용을 넣으면 얼마나 나아지는가.

lgbm 이 logreg 를 크게 못 이기면 피처가 부족한 것이고, 지나치게 이기면
(ROC-AUC 0.99 같은) 누수를 의심해야 한다.

지표는 PR-AUC 를 본다. 사기율이 3.5% 라 ROC-AUC 는 낙관적으로 나온다 —
음성이 압도적이라 대부분을 음성으로 찍기만 해도 점수가 오른다. PR-AUC 는
양성 쪽만 보므로 불균형에서 정직하다. 바닥값이 기저 사기율(0.035)과 같아
'모델이 무엇이든 배웠는가' 를 바로 읽을 수 있다.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.logging import get_logger
from src.ml.dataset import load_training
from src.ml.features import build, fit_domains

logger = get_logger(__name__)

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "ieee-cis-baseline"

# 판별력을 확인한 것만 읽는다. features.build 가 여기에 amount_band 를
# 더하고, dataset.load 가 transaction_hour 와 has_identity 를 만든다.
COLUMNS = [
    "transaction_amt",
    "product_cd",
    "card4",
    "card6",
    "purchaser_email_domain",
    "recipient_email_domain",
]

SEED = 42


def _logreg() -> Pipeline:
    """범주형은 원핫, 수치는 표준화한다.

    선형 모델이라 스케일이 다르면 계수가 왜곡된다. 트리 모델에는 둘 다
    필요 없어서 lgbm 은 원본을 그대로 받는다.

    curated 피처에서만 돈다(FEATURE_SETS 참고). 이 모델을 두는 이유가
    계수를 읽을 수 있다는 것인데, 의미를 모르는 익명 컬럼 378 개를 넣으면
    그 이유가 사라진다.

    handle_unknown='ignore' 는 방어선이다. 도메인 범주는 fit_domains 로
    고정하지만, 그것을 빠뜨렸을 때 추론이 죽는 대신 그 범주를 0 으로
    처리하고 넘어간다.
    """
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            lambda d: d.select_dtypes(["category", "boolean"]).columns,
                        ),
                        (
                            "num",
                            StandardScaler(),
                            lambda d: d.select_dtypes(["number"]).columns,
                        ),
                    ]
                ),
            ),
            # 사기가 3.5% 뿐이라 가중치를 주지 않으면 전부 정상으로 찍는 쪽이
            # 손실이 작아진다. balanced 로 양성에 가중치를 준다.
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )


def _lgbm() -> LGBMClassifier:
    """category dtype 받을 수 있다. 원핫인코딩을 적용하지 않는다.

    베이스라인이라 하이퍼파라미터를 튜닝하지 않는다. 기본값에서 출발해야
    이후 튜닝이 얼마나 기여했는지 확인할 수 잇다.
    """
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=SEED,
        verbose=-1,
    )


@dataclass(frozen=True)
class Baseline:
    """모델과 해당 피처 집합을 담고 있는 dataclass이다.

    항목이 늘어날 자리다 — 하이퍼파라미터 탐색이나 저장 방식이 붙으면
    여기에 필드로 들어온다. 튜플로 두면 그때 읽기 어려워진다.
    """

    name: str
    factory: Callable[[], object]
    # 이 모델을 돌릴 피처 집합. 비어 있지 않아야 한다.
    feature_sets: frozenset[str] = field(default_factory=lambda: frozenset({"curated"}))

    def runs_on(self, tag: str) -> bool:
        return tag in self.feature_sets


# logreg 가 curated 에만 있는 것은 메모리 때문이 아니라 역할 때문이다.
# 선형 모델을 두는 이유는 계수를 읽어 관계를 설명할 수 있다는 것인데,
# 익명 컬럼을 넣으면 설명할 것이 없어진다. 해석 가능한 모델에서는 피처를
# 걸러 쓰는 것이 표준 관행이기도 하다.
#
# 그래서 비교는 이렇게 읽는다.
#   dummy          바닥
#   logreg-curated 해석 가능한 피처만으로 어디까지
#   lgbm-curated   같은 피처에서 비선형이 얼마나 보태는가
#   lgbm-all       익명 컬럼이 얼마나 보태는가
BASELINES = (
    Baseline(
        name="dummy",
        factory=lambda: DummyClassifier(strategy="prior"),
        feature_sets=frozenset({"curated", "all"}),
    ),
    Baseline(name="logreg", factory=_logreg),
    Baseline(name="lgbm", factory=_lgbm, feature_sets=frozenset({"curated", "all"})),
)


def evaluate(y_true: pd.Series, scores: pd.Series) -> dict[str, float]:
    """PR-AUC 와 ROC-AUC. 판단은 PR-AUC 로 한다."""
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def main(all_columns: bool = False) -> dict[str, dict[str, float]]:
    """all_columns 면 익명 컬럼(V*, C*, D*, M*)까지 전부 읽는다.

    베이스라인과 비교해 익명 컬럼이 실제로 얼마나 기여하는지 재기 위한
    것이다. 익명 컬럼이 사기 판별에 영향을 얼마나 주는 판별하고 추후
    dbt를 통해 추가로 등록할지 결정하기 위해 작성했다.
    """
    columns = None if all_columns else COLUMNS
    tag = "all" if all_columns else "curated"

    train = load_training("train", columns=columns)
    valid = load_training("valid", columns=columns)

    # 도메인 범주는 학습 데이터에서 뽑아 valid 에도 같은 것을 쓴다. 이것을
    # 빠뜨리면 train 19 종 / valid 7 종으로 갈려 평가가 낮게 나온다.
    domains = fit_domains(train.X)
    X_train = build(train.X, domains=domains)
    X_valid = build(valid.X, domains=domains)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = {}
    for spec in BASELINES:
        if not spec.runs_on(tag):
            logger.info(
                "%-7s 건너뜀 — %s 피처 집합에서는 돌리지 않는다.", spec.name, tag
            )
            continue

        name, factory = spec.name, spec.factory

        # run 이름에 피처 집합을 붙인다. UI 에서 curated 와 all 을 나란히
        # 놓고 비교하려면 이름만으로 구분되어야 한다.
        with mlflow.start_run(run_name=f"{name}-{tag}"):
            model = factory()
            model.fit(X_train, train.y)
            scores = model.predict_proba(X_valid)[:, 1]

            metrics = evaluate(valid.y, scores)
            results[name] = metrics

            mlflow.log_param("model", name)
            mlflow.log_param("feature_set", tag)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_metric("train_rows", len(X_train))
            mlflow.log_metric("valid_rows", len(X_valid))
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            logger.info(
                "%-7s PR-AUC %.4f  ROC-AUC %.4f",
                name,
                metrics["pr_auc"],
                metrics["roc_auc"],
            )

    baseline = float(valid.y.mean())
    logger.info("기저 사기율 %.4f — PR-AUC 가 이 값을 넘어야 배운 것이다.", baseline)
    return results


if __name__ == "__main__":
    import sys

    main(all_columns="--all-columns" in sys.argv)
