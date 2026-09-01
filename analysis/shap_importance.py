"""익명 컬럼 중 무엇이 기여하는지 SHAP 으로 확인한다 (일회성 조사 스크립트).

파이프라인에 포함되지 않는다. 어떤 피처를 dbt 로 승격할지 정한 근거를
남기기 위해 보관한다.

lgbm-all 이 PR-AUC 를 2.9배 올리는데 394 개 중 무엇이 기여하는지 알 수
없었다. LightGBM 내장 importance 는 "분기에 몇 번 쓰였나"라 자주 쓰이지만
기여가 작은 컬럼이 위로 올라온다. SHAP 은 각 예측에 얼마나 기여했는지를
재므로 크기 비교가 성립한다.

결론: 개당 기여도로 보면 V* 가 가장 낮다.

    계열            개수    기여    개당
    V*  (익명)      339   32.7%   0.10%
    명시 피처        17   27.7%   1.63%
    C*  (카운팅)     14   19.8%   1.41%
    D*  (시간델타)   15   12.6%   0.84%
    M*  (매치)        9    7.1%   0.79%

세 가지를 읽었다.

1. curated 선정이 맞았다. transaction_amt 1위, card6 4위,
   purchaser_email_domain 6위로 판별력을 재고 고른 것이 실제로 상위다.
2. dbt 승격을 검토한다면 C* 와 D* 다. 개당 기여가 V* 의 8~14배라 같은
   노력으로 얻는 것이 많다. 다만 이 컬럼들이 무엇인지는 확인되지 않았다 —
   아래 주의를 볼 것.
3. 상위 50 개가 전체 기여의 80.9% 다. 394 개를 다 읽지 않아도 되므로
   조회와 학습을 줄일 여지가 있다. 90 개는 기여가 정확히 0 이다.

주의: 계열 이름은 커뮤니티 통설이지 확인된 사실이 아니다. 대회 주최측은
익명 컬럼의 의미를 공개하지 않았고, "C 는 카운팅, D 는 시간 델타"는
참가자들이 관찰로 붙인 해석이다. 실제로 재보니 맞지 않는다.

    C1  이 같은 카드 안에서 단조 증가       74.7%   (카운터라면 100%)
    C13 이 같은 카드 안에서 단조 증가       66.8%
    D1  증가분이 실제 경과일수와 일치       33.4%
    D1  이 거래 사이에 변하지 않음          32.0%   (델타라면 변해야 한다)

카드 엔티티를 card1 하나에서 card1~5 + addr1 조합으로 좁혀도 같았다. 값의
형태는 통설과 맞는다 — C* 는 0 이상 정수, D* 는 -83~1091 의 정수로 일
단위 범위다. 하지만 동작이 다르다. 엔티티 키를 잘못 잡았을 수도 있고
통설이 부정확할 수도 있는데, 이 데이터로는 가릴 수 없다.

그래서 "SQL 윈도우로 재현할 수 있다"고 단정하면 안 된다. 승격하려면 그
전에 각 컬럼이 무엇인지 따로 조사해야 한다. 기여도가 높다는 사실(이것은
실측이다)과 재현 방법을 안다는 것(이것은 아직 모른다)은 다른 이야기다.

shap 은 의존성에 넣지 않았다. 분석 도구지 파이프라인 구성요소가 아니라
결과만 얻으면 되기 때문이다. 실행할 때만 --with 로 끌어온다.

실행: uv run --with shap python -m analysis.shap_importance
"""

import numpy as np
import pandas as pd
import shap

from src.common.logging import get_logger
from src.ml.dataset import load_training
from src.ml.features import build
from src.ml.model_store import load

logger = get_logger(__name__)

MODEL = "lgbm-all"

# 전수로 돌리면 82,325 행 x 394 열이라 오래 걸린다. TreeExplainer 는 트리
# 구조를 직접 읽어 정확한 값을 내므로, 샘플링은 속도 문제이지 근사가
# 아니다. 1만 행이면 계열별 순위가 안정적으로 나온다.
SAMPLE_SIZE = 10_000
SEED = 42


def family(column: str) -> str:
    """컬럼을 접두사로 묶는다.

    개별 컬럼의 기여보다 계열별 개당 기여가 판단에 쓸모 있다 — V* 는
    339 개나 되어 합계만 보면 커 보인다.

    괄호 안의 이름은 커뮤니티 통설이며 확인되지 않았다(모듈 docstring 의
    주의 참고). 묶는 기준은 접두사 자체이므로 이름이 틀려도 집계는 유효하다.
    """
    if column.startswith("V"):
        return "V* (익명)"
    if column.startswith("C"):
        return "C* (카운팅)"
    if column.startswith("D"):
        return "D* (시간델타)"
    if column.startswith("M"):
        return "M* (매치)"
    if column.startswith("id_"):
        return "id_* (identity)"
    return "명시 피처"


def main() -> pd.Series:
    bundle = load(MODEL)
    valid = load_training("valid")
    X = bundle.align(build(valid.X, domains=bundle.domains))

    sample = X.sample(n=min(SAMPLE_SIZE, len(X)), random_state=SEED)
    values = shap.TreeExplainer(bundle.model).shap_values(sample)
    # 이진 분류에서 버전에 따라 클래스별 리스트를 돌려준다. 양성 쪽을 쓴다.
    if isinstance(values, list):
        values = values[1]

    importance = pd.Series(np.abs(values).mean(axis=0), index=X.columns)
    importance = importance.sort_values(ascending=False)
    total = importance.sum()

    logger.info("=== 기여도 상위 20 ===")
    for rank, (name, value) in enumerate(importance.head(20).items(), 1):
        logger.info("%2d. %-28s %.5f", rank, name, value)

    logger.info("=== 계열별 ===")
    grouped = importance.groupby(importance.index.map(family)).agg(["sum", "count"])
    grouped["기여%"] = (grouped["sum"] / total * 100).round(1)
    grouped["개당%"] = (grouped["기여%"] / grouped["count"]).round(2)
    for name, row in grouped.sort_values("기여%", ascending=False).iterrows():
        logger.info(
            "%-14s %3d 개  기여 %5.1f%%  개당 %.2f%%",
            name,
            row["count"],
            row["기여%"],
            row["개당%"],
        )

    logger.info("=== 누적 ===")
    for n in (10, 20, 50, 100, 200):
        logger.info("상위 %3d 개 -> %.1f%%", n, importance.head(n).sum() / total * 100)
    logger.info("기여가 0 인 컬럼: %d / %d", int((importance == 0).sum()), len(importance))

    return importance


if __name__ == "__main__":
    main()