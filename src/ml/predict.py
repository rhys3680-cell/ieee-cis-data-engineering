"""저장한 모델로 점수를 낸다.

배치 추론과 API 가 이 함수를 부른다. BigQuery 도 HTTP 도 모르고 DataFrame 만
받는다 — 부르는 쪽이 어디서 읽어왔든 상관없다.

나누는 이유는 학습 때와 같은 형태로 입력을 맞추는 코드가 한 곳에만 있어야
하기 때문이다. 파생 규칙이나 컬럼 정렬이 두 곳에 있으면 언젠가 갈라지는데,
에러가 나지 않고 점수만 조용히 틀린다. 라벨 없는 구간을 채점하는
파이프라인이라 나중에도 알아채기 어렵다.
"""

import numpy as np
import pandas as pd

from src.ml.features import build
from src.ml.model_store import Bundle


def prepare(df: pd.DataFrame, bundle: Bundle) -> pd.DataFrame:
    """원본을 학습 때의 입력 형태로 맞춘다.

    파생을 얹고 컬럼 구성과 순서를 정렬한다. 점수 대신 행렬 자체가 필요한
    경우(SHAP 같은 설명 도구)를 위해 score 에서 분리해 둔다.
    """
    return bundle.align(build(df, domains=bundle.domains))


def score(df: pd.DataFrame, bundle: Bundle) -> np.ndarray:
    """사기 점수(0~1)를 돌려준다. 행 순서는 입력과 같다.

    임계값은 여기서 적용하지 않는다. 점수와 판정을 나누면 임계값이 바뀌어도
    재추론이 필요 없고, 소비처마다 다른 임계값을 쓸 수 있다.
    """
    return bundle.model.predict_proba(prepare(df, bundle))[:, 1]
