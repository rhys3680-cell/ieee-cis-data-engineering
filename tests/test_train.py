"""베이스라인 구성의 계약.

BigQuery 없이 돈다. 학습은 여기서 하지 않는다 — 지표가 얼마인지는 실험의
결과이지 코드의 계약이 아니다. 여기서 보는 것은 '어떤 모델이 어떤 피처
집합에서 도는가' 뿐이다.
"""

from src.ml.features import FeatureSet
from src.ml.train import BASELINES


def test_모든_모델이_적어도_한_곳에서_돈다():
    """feature_sets 가 비면 그 모델은 영영 돌지 않는다.

    에러가 나지 않고 조용히 건너뛰므로, 비교 표에서 한 줄이 사라진 것을
    눈치채지 못한다.
    """
    for spec in BASELINES:
        assert spec.feature_sets, f"{spec.name}: feature_sets 가 비어 있다"


def test_curated_에서는_셋_다_돈다():
    """dummy 없이는 바닥을 모르고, logreg 없이는 비선형의 기여를 못 잰다."""
    names = {s.name for s in BASELINES if s.runs_on(FeatureSet.CURATED)}
    assert names == {"dummy", "logreg", "lgbm"}


def test_all_에서는_logreg_를_뺀다():
    """선형 모델을 두는 이유가 계수를 읽는 것인데, 익명 컬럼 378 개를
    넣으면 설명할 것이 없어진다. 메모리 때문이 아니라 역할 때문이다."""
    names = {s.name for s in BASELINES if s.runs_on(FeatureSet.ALL)}
    assert "logreg" not in names
    assert names == {"dummy", "lgbm"}


def test_바닥을_재는_모델은_모든_집합에서_돈다():
    """비교 대상이 바뀌어도 바닥은 같은 자리에 있어야 한다."""
    dummy = next(s for s in BASELINES if s.name == "dummy")
    assert dummy.runs_on(FeatureSet.CURATED) and dummy.runs_on(FeatureSet.ALL)