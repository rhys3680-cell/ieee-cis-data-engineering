"""저장한 모델을 다시 읽을 때의 계약.

모델만으로는 추론을 재현할 수 없다. 도메인 목록과 컬럼 구성이 학습 때와
같아야 하는데, 어긋나도 에러가 나지 않고 점수만 틀린다. 그 어긋남을 여기서
막는다.

BigQuery 없이 돈다.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.ml.features import FeatureSet
from src.ml.model_store import FORMAT_VERSION, Bundle, load, save


class _StubModel:
    """predict_proba 만 있으면 Scorer 로 충분하다는 것을 보이는 대역."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def _bundle(columns: list[str] | None = None) -> Bundle:
    return Bundle(
        model=_StubModel(),
        domains={"purchaser_email_domain": ["gmail.com"]},
        columns=columns or ["a", "b", "c"],
        feature_set=FeatureSet.CURATED,
        metrics={"pr_auc": 0.1842},
        trained_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


def test_저장하고_읽으면_그대로다(tmp_path):
    """추론에 필요한 셋이 왕복에서 살아남아야 한다."""
    save(_bundle(), "t", directory=tmp_path)
    got = load("t", directory=tmp_path)

    assert got.columns == ["a", "b", "c"]
    assert got.domains == {"purchaser_email_domain": ["gmail.com"]}
    assert got.feature_set is FeatureSet.CURATED
    assert got.trained_at == datetime(2026, 9, 1, tzinfo=UTC)


def test_컬럼이_빠지면_실패한다():
    """조용히 채우면 그 열이 전부 결측인 채로 점수가 나온다.

    정상인지 사고인지 구분할 수 없으므로 멈추는 편이 낫다.
    """
    with pytest.raises(ValueError, match="컬럼"):
        _bundle().align(pd.DataFrame({"a": [1], "b": [2]}))


def test_컬럼_순서를_학습_때로_되돌린다():
    """BigQuery 가 다른 순서로 주면 모델이 다른 자리의 값을 읽는다."""
    shuffled = pd.DataFrame({"c": [3], "a": [1], "b": [2]})
    assert list(_bundle().align(shuffled).columns) == ["a", "b", "c"]


def test_모르는_컬럼은_버린다():
    """소스에 컬럼이 추가되어도 학습 때의 구성으로 맞춘다."""
    extra = pd.DataFrame({"a": [1], "b": [2], "c": [3], "새컬럼": [4]})
    assert list(_bundle().align(extra).columns) == ["a", "b", "c"]


def test_저장_형식이_다르면_거부한다(tmp_path):
    """옛 파일을 새 코드로 읽으면 필드가 조용히 어긋난다."""
    stale = Bundle(
        model=_StubModel(),
        domains={},
        columns=["a"],
        feature_set=FeatureSet.ALL,
        metrics={},
        trained_at=datetime(2026, 9, 1, tzinfo=UTC),
        format_version=FORMAT_VERSION + 1,
    )
    save(stale, "stale", directory=tmp_path)

    with pytest.raises(ValueError, match="저장 형식"):
        load("stale", directory=tmp_path)


def test_없는_모델은_안내와_함께_실패한다(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load("없음", directory=tmp_path)