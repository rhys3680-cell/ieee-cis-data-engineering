"""학습한 모델을 저장하고 불러온다.

모델 저장 -> 학습
모델 불러오기 -> 추론

모델의 입출력의 일관성을 위해서 사용한다. 도메인 목록의 종류, 모델이 학습한 컬럼 순서
를 하나로 묶어 범주가 데이터마다 달라지거나 컬럼 순서가 달라지는 것을 막는다.

형식은 joblib을 사용한다. mlflow.log_model 도 있지만 로드하는 쪽에 mlflow 가
필요해진다 - Cloud Run 이미지에 트래킹 라이브러리를 넣지 않으려면 파일
하나만 읽는 편이 가볍다. MLflow 는 실험 기록에만 쓴다.

지금은 로컬에 저장한다. Airflow 나 Cloud Run 이 읽어야 하는 시점(배치 추론)
에 GCS 로 옮긴다. save/load 가 경로만 받으므로 그때 바뀌는 것은 경로뿐이다.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT
from src.common.logging import get_logger
from src.ml.features import FeatureSet

logger = get_logger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"

# 저장 형식이 바뀌면 수정한다. 옛 파일을 새 코드로 읽으면 에러를 발생시켜
# 작업자가 수정할 수 있도록 한다.
FORMAT_VERSION = 1


class Scorer(Protocol):
    """점수를 내는 것. 학습 라이브러리를 가리지 않는다.

    LGBMClassifier 와 sklearn Pipeline 은 공통 조상이 없어 한 타입으로 묶을
    수 없다. 이 코드가 실제로 요구하는 것은 predict_proba 하나뿐이므로
    그것만 조건으로 둔다.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class Bundle:
    """학습한 모델과, 그 모델에 같은 형태의 입력을 넣기 위해 필요한 것.

    모델 파일만으로는 추론을 재현할 수 없다. 학습 때와 다른 범주 목록을
    쓰면 도메인이 other 로 뭉개지고, 컬럼 순서가 다르면 모델이 다른 자리의
    값을 읽는다. 둘 다 에러 없이 점수만 틀린다.

        model       예측기. predict_proba 를 가진 것이면 된다.
        domains     fit_domains 가 뽑은 범주 목록. build 에 그대로 넘긴다.
        columns     학습 때의 열 구성과 순서. align 이 이것으로 맞춘다.

    나머지는 이 파일이 무엇인지 알려주는 메타데이터다. MLflow 에도 남지만
    모델을 GCS 로 옮기면 트래킹 DB 는 따라가지 않으므로, 파일 자체가 자기를
    설명해야 한다.
    """

    model: Scorer
    domains: dict[str, list[str]]
    columns: list[str]
    feature_set: FeatureSet
    metrics: dict[str, float]
    trained_at: datetime
    format_version: int = FORMAT_VERSION

    def align(self, X: pd.DataFrame) -> pd.DataFrame:
        """추론 입력을 학습 때의 열 구성으로 맞춘다.

        빠진 열이 있으면 실패한다. 조용히 채우면 그 열이 전부 결측인 채로
        점수가 나오는데, 그것이 정상인지 사고인지 구분할 수 없다.
        """
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"학습 때 있던 컬럼이 입력에 없다: {missing}")
        return X[self.columns]


def save(bundle: Bundle, name: str, directory: Path = MODEL_DIR) -> Path:
    """번들을 파일 하나로 저장한다. 경로를 돌려준다."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(bundle, path)

    size_mb = path.stat().st_size / 1024**2
    logger.info(
        "모델 저장: %s (%.1f MB, 피처 %d 개)", path, size_mb, len(bundle.columns)
    )
    return path


def load(name: str, directory: Path = MODEL_DIR) -> Bundle:
    """저장한 번들을 읽는다."""
    path = directory / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없다. train 을 먼저 돌려야 한다.")

    bundle = joblib.load(path)

    if bundle.format_version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: 저장 형식 {bundle.format_version}, 코드는 {FORMAT_VERSION} 를 "
            "기대한다. 다시 학습해야 한다."
        )

    logger.info(
        "모델 로드: %s (%s, %s 학습, PR-AUC %.4f)",
        path.name,
        bundle.feature_set,
        bundle.trained_at.date(),
        bundle.metrics.get("pr_auc", float("nan")),
    )
    return bundle


def now() -> datetime:
    """학습 시각. 어느 시점 데이터로 만든 모델인지 남긴다.

    UTC 로 고정한다. 로컬 시각으로 두면 다른 시간대에서 만든 모델과 순서를
    비교할 수 없다.
    """
    return datetime.now(UTC)
