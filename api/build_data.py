"""화면이 읽을 JSON 을 미리 만든다.

    uv run python -m api.build_data

임계값 곡선은 모델을 재학습해야 바뀌는 값이라 요청마다 계산할 이유가 없다.
구워 두면 컨테이너에 모델도 BigQuery 클라이언트도 lightgbm 도 필요 없어져
이미지가 가벼워지고, Cloud Run 콜드 스타트가 수십 초에서 1초 미만이 된다.

모델을 바꾸면 이것을 다시 돌리고 배포한다. 그 시점이 곧 화면의 숫자가
바뀌는 시점이라, 자동으로 갱신되지 않는 편이 오히려 안전하다.

일별 추이처럼 매일 바뀌는 것이 붙으면 그때는 API 가 BigQuery 를 조회해야
한다. 정적인 것과 동적인 것을 나눠 두는 이유다.
"""

import json
from pathlib import Path

from src.common.config import PROJECT_ROOT
from src.common.logging import get_logger

logger = get_logger(__name__)

OUT_PATH = PROJECT_ROOT / "api" / "static" / "curve.json"
MODEL = "lgbm-all"


def build(model_name: str = MODEL) -> dict:
    from src.ml.dataset import load_training
    from src.ml.features import FeatureSet
    from src.ml.model_store import load
    from src.ml.predict import score
    from src.ml.threshold import curve
    from src.ml.train import COLUMNS

    bundle = load(model_name)
    columns = None if bundle.feature_set is FeatureSet.ALL else COLUMNS
    valid = load_training("valid", columns=columns)

    scores = score(valid.X, bundle)
    amounts = valid.X["transaction_amt"]
    table = curve(valid.y, scores, amounts)

    return {
        "summary": {
            "model": model_name,
            "split": "valid",
            "rows": len(valid),
            "frauds": int(valid.y.sum()),
            # 아무것도 막지 않을 때의 손실. 비교 기준이 없으면 총비용이
            # 큰지 작은지 알 수 없다.
            "fraud_amount": round(float(amounts[valid.y == 1].sum()), 2),
            "trained_at": bundle.trained_at.isoformat(timespec="seconds"),
            "pr_auc": round(bundle.metrics.get("pr_auc", 0.0), 4),
        },
        "rows": table.to_dict("records"),
    }


def main(out: Path = OUT_PATH) -> Path:
    payload = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    logger.info(
        "%s (%d 행, %.1f KB)", out, len(payload["rows"]), out.stat().st_size / 1024
    )
    return out


if __name__ == "__main__":
    main()