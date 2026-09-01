"""임계값 화면의 API 와 정적 파일 서빙.

    uv run uvicorn api.main:app --reload

임계값은 통계가 아니라 비용으로 정하는 문제라(놓친 사기 대 막은 정상 거래)
슬라이더로 손실이 어떻게 변하는지 보이는 편이 낫다. BI 도구로는 안 되는
상호작용이라 여기만 웹으로 만든다.

정적 파일을 함께 내주고 Cloud Run 컨테이너 하나에 올린다. 프론트를 따로
호스팅하면 도메인이 갈려 CORS 를 열어야 하는데, 화면 하나에 치를 비용이
아니다.

`src/` 밖에 두는 이유는 배포 단위가 다르기 때문이다. Cloud Run 이미지에는
`src/ml/` 과 `api/` 만 들어가고 DAG 이나 dbt 는 필요 없다.
"""

from functools import lru_cache

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.common.config import PROJECT_ROOT
from src.common.logging import get_logger

logger = get_logger(__name__)

STATIC_DIR = PROJECT_ROOT / "api" / "static"
MODEL = "lgbm-all"

app = FastAPI(title="IEEE-CIS 임계값", docs_url="/api/docs")


@lru_cache(maxsize=1)
def _curve() -> list[dict]:
    """비용 곡선을 한 번만 계산해 들고 있는다.

    fp_cost 에 의존하지 않는 원자료다. 총비용은 화면이
    missed_amount + false_alarms * fp_cost 로 계산하므로, 슬라이더를 움직여도
    서버에 다시 묻지 않는다.

    지연 임포트를 쓴다. 모듈을 읽는 것만으로 BigQuery 와 모델이 딸려오면
    임포트가 느려지고, 헬스체크조차 그 비용을 치른다.
    """
    from src.ml.dataset import load_training
    from src.ml.model_store import load
    from src.ml.predict import score
    from src.ml.threshold import curve

    bundle = load(MODEL)
    valid = load_training("valid")
    scores = score(valid.X, bundle)

    table = curve(valid.y, scores, valid.X["transaction_amt"])
    logger.info("비용 곡선 %d 행 준비", len(table))
    return table.to_dict("records")


@lru_cache(maxsize=1)
def _summary() -> dict:
    from src.ml.dataset import load_training

    valid = load_training("valid", columns=["transaction_amt"])
    amounts = valid.X["transaction_amt"]
    return {
        "model": MODEL,
        "split": "valid",
        "rows": len(valid),
        "frauds": int(valid.y.sum()),
        # 아무것도 막지 않을 때의 손실. 비교 기준이 없으면 총비용이 큰지
        # 작은지 알 수 없다.
        "fraud_amount": round(float(amounts[valid.y == 1].sum()), 2),
    }


@app.get("/api/curve")
def get_curve() -> dict:
    """임계값별 결과. 99 행에 9 KB 라 화면이 한 번 받아 두고 쓴다."""
    return {"summary": _summary(), "rows": _curve()}


@app.get("/api/health")
def health() -> dict:
    """모델을 건드리지 않는다. 컨테이너가 떴는지만 본다."""
    return {"status": "ok"}


# 정적 파일은 마지막에 마운트한다. 루트에 걸리므로 먼저 두면 /api/* 를
# 가로챈다. 프론트를 빌드하기 전에는 디렉터리가 없을 수 있다.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    logger.warning("%s 가 없다. API 만 서빙한다.", STATIC_DIR)
