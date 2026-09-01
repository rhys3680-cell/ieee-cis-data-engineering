"""임계값 화면을 서빙한다.

    uv run uvicorn api.main:app --reload    # localhost:8000

임계값은 통계가 아니라 비용으로 정하는 문제라(놓친 사기 대 막은 정상 거래)
슬라이더로 손실이 어떻게 변하는지 보이는 편이 낫다. BI 도구로는 안 되는
상호작용이라 여기만 웹으로 만든다.

**모델을 들고 있지 않다.** 화면이 쓰는 곡선은 `api/build_data.py` 가 미리
구워 둔 `static/curve.json` 이다. 임계값 곡선은 재학습해야 바뀌는 값이라
요청마다 계산할 이유가 없고, 그래서 이 컨테이너에는 lightgbm 도 BigQuery
클라이언트도 필요 없다. 콜드 스타트가 수십 초에서 1초 미만이 된다.

일별 추이처럼 매일 바뀌는 것이 붙으면 그때 여기에 조회 라우트를 더한다.
정적인 것과 동적인 것을 나눠 두는 이유다.

`src/` 밖에 두는 것은 배포 단위가 달라서다. Cloud Run 이미지에는 `api/` 만
들어가고 DAG 이나 dbt 는 필요 없다.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# src 를 임포트하지 않는다. 경로 하나를 위해 pydantic-settings 를 끌어오면
# 컨테이너에 프로젝트 의존성이 통째로 따라온다. 이 모듈이 아는 것은 자기
# 옆의 static/ 뿐이다.
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="IEEE-CIS 임계값", docs_url="/api/docs")


@app.get("/api/health")
def health() -> dict:
    """컨테이너가 떴는지만 본다. 데이터를 건드리지 않는다."""
    return {"status": "ok", "curve": (STATIC_DIR / "curve.json").is_file()}


# 정적 파일은 마지막에 마운트한다. 루트에 걸리므로 라우트 정의보다 먼저
# 두면 /api/* 를 가로챈다.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:  # pragma: no cover - 빌드 전에는 없을 수 있다
    logger.warning("%s 가 없다. build_data 를 먼저 돌려야 한다.", STATIC_DIR)

    @app.get("/")
    def missing() -> JSONResponse:
        return JSONResponse(
            {"error": "화면이 빌드되지 않았다. uv run python -m api.build_data"},
            status_code=503,
        )


if __name__ == "__main__":
    # 컨테이너 진입점. Cloud Run 이 PORT 를 주입하므로 여기서 읽는다 —
    # Dockerfile 의 CMD 에 포트를 박으면 Cloud Run 이 다른 값을 줄 때 뜨지
    # 않고, shell 형식으로 ${PORT} 를 쓰면 SIGTERM 을 받지 못한다.
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))