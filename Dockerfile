# 임계값 화면. Cloud Run 컨테이너 하나에 API 와 정적 파일을 함께 올린다.
#
# 프로젝트 의존성을 설치하지 않는다. 화면이 읽는 곡선은 build_data 가 미리
# 구워 둔 JSON 이라 여기에는 모델도 BigQuery 클라이언트도 lightgbm 도 필요
# 없다. fastapi 와 uvicorn 둘이면 된다.
#
#   uv run python -m api.build_data    # static/curve.json 갱신 (빌드 전에)
#   docker build -t threshold-ui .
#   docker run -p 8080:8080 threshold-ui
FROM python:3.12-slim

# 파이썬 로그를 버퍼링하지 않는다. Cloud Run 로그에 즉시 나와야 장애를
# 볼 수 있다.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir \
        "fastapi>=0.141.1" \
        "uvicorn>=0.52.4"

# api/ 만 넣는다. src/, dags/, dbt/ 는 이 이미지의 관심사가 아니다.
# static/curve.json 이 없으면 화면이 503 을 내므로 빌드 전에 build_data 를
# 돌려야 한다.
COPY api/ /app/api/

# 루트로 돌리지 않는다.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# Cloud Run 이 PORT 를 주입한다. 로컬에서는 8080 을 쓴다.
ENV PORT=8080
EXPOSE 8080

# exec 형식이라야 프로세스가 PID 1 이 되어 Cloud Run 의 SIGTERM 을 받는다.
# shell 형식으로 ${PORT} 를 펼치면 그 신호를 놓쳐 정지할 때 강제 종료된다.
# 포트는 api/main.py 가 환경변수에서 읽는다.
CMD ["python", "-m", "api.main"]