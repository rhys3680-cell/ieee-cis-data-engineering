"""로깅 설정.

파이프라인 로그는 나중에 Airflow 태스크 로그로 수집된다.
표준 출력으로만 내보내고 파일 핸들러는 따로 두지 않는다.
"""

import logging
import sys

from src.common.config import get_settings

_configured = False


def setup_logging() -> None:
    """루트 로거를 한번만 설정한다."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
