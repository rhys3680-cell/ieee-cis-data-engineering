"""GCS에 날짜 파티션 Parquet을 올리고 읽는다.

로컬 파일을 거치지 않는다. DataFrame을 메모리에서 직렬화해 바로 올린다.

    gs://<bucket>/transactions/dt=2018-01-15/part.parquet
"""

import io
from datetime import date
from functools import lru_cache

import pandas as pd
from google.cloud import storage

from src.common.config import get_settings
from src.common.logging import get_logger
from src.extract.schema import bigquery_schema

logger = get_logger(__name__)


@lru_cache
def _bucket() -> storage.Bucket:
    """버킷 핸들. 클라이언트 생성 비용이 있어 한 번만 만든다."""
    s = get_settings()
    if not s.gcs_bucket:
        raise ValueError("GCS_BUCKET이 설정되지 않음. .env를 확인해야 함.")
    return storage.Client(project=s.gcp_project_id).bucket(s.gcs_bucket)


def blob_path(dataset: str, dt: date) -> str:
    """버킷 안에서의 객체 경로. gs:// 접두사는 붙이지 않는다."""
    return f"{dataset}/dt={dt.isoformat()}/part.parquet"


def gcs_uri(dataset: str, dt: date | None = None) -> str:
    """gs:// URI. dt를 생략하면 데이터셋 전체를 가리키는 와일드카드."""
    bucket = get_settings().gcs_bucket
    if dt is None:
        return f"gs://{bucket}/{dataset}/dt=*/part.parquet"
    return f"gs://{bucket}/{blob_path(dataset, dt)}"


def _check_schema(df: pd.DataFrame, dataset: str) -> None:
    """스키마와 어긋나면 올리기 전에 막는다.

    BigQuery 는 컬럼이 추가되거나 타입이 다르면 로드를 거부하지만,
    컬럼이 빠진 것은 통과시키고 NULL 로 채운다. 소스에서 컬럼이 사라져도
    적재가 성공해 며칠 뒤에야 발견하게 되므로 여기서 확인한다.
    """
    expected = {c for c, _ in bigquery_schema(dataset)}
    actual = set(df.columns)

    missing = expected - actual
    if missing:
        raise ValueError(
            f"{dataset}: 스키마에 있는 컬럼 {len(missing)}개가 없다. "
            f"BigQuery 는 이를 NULL 로 채우고 통과시킨다. {sorted(missing)[:5]}"
        )

    extra = actual - expected
    if extra:
        raise ValueError(
            f"{dataset}: 스키마에 없는 컬럼 {len(extra)}개. {sorted(extra)[:5]}"
        )


def upload_partition(df: pd.DataFrame, dataset: str, dt: date) -> int:
    """하루치를 GCS에 Parquet로 올린다. 반환값은 업로드된 바이트 수.

    로컬 파일을 거치지 않는다.

    파티션 키를 컬럼으로 남긴다. 경로의 dt=는 사람이 읽기 위한 것이고,
    BigQuery 네이티브 파티션 테이블은 파일 안의 컬럼을 본다.
    """
    _check_schema(df, dataset)

    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)

    blob = _bucket().blob(blob_path(dataset, dt))
    blob.upload_from_file(buf, content_type="application/octet-stream")

    logger.debug("업로드 %s %s: %d 행, %.1f KB", dataset, dt, len(df), blob.size / 1024)
    return blob.size


def read_partition(dataset: str, dt: date) -> pd.DataFrame:
    """하루치를 읽는다. 없으면 빈 DataFrame을 반환한다.

    train과 test 사이에 30일 공백이 있어 데이터가 없는 날짜가 존재한다.
    Airflow 백필이 그 날짜를 부르는 것은 정상이므로 예외를 만들지 않는다.
    """
    blob = _bucket().blob(blob_path(dataset, dt))
    if not blob.exists():
        logger.info("파티션 없음: %s %s", dataset, dt)
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


def list_partitions(dataset: str) -> list[date]:
    """올라가 있는 파티션 날짜. 적재 상태 확인과 백필 대상 파악에 쓴다."""
    prefix = f"{dataset}/dt="
    out = []
    for blob in _bucket().list_blobs(prefix=prefix):
        # transactions/dt=2018-01-15/part.parquet -> 2018-01-15
        token = blob.name.split("/")[1]
        out.append(date.fromisoformat(token.removeprefix("dt=")))
    return sorted(out)
