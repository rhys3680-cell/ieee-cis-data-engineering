"""프로젝트 전역 설정.

값은 .env에서 읽고, 없으면 기본값을 쓴다.
경로는 모두 프로젝트 루트 기준 경로를 사용한다.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/common/config.py -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # --- 로컬 경로 ---
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))

    # --- GCP (적재 단계에서 사용)
    gcp_project_id: str = ""
    gcp_location: str = "asia-northeast3"
    gcs_bucket: str = ""

    bq_dataset_raw: str = "ieee_raw"
    bq_dataset_staging: str = "ieee_staging"
    bq_dataset_mart: str = "ieee_mart"

    log_level: str = "INFO"

    @field_validator("data_raw_dir", "data_processed_dir")
    @classmethod
    def _to_absolute(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v


@lru_cache
def get_settings() -> Settings:
    return Settings()
