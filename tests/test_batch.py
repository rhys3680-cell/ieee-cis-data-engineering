"""배치 채점의 계약.

BigQuery 없이 도는 것만 여기서 본다. 실제 적재는 파티션 데코레이터에
달려 있는데, 그것은 src/load/bigquery.py 와 같은 성질이라 이미 실험으로
확인했다(같은 날짜를 다시 돌려도 행이 늘지 않는다).
"""

from datetime import date

from src.ml.batch import PARTITION_FIELD, SCHEMA, table_id


def test_파티션_데코레이터를_붙인다():
    """없으면 WRITE_TRUNCATE 가 테이블 전체를 교체한다.

    하루치를 넣었을 뿐인데 기존 파티션이 전부 사라지는데, 에러가 나지 않아
    한참 뒤에 발견한다.
    """
    assert table_id(date(2018, 7, 2)).endswith("$20180702")


def test_데코레이터_없이도_부를_수_있다():
    """테이블 생성과 조회는 데코레이터 없는 이름을 쓴다."""
    assert "$" not in table_id()


def test_점수와_모델_버전을_함께_남긴다():
    """모델 컬럼이 없으면 점수 분포가 바뀌었을 때 모델이 바뀐 것인지
    데이터가 바뀐 것인지 구분할 수 없다."""
    fields = {f.name for f in SCHEMA}
    assert {"score", "model", "model_trained_at"} <= fields


def test_판정을_저장하지_않는다():
    """임계값은 비용 가정에 따라 움직인다. 판정을 구워 넣으면 가정이 바뀔
    때마다 50만 건을 다시 채점해야 한다."""
    fields = {f.name for f in SCHEMA}
    assert not fields & {"is_fraud", "blocked", "threshold", "prediction"}


def test_파티션_필드가_스키마에_있다():
    """time_partitioning 이 가리키는 컬럼이 없으면 테이블 생성이 실패한다."""
    assert PARTITION_FIELD in {f.name for f in SCHEMA}