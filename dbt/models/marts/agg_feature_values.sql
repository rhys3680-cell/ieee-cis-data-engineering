/*
  SHAP 상위 피처의 최빈값별 사기율.

  agg_feature_buckets 가 분위수로 자르는데, 그러면 극단값이 뭉개진다.
  C13 은 0 일 때 사기율이 14.1% 로 평균의 4배인데 Q1(0~1)에 섞이면 6.3% 가
  된다. 값 하나가 위험을 가르는 경우가 있어 따로 낸다.

  익명 컬럼이 대부분 정수라 값별 집계가 성립한다. 연속값이면 이 방식이
  의미가 없지만, C* 와 D* 는 0 이상 정수이고 고유값도 수천 개 수준이다.

  자주 나오는 값만 본다. 꼬리까지 내면 표가 수천 행이 되고, 건수가 적은
  값은 사기율이 우연히 튄다.
*/

{{ config(materialized='table') }}

{% set features = [
    'C13', 'C14', 'V70', 'V294', 'C11', 'C1', 'D2', 'D3', 'C5', 'V258'
] %}

with source as (
    select * from {{ ref('stg_transactions') }}
    where source_split = 'train'
),

{% for col in features %}
counted_{{ col }} as (
    select
        '{{ col }}' as feature,
        cast({{ col }} as string) as value,
        count(*) as tx_count,
        countif(is_fraud = 1) as fraud_count,
        avg(is_fraud) as fraud_rate
    from source
    where {{ col }} is not null
    group by 1, 2
    having count(*) >= {{ var('min_bucket_rows') }}
    order by tx_count desc
    limit {{ var('top_values_per_feature') }}
){% if not loop.last %},{% endif %}
{% endfor %}

{% for col in features %}
select * from counted_{{ col }}
{% if not loop.last %}union all{% endif %}
{% endfor %}