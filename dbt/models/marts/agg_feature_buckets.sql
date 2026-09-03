/*
  SHAP 상위 피처의 값 구간별 사기율.

  익명 컬럼은 이름으로 설명할 수 없다. 대회 주최측이 의미를 공개하지 않았고
  "C 는 카운팅"같은 통설도 데이터로 확인되지 않았다(analysis/shap_importance.py).
  대신 어떤 값에서 사기율이 튀는지 보여준다 — 이름을 몰라도 그것은 실측이다.

  구간은 분위수로 자른다. 값 범위를 모르므로 고정 경계를 쓸 수 없다. 다만
  분위수는 극단값을 뭉갠다 — C13 은 0 일 때 사기율이 14.1% 인데 Q1(0~1)에
  섞이면 6.3% 로 낮아진다. 그 신호는 agg_feature_values 가 따로 낸다.

  train 만 본다. test 는 is_fraud 가 NULL 이라 사기율이 정의되지 않는다.

  대상 컬럼은 SHAP 기여도 상위다. 394 개를 전부 내면 표가 커지고, 상위
  10 개가 전체 기여의 34.6% 를 차지한다.
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
bucketed_{{ col }} as (
    select
        '{{ col }}' as feature,
        case
            when {{ col }} is null then 'null'
            -- 결측을 분리해 분위수를 매긴다. 섞으면 NULL 이 한쪽 끝으로
            -- 몰려 구간이 왜곡된다.
            else format('Q%d', ntile(5) over (
                partition by {{ col }} is null order by {{ col }}))
        end as bucket,
        {{ col }} as value,
        is_fraud
    from source
){% if not loop.last %},{% endif %}
{% endfor %}

, unioned as (
    {% for col in features %}
    select * from bucketed_{{ col }}
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
)

select
    feature,
    bucket,

    count(*)            as tx_count,
    countif(is_fraud = 1) as fraud_count,
    avg(is_fraud)       as fraud_rate,
    min(value)          as value_min,
    max(value)          as value_max

from unioned
group by 1, 2
-- 사기 사례가 몇 건뿐인 구간은 비율이 우연히 튄다. V294 의 결측은 12 건인데
-- 사기율 16.7% 로 나오는데, 그리면 신호처럼 보이지만 표본이 없는 것이다.
having count(*) >= {{ var('min_bucket_rows') }}
order by feature, bucket