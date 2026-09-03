/*
  배치 추론 상태. 모델이 정상으로 도는지 보는 지표다.

  라벨이 없어 정확도를 잴 수 없다. test 구간은 is_fraud 가 NULL 이고 앞으로도
  채워지지 않으므로, 여기서 감시하는 것은 '맞혔는가'가 아니라 '평소와 같은가'
  이다. 실무에서도 라벨은 조사가 끝나야 오므로 그 사이의 감시는 이 형태다.

  차단율을 기준 임계값 하나로만 낸다. 임계값은 비용 가정에 따라 움직이지만,
  추이를 보려면 고정된 기준이 있어야 한다. 0.43 은 fp_cost $5 일 때의 값이고
  가정이 바뀌면 여기도 바꾼다 — 화면에서 만지는 임계값과는 별개다.

  점수 분포를 함께 낸다. 차단율만 보면 임계값 근처의 변화만 잡히는데,
  분포가 통째로 밀리면 그것이 먼저 드러난다.
*/

{{ config(materialized='table') }}

with scored as (
    select
        transaction_date,
        model,
        score
    from {{ source('ieee_raw', 'predictions') }}
),

daily as (
    select
        transaction_date,
        model,

        count(*) as scored_count,
        countif(score >= {{ var('block_threshold') }}) as blocked_count,
        countif(score >= {{ var('block_threshold') }}) / count(*) as blocked_rate,

        -- 분포. 평균만 보면 꼬리가 두꺼워지는 것을 놓친다.
        avg(score) as score_avg,
        approx_quantiles(score, 100)[offset(50)] as score_p50,
        approx_quantiles(score, 100)[offset(90)] as score_p90,
        approx_quantiles(score, 100)[offset(99)] as score_p99

    from scored
    group by 1, 2
)

select
    *,

    -- 직전 7일 평균 대비 비율. 1 에서 크게 벗어나면 이상이다.
    -- 자기 자신을 빼야 급변을 잡을 수 있으므로 1 PRECEDING 까지만 본다.
    -- agg_pipeline_daily 와 같은 방식이다.
    safe_divide(
        blocked_rate,
        avg(blocked_rate) over (
            partition by model
            order by transaction_date
            rows between 7 preceding and 1 preceding
        )
    ) as blocked_rate_vs_baseline

from daily