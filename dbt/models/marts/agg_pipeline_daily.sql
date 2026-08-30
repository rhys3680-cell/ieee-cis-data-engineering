/*
  파이프라인 상태. 적재가 정상인지 보는 지표다.

  거래 지표와 성격이 다르다. 여기서 보는 것은 '데이터가 제대로 들어왔는가'
  이지 '사기가 얼마나 있는가'가 아니다. 결측률 같은 값은 세그먼트로 자르면
  의미가 흐려지므로 날짜로만 자른다.

  일별 건수가 이전 7일 중앙값에서 크게 벗어나거나 결측률이 갑자기 오르면
  소스나 적재에 문제가 생긴 것이다. 그 판단은 대시보드에서 하고, 여기서는 
  판단 근거가 되는 값만 내보낸다.
*/

with daily as (
    select
        transaction_date,
        source_split,

        count(*) as tx_count,
        countif(is_fraud is not null) as labeled_count,

        -- 결측률. 소스 스키마가 바뀌면 여기서 먼저 드러난다.
        countif(purchaser_email_domain is null) / count(*) as email_null_rate,
        countif(addr1 is null) / count(*)                  as addr_null_rate,
        countif(card2 is null) / count(*)                  as card2_null_rate,
        countif(has_identity) / count(*)                   as identity_rate,

        -- 신선도. 이 파티션이 마지막으로 적재된 시각.
        max(ingested_at) as last_ingested_at

    from {{ ref('fct_transactions') }}
    group by 1, 2
)

select
    *,

    -- 이전 7일 평균 대비 비율. 1 에서 크게 벗어나면 이상이다.
    -- 자기 자신을 제외해야 급변을 잡을 수 있으므로 1 PRECEDING 까지만 본다.
    --
    -- 중앙값이 이상치에 덜 흔들리지만 BigQuery 의 percentile_cont 는
    -- 분석 함수로 쓸 때 ORDER BY 를 허용하지 않아 이동 중앙값을 낼 수 없다.
    -- 급변 감지가 목적이라 평균으로도 충분하고, 오히려 반응이 빠르다.
    safe_divide(
        tx_count,
        avg(tx_count) over (
            partition by source_split
            order by transaction_date
            rows between 7 preceding and 1 preceding
        )
    ) as tx_count_vs_baseline

from daily
