/*
  점수 x 금액 사분면. 임계값 하나로 자를 때 무엇이 빠져나가는지 본다.

  점수만 보면 "얼마나 의심스러운가"만 남고, 금액을 함께 보면 "틀렸을 때
  얼마를 잃는가"가 드러난다. 두 축이 다른 것을 말하므로 조합하면 대응이
  갈린다.

    점수 높음 x 금액 높음   최우선 검토. 틀리면 손실이 크다.
    점수 높음 x 금액 낮음   자동 차단 후보. 건수는 많고 금액은 작다.
    점수 낮음 x 금액 높음   놓치면 가장 아픈 구간. 임계값 하나로는 빠져나간다.
    점수 낮음 x 금액 낮음   통과.

  세 번째 칸이 이 모델을 만든 이유다. test 32일 기준으로 건수는 7,308 건인데
  금액 합계가 449만 달러로 가장 크다. 임계값을 금액대별로 다르게 둘지
  판단하려면 이 값이 필요하다.

  라벨이 없어 사기율은 낼 수 없다. train 구간에는 예측이 없고 test 구간에는
  정답이 없다 — 둘이 겹치지 않으므로 '이 칸의 실제 사기율'은 정의되지 않는다.
  금액과 건수만 낸다.
*/

{{ config(materialized='table') }}

with scored as (
    select
        p.transaction_date,
        p.score,
        t.transaction_amt,
        t.product_cd
    from {{ source('ieee_raw', 'predictions') }} p
    join {{ ref('stg_transactions') }} t using (transaction_id)
)

select
    transaction_date,
    product_cd,

    -- 기준 임계값. 화면에서 슬라이더로 만지는 값과 별개로, 날짜 간 비교를
    -- 하려면 고정된 기준이 있어야 한다.
    case when score >= {{ var('block_threshold') }} then 'high' else 'low' end
        as score_band,

    -- 금액 경계는 상위 10% 근처다. 사기율이 U 자를 그려 소액과 고액이 모두
    -- 높지만, 손실은 금액에 비례하므로 여기서는 금액만 본다.
    case when transaction_amt >= {{ var('high_amount') }} then 'high' else 'low' end
        as amount_band,

    count(*)                    as tx_count,
    sum(transaction_amt)        as amount_total,
    avg(transaction_amt)        as amount_avg,
    min(score)                  as score_min,
    max(score)                  as score_max

from scored
group by 1, 2, 3, 4