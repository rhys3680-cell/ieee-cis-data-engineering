/*
  거래 집계. 대시보드가 매번 110만 행을 스캔하지 않도록 미리 줄인다.

  날짜 x 시각 x 제품 x 기기로 자른다. 조합해도 4만 행 남짓이라 부담이 없고,
  대시보드는 여기서 필요한 축만 다시 GROUP BY 하면 된다. 특정 조회가 
  느려지면 그 때 전용 집계를 추가한다.

  is_fraud는 test 구간에서 NULL이다. AVG(is_fraud)는 NULL을 무시하므로
  train만 계산된다. 라벨 없는 구간의 사기율은 정의되지 않는다. 
  대신 labeled_count를 함께 내보내 분모를 확인할 수 있게 한다.
*/

select
    transaction_date,
    transaction_hour,
    source_split,
    product_cd,
    coalesce(device_type, 'unknown') as device_type,

    count(*)                              as tx_count,
    countif(is_fraud is not null)         as labeled_count,
    countif(is_fraud = 1)                 as fraud_count,
    avg(is_fraud)                         as fraud_rate,

    sum(transaction_amt)                          as amt_sum,
    avg(transaction_amt)                          as amt_avg,
    sum(if(is_fraud = 1, transaction_amt, 0))     as fraud_amt,

    countif(has_identity)                 as with_identity_count

from {{ ref('fct_transactions') }}
group by 1, 2, 3, 4, 5