/*
  거래 팩트. transaction에 identity를 붙인다.

  컬럼을 명시한다. staging은 소스를 그대로 통과시키지만 여기는 소비처와의
  계약이라, 승인되지 않은 컬럼이 대시보드나 외부 공유까지 가지 않도록 한다.

  익명 컬럼(V1~V339, C1~C14, D1~D15, M1~M9)은 올리지 않는다. 의미를 알 수
  없어서 대시보드에는 부적합하다. ML 피처로는 참조할 수 있도록 features 레이어에서 
  staging을 직접 참조해서 사용한다.

  조인은 1:1이다. identity는 전체 거래의 약 25%이므로 LEFT JOIN이고,
  행 수는 stg_transactions와 같도록 테스트로 고정한다.
*/

with transactions as (
    select * from {{ ref('stg_transactions') }}
),

identity as (
    select * from {{ ref('stg_identity') }}
)

select 
    -- 식별 / 시간
    t.transaction_id,
    t.transaction_date,
    t.source_split,

    -- 하루 중 시각. 기준일이 임의값이라 절대 시각은 의미 없고,
    -- 패턴 비교에만 사용한다.
    mod(div(t.transaction_dt, 3600), 24) as transaction_hour,

    -- 거래
    t.transaction_amt,
    t.product_cd,

    -- 카드
    t.card1, t.card2, t.card3, t.card4, t.card5, t.card6,

    -- 주소 / 거리
    t.addr1, t.addr2, t.dist1, t.dist2,

    -- 이메일
    t.purchaser_email_domain,
    t.recipient_email_domain,

    -- 기기. identity 가 있는 거래에만 존재한다.
    i.device_type,
    i.device_info,
    i.id_30 as device_os,
    i.id_31 as device_browser,
    i.transaction_id is not null as has_identity,

    -- 라벨. test 구간은 조사 중이라 NULL 이다.
    t.is_fraud,

    t.ingested_at

from transactions t
left join identity i
    on t.transaction_id = i.transaction_id