{{
    config(
        materialized = 'view'
    )
}}

/*
    거래 원본을 표준화한다. 이 계층은 이름과 타입만 정리하고 값은 바꾸지 않는다.
    비즈니스 로직과 컬럼 선별은 mart의 역할이다.

    SELECT * 을 쓰는 이유:
        익명 컬럼(V1~V339, C1~C14, D1~D15, M1~M9)이 387개 중 이름이 바뀔 게
        없다. 실제로 재정의하는 것은 7개 뿐이고, 나머지를 손으로 나열하면
        오히려 누락과 오타가 발생한다.

        소스에 컬럼이 추가되면 여기를 그대로 통과한다. validation은 두 곳에서 진행한다.
        적재 단계(src/load/gcs.py)가 업로드 전에 컬럼 집합을 스키마와 대조해 추가 및 누락을 막고, mart는 컬럼을 명시해 승인되지 않은 컬럼이 소비 레이어까지 가지 못하도록 막는다.
 */

select
    -- 카멜케이스 / 접두사 표기를 snake_case 로 통일한다.
    TransactionID   as transaction_id,
    TransactionDT   as transaction_dt,
    TransactionAmt  as transaction_amt,
    ProductCD       as product_cd,
    isFraud         as is_fraud,
    P_emaildomain   as purchaser_email_domain,
    R_emaildomain   as recipient_email_domain,

    -- 나머지는 이미 소문자이거나 익명 컬럼이라 그대로 둔다.
    * except (
        TransactionID,
        TransactionDT,
        TransactionAmt,
        ProductCD,
        isFraud,
        P_emaildomain,
        R_emaildomain
    )

from {{ source('ieee_raw', 'transactions') }}