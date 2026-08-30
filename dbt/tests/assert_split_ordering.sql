/*
  train 이 valid 보다 먼저 끝나야 한다.

  이 모델의 핵심 계약이다. valid_start 를 잘못 바꾸거나 case 순서가
  뒤집히면 미래 데이터를 학습하게 되는데, 성능이 너무 좋으면 의심해봐야 한다. 겹치면 행을 반환해 실패한다.
*/

with bounds as (
    select
        max(if(ml_split = 'train', transaction_date, null)) as train_end,
        min(if(ml_split = 'valid', transaction_date, null)) as valid_start
    from {{ ref('dim_split') }}
)

select * from bounds
where train_end >= valid_start
