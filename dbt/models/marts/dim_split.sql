/*
  ML 학습용 날짜 분할. 행 하나가 날짜 하나다.

  시간 순서로 자른다. 무작위로 나누면 같은 카드의 나중 거래가 train에 
  들어가 미래 데이터가 학습에 사용되게 된다. 시계열 데이터이므로 주의해야
  한다.

  경계는 train 마지막 4주를 valid 로 둔다. 경계일을 valid 에 포함하므로
  (>=) 실제로는 29일(2018-05-04~06-01), train 153일이 된다.
    - valid 82,325 행 / 사기 2,868 건. PR-AUC 를 안정적으로 잴 수 있다.
    - train 508,215 행이 남아 학습량 손실이 14% 에 그친다.
    - 4주면 요일 주기가 온전히 들어간다. 사기율이 시각대별로 3배
      차이나므로 주기가 깨진 구간을 valid 로 쓰면 평가가 흔들린다.
    - 후보 21/28/35/42일 모두 사기율 3.4~3.7% 로 안정적이었다. 어느 쪽도
      편향되지 않아 크기만 보고 골랐다.
  
  test는 is_fraud가 NULL이라 평가에 사용할 수 없다. 분할에는 넣되, 학습과 
  평가에서는 제외한다.

  fct_transactions가 아니라 별도 dim인 이유는 피처 모델과 학습 코드가 
  같은 경계를 참조하게 하기 위해서다. 날짜를 양쪽에 모두 포함하면 실험마다 
  달라진다. 경계를 바꿔 실험할 때는 dbt_project.yml의 valid_start만 수정한다.
*/

select
    transaction_date,

    case
        when source_split = 'test' then 'test'
        when transaction_date >= date('{{ var("valid_start") }}') then 'valid'
        else 'train'
    end as ml_split

from {{ ref('stg_transactions') }}
group by 1, source_split
