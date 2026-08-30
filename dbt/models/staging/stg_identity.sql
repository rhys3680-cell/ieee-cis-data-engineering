{{
  config(
    materialized = 'view'
  )
}}

/*
  기기·네트워크 정보를 표준화한다. 전체 거래의 약 25%에만 존재한다.

  id_01~id_38은 이미 snake_case이고 의미가 익명화되어 있어 그대로 둔다.
  바꾸는 것은 조인 키와 기기 컬럼 두 개뿐이다.
*/

select
    TransactionID   as transaction_id,
    DeviceType      as device_type,
    DeviceInfo      as device_info,

    * except (
        TransactionID,
        DeviceType,
        DeviceInfo
    )

from {{ source('ieee_raw', 'identity') }}
