# 优惠券表 t_coupon 表结构说明

## 主键
t_coupon 表的主键为 coupon_id，采用自增策略。

## 关键字段
t_coupon 表的关键字段为 coupon_code，业务上要求唯一。

## 索引
t_coupon 表在 idx_coupon 上建立了索引，用于加速查询。

## 分库分表
t_coupon 表按 coupon_id 哈希分 16 张表。
