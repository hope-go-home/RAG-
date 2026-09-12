# 支付表 t_payment 表结构说明

## 主键
t_payment 表的主键为 payment_id，采用自增策略。

## 关键字段
t_payment 表的关键字段为 trade_no，业务上要求唯一。

## 索引
t_payment 表在 idx_trade_no 上建立了索引，用于加速查询。

## 分库分表
t_payment 表按 payment_id 哈希分 16 张表。
