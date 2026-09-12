# 订单表 t_order 表结构说明

## 主键
t_order 表的主键为 order_id，采用自增策略。

## 关键字段
t_order 表的关键字段为 order_no，业务上要求唯一。

## 索引
t_order 表在 idx_order_no 上建立了索引，用于加速查询。

## 分库分表
t_order 表按 order_id 哈希分 16 张表。
