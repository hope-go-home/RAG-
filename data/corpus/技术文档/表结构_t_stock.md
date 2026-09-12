# 库存表 t_stock 表结构说明

## 主键
t_stock 表的主键为 stock_id，采用自增策略。

## 关键字段
t_stock 表的关键字段为 warehouse_code，业务上要求唯一。

## 索引
t_stock 表在 idx_warehouse 上建立了索引，用于加速查询。

## 分库分表
t_stock 表按 stock_id 哈希分 16 张表。
