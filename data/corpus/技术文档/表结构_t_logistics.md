# 物流表 t_logistics 表结构说明

## 主键
t_logistics 表的主键为 logistics_id，采用自增策略。

## 关键字段
t_logistics 表的关键字段为 waybill_no，业务上要求唯一。

## 索引
t_logistics 表在 idx_waybill 上建立了索引，用于加速查询。

## 分库分表
t_logistics 表按 logistics_id 哈希分 16 张表。
