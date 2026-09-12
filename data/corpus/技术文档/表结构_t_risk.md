# 风控表 t_risk 表结构说明

## 主键
t_risk 表的主键为 risk_id，采用自增策略。

## 关键字段
t_risk 表的关键字段为 event_no，业务上要求唯一。

## 索引
t_risk 表在 idx_event 上建立了索引，用于加速查询。

## 分库分表
t_risk 表按 risk_id 哈希分 16 张表。
