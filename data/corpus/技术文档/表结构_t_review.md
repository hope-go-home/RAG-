# 评价表 t_review 表结构说明

## 主键
t_review 表的主键为 review_id，采用自增策略。

## 关键字段
t_review 表的关键字段为 review_no，业务上要求唯一。

## 索引
t_review 表在 idx_review 上建立了索引，用于加速查询。

## 分库分表
t_review 表按 review_id 哈希分 16 张表。
