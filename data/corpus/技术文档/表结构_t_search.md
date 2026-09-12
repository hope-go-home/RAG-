# 搜索表 t_search 表结构说明

## 主键
t_search 表的主键为 search_id，采用自增策略。

## 关键字段
t_search 表的关键字段为 keyword，业务上要求唯一。

## 索引
t_search 表在 idx_keyword 上建立了索引，用于加速查询。

## 分库分表
t_search 表按 search_id 哈希分 16 张表。
