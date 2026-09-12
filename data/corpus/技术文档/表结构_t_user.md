# 用户表 t_user 表结构说明

## 主键
t_user 表的主键为 user_id，采用自增策略。

## 关键字段
t_user 表的关键字段为 username，业务上要求唯一。

## 索引
t_user 表在 idx_username 上建立了索引，用于加速查询。

## 分库分表
t_user 表按 user_id 哈希分 16 张表。
