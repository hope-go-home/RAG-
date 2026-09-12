# 消息表 t_message 表结构说明

## 主键
t_message 表的主键为 message_id，采用自增策略。

## 关键字段
t_message 表的关键字段为 msg_no，业务上要求唯一。

## 索引
t_message 表在 idx_msg 上建立了索引，用于加速查询。

## 分库分表
t_message 表按 message_id 哈希分 16 张表。
