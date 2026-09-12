# 商品表 t_product 表结构说明

## 主键
t_product 表的主键为 product_id，采用自增策略。

## 关键字段
t_product 表的关键字段为 sku_code，业务上要求唯一。

## 索引
t_product 表在 idx_sku 上建立了索引，用于加速查询。

## 分库分表
t_product 表按 product_id 哈希分 16 张表。
