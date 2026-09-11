# 企业知识库智能问答系统API接入文档

**版本：V2.1**
**更新日期：2024年1月1日**
**基础URL：https://api.knowledge-base.company.com/v2**

---

## 一、概述

本文档描述企业知识库智能问答系统的RESTful API接口，供内部系统和第三方应用接入使用。系统提供文档管理、知识检索和智能问答等功能。

---

## 二、认证方式

### 2.1 API Key认证

所有API请求必须在HTTP Header中携带有效的API Key进行认证。

**请求头格式：**

```
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

**获取API Key：**

1. 登录知识库管理后台
2. 进入"系统设置"→"API管理"
3. 点击"创建API Key"
4. 复制生成的API Key并妥善保管

**注意事项：**
- API Key具有有效期，默认为365天
- 不同环境（开发/测试/生产）使用不同的API Key
- API Key泄露应立即重新生成

---

## 三、API接口列表

### 3.1 用户查询接口

查询系统中的用户信息。

**接口地址：** `GET /users`

**请求参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| page | integer | 否 | 页码，默认1 |
| page_size | integer | 否 | 每页数量，默认20，最大100 |
| keyword | string | 否 | 搜索关键词 |
| department | string | 否 | 部门筛选 |

**请求示例：**

```python
import requests

url = "https://api.knowledge-base.company.com/v2/users"
headers = {
    "Authorization": "Bearer your_api_key_here",
    "Content-Type": "application/json"
}
params = {
    "page": 1,
    "page_size": 20,
    "keyword": "张三"
}

response = requests.get(url, headers=headers, params=params)
print(response.json())
```

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "total": 150,
    "page": 1,
    "page_size": 20,
    "items": [
      {
        "user_id": "U20240001",
        "name": "张三",
        "department": "技术部",
        "position": "高级工程师",
        "email": "zhangsan@company.com",
        "phone": "138-xxxx-xxxx",
        "status": "active",
        "created_at": "2024-01-15T09:00:00Z"
      }
    ]
  }
}
```

---

### 3.2 文档上传接口

上传文档到知识库系统。

**接口地址：** `POST /documents/upload`

**请求参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| file | binary | 是 | 文件内容（multipart/form-data） |
| category | string | 是 | 文档分类 |
| tags | array | 否 | 文档标签 |
| description | string | 否 | 文档描述 |
| access_level | string | 否 | 访问级别：public/internal/confidential |

**请求示例：**

```python
import requests

url = "https://api.knowledge-base.company.com/v2/documents/upload"
headers = {
    "Authorization": "Bearer your_api_key_here"
}

files = {
    "file": ("考勤管理制度.md", open("考勤管理制度.md", "rb"), "text/markdown")
}
data = {
    "category": "规章制度",
    "tags": "考勤,管理制度",
    "description": "公司考勤管理制度V3.2",
    "access_level": "internal"
}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.json())
```

**响应示例：**

```json
{
  "code": 200,
  "message": "文档上传成功",
  "data": {
    "doc_id": "DOC20240101001",
    "filename": "考勤管理制度.md",
    "category": "规章制度",
    "file_size": 15360,
    "chunk_count": 25,
    "status": "processing",
    "created_at": "2024-01-01T10:30:00Z",
    "estimated_processing_time": 60
  }
}
```

**错误响应：**

```json
{
  "code": 400,
  "message": "文件格式不支持",
  "error": {
    "detail": "支持的文件格式：md, txt, pdf, docx, xlsx"
  }
}
```

---

### 3.3 文档查询接口

查询已上传的文档信息。

**接口地址：** `GET /documents`

**请求参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| page | integer | 否 | 页码，默认1 |
| page_size | integer | 否 | 每页数量，默认20 |
| category | string | 否 | 按分类筛选 |
| keyword | string | 否 | 搜索关键词 |
| status | string | 否 | 文档状态：processing/ready/failed |

**请求示例：**

```bash
curl -X GET "https://api.knowledge-base.company.com/v2/documents?category=规章制度&page=1&page_size=10" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json"
```

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "total": 50,
    "page": 1,
    "page_size": 10,
    "items": [
      {
        "doc_id": "DOC20240101001",
        "filename": "考勤管理制度.md",
        "category": "规章制度",
        "file_size": 15360,
        "chunk_count": 25,
        "status": "ready",
        "created_at": "2024-01-01T10:30:00Z",
        "updated_at": "2024-01-01T10:31:00Z"
      }
    ]
  }
}
```

---

### 3.4 知识检索接口

基于语义检索知识库内容。

**接口地址：** `POST /search`

**请求参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| query | string | 是 | 查询内容 |
| category | string | 否 | 限定分类 |
| top_k | integer | 否 | 返回结果数量，默认5，最大20 |
| score_threshold | float | 否 | 相似度阈值，默认0.7 |

**请求示例：**

```python
import requests

url = "https://api.knowledge-base.company.com/v2/search"
headers = {
    "Authorization": "Bearer your_api_key_here",
    "Content-Type": "application/json"
}
payload = {
    "query": "请假需要提前几天申请",
    "category": "规章制度",
    "top_k": 5,
    "score_threshold": 0.7
}

response = requests.post(url, headers=headers, json=payload)
results = response.json()

for item in results["data"]["results"]:
    print(f"文档: {item['doc_name']}")
    print(f"内容: {item['content']}")
    print(f"相似度: {item['score']}")
    print("---")
```

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "query": "请假需要提前几天申请",
    "results": [
      {
        "doc_id": "DOC20240101001",
        "doc_name": "考勤管理制度.md",
        "chunk_id": "chunk_21",
        "content": "第二十四条 事假规定：事假需提前1个工作日申请，紧急情况可当日申请但需说明原因。",
        "score": 0.95,
        "metadata": {
          "category": "规章制度",
          "article": "第二十四条",
          "section": "请假制度"
        }
      },
      {
        "doc_id": "DOC20240101002",
        "doc_name": "HR常见问题.md",
        "chunk_id": "chunk_15",
        "content": "Q: 请假审批流程是怎样的？\nA: 请假需提前通过OA系统提交申请，事假提前1天，年假提前3天...",
        "score": 0.88,
        "metadata": {
          "category": "FAQ",
          "type": "Q&A"
        }
      }
    ]
  }
}
```

---

### 3.5 智能问答接口

基于知识库进行智能问答。

**接口地址：** `POST /chat`

**请求参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| question | string | 是 | 用户问题 |
| conversation_id | string | 否 | 会话ID（用于多轮对话） |
| category | string | 否 | 限定知识范围 |
| max_tokens | integer | 否 | 最大生成长度，默认500 |
| temperature | float | 否 | 生成温度，默认0.7 |

**请求示例：**

```python
import requests

url = "https://api.knowledge-base.company.com/v2/chat"
headers = {
    "Authorization": "Bearer your_api_key_here",
    "Content-Type": "application/json"
}
payload = {
    "question": "公司年假有多少天？",
    "category": "规章制度",
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()

print(f"回答: {result['data']['answer']}")
print(f"参考来源: {result['data']['references']}")
```

**响应示例：**

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "conversation_id": "conv_20240101001",
    "question": "公司年假有多少天？",
    "answer": "根据公司考勤管理制度规定，年假天数如下：工作满1年不满10年的，年休假5天；工作满10年不满20年的，年休假10天；工作满20年的，年休假15天。年假需在当年12月31日前使用完毕，未休完的按日工资的300%折算补偿。",
    "references": [
      {
        "doc_id": "DOC20240101001",
        "doc_name": "考勤管理制度.md",
        "chunk_id": "chunk_26",
        "content": "第二十六条 年假规定：工作满1年不满10年的，年休假5天；工作满10年不满20年的，年休假10天；工作满20年的，年休假15天。",
        "score": 0.96
      }
    ],
    "confidence": 0.95,
    "generated_at": "2024-01-01T10:35:00Z"
  }
}
```

**多轮对话示例：**

```python
# 第一轮对话
payload = {
    "question": "请假需要提前几天申请？",
    "category": "规章制度"
}
response1 = requests.post(url, headers=headers, json=payload)
conversation_id = response1.json()["data"]["conversation_id"]

# 第二轮对话（使用conversation_id保持上下文）
payload = {
    "question": "那病假呢？",
    "conversation_id": conversation_id,
    "category": "规章制度"
}
response2 = requests.post(url, headers=headers, json=payload)
```

---

### 3.6 文档删除接口

删除知识库中的文档。

**接口地址：** `DELETE /documents/{doc_id}`

**请求示例：**

```bash
curl -X DELETE "https://api.knowledge-base.company.com/v2/documents/DOC20240101001" \
  -H "Authorization: Bearer your_api_key_here"
```

**响应示例：**

```json
{
  "code": 200,
  "message": "文档删除成功",
  "data": {
    "doc_id": "DOC20240101001",
    "deleted_at": "2024-01-02T10:00:00Z"
  }
}
```

---

## 四、错误码说明

| 错误码 | 说明 | 处理建议 |
|--------|------|---------|
| 200 | 成功 | - |
| 400 | 请求参数错误 | 检查请求参数 |
| 401 | 认证失败 | 检查API Key |
| 403 | 权限不足 | 申请相应权限 |
| 404 | 资源不存在 | 检查资源ID |
| 429 | 请求频率超限 | 降低请求频率 |
| 500 | 服务器内部错误 | 联系技术支持 |
| 503 | 服务暂不可用 | 稍后重试 |

---

## 五、请求频率限制

为保证系统稳定性，API请求频率限制如下：
- 认证接口：10次/分钟
- 查询接口：100次/分钟
- 写入接口：30次/分钟
- 问答接口：20次/分钟

超出限制将返回429错误码，请合理安排请求频率。

---

## 六、SDK支持

系统提供以下语言的SDK：
- Python SDK：`pip install kb-sdk-python`
- Java SDK：Maven依赖 `com.company:kb-sdk-java`
- JavaScript SDK：`npm install kb-sdk-js`

SDK封装了认证、请求签名、错误处理等功能，推荐使用。

---

## 七、技术支持

- 技术文档：https://docs.knowledge-base.company.com
- API调试工具：https://api-debug.knowledge-base.company.com
- 技术支持邮箱：api-support@company.com
- 技术支持电话：400-xxx-xxxx

---

**文档维护：技术部**
**最后更新：2024年1月1日**
