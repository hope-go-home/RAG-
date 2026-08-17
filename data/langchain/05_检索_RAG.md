# 05 - 检索：RAG（检索增强生成）

---

## 一、什么是 RAG？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

### 传统 LLM 的问题

- 知识截止于训练数据
- 无法知道你的私有数据
- 可能产生幻觉

### RAG 的解决思路

```
用户提问 → 从知识库检索相关内容 → 把检索结果 + 问题 一起给 LLM → LLM 基于资料回答
```

### RAG 完整流程

```
                     ┌─────────────────┐
                     │   文档加载器     │ ← PDF/Word/网页/数据库
                     └────────┬────────┘
                              ↓
                     ┌─────────────────┐
                     │   文本分割器     │ ← 切成小块
                     └────────┬────────┘
                              ↓
                     ┌─────────────────┐
                     │   嵌入+向量存储  │ ← 转成向量存到数据库
                     └────────┬────────┘
                              ↓
    用户提问 → ┌──────────────┴──────────────┐
               │         检索器              │ ← 找到最相关的文档块
               └──────────────┬──────────────┘
                              ↓
               ┌────────────────────────────┐
               │   Prompt（问题 + 资料）     │
               └──────────────┬─────────────┘
                              ↓
                           ┌──────┐
                           │ LLM  │ ← 基于资料回答
                           └──────┘
```

---

## 二、Document Loaders（文档加载器）

### 加载各类文档

```bash
pip install langchain-community
pip install pypdf          # PDF
pip install python-docx    # Word
pip install beautifulsoup4  # 网页
```

```python
# 文本文件
from langchain_community.document_loaders import TextLoader
loader = TextLoader("my_doc.txt")
docs = loader.load()

# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("report.pdf")
docs = loader.load()

# Word
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("doc.docx")
docs = loader.load()

# CSV
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()

# 网页
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# 多个文件
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs/", glob="**/*.txt")
docs = loader.load()
```

### Document 对象结构

```python
# 每个文档包含：
doc.page_content   # 文本内容
doc.metadata       # 元数据（来源、页码等）
```

---

## 三、Text Splitters（文本分割器）

将长文本切成小块，方便检索和嵌入。

### 常用分割器

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# 按字符递归分割（推荐）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每块最大字符数
    chunk_overlap=50,     # 块间重叠字符数
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]  # 分割优先级
)

# 按 token 分割（更精确）
token_splitter = TokenTextSplitter(
    chunk_size=200,       # 每块最大 token 数
    chunk_overlap=20
)

# 使用
chunks = splitter.split_documents(docs)
# 或
chunks = splitter.split_text("很长的文本内容...")
```

### 分割参数的影响

```
chunk_size=500, chunk_overlap=50:
                        ┌─ 块1 (500字符) ─┐
                                           │← 重叠50 →│
                                          ┌─ 块2 (500字符) ─┐
```

**chunk_size：** 越小检索越精确，但丢失上下文
**chunk_overlap：** 避免在中间切断重要信息

---

## 四、Embeddings（嵌入模型）

把文本转换为向量（一堆浮点数），让"语义相近"的文本向量距离更近。

```bash
pip install langchain-openai
```

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # OpenAI 嵌入模型
)

# 单个文本转向量
vector = embeddings.embed_query("今天天气真好")
print(len(vector))  # 1536 维向量

# 批量转
vectors = embeddings.embed_documents([
    "今天天气真好",
    "明天可能会下雨"
])
```

### 本地嵌入

```python
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

---

## 五、Vector Stores（向量数据库）

存储向量并支持相似度搜索。

```bash
pip install langchain-chroma
```

### Chroma（简单，适合入门）

```python
from langchain_chroma import Chroma

# 创建向量库
vectorstore = Chroma.from_documents(
    documents=chunks,           # 文档块
    embedding=embeddings,       # 嵌入模型
    persist_directory="./chroma_db"  # 持久化目录
)

# 搜索最相关的 k 个文档
results = vectorstore.similarity_search(
    "量子计算是什么？",
    k=3  # 返回最相似的3个
)

for doc in results:
    print(doc.page_content)
    print("---")

# 带分数的搜索
results = vectorstore.similarity_search_with_score(
    "量子计算是什么？", k=3
)
for doc, score in results:
    print(f"分数: {score:.3f}")
    print(doc.page_content)
```

### FAISS（高性能）

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./faiss_index")

# 加载
vectorstore = FAISS.load_local("./faiss_index", embeddings,
                               allow_dangerous_deserialization=True)
```

---

## 六、Retrievers（检索器）

### 基础检索器

```python
# 从向量库创建检索器
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 返回前3个结果
)

# 使用
docs = retriever.invoke("量子计算是什么？")
```

### 多种检索策略

```python
# MMR（最大边际相关性）：保证结果多样性
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

# 按分数阈值
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
```

---

## 七、完整 RAG 链条

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
loader = TextLoader("知识库.txt")
docs = loader.load()

# 2. 分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 嵌入+存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 提示词模板
template = """基于以下资料回答问题。如果你不知道答案，就说不知道。

资料：
{context}

问题：{question}

回答："""
prompt = ChatPromptTemplate.from_template(template)

# 6. 构建 RAG 链
llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,   # 检索并格式化
        "question": RunnablePassthrough()      # 原样传递问题
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 使用
result = rag_chain.invoke("什么是量子计算？")
print(result)
```

---

## 八、RAG 进阶技巧

### 带来源引用

```python
def format_with_sources(docs):
    sources = []
    for i, doc in enumerate(docs):
        sources.append(f"[{i+1}] {doc.page_content}\n(来源: {doc.metadata.get('source', '未知')})")
    return "\n\n".join(sources)

rag_chain = (
    {"context": retriever | format_with_sources, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

### 多轮对话 RAG

```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate([
    ("system", "基于资料回答问题：{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])
```

---

## 九、总结：RAG 核心组件

| 组件 | 作用 | 常用实现 |
|------|------|---------|
| **Document Loaders** | 加载各种格式文档 | TextLoader, PyPDFLoader, WebBaseLoader |
| **Text Splitters** | 切分文档 | RecursiveCharacterTextSplitter |
| **Embeddings** | 文本→向量 | OpenAIEmbeddings, OllamaEmbeddings |
| **Vector Stores** | 存储+搜索向量 | Chroma, FAISS, Pinecone |
| **Retrievers** | 从向量库检索 | as_retriever(), MMR |
| **Chain** | 串联所有组件 | LCEL `\|` |

---

**下一篇 → [06_智能体_Agents](06_智能体_Agents.md)**
