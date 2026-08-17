"""
05_rag.py - RAG 检索增强生成演示
需要先创建一个 test_knowledge.txt 文件
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 准备知识库
knowledge = """
机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进。
监督学习使用标注数据进行训练，常见算法有线性回归、决策树、SVM。
无监督学习使用未标注数据，常见算法有K-means聚类、PCA降维。
深度学习是机器学习的子集，使用多层神经网络。
Transformer是2017年提出的深度学习架构，是GPT和BERT的基础。
Python是机器学习和深度学习最流行的编程语言。
"""

with open("test_knowledge.txt", "w", encoding="utf-8") as f:
    f.write(knowledge)

print("知识库已创建\n")

# 2. 加载和分割
loader = TextLoader("test_knowledge.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)
print(f"分割成 {len(chunks)} 个文档块\n")

# 3. 向量化 + 存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. RAG 链
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    "基于以下资料回答问题：\n{context}\n\n问题：{question}\n回答："
)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 测试
questions = [
    "什么是监督学习？",
    "深度学习是什么？",
    "Transformer是什么时候提出的？"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}\n")
