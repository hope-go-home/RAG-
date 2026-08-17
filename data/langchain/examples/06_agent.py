"""
06_agent.py - 智能体演示
需要设置 OPENAI_API_KEY
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. 定义工具
@tool
def calculate(expression: str) -> str:
    """执行数学计算。输入数学表达式如 '2**10'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"错误: {e}"

@tool
def get_fact(topic: str) -> str:
    """返回关于某个主题的一个有趣事实"""
    facts = {
        "python": "Python 是由 Guido van Rossum 于 1991 年创建的。",
        "月亮": "月亮每年远离地球约 3.8 厘米。",
        "AI": "人工智能的概念最早在 1956 年的达特茅斯会议上提出。"
    }
    return facts.get(topic.lower(), f"没有关于 '{topic}' 的事实。")

tools = [calculate, get_fact]

# 2. 创建 Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，用工具回答问题。回答要简洁。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 3. 测试
print("=== Agent 演示 ===\n")

questions = [
    "计算 1234 * 5678 等于多少？",
    "告诉我关于 Python 的一个事实",
    "先算 2的10次方，然后告诉我关于 AI 的事实"
]

for q in questions:
    print(f"\n用户: {q}")
    result = agent_executor.invoke({"input": q})
    print(f"助手: {result['output']}")
    print("-" * 40)
