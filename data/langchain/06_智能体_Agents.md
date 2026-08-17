# 06 - 智能体：Agents

---

## 一、什么是 Agent？

**Agent（智能体）** = 让 LLM 自己**决定**调用什么工具来完成你的任务。

### 传统 Chain vs Agent

```
Chain:   输入 → [固定流程] → 输出
Agent:   输入 → [LLM 思考 → 调用工具 → 观察结果 → 再思考 → ...] → 输出
```

### Agent 的核心循环

```
用户请求
     ↓
┌─ LLM 思考（需要调用什么工具？）
│       ↓
│   调用工具（搜索/计算/查数据库...）
│       ↓
│   观察结果
│       ↓
│   还没完成？→ 回到 LLM 思考
│       ↓
│   完成 → 输出最终答案
└────────┘
```

这个循环叫做 **ReAct（Reasoning + Acting）** 模式。

---

## 二、Tools（工具）

### 内置工具

```bash
pip install langchain-community
```

```python
from langchain_community.tools import (
    DuckDuckGoSearchRun,     # 网页搜索
    CalculatorTool,          # 计算器
    WikipediaQueryRun,       # Wikipedia
    ShellTool,               # 执行命令
)

# 搜索工具
search = DuckDuckGoSearchRun()
result = search.invoke("2024年奥运会")
print(result)

# 计算器
calc = CalculatorTool()
result = calc.invoke("2 ** 10")  # 1024

# Wikipedia
wiki = WikipediaQueryRun()
result = wiki.invoke("Python")
```

### 自定义工具

```python
from langchain_core.tools import tool

# 简单工具（用装饰器）
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息"""
    # 这里可以调用真实天气 API
    return f"{city}的天气：晴天，25°C"

# 复杂工具
@tool
def multiply(a: int, b: int) -> int:
    """计算两个数的乘积"""
    return a * b

# 测试
result = get_weather.invoke({"city": "北京"})
print(result)
```

### 工具的重要字段

```python
@tool
def my_tool(param1: str, param2: int) -> str:
    """工具的详细描述（LLM 根据这个决定是否调用你）"""
    # 参数类型注解帮助 LLM 正确传参
    return f"结果: {param1}, {param2}"

# 查看工具的信息
print(my_tool.name)        # "my_tool"
print(my_tool.description) # "工具的详细描述..."
print(my_tool.args)        # 参数 schema
```

---

## 三、创建 Agent

### 方式1：create_react_agent（推荐）

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# 工具列表
tools = [get_weather, multiply, search]

# 提示词
prompt = PromptTemplate.from_template(
    """你是一个助手，可以用工具回答问题。

工具：
{tools}

工具名称：{tool_names}

请按以下格式回答：
思考：你需要做什么
行动：工具名称
行动输入：工具参数
观察：工具结果
...
最终答案：你的回答

问题：{input}

{agent_scratchpad}"""
)

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 显示思考过程
    handle_parsing_errors=True,
    max_iterations=5       # 最多思考-行动轮次
)

# 使用
result = agent_executor.invoke({"input": "北京的天气怎么样？顺便算一下 123 * 456"})
print(result["output"])
```

### 方式2：create_openai_tools_agent（OpenAI 函数调用）

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，用工具回答问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

result = agent_executor.invoke({"input": "搜索 Python 的最新版本"})
```

---

## 四、Agent 的思考过程

```
> 进入新的 AgentExecutor 链...
思考：用户想知道北京天气和计算 123*456，我需要分别调用两个工具。
行动：get_weather
行动输入：{"city": "北京"}
观察：北京的天气：晴天，25°C

思考：天气查到了，现在计算乘法。
行动：multiply
行动输入：{"a": 123, "b": 456}
观察：56088

思考：两个结果都有了，可以回答。
最终答案：北京的天气是晴天，25°C。123 × 456 = 56088。

> 链结束。
```

---

## 五、预定义 Agent 类型

```python
from langchain.agents import (
    create_tool_calling_agent,   # 函数调用（推荐）
    create_react_agent,          # ReAct 模式
    create_self_ask_with_search, # Self-Ask
    create_json_chat_agent       # JSON 格式
)
```

| Agent类型 | 适用场景 |
|-----------|---------|
| `create_tool_calling_agent` | 模型支持函数调用（GPT-4, Claude 3） |
| `create_react_agent` | 通用 ReAct 模式 |
| `create_self_ask_with_search` | 需要多步推理的问题 |
| `create_json_chat_agent` | 需要结构化输出 |

---

## 六、Agent 实战：数据分析助手

```python
import pandas as pd
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

# 数据分析工具
@tool
def query_data(sql: str) -> str:
    """用 SQL 查询 CSV 数据。表名: data"""
    df = pd.read_csv("sales.csv")
    try:
        result = df.query(sql)
        return result.to_string()
    except Exception as e:
        return f"错误: {e}"

@tool
def plot_chart(chart_type: str, x_col: str, y_col: str) -> str:
    """生成图表。chart_type: line/bar/scatter"""
    import matplotlib.pyplot as plt
    df = pd.read_csv("sales.csv")
    fig, ax = plt.subplots()
    
    if chart_type == "line":
        ax.plot(df[x_col], df[y_col])
    elif chart_type == "bar":
        ax.bar(df[x_col], df[y_col])
    
    plt.savefig("chart.png")
    return "图表已保存为 chart.png"

tools = [query_data, plot_chart]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是数据分析助手，用工具分析数据。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "哪个品类的销售额最高？画个柱状图"
})
```

---

## 七、Agent + Memory（智能体带记忆）

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5
)

result = agent_executor.invoke({"input": "我叫张三"})
result = agent_executor.invoke({"input": "搜索最新的 AI 新闻"})
result = agent_executor.invoke({"input": "我叫什么名字？"})  # ✅ 记得叫张三
```

---

## 八、Agent 执行控制

```python
# 限制执行时间
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,          # 最多5轮思考-行动
    max_execution_time=30,     # 最多30秒
    early_stopping_method="generate"  # 超时后强制生成答案
)

# 处理错误
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # 解析失败时自动重试
    return_intermediate_steps=True  # 返回中间步骤（调试用）
)

# 获取完整执行过程
result = agent_executor.invoke({"input": "搜索 AI 新闻"})
for step in result["intermediate_steps"]:
    action, observation = step
    print(f"行动: {action.tool}({action.tool_input})")
    print(f"观察: {observation}")
```

---

## 九、总结

```
Agent = LLM（思考）+ Tools（工具）
   ↓
LLM 决定：什么时候用什么工具、什么时候给出最终答案
```

| 概念 | 说明 |
|------|------|
| **Tool** | 一个可调用的函数（搜索、计算、API） |
| **Agent** | LLM + Tools 的组合，自主决策 |
| **AgentExecutor** | 执行 Agent 循环的引擎 |
| **ReAct** | 思考→行动→观察→再思考的循环模式 |

---

**下一篇 → [07_回调_Callbacks](07_回调_Callbacks.md)**
