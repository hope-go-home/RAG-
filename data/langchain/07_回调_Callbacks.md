# 07 - 回调：Callbacks

---

## 一、什么是 Callback？

**Callback（回调）** 让你在 Chain 运行时"监听"和"干预"各个阶段的事件。

### 你能做什么？

- 📝 **记录日志** —— 记录 LLM 的输入输出
- 📊 **监控性能** —— 计算 token 用量、响应时间
- 🔄 **实时更新 UI** —— 在 Streamlit 中显示进度
- 🐛 **调试** —— 查看中间结果

---

## 二、BaseCallbackHandler

实现自己的回调处理器：

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallback(BaseCallbackHandler):
    """自定义回调处理器"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始调用时触发"""
        print(f"🤖 LLM 收到输入: {prompts[:50]}...")

    def on_llm_end(self, response, **kwargs):
        """LLM 返回结果时触发"""
        print(f"✅ LLM 输出: {response.generations[0][0].text[:50]}...")

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Chain 开始执行时触发"""
        print(f"🔗 Chain 开始: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        """Chain 执行完成时触发"""
        print(f"✅ Chain 完成: {outputs}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """工具开始调用时触发"""
        print(f"🔧 工具调用: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """工具返回结果时触发"""
        print(f"✅ 工具返回: {output[:50]}...")

    def on_llm_error(self, error, **kwargs):
        """LLM 出错时触发"""
        print(f"❌ LLM 错误: {error}")

    def on_chain_error(self, error, **kwargs):
        """Chain 出错时触发"""
        print(f"❌ Chain 错误: {error}")
```

---

## 三、使用 Callback

### 在 invoke 时传入

```python
callback = MyCallback()

# 单次调用传入
result = llm.invoke("你好", config={"callbacks": [callback]})

# Chain 调用传入
chain = prompt | llm | parser
result = chain.invoke(
    {"question": "什么是机器学习"},
    config={"callbacks": [callback]}
)
```

### 在构建时绑定

```python
# 给 LLM 绑定回调
llm_with_callback = llm.bind(callbacks=[MyCallback()])
chain = prompt | llm_with_callback | parser
```

### 全局设置

```python
from langchain.globals import set_llm_cache

# 全局回调
import langchain.callbacks as cb
with cb.collect_runs() as cb_ctx:
    result = chain.invoke({"question": "你好"})
    print(cb_ctx.traced_runs)
```

---

## 四、内置 Callback 处理器

### ConsoleCallbackHandler（终端输出）

```python
from langchain_core.callbacks import ConsoleCallbackHandler

result = llm.invoke("你好", config={"callbacks": [ConsoleCallbackHandler()]})
```

### StdOutCallbackHandler（详细日志）

```python
from langchain_core.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()
result = chain.invoke({"input": "你好"}, config={"callbacks": [handler]})
```

### FileCallbackHandler（输出到文件）

```python
import logging
from langchain_core.callbacks import FileCallbackHandler

logging.basicConfig(filename="langchain.log")
handler = FileCallbackHandler()
```

---

## 五、Streamlit 实时显示

```python
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

class StreamlitCallback(BaseCallbackHandler):
    def __init__(self):
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """流式输出时触发（实时显示每个 token）"""
        self.text += token
        self.container.markdown(self.text)

    def on_tool_start(self, serialized, input_str, **kwargs):
        st.info(f"🔧 调用工具: {serialized.get('name', 'unknown')}")

    def on_tool_end(self, output, **kwargs):
        st.success(f"✅ 工具返回: {output[:100]}")

# 使用
callback = StreamlitCallback()

st.title("AI 助手")
query = st.text_input("请输入问题")

if query:
    with st.spinner("思考中..."):
        result = llm.invoke(query, config={"callbacks": [callback]})
        st.success("完成！")
```

---

## 六、Callback 管理器

```python
from langchain_core.callbacks import CallbackManager

# 组合多个回调
manager = CallbackManager([
    MyCallback(),
    StdOutCallbackHandler()
])

result = llm.invoke("你好", config={"callbacks": manager})
```

---

## 七、常用事件一览

| 事件方法 | 触发时机 | 常见用途 |
|---------|---------|---------|
| `on_llm_start` | LLM 开始调用 | 记录输入、开始计时 |
| `on_llm_end` | LLM 返回结果 | 记录输出、计算耗时 |
| `on_llm_new_token` | 流式输出每个 token | 实时显示文本 |
| `on_llm_error` | LLM 出错 | 错误处理 |
| `on_chain_start` | Chain 开始 | 日志记录 |
| `on_chain_end` | Chain 完成 | 结果记录 |
| `on_tool_start` | 工具开始调用 | 显示工具使用 |
| `on_tool_end` | 工具返回结果 | 记录工具结果 |
| `on_tool_error` | 工具出错 | 错误处理 |
| `on_retriever_start` | 检索开始 | 日志 |
| `on_retriever_end` | 检索完成 | 显示检索结果 |

---

## 八、实用示例：Token 计数

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        # 简单估算（实际应使用模型的 tokenizer）
        for p in prompts:
            self.input_tokens += len(p.split())

    def on_llm_end(self, response, **kwargs):
        for gen in response.generations:
            for g in gen:
                self.output_tokens += len(g.text.split())

    def get_summary(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total": self.input_tokens + self.output_tokens
        }

# 使用
counter = TokenCounter()
result = llm.invoke("写一篇关于机器学习的短文", config={"callbacks": [counter]})
print(counter.get_summary())
```

---

**下一篇 → [08_完整实战项目](08_完整实战项目.md)**
