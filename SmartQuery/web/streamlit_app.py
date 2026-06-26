import json
import streamlit as st
import requests
import uuid

API_BASE = "http://localhost:8000"

# ------------------ session 初始化 ------------------ #

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# ------------------ 页面配置 ------------------ #

st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Agentic RAG - 专业知识智能问答系统")
st.caption("LangGraph Agentic RAG · 混合检索（稠密 + 稀疏）+ 文档评分 + 幻觉检查")

# ------------------ 双栏布局 ------------------ #

left_col, right_col = st.columns([1, 2])

# ================== 左栏：思考过程 + 来源引用 ================== #

with left_col:
    st.subheader("Think 思考过程")
    thinking_area = st.empty()

    st.divider()
    st.subheader("Stats 检索统计")
    stats_area = st.empty()

    st.divider()
    st.subheader("Trace 检索链路")
    st.caption("dense / sparse 排名 → RRF 融合 → Rerank")
    sources_area = st.empty()

    st.divider()
    st.subheader("Upload 文档上传")
    uploaded = st.file_uploader("选择文件", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    if uploaded:
        file_key = f"{uploaded.name}_{uploaded.size}"
        if file_key != st.session_state.last_uploaded:
            try:
                files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
                with st.spinner("正在上传并入库..."):
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
                if resp.ok:
                    st.success(resp.json()["message"])
                    st.session_state.last_uploaded = file_key
                else:
                    st.error(f"上传失败：{resp.status_code} {resp.text}")
            except requests.exceptions.Timeout:
                st.error("上传超时，文件可能较大，请重试")
            except requests.exceptions.ConnectionError:
                st.error("无法连接后端，请确认 python app.py 已启动")
            except Exception as e:
                st.error(f"上传出错：{e}")

    # 历史记录
    st.divider()
    st.subheader("History 历史记录")
    try:
        resp = requests.get(f"{API_BASE}/history/{st.session_state.session_id}", timeout=5)
        if resp.ok:
            records = resp.json()
            if records:
                for i, msg in enumerate(reversed(records[-20:])):
                    q_preview = msg['question'][:50]
                    with st.expander(f"Q: {q_preview}{'...' if len(msg['question']) > 50 else ''}", expanded=False):
                        st.markdown(f"**问：** {msg['question']}")
                        st.markdown(f"**答：** {msg['answer']}")
                        st.caption(f"时间：{msg.get('created_at', '')}")
            else:
                st.caption("暂无历史记录")
    except Exception:
        st.caption("历史记录加载失败")

# ================== 右栏：聊天区 ================== #

with right_col:
    # 渲染已有消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 聊天输入（SSE 流式消费）
    if prompt := st.chat_input("输入你的问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer_placeholder = st.chat_message("assistant").empty()
        thinking_log = []

        try:
            resp = requests.post(
                f"{API_BASE}/chat/stream",
                json={"question": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=120,
            )

            full_answer = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                # --- 思考步骤 ---
                if etype == "thinking":
                    info = event.get("info", "")
                    thinking_log.append(f"[OK] {info}")
                    thinking_area.markdown("\n\n".join(thinking_log))

                # --- 检索统计 ---
                elif etype == "stats":
                    stats_area.markdown(
                        f"稠密命中 **{event.get('dense_hits','?')}** → "
                        f"稀疏命中 **{event.get('sparse_hits','?')}** → "
                        f"RRF 融合 **{event.get('fused_count','?')}** → "
                        f"Rerank **{event.get('reranked_count','?')}**"
                    )

                # --- 检索来源（链路追踪卡片：dense#X + sparse#Y → RRF → rerank#Z） ---
                elif etype == "sources":
                    sources_list = event.get("sources", [])
                    if sources_list:
                        cards = []
                        for i, s in enumerate(sources_list):
                            d = s.get("dense_rank") or "-"
                            sr = s.get("sparse_rank") or "-"
                            rrf = s.get("rrf_score", 0)
                            rerank = s.get("rerank_score", 0)
                            text = s.get("text", "")[:160]
                            cards.append(
                                f"**{i+1}.** dense#{d} + sparse#{sr} → RRF `{rrf:.3f}` → rerank#{i+1} `{rerank:.3f}`\n"
                                f"> {text}..."
                            )
                        sources_area.markdown("\n\n---\n".join(cards))

                # --- 文档评分 ---
                elif etype == "grades":
                    grades = event.get("grades", [])
                    need = event.get("need_retrieve", False)
                    gs = []
                    for g in grades:
                        gs.append(f"{'[PASS]' if g.get('relevance',0)>=3 else '[DROP]'} 文档{g.get('doc_index','?')} 相关度{g.get('relevance','?')}/5：{g.get('reason','')}")
                    if need:
                        gs.append(" 相关文档不足，触发重检...")
                    sources_area.markdown("\n\n".join(gs))

                # --- 反思 ---
                elif etype == "reflection":
                    thinking_log.append(f"[Reflect] {event.get('issues', '无问题')}（完整度 {event.get('completeness_score', '?')}/5）")
                    thinking_area.markdown("\n\n".join(thinking_log))

                # --- Token 流式（打字机效果） ---
                # --- 最终答案（兼容非流式模式） ---
                elif etype == "answer":
                    full_answer = event.get("answer", "")
                    answer_placeholder.markdown(full_answer)

                # --- 完成（去掉光标，持久化消息） ---
                elif etype == "done":
                    answer_placeholder.markdown(full_answer if full_answer else "（未生成回答）")
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})

                # --- 错误 ---
                elif etype == "error":
                    thinking_log.append(f"[ERROR] {event.get('message', '')}")
                    thinking_area.markdown("\n\n".join(thinking_log))
                    answer_placeholder.error(event.get("message", "服务端错误"))

        except requests.exceptions.Timeout:
            thinking_log.append(" 请求超时")
            thinking_area.markdown("\n".join(thinking_log))
            answer_placeholder.error("请求超时，请重试")
        except requests.exceptions.ConnectionError:
            thinking_log.append(" 无法连接后端")
            thinking_area.markdown("\n".join(thinking_log))
            answer_placeholder.error("无法连接后端，请确认 python app.py 已启动")
        except Exception as e:
            thinking_log.append(f" 出错：{e}")
            thinking_area.markdown("\n".join(thinking_log))
            answer_placeholder.error(f"请求出错：{e}")
