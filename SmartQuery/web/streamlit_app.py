import json
import streamlit as st
import requests
import time
import uuid

API_BASE = "http://localhost:8000"

# ------------------ session 初始化 ------------------ #

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# SSE 流式期间对 st.empty() 占位符的更新节流（避免 Streamlit 高频渲染 DOM 竞态 bug）
_last_render = {}


def _should_render(key: str, interval: float = 0.3) -> bool:
    """同一占位符在 interval 秒内最多更新一次"""
    now = time.time()
    if now - _last_render.get(key, 0) >= interval:
        _last_render[key] = now
        return True
    return False

# ------------------ 页面配置 ------------------ #

st.set_page_config(page_title="RAG 智能问答", layout="wide")
st.title("RAG - 专业知识智能问答系统")
st.caption("LangGraph · 混合检索（稠密 + 稀疏）+ Rerank 重排序")

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
    uploaded = st.file_uploader("选择文件（可多选）", type=["pdf", "docx", "txt", "md"],
                                accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        keys = {f"{f.name}_{f.size}" for f in uploaded}
        if keys != st.session_state.last_uploaded:
            try:
                files = [("files", (f.name, f.read(), f.type)) for f in uploaded]
                with st.spinner("正在上传并入库..."):
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=300)
                if resp.ok:
                    data = resp.json()
                    for r in data.get("results", []):
                        if r["status"] == "success":
                            st.success(f"{r['file']}：{r['message']}")
                        elif r["status"] == "duplicate":
                            st.warning(f"{r['file']}：{r['message']}")
                        else:
                            st.error(f"{r['file']}：{r['message']}")
                    st.session_state.last_uploaded = keys
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
                for msg in reversed(records[-20:]):
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
                # --- 思考步骤（节流渲染） ---
                if etype == "thinking":
                    info = event.get("info", "")
                    thinking_log.append(f"[OK] {info}")
                    if _should_render("thinking"):
                        thinking_area.markdown("\n\n".join(thinking_log))

                # --- 检索统计（节流渲染） ---
                elif etype == "stats":
                    if _should_render("stats"):
                        stats_area.markdown(
                            f"稠密命中 **{event.get('dense_hits','?')}** → "
                            f"稀疏命中 **{event.get('sparse_hits','?')}** → "
                            f"RRF 融合 **{event.get('fused_count','?')}** → "
                            f"Rerank **{event.get('reranked_count','?')}**"
                        )

                # --- 检索来源（链路追踪卡片：dense#X + sparse#Y → RRF → rerank#Z，节流渲染） ---
                elif etype == "sources":
                    sources_list = event.get("sources", [])
                    if sources_list and _should_render("sources"):
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

                # --- token 级流式：逐步拼接显示 ---
                elif etype == "token":
                    full_answer += event.get("content", "")
                    if _should_render("answer_stream", interval=0.1):
                        answer_placeholder.markdown(full_answer + " ")

                # --- 最终答案（非流式兼容：直接展示完整答案） ---
                elif etype == "answer":
                    full_answer = event.get("answer", "") or full_answer
                    answer_placeholder.markdown(full_answer)

                # --- 完成（去掉光标，持久化消息；强制渲染被节流跳过的最后状态） ---
                elif etype == "done":
                    thinking_area.markdown("\n\n".join(thinking_log))
                    final_answer = full_answer if full_answer and full_answer.strip() else "抱歉，无法生成回答。请尝试换一种方式提问。"
                    answer_placeholder.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

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
