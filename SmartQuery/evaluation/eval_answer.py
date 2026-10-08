"""
RAG 生成质量 LLM-as-Judge 离线评估脚本（WixQA 版）
==================================================
基于 golden_wixqa.json（200 题，每题带 reference 标准答案 + gold_sources），
对 RAG 系统的「生成侧」质量做端到端评估：

  1. 答案正确率   correctness：裁判对 系统回答 vs 参考答案 给 correct/partial/incorrect
  2. 忠实度       faithfulness：裁判基于【检索到的上下文】判断回答是否都有依据（0~1）
  3. 幻觉率       hallucination：含 ≥1 条无上下文支持的声明 的答案占比
  4. 语义相似度   semantic_sim：系统回答与参考答案的 embedding 余弦相似度

注意：忠实度/幻觉必须用「检索到的上下文」判定，不能用 reference（那是标准答案，
不是系统实际看到的资料）。这是旧版脚本最大的方法论错误，本版已修正。

用法：
  python SmartQuery/evaluation/eval_answer.py                 # 全量
  python SmartQuery/evaluation/eval_answer.py --limit 50      # 抽样 50
  python SmartQuery/evaluation/eval_answer.py --limit 5 --verbose

输出：evaluation/report/answer_eval.json 与 answer_eval.md
"""

import argparse
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SmartQuery.backend.config import (
    JUDGE_API_KEY,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
)
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVAL_DIR / "golden_wixqa.json"
REPORT_DIR = EVAL_DIR / "report"

# 上下文超长时的安全上限（必须 ≥ 生成时实际喂给模型的上下文，否则裁判看到的资料
# 比生成时少，会把有依据的陈述误判为幻觉）。10 个父块一般 < 3 万字符。
MAX_CONTEXT_CHARS = 48000
# 忠实度低于该阈值视为「该答案存在幻觉风险」
FAITHFULNESS_FLOOR = 0.9


# ---------------- 裁判模型 ----------------

judge_llm = ChatOpenAI(
    model=JUDGE_MODEL,
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
    temperature=0,
)


CORRECTNESS_PROMPT = """你是严格的 RAG 评估专家。请比较「系统回答」与「参考答案」，判断系统回答是否正确。

用户问题：{question}

参考答案：
{reference}

系统回答：
{answer}

判定标准：
- correct：核心事实与参考答案一致，能正确解决用户问题（表述不同不影响）。
- partial：部分正确，但遗漏了关键信息或存在小错误。
- incorrect：答错、答非所问、或声称无法回答但参考答案给出了答案。

附加评分：给 1-5 的总体正确性分数。

仅输出 JSON（不要 markdown 包裹）：
{{"verdict": "correct|partial|incorrect", "score": N, "reason": "..."}}"""


FAITHFULNESS_PROMPT = """你是严格的 RAG 忠实度评估专家。下面是系统检索到的【资料】和它给出的【回答】。
请判断回答中的每一处事实性陈述是否都能在资料中找到依据。

用户问题：{question}

资料：
{context}

回答：
{answer}

请：
1. 找出回答中【无法被资料支持】的陈述（幻觉），逐条列出；若全部有依据则为空列表。
2. 给出忠实度分数 0.0-1.0：回答中受资料支持的内容占比（1.0=完全有依据，0.0=完全编造）。

仅输出 JSON（不要 markdown 包裹）：
{{"faithfulness": 0.0, "unsupported_claims": ["..."], "reason": "..."}}"""


def _safe_json_parse(raw: str, default: dict | None = None) -> dict:
    if default is None:
        default = {}
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return default


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
def call_judge(prompt: str) -> dict:
    resp = judge_llm.invoke([("human", prompt)])
    return _safe_json_parse(resp.content, {})


# ---------------- 余弦相似度 ----------------

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_similarity(text_a: str, text_b: str) -> float:
    try:
        from SmartQuery.rag.embedding import embed_query
        return _cosine(embed_query(text_a), embed_query(text_b))
    except Exception:
        return 0.0


# ---------------- RAG Agent 调用 ----------------

def query_rag_agent(question: str, chat_history: list[dict] | None = None) -> dict:
    """调用 Agent，返回 {answer, context, sources, intent}"""
    from SmartQuery.rag.agent import app
    result = app.invoke({
        "question": question,
        "search_query": "",
        "context": [],
        "answer": "",
        "intent": "",
        "chat_history": chat_history or [],
        "thinking_steps": [],
        "retrieval_sources": [],
        "retrieval_stats": {},
    })
    return {
        "answer": result.get("answer", ""),
        "context": result.get("context", []) or [],
        "sources": result.get("retrieval_sources", []) or [],
        "intent": result.get("intent", ""),
    }


# ---------------- 数据加载 ----------------

def load_golden() -> list[dict]:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def load_checkpoint(path: Path) -> dict[str, dict]:
    """读取断点续跑缓存（JSONL，每题一行），返回 {question: 结果记录}。"""
    done: dict[str, dict] = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("question"):
                done[rec["question"]] = rec
    return done


# ---------------- 单题评估 ----------------

def evaluate_one(item: dict) -> dict:
    question = item["question"]
    reference = item.get("reference", "")

    gen = query_rag_agent(question)
    answer = gen["answer"]
    context_text = "\n\n".join(gen["context"])
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    # 正确性
    cor = call_judge(CORRECTNESS_PROMPT.format(
        question=question, reference=reference, answer=answer))
    verdict = cor.get("verdict", "incorrect")
    if verdict not in ("correct", "partial", "incorrect"):
        verdict = "incorrect"
    try:
        cor_score = max(1, min(5, int(cor.get("score", 1))))
    except (TypeError, ValueError):
        cor_score = 1

    # 忠实度（无上下文时跳过，记为 None）
    if context_text.strip():
        fth = call_judge(FAITHFULNESS_PROMPT.format(
            question=question, context=context_text, answer=answer))
        try:
            faith = max(0.0, min(1.0, float(fth.get("faithfulness", 0.0))))
        except (TypeError, ValueError):
            faith = 0.0
        unsupported = fth.get("unsupported_claims", []) or []
        if not isinstance(unsupported, list):
            unsupported = [str(unsupported)]
    else:
        faith = None
        unsupported = []

    sim = semantic_similarity(answer, reference)

    return {
        "question": question,
        "doc_type": item.get("doc_type", ""),
        "reference": reference,
        "answer": answer,
        "intent": gen["intent"],
        "n_context": len(gen["context"]),
        "context_chars": len(context_text),
        "verdict": verdict,
        "correctness_score": cor_score,
        "correctness_reason": cor.get("reason", ""),
        "faithfulness": faith,
        "unsupported_claims": unsupported,
        "has_hallucination": bool(unsupported),
        "semantic_sim": round(sim, 4),
    }


# ---------------- 报告 ----------------

def _avg(values: list) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def generate_report(rows: list[dict]) -> str:
    n = len(rows)
    acc = sum(1 for r in rows if r["verdict"] == "correct") / n if n else 0
    partial = sum(1 for r in rows if r["verdict"] == "partial") / n if n else 0
    wrong = sum(1 for r in rows if r["verdict"] == "incorrect") / n if n else 0
    avg_cor = _avg([r["correctness_score"] for r in rows])
    avg_faith = _avg([r["faithfulness"] for r in rows])
    halluc = sum(1 for r in rows if r["has_hallucination"]) / n if n else 0
    avg_sim = _avg([r["semantic_sim"] for r in rows])

    lines = [
        "# RAG 生成质量评估报告（WixQA · LLM-as-Judge）",
        "",
        f"- 评估问题数：{n}",
        f"- 裁判模型：{JUDGE_MODEL}",
        "- 生成模型：见 QWEN_MODEL 配置",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 答案正确率 (correct) | {acc:.1%} |",
        f"| 部分正确 (partial) | {partial:.1%} |",
        f"| 错误 (incorrect) | {wrong:.1%} |",
        f"| 正确率 (correct+partial) | {acc + partial:.1%} |",
        f"| 平均正确性分 (1-5) | {avg_cor:.2f} |",
        f"| 平均忠实度 (0-1) | {avg_faith:.2f} |",
        f"| 幻觉率 (含无支持声明) | {halluc:.1%} |",
        f"| 平均语义相似度 (vs 参考答案) | {avg_sim:.3f} |",
        "",
        "## 按文档类型分组",
        "",
        "| doc_type | 题数 | 正确率 | 平均正确分 | 平均忠实度 | 幻觉率 |",
        "|----------|------|--------|-----------|-----------|--------|",
    ]
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["doc_type"] or "-", []).append(r)
    for dt, rs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        m = len(rs)
        a = sum(1 for r in rs if r["verdict"] == "correct") / m
        h = sum(1 for r in rs if r["has_hallucination"]) / m
        lines.append(
            f"| {dt} | {m} | {a:.1%} | {_avg([r['correctness_score'] for r in rs]):.2f} "
            f"| {_avg([r['faithfulness'] for r in rs]):.2f} | {h:.1%} |"
        )

    lines += [
        "",
        "## 逐题明细",
        "",
        "| # | doc_type | 问题 | 判定 | 正确分 | 忠实度 | 幻觉 | 相似度 |",
        "|---|----------|------|------|--------|--------|------|--------|",
    ]
    for i, r in enumerate(rows, 1):
        faith = "-" if r["faithfulness"] is None else f"{r['faithfulness']:.2f}"
        q = r["question"][:36] + ("..." if len(r["question"]) > 36 else "")
        lines.append(
            f"| {i} | {r['doc_type']} | {q} | {r['verdict']} | {r['correctness_score']} "
            f"| {faith} | {'Y' if r['has_hallucination'] else ''} | {r['semantic_sim']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser(description="RAG 生成质量 LLM-as-Judge 评估（WixQA）")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 个问题（0=全部）")
    parser.add_argument("--workers", type=int, default=4, help="并发题数（LLM 网络等待可并行）")
    parser.add_argument("--checkpoint", default=str(REPORT_DIR / "answer_eval.checkpoint.jsonl"),
                        help="断点续跑缓存（JSONL）；已完成的题自动跳过")
    parser.add_argument("--no-resume", action="store_true", help="忽略缓存、全部重跑")
    parser.add_argument("--dry-run", action="store_true", help="只打印待跑题数，不调用模型")
    parser.add_argument("--verbose", action="store_true", help="逐题打印分数")
    args = parser.parse_args()

    from SmartQuery.backend.database.milvus import connect_milvus
    connect_milvus()

    golden = load_golden()
    if args.limit:
        golden = golden[: args.limit]
    if not golden:
        print("没有可评估的问题，退出")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    index_of = {it["question"]: i for i, it in enumerate(golden)}
    ckpt_path = Path(args.checkpoint)
    done_map = {} if args.no_resume else load_checkpoint(ckpt_path)
    todo = [it for it in golden if it["question"] not in done_map]

    print(f"裁判模型：{JUDGE_MODEL} | 总题 {len(golden)} | 缓存已完成 {len(done_map)} | "
          f"待跑 {len(todo)} | 并发 {args.workers}")
    if args.dry_run:
        print("[dry-run] 仅统计，不调用模型。去掉 --dry-run 即开始补跑。")
        return
    if not todo:
        print("所有题目均已完成（命中缓存），直接生成报告。")
    print("[1/2] 逐题生成 + 裁判评分 ...")
    t0 = time.time()

    new_rows: list[dict] = []
    print_lock = threading.Lock()
    ckpt_lock = threading.Lock()
    total_todo = len(todo)

    def _save(row: dict) -> None:
        try:
            with ckpt_lock:
                with open(ckpt_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _record(done: int, row: dict) -> None:
        new_rows.append(row)
        _save(row)
        q = row["question"][:50] + ("..." if len(row["question"]) > 50 else "")
        faith = "-" if row["faithfulness"] is None else f"{row['faithfulness']:.2f}"
        with print_lock:
            print(f"  [{done}/{total_todo}] {row['verdict']:9s} 正确分={row['correctness_score']} "
                  f"忠实度={faith} 相似度={row['semantic_sim']:.3f} | {q}")

    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(evaluate_one, it): it["question"] for it in todo}
            done = 0
            for fut in as_completed(futures):
                done += 1
                q = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  [warn] 评估失败：{q[:40]} — {e}")
                    continue
                _record(done, row)
    else:
        for it in todo:
            print(f"  {it['question'][:60]}")
            try:
                row = evaluate_one(it)
            except Exception as e:
                print(f"    [warn] 评估失败：{e}")
                continue
            _record(len(new_rows), row)

    rows = list(done_map.values()) + new_rows
    rows.sort(key=lambda r: index_of.get(r["question"], 10 ** 9))

    elapsed = time.time() - t0
    print(f"[2/2] 生成报告 ... ({elapsed:.1f}s)")

    n = len(rows)
    summary = {
        "question_count": n,
        "judge_model": JUDGE_MODEL,
        "accuracy": round(sum(1 for r in rows if r["verdict"] == "correct") / n, 4) if n else 0,
        "correct_or_partial": round(
            sum(1 for r in rows if r["verdict"] in ("correct", "partial")) / n, 4) if n else 0,
        "avg_correctness_score": round(_avg([r["correctness_score"] for r in rows]), 4),
        "avg_faithfulness": round(_avg([r["faithfulness"] for r in rows]), 4),
        "hallucination_rate": round(
            sum(1 for r in rows if r["has_hallucination"]) / n, 4) if n else 0,
        "avg_semantic_sim": round(_avg([r["semantic_sim"] for r in rows]), 4),
        "elapsed_seconds": round(elapsed, 1),
    }

    (REPORT_DIR / "answer_eval.json").write_text(
        json.dumps({"summary": summary, "per_question": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (REPORT_DIR / "answer_eval.md").write_text(generate_report(rows), encoding="utf-8")

    print("\n===== 生成质量评估结果 =====")
    print(f"正确率 {summary['accuracy']:.1%}  "
          f"(+部分正确 {summary['correct_or_partial']:.1%})  "
          f"平均忠实度 {summary['avg_faithfulness']:.2f}  "
          f"幻觉率 {summary['hallucination_rate']:.1%}  "
          f"相似度 {summary['avg_semantic_sim']:.3f}")
    print(f"报告已生成：{REPORT_DIR / 'answer_eval.md'}")


if __name__ == "__main__":
    main()
