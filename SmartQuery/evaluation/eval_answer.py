"""
RAG 回答质量 LLM-as-Judge 离线评估脚本
======================================
基于 golden_set.json 标注集，对 RAG 系统的回答质量进行多维度评估：

  1. 忠实度 (Faithfulness)  1-5 分：回答是否基于检索到的文档内容，有无幻觉
  2. 相关性 (Relevance)     1-5 分：回答是否针对用户问题
  3. 完整性 (Completeness)  1-5 分：回答是否覆盖了关键信息

同时计算短语匹配分数：gold_phrases 中有多少出现在回答中。

用法：
  python SmartQuery/evaluation/eval_answer.py
  python SmartQuery/evaluation/eval_answer.py --limit 5
  python SmartQuery/evaluation/eval_answer.py --verbose

输出：evaluation/report/answer_eval.md 和 answer_eval.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SmartQuery.backend.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = EVAL_DIR / "golden_set.json"
REPORT_DIR = EVAL_DIR / "report"

JUDGE_PROMPT = """你是一个严格的RAG系统评估专家。请对以下问答对进行评分。

用户问题：{question}
参考答案：{reference}
系统回答：{answer}

请从以下三个维度评分（1-5分）：
1. 忠实度：回答是否基于检索到的文档内容，有无幻觉
2. 相关性：回答是否针对用户问题
3. 完整性：回答是否覆盖了关键信息

输出JSON格式：
{{"faithfulness": N, "relevance": N, "completeness": N, "reason": "..."}}"""


# ---------------- LLM 评审 ----------------

judge_llm = ChatOpenAI(
    model=QWEN_MODEL,
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    temperature=0,
)


def _safe_json_parse(raw: str, default: dict | None = None) -> dict:
    if default is None:
        default = {}
    text = raw.strip()
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
def call_judge_llm(question: str, reference: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    resp = judge_llm.invoke([("human", prompt)])
    return _safe_json_parse(resp.content, {})


# ---------------- 短语匹配 ----------------

def phrase_match_score(answer: str, gold_phrases: list[str]) -> tuple[float, list[str]]:
    if not gold_phrases:
        return 1.0, []
    matched = [p for p in gold_phrases if p.lower() in answer.lower()]
    return len(matched) / len(gold_phrases), matched


# ---------------- RAG Agent 调用 ----------------

def query_rag_agent(question: str) -> str:
    from SmartQuery.rag.agent import app
    result = app.invoke({
        "question": question,
        "search_query": "",
        "context": [],
        "answer": "",
        "intent": "",
        "chat_history": [],
        "thinking_steps": [],
        "retrieval_sources": [],
        "retrieval_stats": {},
    })
    return result.get("answer", "")


# ---------------- 数据加载 ----------------

def load_golden() -> list[dict]:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


# ---------------- 报告生成 ----------------

def _score_distribution(values: list[int]) -> dict[int, int]:
    dist = {i: 0 for i in range(1, 6)}
    for v in values:
        if v in dist:
            dist[v] += 1
    return dist


def generate_report(
    per_question: list[dict],
    all_faithfulness: list[int],
    all_relevance: list[int],
    all_completeness: list[int],
    all_phrase: list[float],
    level_stats: dict,
) -> str:
    n = len(per_question)
    avg_f = sum(all_faithfulness) / n if n else 0
    avg_r = sum(all_relevance) / n if n else 0
    avg_c = sum(all_completeness) / n if n else 0
    avg_p = sum(all_phrase) / n if n else 0

    dist_f = _score_distribution(all_faithfulness)
    dist_r = _score_distribution(all_relevance)
    dist_c = _score_distribution(all_completeness)

    lines = [
        "# RAG 回答质量评估报告（LLM-as-Judge）",
        "",
        f"- 评估问题数：{n}",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 平均分",
        "",
        "| 维度 | 平均分 |",
        "|------|--------|",
        f"| 忠实度 | {avg_f:.2f} |",
        f"| 相关性 | {avg_r:.2f} |",
        f"| 完整性 | {avg_c:.2f} |",
        f"| 短语匹配 | {avg_p:.2%} |",
        "",
        "## 分数分布",
        "",
        "| 分数 | 忠实度 | 相关性 | 完整性 |",
        "|------|--------|--------|--------|",
    ]
    for score in range(1, 6):
        lines.append(f"| {score} | {dist_f[score]} | {dist_r[score]} | {dist_c[score]} |")

    lines += [
        "",
        "## 按问题层级分组",
        "",
        "| 层级 | 题数 | 忠实度 | 相关性 | 完整性 |",
        "|------|------|--------|--------|--------|",
    ]
    for lv, st in level_stats.items():
        cnt = st["count"]
        lines.append(
            f"| {lv} | {cnt} "
            f"| {sum(st['faithfulness']) / cnt:.2f} "
            f"| {sum(st['relevance']) / cnt:.2f} "
            f"| {sum(st['completeness']) / cnt:.2f} |"
        )

    lines += [
        "",
        "## 逐问题明细",
        "",
        "| # | 层级 | 问题 | 忠实度 | 相关性 | 完整性 | 短语匹配 |",
        "|---|------|------|--------|--------|--------|----------|",
    ]
    for i, row in enumerate(per_question, 1):
        lines.append(
            f"| {i} | {row.get('level', '')} "
            f"| {row['question'][:40]}{'...' if len(row['question']) > 40 else ''} "
            f"| {row['faithfulness']} | {row['relevance']} | {row['completeness']} "
            f"| {row['phrase_score']:.0%} ({row['matched_count']}/{row['gold_count']}) |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser(description="RAG 回答质量 LLM-as-Judge 评估")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 个问题（0=全部）")
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

    per_question = []
    all_faithfulness: list[int] = []
    all_relevance: list[int] = []
    all_completeness: list[int] = []
    all_phrase: list[float] = []
    level_stats: dict[str, dict] = {}

    total = len(golden)
    print(f"[1/2] 评估 {total} 个问题 ...")
    t0 = time.time()

    for i, item in enumerate(golden, 1):
        question = item["question"]
        reference = item.get("reference", "、".join(item.get("gold_phrases", [])))
        gold_phrases = item.get("gold_phrases", [])
        level = item.get("level", "")

        print(f"  [{i}/{total}] {question[:60]}")

        # 调用 RAG Agent 获取回答
        answer = query_rag_agent(question)

        # LLM 评审打分
        scores = call_judge_llm(question, reference, answer)
        faithfulness = max(1, min(5, int(scores.get("faithfulness", 3))))
        relevance = max(1, min(5, int(scores.get("relevance", 3))))
        completeness = max(1, min(5, int(scores.get("completeness", 3))))
        reason = scores.get("reason", "")

        # 短语匹配
        phrase_score, matched = phrase_match_score(answer, gold_phrases)

        row = {
            "question": question,
            "reference": reference,
            "answer": answer,
            "level": level,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "reason": reason,
            "phrase_score": phrase_score,
            "matched_phrases": matched,
            "matched_count": len(matched),
            "gold_count": len(gold_phrases),
        }
        per_question.append(row)

        all_faithfulness.append(faithfulness)
        all_relevance.append(relevance)
        all_completeness.append(completeness)
        all_phrase.append(phrase_score)

        if level not in level_stats:
            level_stats[level] = {"count": 0, "faithfulness": [], "relevance": [], "completeness": []}
        level_stats[level]["count"] += 1
        level_stats[level]["faithfulness"].append(faithfulness)
        level_stats[level]["relevance"].append(relevance)
        level_stats[level]["completeness"].append(completeness)

        if args.verbose:
            print(f"    忠实度={faithfulness} 相关性={relevance} 完整性={completeness} "
                  f"短语匹配={phrase_score:.0%} | {reason}")

    elapsed = time.time() - t0
    print(f"[2/2] 生成报告 ... ({elapsed:.1f}s)")

    # 写 JSON
    json_path = REPORT_DIR / "answer_eval.json"
    json_data = {
        "summary": {
            "faithfulness_avg": round(sum(all_faithfulness) / len(all_faithfulness), 2),
            "relevance_avg": round(sum(all_relevance) / len(all_relevance), 2),
            "completeness_avg": round(sum(all_completeness) / len(all_completeness), 2),
            "phrase_match_avg": round(sum(all_phrase) / len(all_phrase), 4),
            "faithfulness_dist": _score_distribution(all_faithfulness),
            "relevance_dist": _score_distribution(all_relevance),
            "completeness_dist": _score_distribution(all_completeness),
        },
        "level_stats": {
            lv: {
                "count": st["count"],
                "faithfulness_avg": round(sum(st["faithfulness"]) / st["count"], 2),
                "relevance_avg": round(sum(st["relevance"]) / st["count"], 2),
                "completeness_avg": round(sum(st["completeness"]) / st["count"], 2),
            }
            for lv, st in level_stats.items()
        },
        "per_question": per_question,
        "question_count": len(per_question),
        "elapsed_seconds": round(elapsed, 1),
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 写 Markdown
    md_path = REPORT_DIR / "answer_eval.md"
    md_path.write_text(
        generate_report(per_question, all_faithfulness, all_relevance, all_completeness,
                        all_phrase, level_stats),
        encoding="utf-8",
    )

    # 打印摘要
    n = len(per_question)
    print("\n===== 回答质量评估结果 =====")
    print(f"忠实度: {sum(all_faithfulness) / n:.2f}  "
          f"相关性: {sum(all_relevance) / n:.2f}  "
          f"完整性: {sum(all_completeness) / n:.2f}  "
          f"短语匹配: {sum(all_phrase) / n:.0%}")
    print(f"报告已生成：{md_path}")


if __name__ == "__main__":
    main()
