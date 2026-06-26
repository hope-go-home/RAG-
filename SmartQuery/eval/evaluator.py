"""RAG 系统自评估：检索命中、精确率、召回率、F1"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from SmartQuery.rag.agent import app as rag_agent
from SmartQuery.backend.database.milvus import connect_milvus

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.json")


def run_evaluation():
    connect_milvus()
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total_precision = 0
    total_recall = 0
    passed = 0

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        ground_truth = case["ground_truth"]
        expected = case.get("expected_docs", [])
        keywords = case.get("relevant_keywords", [])
        total_relevant = case.get("total_relevant", 5)

        result = rag_agent.invoke({"question": question})
        answer = result.get("answer", "")
        context = result.get("context", [])
        grades = result.get("document_grades", [])
        sources = result.get("retrieval_sources", [])

        # --- 检索命中检测 ---
        found_docs = []
        for doc in expected:
            for c in context:
                if doc[:15] in c:
                    found_docs.append(doc)
                    break

        # --- 精确率 / 召回率 ---
        # 精确率 = 检索结果中包含相关关键词的文档数 / 检索结果总数
        # 召回率 = 检索结果中包含相关关键词的文档数 / 知识库中预计相关文档数
        relevant_retrieved = 0
        for c in context:
            if any(kw in c for kw in keywords):
                relevant_retrieved += 1

        precision = relevant_retrieved / len(context) if len(context) > 0 else 0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_precision += precision
        total_recall += recall

        doc_ok = len(found_docs) == len(expected) if expected else bool(context)
        has_answer = bool(answer.strip())
        ok = has_answer and doc_ok

        if ok:
            passed += 1

        print(f"\n{'='*60}")
        print(f"第 {i} 题: {question}")
        print(f"  标准答案: {ground_truth[:80]}...")
        print(f"  系统答案: {answer[:120]}...")
        print(f"  检索到 {len(context)} 篇 | 相关 {relevant_retrieved} 篇 | 预计命中 {len(found_docs)}/{len(expected)}")
        print(f"  精确率: {precision:.2f} | 召回率: {recall:.2f} | F1: {f1:.2f}")
        print(f"  评分通过: {len([g for g in grades if g.get('relevance',0) >= 3])}/{len(grades)}")
        if sources:
            td = sources[0].get("dense_rank", "-")
            ts = sources[0].get("sparse_rank", "-")
            tr = sources[0].get("rerank_score", 0)
            print(f"  Top1: dense#{td} + sparse#{ts} -> rerank {tr:.3f}")
        print(f"  命中: {'YES' if ok else 'NO'}")

    avg_p = total_precision / len(test_cases)
    avg_r = total_recall / len(test_cases)
    avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0

    print(f"\n{'='*60}")
    print(f"检索命中率: {passed}/{len(test_cases)}")
    print(f"平均精确率: {avg_p:.2f}")
    print(f"平均召回率: {avg_r:.2f}")
    print(f"平均 F1:     {avg_f1:.2f}")
    return {"passed": passed, "total": len(test_cases), "precision": avg_p, "recall": avg_r, "f1": avg_f1}


if __name__ == "__main__":
    run_evaluation()
