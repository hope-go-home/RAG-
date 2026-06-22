import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from SmartQuery.rag.agent import app as rag_agent

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.json")


def run_evaluation():
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in test_cases:
        print(f"评估中: {case['question']}")
        result = rag_agent.invoke({"question": case["question"]})
        questions.append(case["question"])
        answers.append(result["answer"])
        contexts.append(result.get("context", []))
        ground_truths.append(case["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    print("\n========== 评估结果 ==========")
    print(f"Faithfulness（忠实性）:       {score['faithfulness']:.4f}")
    print(f"Answer Relevancy（回答相关性）: {score['answer_relevancy']:.4f}")
    print(f"Context Recall（上下文召回率）: {score['context_recall']:.4f}")
    print(f"Context Precision（上下文精确率）: {score['context_precision']:.4f}")
    print("================================")

    return score


if __name__ == "__main__":
    run_evaluation()
