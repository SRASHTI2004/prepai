import time
import json
from datetime import datetime
from app.query import query
from app.retrieval import hybrid_search
from app.llm import get_eval_llm


GOLDEN_DATASET = [
    {
        "question": "What is system design?",
        "ground_truth": "System design is the process of defining architecture components and data flow for a system to satisfy requirements"
    },
    {
        "question": "What is the difference between SQL and NoSQL?",
        "ground_truth": "SQL databases are relational and use structured schemas while NoSQL databases are non-relational and flexible"
    },
    {
        "question": "What is dynamic programming?",
        "ground_truth": "Dynamic programming is an optimization technique that solves complex problems by breaking them into overlapping subproblems"
    },
    {
        "question": "What is a vector database?",
        "ground_truth": "A vector database stores high dimensional embeddings and enables similarity search using distance metrics like cosine similarity"
    },
    {
        "question": "What is RAG?",
        "ground_truth": "RAG stands for Retrieval Augmented Generation which combines retrieval systems with LLMs to answer questions from documents"
    },
]


def score_answer(question: str, answer: str, 
                 context: str, ground_truth: str) -> dict:
    llm = get_eval_llm()

    prompt = f"""You are evaluating a RAG system. Score the answer on these 2 metrics.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context: {context[:500]}
Generated Answer: {answer}

Score each metric from 0.0 to 1.0:

1. Faithfulness — Is the answer grounded in the retrieved context?
   1.0 = completely grounded, 0.0 = completely hallucinated

2. Answer Relevancy — Does the answer actually answer the question?
   1.0 = perfectly answers question, 0.0 = completely off topic

Respond in this EXACT format, nothing else:
faithfulness: <score>
answer_relevancy: <score>"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    try:
        lines = text.split("\n")
        faith = float(lines[0].split(":")[1].strip())
        rel = float(lines[1].split(":")[1].strip())
        return {
            "faithfulness": round(faith, 3),
            "answer_relevancy": round(rel, 3)
        }
    except Exception:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0
        }


def run_eval() -> dict:
    print("\n" + "="*50)
    print("RUNNING EVALUATION")
    print("="*50)

    all_faith = []
    all_rel = []
    detailed = []

    for i, item in enumerate(GOLDEN_DATASET):
        print(f"\nQ{i+1}: {item['question']}")

        # Get answer
        result = query(item["question"])
        answer = result["answer"]

        # Get context
        chunks, _ = hybrid_search(item["question"])
        context = " ".join([
            c.payload["content"] for c in chunks
        ])

        # Score
        scores = score_answer(
            item["question"],
            answer,
            context,
            item["ground_truth"]
        )

        print(f"  Faithfulness:     {scores['faithfulness']}")
        print(f"  Answer Relevancy: {scores['answer_relevancy']}")

        all_faith.append(scores["faithfulness"])
        all_rel.append(scores["answer_relevancy"])

        detailed.append({
            "question": item["question"],
            "answer": answer,
            "scores": scores
        })

        time.sleep(1)

    # Averages
    avg_faith = round(sum(all_faith) / len(all_faith), 3)
    avg_rel = round(sum(all_rel) / len(all_rel), 3)

    print("\n" + "="*50)
    print("FINAL SCORES")
    print("="*50)
    print(f"Avg Faithfulness:     {avg_faith}")
    print(f"Avg Answer Relevancy: {avg_rel}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "faithfulness": avg_faith,
        "answer_relevancy": avg_rel,
        "num_questions": len(GOLDEN_DATASET),
        "detailed": detailed
    }

    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved to eval_results.json")
    return output


if __name__ == "__main__":
    run_eval()