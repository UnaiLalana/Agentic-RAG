#!/usr/bin/env python3
"""
Evaluation script for the RAG system.

Computes retrieval quality metrics:
- Hit Rate @k: fraction of queries where the relevant passage is in top-k
- Mean Reciprocal Rank (MRR): average of 1/rank of first relevant result
- Precision @k: fraction of retrieved chunks that are relevant

Usage:
    python evaluate.py [--api-url http://localhost:5000] [--output results/eval_results.json]
"""

import json
import argparse
import os
import sys
from datetime import datetime
import requests


def compute_hit_rate(results: list, k: int) -> float:
    """Fraction of queries where at least one relevant chunk is in top-k."""
    hits = 0
    for r in results:
        retrieved_docs = [s["filename"] for s in r["sources"][:k]]
        if r["source_doc"] in retrieved_docs:
            hits += 1
    return hits / len(results) if results else 0.0


def compute_mrr(results: list, k: int) -> float:
    """Mean Reciprocal Rank: average 1/rank of first relevant result."""
    reciprocal_ranks = []
    for r in results:
        retrieved_docs = [s["filename"] for s in r["sources"][:k]]
        try:
            rank = retrieved_docs.index(r["source_doc"]) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_precision_at_k(results: list, k: int) -> float:
    """Average fraction of top-k results that are relevant."""
    precisions = []
    for r in results:
        retrieved_docs = [s["filename"] for s in r["sources"][:k]]
        relevant = sum(1 for d in retrieved_docs if d == r["source_doc"])
        precisions.append(relevant / k)
    return sum(precisions) / len(precisions) if precisions else 0.0


def run_evaluation(api_url: str, dataset_path: str, output_path: str):
    """Run the full evaluation pipeline."""
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"📊 Loaded {len(dataset)} evaluation questions")
    print(f"🔗 API URL: {api_url}")
    print()

    # Run queries
    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        print(f"  [{i}/{len(dataset)}] {question[:60]}...")

        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"question": question, "top_k": 5},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()

            results.append({
                "id": item["id"],
                "question": question,
                "expected_answer": item["expected_answer"],
                "source_doc": item["source_doc"],
                "source_passage": item.get("source_passage", ""),
                "generated_answer": data.get("answer", ""),
                "sources": data.get("sources", []),
            })
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "id": item["id"],
                "question": question,
                "expected_answer": item["expected_answer"],
                "source_doc": item["source_doc"],
                "source_passage": item.get("source_passage", ""),
                "generated_answer": f"ERROR: {str(e)}",
                "sources": [],
            })

    # Compute metrics for k ∈ {1, 3, 5}
    metrics = {}
    for k in [1, 3, 5]:
        metrics[f"k={k}"] = {
            "hit_rate": round(compute_hit_rate(results, k), 4),
            "mrr": round(compute_mrr(results, k), 4),
            "precision_at_k": round(compute_precision_at_k(results, k), 4),
        }

    # Print results
    print("\n" + "=" * 60)
    print("📈 RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    for k_label, m in metrics.items():
        print(f"\n  {k_label}:")
        print(f"    Hit Rate:     {m['hit_rate']:.4f}")
        print(f"    MRR:          {m['mrr']:.4f}")
        print(f"    Precision@k:  {m['precision_at_k']:.4f}")

    # Qualitative analysis — success and failure cases
    successes = [r for r in results if any(s["filename"] == r["source_doc"] for s in r["sources"][:3])]
    failures = [r for r in results if not any(s["filename"] == r["source_doc"] for s in r["sources"][:5])]

    qualitative = {
        "success_cases": [],
        "failure_cases": [],
    }

    for r in successes[:3]:
        qualitative["success_cases"].append({
            "question": r["question"],
            "expected_source": r["source_doc"],
            "retrieved_sources": [s["filename"] for s in r["sources"][:3]],
            "generated_answer_preview": r["generated_answer"][:200],
            "explanation": "The correct source document was retrieved in the top-3 results.",
        })

    for r in failures[:2]:
        qualitative["failure_cases"].append({
            "question": r["question"],
            "expected_source": r["source_doc"],
            "retrieved_sources": [s["filename"] for s in r["sources"][:5]],
            "generated_answer_preview": r["generated_answer"][:200],
            "explanation": "The expected source document was not found in the top-5 results.",
        })

    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_questions": len(dataset),
        "metrics": metrics,
        "qualitative_analysis": qualitative,
        "detailed_results": results,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System Retrieval Evaluation")
    parser.add_argument("--api-url", default="http://localhost:5000", help="Flask API URL")
    parser.add_argument("--dataset", default="eval/eval_dataset.json", help="Evaluation dataset path")
    parser.add_argument("--output", default="results/eval_results.json", help="Output path for results")
    args = parser.parse_args()

    run_evaluation(args.api_url, args.dataset, args.output)
