#!/usr/bin/env python3
"""LongMemEval evaluation script for Hypatia.

Computes retrieval metrics (R@k, NDCG), optionally generates QA answers with an
LLM, and optionally runs GPT-4o judge evaluation.

Usage:
    # Retrieval metrics only
    python3 scripts/longmemeval_eval.py --results longmemeval_m_results.jsonl --retrieval-only

    # Full evaluation with LLM generation + judge
    python3 scripts/longmemeval_eval.py --results longmemeval_m_results.jsonl \
        --generate --llm-endpoint https://ai.gitee.com/v1/chat/completions \
        --llm-model Kimi-K2-Thinking --llm-key-env CHAT_SK \
        --judge --judge-key-env OPENAI_API_KEY
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import httpx

# ── Ability mapping ───────────────────────────────────────────────────

TYPE_TO_ABILITY = {
    "single-session-user": "information_extraction",
    "single-session-assistant": "information_extraction",
    "single-session-preference": "information_extraction",
    "multi-session": "multi_session_reasoning",
    "knowledge-update": "knowledge_updates",
    "temporal-reasoning": "temporal_reasoning",
    "abstention": "abstention",
}

ABILITY_ORDER = [
    "information_extraction",
    "multi_session_reasoning",
    "knowledge_updates",
    "temporal_reasoning",
]

TYPE_ORDER = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
]

# ── Data loading ──────────────────────────────────────────────────────


def load_results(path: str) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ── Retrieval metrics ─────────────────────────────────────────────────


def compute_recall_at_k(retrieved: list[str], expected: list[str], k: int) -> bool:
    """Whether any expected item appears in top-k retrieved items."""
    if not expected:
        return True  # no ground truth → vacuously true
    top_k = retrieved[:k]
    return any(e in top_k for e in expected)


def compute_ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """NDCG@k for binary relevance."""
    if not expected:
        return 1.0
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in expected:
            dcg += 1.0 / (i + 1).bit_length()  # log2(i+1)
    # Ideal DCG: all relevant at top
    idcg = sum(1.0 / (i + 1).bit_length() for i in range(min(len(expected), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(results: list[dict]) -> dict:
    """Compute retrieval metrics grouped by ability and question type."""
    metrics = {
        "fts": {"overall": [], "by_ability": defaultdict(list), "by_type": defaultdict(list)},
        "vec": {"overall": [], "by_ability": defaultdict(list), "by_type": defaultdict(list)},
    }

    for r in results:
        ability = r.get("ability", TYPE_TO_ABILITY.get(r["question_type"], "unknown"))
        if ability == "abstention":
            continue

        qtype = r["question_type"]
        expected_sessions = r.get("answer_session_ids", [])

        for method in ["fts", "vec"]:
            sessions_key = f"{method}_retrieved_sessions"
            retrieved = r.get(sessions_key) or []

            for k in [1, 3, 5, 10, 50]:
                recall = compute_recall_at_k(retrieved, expected_sessions, k)
                ndcg = compute_ndcg_at_k(retrieved, expected_sessions, k)
                entry = {"r@k": recall, f"ndcg@{k}": ndcg, "k": k}

                metrics[method]["overall"].append({"recall": recall, "ndcg": ndcg, "k": k})
                metrics[method]["by_ability"][ability].append(
                    {"recall": recall, "ndcg": ndcg, "k": k}
                )
                metrics[method]["by_type"][qtype].append({"recall": recall, "ndcg": ndcg, "k": k})

    return metrics


def format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator * 100:.1f}%"


def print_retrieval_report(results: list[dict], metrics: dict):
    print()
    print("=" * 70)
    print("  LongMemEval M — Hypatia Retrieval Results")
    print("=" * 70)

    total = len(results)
    abstention_count = sum(1 for r in results if r.get("ability") == "abstention")
    eval_count = total - abstention_count
    print(f"  Total questions: {total}")
    print(f"  Evaluated (excl. abstention): {eval_count}")
    print()

    for method, label in [("fts", "FTS (BM25)"), ("vec", "Vector (cosine)")]:
        by_ability = metrics[method]["by_ability"]
        if not by_ability:
            continue

        print(f"  {label} — By Ability (session-level)")
        print(f"  {'Ability':<25} {'N':>5} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8}")
        print("  " + "─" * 66)

        total_counts = {k: [0, 0] for k in [1, 5, 10, 50]}

        for ability in ABILITY_ORDER:
            entries = by_ability.get(ability, [])
            if not entries:
                continue
            n = len([e for e in entries if e["k"] == 1])
            if n == 0:
                continue
            row = {}
            for k in [1, 5, 10, 50]:
                hits = sum(1 for e in entries if e["k"] == k and e["recall"])
                row[k] = hits
                total_counts[k][0] += hits
                total_counts[k][1] += n

            print(
                f"  {ability.replace('_', ' '):<25} {n:>5} "
                f"{format_pct(row[1], n):>8} {format_pct(row[5], n):>8} "
                f"{format_pct(row[10], n):>8} {format_pct(row[50], n):>8}"
            )

        print("  " + "─" * 66)
        if total_counts[1][1] > 0:
            print(
                f"  {'OVERALL':<25} {total_counts[1][1]:>5} "
                f"{format_pct(*total_counts[1]):>8} {format_pct(*total_counts[5]):>8} "
                f"{format_pct(*total_counts[10]):>8} {format_pct(*total_counts[50]):>8}"
            )
        print()

    # By question type
    print("  FTS — By Question Type")
    print(f"  {'Type':<30} {'N':>5} {'R@5':>8} {'R@10':>8}")
    print("  " + "─" * 55)

    by_type = metrics["fts"]["by_type"]
    for qtype in TYPE_ORDER:
        entries = by_type.get(qtype, [])
        if not entries:
            continue
        n = len([e for e in entries if e["k"] == 1])
        r5 = sum(1 for e in entries if e["k"] == 5 and e["recall"])
        r10 = sum(1 for e in entries if e["k"] == 10 and e["recall"])
        print(f"  {qtype:<30} {n:>5} {format_pct(r5, n):>8} {format_pct(r10, n):>8}")

    print("=" * 70)


# ── QA Generation ─────────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are a helpful assistant with access to conversation history.
Answer the question based on the provided conversation context.
Be concise and direct. If the answer is not in the context, say "I don't know"."""


def generate_answers(
    results: list[dict],
    endpoint: str,
    model: str,
    api_key: str,
    top_k: int = 5,
) -> list[dict]:
    """Generate answers using an LLM with retrieved context."""
    hypotheses = []
    client = httpx.Client(timeout=60)

    for i, r in enumerate(results):
        # Build context from top-k FTS results
        context_parts = []
        top_keys = r.get("fts_top_keys", [])[:top_k]
        for key in top_keys:
            context_parts.append(f"[{key}]")

        context = "\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = (
            f"Based on the following conversation history excerpts, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {r['question']}\n\n"
            f"Provide a concise answer:"
        )

        try:
            resp = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            hypothesis = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            hypothesis = f"ERROR: {e}"

        hypotheses.append(
            {
                "question_id": r["question_id"],
                "question_type": r["question_type"],
                "hypothesis": hypothesis,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(results)} answers generated")

    return hypotheses


# ── LLM-as-Judge ──────────────────────────────────────────────────────

JUDGE_PROMPTS = {
    "default": """You are an expert judge. Determine if the hypothesis contains the correct answer to the question.

Question: {question}
Reference Answer: {answer}
Hypothesis: {hypothesis}

Answer "yes" if the hypothesis contains the correct answer, "no" otherwise.
If the hypothesis only contains a subset of the information, answer "no".

Your answer (yes/no):""",
    "temporal-reasoning": """You are an expert judge. Determine if the hypothesis contains the correct answer.

Question: {question}
Reference Answer: {answer}
Hypothesis: {hypothesis}

Answer "yes" if the hypothesis contains the correct answer, "no" otherwise.
Do NOT penalize off-by-one errors for the number of days.

Your answer (yes/no):""",
    "knowledge-update": """You are an expert judge. Determine if the hypothesis contains the most updated answer.

Question: {question}
Reference Answer: {answer}
Hypothesis: {hypothesis}

If the hypothesis contains some previous information along with an updated answer,
the hypothesis should be considered correct as long as the updated answer matches.

Your answer (yes/no):""",
    "single-session-preference": """You are an expert judge. Determine if the hypothesis correctly recalls and utilizes user information.

Question: {question}
Rubric: {answer}
Hypothesis: {hypothesis}

The model does not need to reflect all points in the rubric. The response is correct
as long as it recalls and utilizes the user's personal information correctly.

Your answer (yes/no):""",
    "abstention": """You are an expert judge. Determine if the model correctly identifies the question as unanswerable.

Question: {question}
Hypothesis: {hypothesis}

Answer "yes" if the model correctly identifies that it cannot answer the question
(e.g., says "I don't know", "I don't have that information", etc.).

Your answer (yes/no):""",
}


def get_judge_prompt(question_type: str, question: str, answer: str, hypothesis: str) -> str:
    is_abstention = "abs" in question_type
    canonical = question_type.replace("_abs", "")

    if is_abstention:
        template = JUDGE_PROMPTS["abstention"]
        return template.format(question=question, hypothesis=hypothesis)
    elif canonical == "temporal-reasoning":
        template = JUDGE_PROMPTS["temporal-reasoning"]
    elif canonical == "knowledge-update":
        template = JUDGE_PROMPTS["knowledge-update"]
    elif canonical == "single-session-preference":
        template = JUDGE_PROMPTS["single-session-preference"]
    else:
        template = JUDGE_PROMPTS["default"]

    return template.format(question=question, answer=answer, hypothesis=hypothesis)


def run_judge(
    results: list[dict],
    hypotheses: list[dict],
    endpoint: str,
    model: str,
    api_key: str,
) -> dict:
    """Run LLM-as-judge evaluation."""
    hyp_map = {h["question_id"]: h["hypothesis"] for h in hypotheses}

    by_type = defaultdict(lambda: [0, 0])  # [correct, total]
    by_ability = defaultdict(lambda: [0, 0])
    client = httpx.Client(timeout=60)

    for r in results:
        qid = r["question_id"]
        hypothesis = hyp_map.get(qid)
        if hypothesis is None:
            continue

        prompt = get_judge_prompt(r["question_type"], r["question"], r["answer"], hypothesis)

        try:
            resp = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            answer_text = data["choices"][0]["message"]["content"].strip().lower()
            is_correct = answer_text.startswith("yes")
        except Exception as e:
            print(f"    WARN: Judge failed for {qid}: {e}")
            is_correct = False

        qtype = r["question_type"].replace("_abs", "")
        ability = r.get("ability", TYPE_TO_ABILITY.get(qtype, "unknown"))

        by_type[qtype][1] += 1
        by_ability[ability][1] += 1
        if is_correct:
            by_type[qtype][0] += 1
            by_ability[ability][0] += 1

    return {"by_type": dict(by_type), "by_ability": dict(by_ability)}


def print_qa_report(judge_results: dict):
    print()
    print("  QA ACCURACY (LLM-as-Judge)")
    print("  " + "─" * 40)

    total_correct = 0
    total_count = 0
    type_correct = 0
    type_count = 0

    print(f"  {'Type':<30} {'N':>5} {'Acc':>8}")
    print("  " + "─" * 45)
    for qtype in TYPE_ORDER:
        correct, n = judge_results["by_type"].get(qtype, [0, 0])
        if n > 0:
            acc = correct / n * 100
            print(f"  {qtype:<30} {n:>5} {acc:>7.1f}%")
            type_correct += correct
            type_count += n

    # Abstention
    abs_correct, abs_n = judge_results["by_ability"].get("abstention", [0, 0])
    if abs_n > 0:
        print(f"  {'abstention':<30} {abs_n:>5} {abs_correct / abs_n * 100:>7.1f}%")

    # Overall
    for ability_data in judge_results["by_ability"].values():
        total_correct += ability_data[0]
        total_count += ability_data[1]

    print("  " + "─" * 45)
    if total_count > 0:
        print(f"  {'Overall':<30} {total_count:>5} {total_correct / total_count * 100:>7.1f}%")
    if type_count > 0:
        print(f"  {'Task-averaged':<30} {type_count:>5} {type_correct / type_count * 100:>7.1f}%")
    if abs_n > 0:
        print(f"  {'Abstention':<30} {abs_n:>5} {abs_correct / abs_n * 100:>7.1f}%")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LongMemEval evaluation for Hypatia")
    parser.add_argument("--results", required=True, help="JSONL results file from Rust benchmark")
    parser.add_argument("--retrieval-only", action="store_true", help="Only compute retrieval metrics")
    parser.add_argument("--generate", action="store_true", help="Generate QA answers with LLM")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-judge evaluation")
    parser.add_argument("--llm-endpoint", default="", help="LLM API endpoint for generation")
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model for generation")
    parser.add_argument("--llm-key-env", default="OPENAI_API_KEY", help="Env var for LLM API key")
    parser.add_argument("--judge-endpoint", default="", help="Judge LLM API endpoint")
    parser.add_argument("--judge-model", default="gpt-4o", help="Judge model")
    parser.add_argument("--judge-key-env", default="OPENAI_API_KEY", help="Env var for judge API key")
    parser.add_argument(
        "--output-hypotheses", default="", help="Output file for hypotheses"
    )
    args = parser.parse_args()

    results = load_results(args.results)

    # Phase A: Retrieval metrics (always)
    metrics = evaluate_retrieval(results)
    print_retrieval_report(results, metrics)

    if args.retrieval_only:
        return

    # Phase B: QA Generation
    hypotheses = None
    if args.generate:
        llm_key = os.environ.get(args.llm_key_env, "")
        if not llm_key:
            print(f"  ERROR: Set {args.llm_key_env} environment variable")
            sys.exit(1)
        endpoint = args.llm_endpoint
        if not endpoint:
            print("  ERROR: --llm-endpoint required for generation")
            sys.exit(1)

        print("\n  Generating answers...")
        hypotheses = generate_answers(results, endpoint, args.llm_model, llm_key)

        out_path = args.output_hypotheses or args.results.replace(".jsonl", "_hypotheses.jsonl")
        with open(out_path, "w") as f:
            for h in hypotheses:
                f.write(json.dumps(h) + "\n")
        print(f"  Hypotheses saved to: {out_path}")

    # Phase C: Judge
    if args.judge:
        if hypotheses is None:
            hyp_path = args.results.replace(".jsonl", "_hypotheses.jsonl")
            if Path(hyp_path).exists():
                hypotheses = load_results(hyp_path)
                print(f"  Loaded hypotheses from: {hyp_path}")
            else:
                print("  ERROR: No hypotheses found. Run with --generate first.")
                sys.exit(1)

        judge_key = os.environ.get(args.judge_key_env, "")
        if not judge_key:
            print(f"  ERROR: Set {args.judge_key_env} environment variable")
            sys.exit(1)
        judge_endpoint = args.judge_endpoint or "https://api.openai.com/v1/chat/completions"

        print("\n  Running LLM-as-judge evaluation...")
        judge_results = run_judge(results, hypotheses, judge_endpoint, args.judge_model, judge_key)
        print_qa_report(judge_results)
        print("=" * 70)


if __name__ == "__main__":
    main()
