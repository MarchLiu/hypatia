#!/usr/bin/env python3
"""LoCoMo Benchmark Evaluation for Hypatia.

Reads JSONL results from the Rust test, computes F1 scores, and optionally
runs LLM judge via OpenRouter API.

Usage:
    python3 scripts/locomo_eval.py --results locomo_results.jsonl
    python3 scripts/locomo_eval.py --results locomo_results.jsonl --sample 200
    python3 scripts/locomo_eval.py --results locomo_results.jsonl --no-judge
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    sys.exit(1)


# ── F1 Score (deterministic) ─────────────────────────────────────────

def simple_stem(word: str) -> str:
    """Approximate Porter stemming for English words."""
    word = word.lower().strip()
    if len(word) <= 3:
        return word

    # Remove common suffixes
    suffixes = [
        "ication", "ation", "tion", "sion", "ment", "ness", "ence",
        "ance", "ible", "able", "ful", "ous", "ive", "ing", "ied",
        "ies", "ed", "er", "ly", "al", "es",
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]

    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def normalize_tokens(text: str) -> set[str]:
    """Tokenize and stem text for F1 comparison."""
    # Remove punctuation, lowercase, split
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text.split()
    # Remove very common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "it", "its", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "she", "they",
        "them", "their", "and", "or", "but", "not", "no", "if",
        "then", "so", "as", "up", "out", "about", "into", "over",
    }
    stemmed = set()
    for token in tokens:
        if token not in stop_words and len(token) > 1:
            stemmed.add(simple_stem(token))
    return stemmed


def compute_f1(pred_text: str, gold_answer: str) -> float:
    """Compute token-level F1 between predicted text and gold answer."""
    pred_tokens = normalize_tokens(pred_text)
    gold_tokens = normalize_tokens(gold_answer)

    if not gold_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# ── LLM Judge ─────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are evaluating a memory retrieval system. Given a question, the gold answer, and the top retrieved context, rate how well the context supports finding the correct answer.

Rate on a scale of 0.0 to 1.0:
- 1.0: Context directly contains the gold answer
- 0.7-0.9: Context contains strongly related information that would help
- 0.4-0.6: Context is partially relevant
- 0.1-0.3: Context is tangentially related
- 0.0: Context is completely irrelevant

Respond with ONLY a single number (e.g., 0.7). No explanation."""


async def judge_single(
    client: httpx.AsyncClient,
    question: str,
    gold_answer: str,
    context: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> float:
    """Get LLM judge score for a single QA pair."""
    user_prompt = f"""Question: {question}
Gold Answer: {gold_answer}
Retrieved Context:
{context[:2000]}

Rate 0.0-1.0:"""

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 10,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                # Extract number from response
                match = re.search(r"(\d+\.?\d*)", text)
                if match:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                return 0.0
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  WARN: Judge failed for '{question[:40]}...': {e}")
                    return -1.0  # sentinel for failure
    return 0.0


async def run_llm_judge(
    results: list[dict],
    model: str,
    api_key: str,
    max_concurrent: int = 5,
) -> list[float]:
    """Run LLM judge on all results."""
    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = []
        for r in results:
            context = "\n".join(r.get("top_texts", [])[:3])
            if not context:
                tasks.append(asyncio.coroutine(lambda: 0.0)())
                continue
            tasks.append(
                judge_single(
                    client,
                    r["question"],
                    r["answer"],
                    context,
                    model,
                    semaphore,
                )
            )

        scores = []
        total = len(tasks)
        completed = 0

        for coro in asyncio.as_completed(tasks):
            score = await coro
            scores.append(score)
            completed += 1
            if completed % 50 == 0:
                valid = [s for s in scores if s >= 0]
                avg = statistics.mean(valid) if valid else 0
                print(f"  Progress: {completed}/{total} (avg: {avg:.2f})")

    return scores


# ── Report ─────────────────────────────────────────────────────────────

CAT_NAMES = {4: "Single-hop", 1: "Multi-hop", 2: "Temporal", 3: "Open-domain"}


def print_report(results: list[dict], f1_scores: list[float], judge_scores: list[float] | None):
    """Print formatted report."""
    by_cat: dict[int, dict] = {}

    for i, r in enumerate(results):
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {
                "n": 0, "r1": 0, "r5": 0, "r10": 0,
                "f1": [], "judge": [],
            }
        d = by_cat[cat]
        d["n"] += 1
        if r.get("recall_at_1"): d["r1"] += 1
        if r.get("recall_at_5"): d["r5"] += 1
        if r.get("recall_at_10"): d["r10"] += 1
        d["f1"].append(f1_scores[i])
        if judge_scores and i < len(judge_scores) and judge_scores[i] >= 0:
            d["judge"].append(judge_scores[i])

    # Latency
    latencies = sorted(r["search_latency_us"] for r in results if r.get("search_latency_us", 0) > 0)
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p99 = latencies[len(latencies) * 99 // 100] if latencies else 0

    print()
    print("=" * 65)
    print("  LoCoMo Benchmark Results for Hypatia")
    print("=" * 65)
    print(f"  Queries: {len(results)} (excl. adversarial Cat 5)")
    print()
    print("  RETRIEVAL (deterministic)")
    print("  " + "-" * 61)
    print(f"  {'Category':<20} {'N':>5} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'F1':>7}")
    print("  " + "-" * 61)

    total = {"n": 0, "r1": 0, "r5": 0, "r10": 0, "f1": [], "judge": []}
    for cat in [4, 1, 2, 3]:
        d = by_cat.get(cat)
        if not d or d["n"] == 0:
            continue
        n = d["n"]
        f1_mean = statistics.mean(d["f1"]) if d["f1"] else 0
        print(f"  {CAT_NAMES[cat]:<20} {n:>5} {d['r1']/n*100:>6.1f}% {d['r5']/n*100:>6.1f}% {d['r10']/n*100:>6.1f}% {f1_mean:>6.3f}")
        for k in ["n", "r1", "r5", "r10"]:
            total[k] += d[k]
        total["f1"].extend(d["f1"])
        total["judge"].extend(d["judge"])

    n = total["n"]
    f1_mean = statistics.mean(total["f1"]) if total["f1"] else 0
    print("  " + "-" * 61)
    print(f"  {'OVERALL':<20} {n:>5} {total['r1']/n*100:>6.1f}% {total['r5']/n*100:>6.1f}% {total['r10']/n*100:>6.1f}% {f1_mean:>6.3f}")

    if judge_scores and total["judge"]:
        print()
        print("  LLM JUDGE")
        print("  " + "-" * 61)
        print(f"  {'Category':<20} {'N':>5} {'Mean':>7} {'Median':>7} {'Std':>7}")
        print("  " + "-" * 61)
        for cat in [4, 1, 2, 3]:
            d = by_cat.get(cat)
            if not d or not d["judge"]:
                continue
            j = d["judge"]
            print(f"  {CAT_NAMES[cat]:<20} {len(j):>5} {statistics.mean(j):>7.2f} {statistics.median(j):>7.2f} {statistics.stdev(j):>7.2f}" if len(j) > 1 else f"  {CAT_NAMES[cat]:<20} {len(j):>5} {statistics.mean(j):>7.2f} {statistics.mean(j):>7.2f} {'N/A':>7}")
        j = total["judge"]
        if len(j) > 1:
            print("  " + "-" * 61)
            print(f"  {'OVERALL':<20} {len(j):>5} {statistics.mean(j):>7.2f} {statistics.median(j):>7.2f} {statistics.stdev(j):>7.2f}")

    print()
    print("  LATENCY")
    print("  " + "-" * 61)
    if p50 > 1000:
        print(f"  Search p50: {p50/1000:.1f} ms")
        print(f"  Search p99: {p99/1000:.1f} ms")
    else:
        print(f"  Search p50: {p50} µs")
        print(f"  Search p99: {p99} µs")

    # Comparison reference
    print()
    print("  REFERENCE (MemPalace on LoCoMo)")
    print("  " + "-" * 61)
    print("  Raw ChromaDB:     R@10 = 60.3%")
    print("  Hybrid v5:        R@10 = 88.9%")
    print("=" * 65)
    print()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoCoMo evaluation for Hypatia")
    parser.add_argument("--results", required=True, help="Path to JSONL results file")
    parser.add_argument("--sample", type=int, default=0, help="Random sample N questions (0=all)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--model", default="z-ai/glm-5.1", help="LLM model for judge")
    parser.add_argument("--output", default=None, help="Save report as JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Load results
    results = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print(f"Loaded {len(results)} results from {args.results}")

    # Sample if requested
    if args.sample > 0 and args.sample < len(results):
        random.seed(args.seed)
        results = random.sample(results, args.sample)
        print(f"Sampled {args.sample} results (seed={args.seed})")

    # Compute F1 scores
    print("Computing F1 scores...")
    f1_scores = []
    for r in results:
        # Use top retrieved texts as the "prediction"
        pred_text = " ".join(r.get("top_texts", [])[:3])
        f1 = compute_f1(pred_text, r["answer"])
        f1_scores.append(f1)

    # Run LLM judge
    judge_scores = None
    if not args.no_judge:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("WARNING: OPENROUTER_API_KEY not set, skipping LLM judge")
            print("  Set it with: export OPENROUTER_API_KEY=sk-...")
        else:
            print(f"Running LLM judge ({args.model})...")
            judge_scores = asyncio.run(
                run_llm_judge(results, args.model, api_key, max_concurrent=5)
            )

    # Print report
    print_report(results, f1_scores, judge_scores)

    # Save JSON report
    if args.output:
        by_cat: dict = {}
        for i, r in enumerate(results):
            cat = r["category"]
            if cat not in by_cat:
                by_cat[cat] = {"n": 0, "r1": 0, "r5": 0, "r10": 0, "f1": []}
            d = by_cat[cat]
            d["n"] += 1
            if r.get("recall_at_1"): d["r1"] += 1
            if r.get("recall_at_5"): d["r5"] += 1
            if r.get("recall_at_10"): d["r10"] += 1
            d["f1"].append(f1_scores[i])

        report = {
            "model": args.model,
            "total_queries": len(results),
            "categories": {},
        }
        for cat, d in by_cat.items():
            n = d["n"]
            report["categories"][str(cat)] = {
                "name": CAT_NAMES.get(cat, f"Cat {cat}"),
                "n": n,
                "recall_at_1": d["r1"] / n if n else 0,
                "recall_at_5": d["r5"] / n if n else 0,
                "recall_at_10": d["r10"] / n if n else 0,
                "f1_mean": statistics.mean(d["f1"]) if d["f1"] else 0,
            }
        if judge_scores:
            report["judge_scores"] = judge_scores

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
