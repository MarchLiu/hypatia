# LoCoMo Benchmark: FTS Coverage Report

> Date: 2026-04-11
> Dataset: LoCoMo (ACL 2024, Snap Research) — 10 multi-session conversations, 1,986 QA pairs

## Background

LoCoMo 是一个用于评估长期对话记忆系统的学术基准测试。每个 QA 对给出一个自然语言问题，要求系统从历史对话中检索出相关信息来回答。

这并非 Hypatia 的设计目标场景。Hypatia 的核心功能是基于图结构（Knowledge + Statement triples）的**结构化查询**，FTS 只是 JSE 引擎中的一个算子。本次测试的目的是记录 FTS 算子单独面对自然语言 QA 时的能力边界。

## Test Configuration

- **数据加载**：10 个对话的所有 turns、session summaries、event summaries、observations 作为 Knowledge entries 载入临时 shelf
- **检索方式**：直接将自然语言问题作为 FTS5 查询（经 sanitize 后）
- **评估指标**：Recall@K — 证据 turn 是否出现在 top-K 搜索结果中

## Results

### Turn-level storage（每个对话 turn 单独存储）

| Category | N | R@1 | R@5 | R@10 |
|----------|---:|-----|-----|------|
| Single-hop (Cat 4) | 841 | 0.1% | 0.1% | 0.1% |
| Multi-hop (Cat 1) | 282 | 0.4% | 0.4% | 0.4% |
| Temporal (Cat 2) | 321 | 0.0% | 0.0% | 0.0% |
| Open-domain (Cat 3) | 96 | 1.0% | 1.0% | 1.0% |
| **OVERALL** | **1,540** | **0.2%** | **0.2%** | **0.2%** |

### Session-level storage（同一 session 的 turns + 日期合并为单文档）

| Category | N | R@1 | R@5 | R@10 |
|----------|---:|-----|-----|------|
| Single-hop (Cat 4) | 841 | 15.8% | 17.5% | 17.5% |
| Multi-hop (Cat 1) | 281 | 15.3% | 18.9% | 19.2% |
| Temporal (Cat 2) | 319 | 14.1% | 15.0% | 15.0% |
| Open-domain (Cat 3) | 96 | 3.1% | 4.2% | 4.2% |
| **OVERALL** | **1,537** | **14.6%** | **16.4%** | **16.5%** |

### Reference (MemPalace on LoCoMo)

| Mode | R@10 |
|------|------|
| Raw ChromaDB (vector) | 60.3% |
| Hybrid v5 (vector + keyword + rerank) | 88.9% |

## Why FTS Fails Here

FTS 的核心能力是**精确/词干匹配**。LoCoMo 的自然语言问答与之存在三重鸿沟：

1. **词汇鸿沟（Lexical Gap）**：问题用词与原文不同。例如：
   - 问题："When did Caroline go to the LGBTQ support group?"
   - 原文："I went to a LGBTQ support group yesterday"
   - "go to" vs "went to" — Porter stemmer 处理不了不规则动词

2. **语义压缩（Semantic Compression）**：答案往往需要推理。例如：
   - 原文："I went to a LGBTQ support group **yesterday**"
   - Session 日期："1:56 pm on 8 May, 2023"
   - 答案："7 May 2023" — 需要 "yesterday + session date" 推理

3. **问题噪声（Query Noise）**：自然语言问题包含大量 FTS 无法利用的功能词（when, did, what, is...），稀释了关键词的权重

## Positioning

这个结果在预期之内。Hypatia 的架构设计不追求解决语义搜索问题：

- **核心功能**：图结构查询（Knowledge nodes + Statement triples + JSE）
- **FTS 定位**：JSE 引擎中的一个算子，用于精确关键词检索
- **不覆盖的场景**：自然语言问答、语义相似度搜索、跨语言检索

语义搜索是一个独立的问题域，需要 embedding 模型 + 向量检索。如果未来有需求，可以作为一个新的存储引擎集成到 Hypatia 的 dual-write 架构中，但这是后续讨论的范畴。

## Reproduction

```bash
# Download LoCoMo data (one time)
curl -sL https://huggingface.co/datasets/Percena/locomo-mc10/resolve/main/raw/locomo10.json \
  -o locomo10.json

# Run Rust test (ingest + search + output JSONL)
LOCOMO_DATA=locomo10.json LOCOMO_RESULTS=locomo_results.jsonl \
  cargo test --test locomo --release -- --nocapture

# Run Python evaluation (F1 + optional LLM judge)
python3 scripts/locomo_eval.py --results locomo_results.jsonl --no-judge
```
