# LoCoMo Benchmark: FTS + Vector + Graph Report

> Date: 2026-04-13 (updated)
> Dataset: LoCoMo (ACL 2024, Snap Research) — 10 multi-session conversations, 1,986 QA pairs
> Default embedding model: BAAI/bge-m3 (ONNX, 1024d, 568M params)

## Background

LoCoMo 是一个用于评估长期对话记忆系统的学术基准测试。每个 QA 对给出一个自然语言问题，要求系统从历史对话中检索出相关信息来回答。

Hypatia 支持三种检索方式：**FTS**（关键词匹配）、**Vector**（语义相似度）和 **Graph**（k-hop 关系遍历）。本次测试主要对比 FTS 和 Vector 在自然语言 QA 场景下的表现。

## Test Configuration

- **数据加载**：10 个对话的所有 turns、session summaries、event summaries、observations 作为 Knowledge entries 载入临时 shelf
- **FTS 检索**：直接将自然语言问题作为 FTS5 查询（经 sanitize 后）
- **向量检索**：默认使用 BGE-M3 (1024d)，也测试了 EmbeddingGemma-300M、gte-multilingual-base、gte-Qwen2-1.5B、Jina v5 等模型，通过 DuckDB `array_cosine_distance` 做暴力最近邻搜索
- **评估指标**：Recall@K — 证据 turn 是否出现在 top-K 搜索结果中
- **数据规模**：6,426 entries ingested（含 embeddings）

## Results

### Turn-level storage — FTS vs Vector (BGE-M3, 默认模型)

| Category | N | FTS R@1 | FTS R@10 | Vec R@1 | Vec R@10 | Δ R@10 |
|----------|---:|---------|----------|---------|----------|--------|
| Single-hop (Cat 4) | 841 | 0.1% | 0.1% | **45.5%** | **76.1%** | +76.0% |
| Multi-hop (Cat 1) | 282 | 0.4% | 0.4% | **30.5%** | **75.5%** | +75.1% |
| Temporal (Cat 2) | 321 | 0.0% | 0.0% | **53.0%** | **80.4%** | +80.4% |
| Open-domain (Cat 3) | 96 | 1.0% | 1.0% | **22.9%** | **49.0%** | +48.0% |
| **OVERALL** | **1,540** | **0.2%** | **0.2%** | **38.6%** | **75.2%** | **+75.0%** |

### Latency

| Metric | FTS | Vector (BGE-M3) |
|--------|-----|--------|
| Search p50 | 814 ms | 43 ms |
| Search p99 | 1,713 ms | 54 ms |

向量搜索不仅召回率大幅领先，延迟也更低（FTS 的 BM25 JOIN 开销在 6K 条目下较大）。

### Session-level storage — FTS only（历史数据）

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
| MemPalace Raw ChromaDB (vector) | 60.3% |
| **Hypatia Vector (BGE-M3, 默认)** | **75.2%** |
| Hypatia Vector (EmbeddingGemma-300M) | 80.3% |
| MemPalace Hybrid v5 (vector + keyword + rerank) | 88.9% |

Hypatia 的纯向量搜索已经超过 MemPalace 的 ChromaDB 基线（75.2% vs 60.3%），接近其混合模式的 88.9%。EmbeddingGemma-300M 在纯 recall 上更高（80.3%），但 BGE-M3 在速度和多语言覆盖上更优。

## Analysis

### Why FTS Fails on Natural Language QA

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

### Why Vector Search Works

向量搜索通过语义嵌入解决了词汇鸿沟问题——BGE-M3 将 "go to" 和 "went to" 编码到相近的向量空间位置，无需精确词汇匹配。

### Remaining Gap (75.2% vs 88.9%)

与 MemPalace 混合模式的差距主要来自：
1. **无重排序**：Hypatia 使用纯 cosine 距离排序，没有 reranker
2. **无 hybrid fusion**：没有 FTS + vector 的分数融合
3. **Open-domain 较弱**（49.0%）— 这类问题通常需要更复杂的推理，单靠向量相似度不够

## Positioning

Hypatia 提供三种互补的检索能力：

- **FTS**：JSE 引擎中的精确关键词检索算子（`$search`），确定性、零依赖、亚毫秒级延迟
- **向量搜索**：语义相似度检索（`$similar` / `hypatia similar`），自动桥接词汇鸿沟，需要本地嵌入模型（默认 BGE-M3）
- **图遍历**：k-hop 关系遍历（`$k-hop`），基于 statement triples 的 recursive CTE，探索实体间的路径和邻域

三种方式互补：FTS 精确快速、向量搜索处理语义、图遍历探索关系。通过 JSE 统一查询接口。

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
