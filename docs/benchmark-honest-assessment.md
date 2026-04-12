# Hypatia Benchmark 诚实地评估

> Date: 2026-04-10
> Status: Internal assessment, not peer-reviewed

## 1. 当前成绩

| Metric | Result |
|--------|--------|
| Recall@1 | 100.0% (20/20) |
| Recall@5 | 100.0% |
| Recall@10 | 100.0% |
| FTS search p50 | 474 µs |
| Scale | 1K knowledge, 2K statements, 20 needles |

## 2. 与 MemPalace 基准的覆盖面对比

| 维度 | MemPalace | Hypatia |
|------|-----------|---------|
| **学术基准** | 4个（LongMemEval/LoCoMo/ConvoMem/MemBench） | 0个 |
| **数据来源** | 真实对话数据（HuggingFace 公开数据集） | 合成数据（固定词库 + seed=42） |
| **问题规模** | 500 + 1,986 + 75,336 + 8,500 | 20 |
| **问题类型** | 6种（单会话/多会话/时序推理/知识更新/偏好/时间） | 1种（关键词匹配） |
| **查询形式** | 自然语言提问 | 从 needle 内容提取的关键词 |
| **评估指标** | R@K, NDCG@K, F1 | 仅 R@K |
| **统计显著性** | 500+ 样本 | 20 样本（1个变化 = 5% 波动） |

Hypatia 只覆盖了 MemPalace 内部基准（`PalaceDataGenerator` 合成测试）的对应物，完全没有触及任何学术基准。

## 2.1 LoCoMo 学术基准结果 (2026-04-12, updated)

在真实学术基准 LoCoMo 上评测，同时测试 FTS 和向量搜索。

| 配置 | 数据规模 | 查询数 |
|------|---------|--------|
| 10 个多会话对话 | 6,426 knowledge entries (含 embeddings) | 1,540 (排除 adversarial) |
| 嵌入模型 | EmbeddingGemma-300M (ONNX, int8, 768d) | — |

### FTS (BM25)

| Category | N | R@1 | R@5 | R@10 |
|----------|---|-----|-----|------|
| Single-hop (Cat 4) | 841 | 0.1% | 0.1% | 0.1% |
| Multi-hop (Cat 1) | 282 | 0.4% | 0.4% | 0.4% |
| Temporal (Cat 2) | 321 | 0.0% | 0.0% | 0.0% |
| Open-domain (Cat 3) | 96 | 1.0% | 1.0% | 1.0% |
| **OVERALL** | **1,540** | **0.2%** | **0.2%** | **0.2%** |

### Vector (EmbeddingGemma-300M, cosine similarity)

| Category | N | R@1 | R@5 | R@10 |
|----------|---|-----|-----|------|
| Single-hop (Cat 4) | 841 | 44.7% | 74.6% | **83.8%** |
| Multi-hop (Cat 1) | 282 | 27.7% | 66.0% | **77.3%** |
| Temporal (Cat 2) | 321 | 51.7% | 77.6% | **84.1%** |
| Open-domain (Cat 3) | 96 | 20.8% | 32.3% | **45.8%** |
| **OVERALL** | **1,540** | **41.6%** | **71.0%** | **80.3%** |

对比 MemPalace：Raw ChromaDB R@10 = 60.3%，Hybrid v5 R@10 = 88.9%。

**结论**：向量搜索将 R@10 从 0.2% 提升到 80.3%，超过 MemPalace 的 ChromaDB 基线（60.3%）。核心原因是嵌入模型消除了词汇鸿沟——语义相似的表述被编码到相邻向量空间。Open-domain（45.8%）仍较低，这类问题需要多跳推理，单靠向量相似度不够。

## 3. 过拟合分析

### 3.1 查询提取对测试数据的直接拟合

实现过程中发生了典型的"教测试题"行为：

1. **第一轮**：查询在 `"uses"/"set to"/"configured"` 等动词处截断 → 95% recall（19/20）
2. 分析失败 needle → 发现查询太短，丢失区分性关键词
3. **第二轮**：改为在 `"with"/"for"/"from"/"in"` 处截断 → Docker needle 的 `"1.2GB"` 导致 FTS5 语法错误
4. 把 `.` 加入 sanitization → 100% recall

这些截断点和 sanitization 规则是**观察了 NEEDLE_TOPICS 之后决定的**。MemPalace 在类似情况下（hybrid v4 从 99.4% → 100%）明确写道："This is teaching to the test." 并建立了 50/450 train/test split 量化过拟合程度（held-out 98.4% vs full 100%）。

Hypatia 目前没有这样的分离机制。

### 3.2 查询是从答案中提取的

Needle 查询是从 needle 内容本身提取的关键词。搜索词**一定**出现在目标文档中。这是 FTS 最擅长的场景——精确关键词匹配，近乎同义反复。

MemPalace 的学术基准使用用户自然语言提问：
- "What database config was discussed?" 需要匹配 "PostgreSQL vacuum autovacuum threshold set to 50 percent"
- "Who manages the backend team?" 需要匹配 "Alice manages the backend engineering group"

这种词汇 gap（lexical gap）是检索系统的核心挑战，我们的测试没有覆盖。

### 3.3 样本量不足

20 个 needle，每个影响 5%。任何一个偶然的成功/失败都会显著改变结果。LongMemEval 有 500 题，单题影响 0.2%。

### 3.4 100% recall 的真实含义

当前的 100% 实际证明的是：**给定一段文本，从中提取关键词，用这些关键词搜索 FTS5，能找到原文。** 这是 FTS 的本职工作，不能推出系统在真实场景下的表现。

## 4. 经得起第三方考验吗？

**目前不能。** 原因：

1. **没有使用公开数据集** — 第三方无法用同一数据集复现和对比
2. **没有 train/test 分离** — 所有调参都是在全量数据上做的
3. **没有自然语言查询测试** — 只测了关键词匹配
4. **没有跨领域泛化测试** — 所有数据都是技术领域
5. **没有 precision 测试** — 只测了 recall，不知道查询返回了多少无关结果

## 5. 改进方向

### 已完成：向量搜索集成 (2026-04-12)

- **嵌入模型**：EmbeddingGemma-300M (ONNX, int8)，295MB，本地推理零外部依赖
- **向量存储**：DuckDB `FLOAT[768]` 列 + `array_cosine_distance` 暴力搜索
- **结果**：LoCoMo R@10 从 0.2%（FTS）提升到 80.3%（vector）
- **后续**：hybrid fusion（FTS + vector 分数融合）可能进一步提升到接近 MemPalace 的 88.9%

### 短期：提高内部基准信度

1. **Train/test split** — 随机分 10 needle 做开发调参，10 needle 做 held-out 评估，报告两个分数
2. **增加 needle 数量** — 至少 100 个，降低单题影响
3. **引入自然语言查询** — 为每个 needle 编写用户视角的查询（不从内容提取），如"那个数据库的配置是什么？"
4. **引入 precision 测试** — 测量查询返回的无关结果比例

### 中期：对接学术基准

5. **对接 LongMemEval** — 数据集公开在 HuggingFace (`xiaowu0162/longmemeval-cleaned`)，可以直接拿来测试
6. **添加会话式场景** — 测试从多轮对话中提取和检索信息的能力

### 长期：能力边界探索

7. **跨领域测试** — 医疗、法律、金融等不同术语体系
8. **多语言测试** — 中英文混合查询
9. **对抗性测试** — 故意使用同义词、缩写、错别字查询

## 6. Hypatia 的定位声明

Hypatia 的设计目标是**可解释、可管理的 AI 外挂记忆系统**，而非通用向量检索引擎。通过 AI 的推理能力配合精确的结构化查询（JSE），在准确性优先的场景下提供确定性记忆管理。

Benchmark 成绩反映的是特定场景下的能力边界。成绩不理想不是系统的问题，而是帮助用户理解这个系统适合做什么、不适合做什么。这是好事。

---

## 附录：MemPalace 基准透明度参考

MemPalace 在方法论诚信方面值得学习：

- 每次模式改进都记录了具体修改和动机
- 建立了 50/450 train/test split 量化过拟合
- LoCoMo 100% 结果明确标注了结构性原因（top-k 超过会话总数）
- 所有结果以 JSONL 格式提交，每题检索结果可审计
- 单命令可复现全部基准

> Source: `milla-jovovich/mempalace` — `benchmarks/BENCHMARKS.md`, `benchmarks/HYBRID_MODE.md`
