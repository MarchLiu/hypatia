# LongMemEval Benchmark for Hypatia

[LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025) 是评估 AI 长期记忆能力的学术基准。
本 benchmark 使用 **LongMemEval M 变体**（~500 sessions/question, ~1.5M tokens），是三个变体中规模最大的，对检索系统构成最大压力。

## 五项核心能力

| 能力 | 题型 |
|------|------|
| 信息提取 (IE) | single-session-user, single-session-assistant, single-session-preference |
| 多会话推理 (MR) | multi-session |
| 知识更新 (KU) | knowledge-update |
| 时序推理 (TR) | temporal-reasoning |
| 弃答 (ABS) | abstention |

共 500 道题，7 种题型。

## 快速开始

### 1. 下载数据

```bash
python3 scripts/longmemeval_download.py --variant m
```

### 2. 运行检索 benchmark

```bash
LONGMEMEVAL_DATA=longmemeval_m.json \
LONGMEMEVAL_RESULTS=longmemeval_m_results.jsonl \
  cargo test --test longmemeval --release -- --nocapture
```

**注意**：M 变体数据量大，摄入和检索阶段耗时较长。需要本地 embedding model（`~/.hypatia/default/` 下的 ONNX 模型）。

### 3. 查看检索指标

```bash
python3 scripts/longmemeval_eval.py \
  --results longmemeval_m_results.jsonl \
  --retrieval-only
```

### 4. 全量评估（QA 生成 + Judge）

```bash
# QA 生成
python3 scripts/longmemeval_eval.py \
  --results longmemeval_m_results.jsonl \
  --generate \
  --llm-endpoint https://ai.gitee.com/v1/chat/completions \
  --llm-model Kimi-K2-Thinking \
  --llm-key-env CHAT_SK

# LLM-as-Judge 评判
python3 scripts/longmemeval_eval.py \
  --results longmemeval_m_results.jsonl \
  --judge \
  --judge-model gpt-4o \
  --judge-key-env OPENAI_API_KEY
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `longmemeval_m.json` | LongMemEval M 数据（下载获得）|
| `tests/longmemeval.rs` | Rust 集成测试：摄入 + 检索 |
| `scripts/longmemeval_download.py` | 数据下载脚本 |
| `scripts/longmemeval_eval.py` | Python 评估脚本：检索指标 + QA 生成 + Judge |
| `longmemeval_m_results.jsonl` | Rust 测试输出的检索结果 |
| `longmemeval_m_hypotheses.jsonl` | LLM 生成的回答 |

## 评估指标

### 检索指标

- **R@k (Recall at k)**：正确答案会话是否出现在 top-k 检索结果中
- 主指标：**R@5**（与论文对齐）

### QA 准确率

- 使用 LLM-as-Judge（GPT-4o）评判生成回答的正确性
- 报告 Overall Accuracy、Task-averaged Accuracy、Abstention Accuracy

## 参考

- [LongMemEval 论文](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub](https://github.com/xiaowu0162/LongMemEval)
- [LongMemEval 数据集 (HuggingFace)](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
