# 项目规划和设计

Hypatia 是一个面向 AI 的记忆管理系统。它使用文本检索（FTS）、向量搜索（Vector）和图遍历（Graph）三种方式管理和检索知识。

## 知识图谱

Hypatia 的知识由两大类组成，
- Knowledge 知识条目，条目之间是独立的，这代表信息空间上的点
- Statement 三元组条目，三元组定义 Knowledge 之间的关系，这代表信息空间上的边

通过 knowledge 和 statement ，知识构成了图。

## 记忆架构

Hypatia 的知识通过 shelves 分组，提供一个默认的 shelves，允许用户导入、导出和引用不同的 shelves。

一个 Shelves 是一个目录，每个目录包含 `data.duckdb`（knowledge 表和 statement 表，含向量列）、`index.sqlite`（全文检索索引）、可选的 `shelf.toml`（嵌入模型配置）和嵌入模型文件。

Hypatia 面向局部环境，因此数据管理也遵循简单原则，例如 knowledge 直接保存 name 作为主键，statement 中直接保存三元组的可读性形式 `subject, predicate, object` 作为主键。content 是一个 json 对象，默认总会包含：

```json
{
    "format": "markdown",
    "data": "",
    "tags": []
}
```
其中
- format 可以是 markdown、json等，data保存对应格式的内容
- tags 是字符串列表，可以为空
- knowledge 保存时，向 sqlite 的 fts 库中插入/更新一条记录，其 key 对应 knowledge 的 name，catalog 是 knowledge，content 是 knowledge 的 content 字典加上 name 和 日期
- statement 保存时，向 sqlite 的 fts 库中插入/更新一条记录，其key 对应 statement 的三元组可读形式 `subject, predicate, object`，注意这里没有左右括号，这是因为它们不对搜索提供有意义的信息
- 三元组中的信息可能本身就包含,或"等字符，因此fts表中保存三元组时，key需要按照csv的规范处理成一行三列的文本
- 知识经常是需要包含其定义的，而 statement 有更多的可能并没有真正的内容，这是一个预留机制

### 知识表

```sql
create table knowledge (
    name text primary key,
    content json,
    embedding float[N],   -- N 由嵌入模型维度决定（BGE-M3 为 1024）
    created_at timestamp default now(),
);
```

```sql
create table statement (
    subject text,
    predicate text,
    object text,
    content json,
    embedding float[N],   -- N 由嵌入模型维度决定
    created_at timestamp default now(),
    tr_start timestamp,
    tr_end timestamp,
    primary key(subject, predicate, object)
);
```

### 嵌入模型配置

通过 `shelf.toml` 配置嵌入模型，支持本地 ONNX 模型和远程 API：

```toml
[embedding]
provider = "local"           # "local" (ONNX) 或 "remote" (HTTP API)
dimensions = 1024            # 向量维度
max_seq_length = 8192        # 最大序列长度
pooling = "mean"             # "mean" / "cls" / "last_token"
```

默认使用 BAAI/bge-m3 (568M params, 1024d, mean pooling)。也支持 EmbeddingGemma-300M、gte 系列、Jina v5 等模型，以及 OpenAI 兼容的远程 API。

### FTI

Hypatia 通过 sqlite 表管理 Full Text Index信息，对应的表是

```sql
CREATE TABLE docs_meta (
    id INTEGER PRIMARY KEY,
    catalog TEXT,
    key TEXT,
    content TEXT
);
create index idx_docs_catalog on docs_meta(catalog);

CREATE VIRTUAL TABLE docs_fts USING fts5(
    content,
    content='docs_meta',
    content_rowid='id'
);
```

## 命令行

代码的生成产物包含
- 提供 API 接口的 Lab
- 提供全功能操作的可执行文件
 - 可执行文件支持一次性的命令调用
 - 可执行文件支持 repl cli 模式

操作功能包括：

- 连接或者断开指定的 Shelves
- 默认情况下 Shelves 最后一级目录名就是该数据集的名字，也允许指定一个名字
- 提供 knowledge 和 statement 查询接口，查询语法遵循 [JSE](JSE_v2.0_AI_Output_Spec.md)
- 提供 knowledge 和 statement 的创建和编辑功能
- 提供 Shelves 导出能力，可以把书架复制到另一个路径

## 信息架构

## Agent 集成

该项目提供 skill和相关的功能，允许集成在 Openclaw、Claude code 等Agent中使用

Hypatia 提供一个 hypatia-query skill，支持将自然语言提问变成 knowledge 和 statement 查询，返回查询结果。

Hypatia 提供钩子，在用户输入前后，检查当前session的对话历史，做以下处理
- 在before hook中读取用户输入，检查用户的输入是否包含一个知识查询？如果有，在所有打开的shelves中查询，整合结果，插入到上下文中
- 在after hook中检查对话历史，如果一个模式反复出现（超过3次），将这个对话提取出来
- 将前述提取的对话整理为三元组，保存/合并到 default Hypatia shelves 中

允许用户明确的通过自然语言编辑和删除三元组和知识条目

Hypatia 的 JSE 指令包含：

- $knowledge 表示对 knowledge 表的查询
- $statement 表示 对 statement 表的查询
- $and 会转化成 Duckdb SQL where 条件中的 and
- $or 会转化成 SQL where 条件中的 or
- $not 会转化成 SQL where 条件中的 not
- $search 会转化成 SQLite FTS 数据中的查询，它始终在 $knowledge 或 $statement 内部使用，与 $and、$or 等指令一致，不单独传入 opts 参数，而是遵循外部的 opts 参数。$search 的产物是对应的子查询，这些子查询限制 knowledge 的 name 或 statement 的 subject、predicate、object 匹配从 FTS 搜索到的 key
- $similar 语义向量搜索，将查询文本编码为向量后通过 DuckDB cosine distance 检索最相似的条目
- $eq 对应等于
- $ne 对应不等于
- $gte 对应 大于等于
- $lte 对应 小于等于
- $gt 对应大于
- $lt 对应小于
- $contains 对应对 duckdb 数据表 content 字段的 contains 查询
- $like 对应 SQL LIKE 模式匹配
- $content 对应 content JSON 字段的键值匹配
- $triple 对三元组的位置匹配，支持通配符 $*
- $k-hop 图遍历查询，基于 statement triples 做 k 跳递归 CTE，探索实体间的关系路径
- $quote 用于封装对查询的延迟解释，对应 LISP 语言的 quote 形式
- 以上针对 duckdb 数据集的查询也包含可选的 opts 字典，其中包含
  - catalog
  - offset
  - limit
- 所有的 limit 默认 100，offset 默认为 0