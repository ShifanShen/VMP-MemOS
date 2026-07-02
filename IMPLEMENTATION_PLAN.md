# VMP-MemOS 实施计划

## 1. 项目定位

本项目目标是实现一个面向长程 LLM Agent 的记忆管理框架，暂定名为 **VMP-MemOS**。

VMP-MemOS 不做一个单纯的向量数据库，也不只是做会话摘要，而是在不同 memory backend 之上实现一层可解释的 **Memory Policy Layer**。该层通过 **Memory Policy Embedding** 判断每条记忆在当前任务中的管理状态，并决定是否执行写入、更新、合并、归档、压缩、召回、置顶、后台整理等操作。

核心思想是：

```text
content_embedding 表示这条记忆在说什么
policy_embedding 表示这条记忆现在应该怎么被管理
```

项目最终需要支持以下研究目标：

```text
1. 构建一个可运行的 Agent Memory 框架原型
2. 支持向量记忆、文件记忆、workspace 记忆等多种 backend
3. 实现可解释的 Memory Policy Embedding
4. 实现 ADD / UPDATE / MERGE / ARCHIVE / COMPRESS / RETRIEVE / PIN / DREAM 等记忆操作
5. 支持多框架 baseline 对比实验
6. 支持 LongMemEval / LoCoMo / MemoryAgentBench 子集评测
7. 输出完整实验日志、指标统计、消融实验结果
```

---

## 2. 核心问题定义

传统 Agent Memory 常见做法是：

```text
历史对话 / 工具轨迹
→ embedding
→ vector database
→ top-k 相似度召回
→ 拼进 prompt
```

该方式存在以下问题：

```text
1. 只关注语义相似度，忽略记忆是否过期
2. 新旧记忆冲突时无法自动更新
3. 记忆重复写入导致 memory growth
4. 召回内容冗余，token 成本高
5. 无法判断某条记忆过去是否真的帮助任务成功
6. 缺少可审计、可解释、可回滚的记忆操作日志
7. 难以支持跨框架统一评测
```

本项目希望解决的问题是：

```text
什么信息应该写入？
什么信息应该更新？
什么信息应该合并？
什么信息应该归档？
什么信息应该压缩？
什么信息应该召回？
什么信息应该进入后台整理？
```

---

## 3. 参考框架与吸收点

### 3.1 Text2Mem

吸收点：

```text
1. 记忆操作标准化
2. JSON IR 表达记忆操作
3. 操作前后验证
4. 可移植到不同 backend
```

本项目不直接复刻 Text2Mem，而是借鉴其 operation language 思想，定义统一 Memory Operation DSL。

---

### 3.2 Mem0

吸收点：

```text
1. 生产级 memory middleware
2. add / search / update / delete 统一 API
3. 支持长期用户记忆
4. 可接 vector / graph / database backend
5. 重视 token cost、latency、memory quality
```

本项目需要将 Mem0 作为强 baseline，并设计 `Mem0Adapter`。同时需要支持：

```text
Mem0
Mem0 + VMP Policy Layer
```

用来证明 VMP 是 backend-agnostic 的增强层。

---

### 3.3 Letta / MemGPT

吸收点：

```text
1. OS-style memory hierarchy
2. 有限 context 与外部 memory 的调度
3. git-backed memory filesystem
4. 版本化、回滚、后台整理
5. sleeptime / dreaming 式异步学习
```

本项目需要吸收其 Memory OS 思想，实现：

```text
1. memory version log
2. operation trace
3. background consolidation
4. PIN / ARCHIVE / DREAM 等操作
```

---

### 3.4 ReMe

吸收点：

```text
1. Memory as File
2. Markdown 文件作为长期记忆
3. 用户可读、可编辑、可审计
4. frontmatter + wikilinks
5. daily memory → digest memory 的整理流程
```

本项目需要实现一个 `FileMemoryBackend`，支持 Markdown workspace。

---

### 3.5 memU

吸收点：

```text
1. workspace runtime 思想
2. INDEX.md / MEMORY.md / SKILL.md 三层结构
3. 将任务轨迹沉淀为可复用 skill
4. 支持 proactive / background memory agent
```

本项目需要支持将 episodic traces 整理为 procedural skill memory。

---

## 4. 总体架构

项目架构如下：

```text
User / Tool / Environment Event
        ↓
Event Collector
        ↓
Memory Candidate Extractor
        ↓
Typed Memory Workspace
        ↓
Content Embedding Builder
        ↓
Policy Feature Builder
        ↓
Memory Policy Encoder
        ↓
Policy Controller
        ↓
Memory Operation Executor
        ↓
Backend Adapters
        ↓
Vector DB / Markdown Files / SQLite / Mem0 / ReMe-style Workspace
```

---

## 5. 模块划分

### 5.1 Event Collector

负责接收来自 agent 的所有事件。

事件类型包括：

```text
user_message
assistant_message
tool_call
tool_result
task_result
user_correction
file_upload
benchmark_sample
system_feedback
```

事件统一结构：

```json
{
  "event_id": "evt_001",
  "session_id": "sess_001",
  "task_id": "task_001",
  "event_type": "user_message",
  "content": "我现在更想主攻 Agent 开发，不想 all in Java。",
  "metadata": {
    "timestamp": "2026-06-23T10:00:00+08:00",
    "source": "conversation",
    "user_id": "default_user"
  }
}
```

---

### 5.2 Memory Candidate Extractor

负责从事件中抽取候选记忆。

候选记忆类型：

```text
semantic      稳定事实、用户偏好、项目背景
episodic      具体任务事件、工具调用轨迹
procedural    可复用流程、debug 经验、操作策略
reflective    成功或失败复盘
resource      文档、代码、网页、图片等外部资源
```

候选记忆结构：

```json
{
  "candidate_id": "cand_001",
  "source_event_id": "evt_001",
  "memory_type": "semantic",
  "content": "用户当前主要求职和研究方向是 Agent 开发，不想 all in Java。",
  "scope": "career/agent-dev",
  "tags": ["career", "agent", "java"],
  "confidence": 0.92,
  "importance": 0.88
}
```

第一阶段可以用规则 + LLM prompt 抽取候选记忆。后续可以加入小模型抽取。

---

### 5.3 Typed Memory Workspace

记忆需要按类型和作用域组织，而不是全部扔进同一个向量库。

推荐 workspace 结构：

```text
memory_workspace/
├── INDEX.md
├── MEMORY.md
├── projects/
│   ├── fancode.md
│   ├── ai-chef.md
│   └── agent-memory.md
├── skills/
│   ├── debug-fastapi/SKILL.md
│   ├── paper-review/SKILL.md
│   └── benchmark-evaluation/SKILL.md
├── episodes/
│   ├── 2026-06-23-task-trace.md
│   └── 2026-06-24-benchmark-trace.md
├── resources/
│   └── papers.md
├── archive/
│   └── stale-memory.md
└── logs/
    ├── operations.jsonl
    ├── retrievals.jsonl
    └── evaluations.jsonl
```

---

### 5.4 Memory Item Schema

每条正式记忆统一存储为 `MemoryItem`。

```json
{
  "id": "mem_001",
  "type": "semantic",
  "scope": "career/agent-dev",
  "content": "用户当前主要求职方向是 Agent 开发和 LLM 应用开发。",
  "summary": "用户职业方向偏 Agent / LLM 应用。",
  "source": {
    "event_id": "evt_001",
    "source_type": "conversation"
  },
  "content_embedding": [],
  "policy_embedding": [],
  "features": {
    "semantic_relevance": 0.0,
    "importance": 0.88,
    "confidence": 0.92,
    "recency": 1.0,
    "stability": 0.7,
    "novelty": 0.9,
    "redundancy": 0.1,
    "contradiction": 0.0,
    "staleness": 0.0,
    "access_frequency": 0.0,
    "success_contribution": 0.0,
    "failure_contribution": 0.0,
    "token_cost": 0.03,
    "scope_match": 1.0,
    "actionability": 0.8,
    "privacy_risk": 0.1
  },
  "metadata": {
    "created_at": "2026-06-23T10:00:00+08:00",
    "last_accessed_at": null,
    "access_count": 0,
    "version": 1,
    "status": "active"
  },
  "links": {
    "related": [],
    "supersedes": [],
    "superseded_by": []
  }
}
```

---

## 6. Memory Policy Embedding 设计

### 6.1 Policy Feature Vector

第一版使用可解释 policy feature vector。

```text
p_i = [
  semantic_relevance,
  importance,
  confidence,
  recency,
  stability,
  novelty,
  redundancy,
  contradiction,
  staleness,
  access_frequency,
  success_contribution,
  failure_contribution,
  token_cost,
  scope_match,
  actionability,
  privacy_risk
]
```

其中：

```text
semantic_relevance：当前 query 与 memory 的语义相关性
importance：长期保存价值
confidence：记忆可靠性
recency：时间新近度
stability：是否稳定，不容易变化
novelty：是否是新信息
redundancy：是否与已有记忆重复
contradiction：是否与已有记忆冲突
staleness：是否过期
access_frequency：历史使用频率
success_contribution：过去是否帮助任务成功
failure_contribution：过去是否导致错误
token_cost：召回成本
scope_match：是否匹配当前用户 / 项目 / 任务
actionability：是否能指导后续行动
privacy_risk：是否存在敏感或不适合长期保存的信息
```

---

### 6.2 特征计算方式

#### semantic_relevance

```text
semantic_relevance = cosine(query_embedding, memory_embedding)
```

归一化到 `[0, 1]`。

#### recency

```text
recency = 0.5 ^ (time_gap / half_life(memory_type))
```

不同 memory type 使用不同 half-life：

```text
semantic: 90-180 days
episodic: 7-30 days
procedural: 180+ days
reflective: 90-180 days
resource: depends on source freshness
```

#### redundancy

```text
redundancy = max cosine(candidate_embedding, existing_memory_embedding)
novelty = 1 - redundancy
```

#### contradiction

第一阶段用 LLM-as-judge 或 NLI 模型判断：

```text
old_memory + new_candidate → contradiction_score
```

输出范围 `[0, 1]`。

#### token_cost

```text
token_cost = min(memory_tokens / budget_tokens, 1.0)
```

#### success_contribution

每次任务结束后根据 task result 更新：

```text
success_contribution_{t+1}
= 0.9 * success_contribution_t + 0.1 * reward_t
```

其中：

```text
reward_t = 1      记忆被使用且任务成功
reward_t = 0      不确定是否有贡献
reward_t = -1     记忆被使用且导致错误
```

---

## 7. Memory Operation DSL

定义统一操作集合。

```text
ADD        写入新记忆
UPDATE     更新已有记忆
MERGE      合并相似记忆
SPLIT      拆分过大的记忆
DELETE     删除错误记忆，第一阶段不物理删除
ARCHIVE    归档低频或过期记忆
EXPIRE     标记过期
LOCK       锁定高置信记忆
PROMOTE    提升为长期核心记忆
DEMOTE     降级为低优先级记忆
COMPRESS   压缩长记忆
RETRIEVE   召回记忆
PIN        固定进入上下文
DREAM      后台整理、反思、重组
VERIFY     验证记忆是否仍成立
IGNORE     不处理
```

统一 operation JSON：

```json
{
  "op_id": "op_001",
  "op": "UPDATE",
  "target_memory_id": "mem_001",
  "source_event_id": "evt_002",
  "reason": "New user preference supersedes older career preference.",
  "policy_score": 0.82,
  "confidence": 0.91,
  "scope": "career/agent-dev",
  "backend": "file",
  "timestamp": "2026-06-23T10:10:00+08:00"
}
```

所有 operation 都必须写入 `logs/operations.jsonl`。

---

## 8. Policy Controller 第一版规则

### 8.1 WriteScore

```text
WriteScore =
0.30 * importance
+ 0.25 * novelty
+ 0.20 * confidence
+ 0.15 * actionability
+ 0.10 * scope_match
- 0.20 * redundancy
- 0.10 * privacy_risk
```

规则：

```text
if WriteScore > 0.65:
    ADD
else:
    IGNORE
```

---

### 8.2 RetrieveScore

```text
RetrieveScore =
0.30 * semantic_relevance
+ 0.20 * importance
+ 0.15 * scope_match
+ 0.10 * confidence
+ 0.10 * success_contribution
+ 0.10 * recency
- 0.15 * contradiction
- 0.05 * redundancy
- 0.05 * token_cost
```

召回策略：

```text
1. 先取 top-50 candidate
2. 计算 RetrieveScore
3. 过滤 archived / expired memory
4. 去重，同一 cluster 最多保留 1-2 条
5. 在 token budget 内选择最终 memory
6. 写入 retrieval log
```

---

### 8.3 UpdateScore

```text
UpdateScore =
0.30 * semantic_similarity_to_existing
+ 0.30 * contradiction
+ 0.20 * recency
+ 0.15 * source_priority
+ 0.05 * confidence
```

规则：

```text
if semantic_similarity_to_existing > 0.70 and contradiction > 0.45:
    UPDATE
```

更新时保留版本，不直接覆盖历史。

---

### 8.4 MergeScore

```text
MergeScore =
0.35 * semantic_similarity
+ 0.30 * redundancy
+ 0.20 * scope_match
+ 0.15 * low_conflict
```

其中：

```text
low_conflict = 1 - contradiction
```

规则：

```text
if MergeScore > 0.75:
    MERGE
```

---

### 8.5 ArchiveScore

第一阶段不做物理删除，只做 archive。

```text
ArchiveScore =
0.25 * staleness
+ 0.25 * redundancy
+ 0.20 * negative_contribution
+ 0.15 * low_importance
+ 0.15 * superseded
```

其中：

```text
low_importance = 1 - importance
```

规则：

```text
if ArchiveScore > 0.80:
    ARCHIVE
```

---

### 8.6 CompressScore

```text
CompressScore =
0.30 * token_cost
+ 0.25 * access_frequency
+ 0.20 * information_density
+ 0.15 * actionability
+ 0.10 * scope_match
```

规则：

```text
if CompressScore > 0.65:
    COMPRESS
```

---

## 9. Backend 设计

### 9.1 BaseMemoryBackend

定义统一接口：

```python
class BaseMemoryBackend:
    def add(self, memory_item): ...
    def update(self, memory_id, patch): ...
    def get(self, memory_id): ...
    def search(self, query, top_k=20, filters=None): ...
    def list(self, filters=None): ...
    def archive(self, memory_id, reason=None): ...
    def delete(self, memory_id): ...
    def persist(self): ...
```

---

### 9.2 VectorMemoryBackend

第一阶段使用本地向量库。

推荐实现：

```text
Chroma 或 Qdrant
```

最小实现可以先用：

```text
SQLite + numpy embeddings
```

---

### 9.3 FileMemoryBackend

使用 Markdown 文件保存记忆。

每个 Markdown 文件包含 frontmatter：

```markdown
---
id: mem_001
type: semantic
scope: career/agent-dev
status: active
version: 1
created_at: 2026-06-23T10:00:00+08:00
importance: 0.88
confidence: 0.92
---

用户当前主要求职方向是 Agent 开发和 LLM 应用开发。
```

---

### 9.4 HybridMemoryBackend

组合：

```text
Markdown file 保存可读记忆
SQLite 保存 metadata
Vector DB 保存 embedding
JSONL 保存 operation traces
```

---

### 9.5 Future Adapters

预留：

```text
Mem0Adapter
LettaAdapter
ReMeAdapter
MemUAdapter
GraphMemoryBackend
```

第一阶段不强制全部接入，但接口必须预留。

---

## 10. Benchmark 设计

### 10.1 Baselines

至少实现以下 baseline：

```text
NoMemory
FullContext
SummaryMemory
NaiveVectorRAG
VectorRAG + Recency
VectorRAG + Importance
VMP-Rule
VMP-Tuned
VMP-Learned
```

后续加入：

```text
Mem0
Mem0 + VMP
ReMe-style FileMemory
ReMe-style FileMemory + VMP
memU-style Workspace
memU-style Workspace + VMP
```

---

### 10.2 推荐 benchmark

第一阶段使用自建小型 benchmark。

样本类型：

```text
1. 用户偏好更新
2. 新旧事实冲突
3. 项目状态变化
4. 多 session 信息整合
5. 旧记忆过期
6. 重复记忆合并
7. 长工具日志压缩
8. 失败经验沉淀为 procedural memory
```

第二阶段接入公开 benchmark 子集：

```text
LongMemEval
LoCoMo
MemoryAgentBench
```

---

## 11. 评测指标

### 11.1 任务效果

```text
QA Accuracy
F1 / EM
LLM-as-Judge Score
Task Success Rate
State Assertion Pass Rate
```

---

### 11.2 召回质量

```text
Recall@k
Precision@k
MRR
NDCG
Evidence Precision
Evidence Sufficiency
Temporal Evidence Accuracy
```

---

### 11.3 写入质量

```text
Write Precision
Write Recall
Useful Memory Ratio
Noise Memory Rate
Memory Admission Accuracy
```

---

### 11.4 更新与冲突处理

```text
Knowledge Update Accuracy
Conflict Resolution Accuracy
Superseded Memory Suppression Rate
Contradictory Retrieval Rate
Preference Update Accuracy
Version Consistency
```

---

### 11.5 遗忘与归档

```text
Selective Forgetting Accuracy
Archive Precision
Expire Accuracy
Stale Memory Usage Rate
Obsolete Memory Retention Rate
```

---

### 11.6 合并与压缩

```text
Merge Precision
Merge Recall
Compression Faithfulness
Compression Coverage
Redundancy Rate
Memory Growth Rate
Information Loss Rate
```

---

### 11.7 成本与效率

```text
Token Cost per Query
Retrieved Token Count
Memory Write Token Cost
LLM Calls per Task
p50 Latency
p95 Latency
Storage Size
Indexing Cost
Consolidation Cost
Cost per Successful Task
```

---

### 11.8 可解释性与可审计性

```text
Operation Log Completeness
Operation Explanation Quality
Version Rollback Success
Human Edit Propagation Accuracy
Memory Inspectability Score
```

---

## 12. 消融实验

需要支持以下 ablation：

```text
VMP-full
VMP w/o recency
VMP w/o contradiction
VMP w/o redundancy
VMP w/o success_contribution
VMP w/o token_cost
VMP w/o background_dream
VMP w/o file_workspace
VMP w/o versioning
```

预期观察：

```text
去掉 contradiction：conflict resolution 下降
去掉 redundancy：memory growth 上升
去掉 success_contribution：task success 下降
去掉 recency：temporal reasoning 下降
去掉 token_cost：召回成本上升
去掉 background_dream：procedural memory 质量下降
```

---

## 13. 推荐项目目录

```text
vmp-memos/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── default.yaml
│   ├── benchmark.yaml
│   └── ablation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/
├── memory_workspace/
│   ├── INDEX.md
│   ├── MEMORY.md
│   ├── projects/
│   ├── skills/
│   ├── episodes/
│   ├── resources/
│   ├── archive/
│   └── logs/
├── src/
│   └── vmp_memos/
│       ├── __init__.py
│       ├── schemas/
│       │   ├── event.py
│       │   ├── memory.py
│       │   ├── operation.py
│       │   └── benchmark.py
│       ├── extraction/
│       │   ├── candidate_extractor.py
│       │   └── prompts.py
│       ├── embeddings/
│       │   ├── base.py
│       │   ├── sentence_transformer.py
│       │   └── openai_embedder.py
│       ├── policy/
│       │   ├── features.py
│       │   ├── encoder.py
│       │   ├── controller.py
│       │   ├── rules.py
│       │   └── learned.py
│       ├── operations/
│       │   ├── executor.py
│       │   ├── validators.py
│       │   └── logger.py
│       ├── backends/
│       │   ├── base.py
│       │   ├── vector_backend.py
│       │   ├── file_backend.py
│       │   ├── hybrid_backend.py
│       │   └── adapters/
│       │       ├── mem0_adapter.py
│       │       ├── reme_adapter.py
│       │       └── memu_adapter.py
│       ├── retrieval/
│       │   ├── retriever.py
│       │   ├── reranker.py
│       │   └── budget.py
│       ├── consolidation/
│       │   ├── dreamer.py
│       │   ├── compressor.py
│       │   └── merger.py
│       ├── benchmark/
│       │   ├── runner.py
│       │   ├── datasets.py
│       │   ├── baselines.py
│       │   ├── metrics.py
│       │   └── reports.py
│       └── utils/
│           ├── tokens.py
│           ├── time.py
│           ├── ids.py
│           └── logging.py
├── scripts/
│   ├── init_workspace.py
│   ├── run_demo.py
│   ├── run_benchmark.py
│   ├── run_ablation.py
│   └── export_report.py
├── tests/
│   ├── test_policy_features.py
│   ├── test_policy_controller.py
│   ├── test_operations.py
│   ├── test_file_backend.py
│   ├── test_vector_backend.py
│   └── test_benchmark_metrics.py
└── outputs/
    ├── runs/
    ├── reports/
    └── figures/
```

---

## 14. 开发阶段规划

### Phase 0：项目初始化

目标：

```text
1. 创建 Python 项目结构
2. 使用 pyproject.toml 管理依赖
3. 配置 ruff / pytest / mypy
4. 创建默认 config
5. 创建 memory_workspace 默认目录
```

验收标准：

```text
python -m pytest 能正常运行
python scripts/init_workspace.py 能创建 workspace
```

---

### Phase 1：核心 schema

实现：

```text
Event
MemoryCandidate
MemoryItem
MemoryOperation
PolicyFeatures
BenchmarkSample
BenchmarkResult
```

要求：

```text
1. 使用 pydantic
2. 所有 schema 可序列化为 JSON
3. 所有对象有稳定 id
4. 支持 JSONL 日志写入
```

验收标准：

```text
pytest tests/test_operations.py
pytest tests/test_policy_features.py
```

---

### Phase 2：FileMemoryBackend

实现：

```text
1. Markdown frontmatter 读写
2. memory item add / update / archive / list
3. INDEX.md 自动更新
4. operations.jsonl 写入
```

验收标准：

```text
python scripts/run_demo.py --backend file
```

运行后能看到：

```text
memory_workspace/MEMORY.md
memory_workspace/logs/operations.jsonl
```

---

### Phase 3：Embedding 与 Vector Backend

实现：

```text
1. BaseEmbedder
2. SentenceTransformerEmbedder
3. VectorMemoryBackend
4. cosine search
5. embedding cache
```

第一版可以用 numpy 实现本地向量检索，后续再换 Qdrant / Chroma。

验收标准：

```text
python scripts/run_demo.py --backend vector
```

能够写入记忆并根据 query 召回。

---

### Phase 4：Policy Feature Builder

实现所有 policy features：

```text
semantic_relevance
importance
confidence
recency
stability
novelty
redundancy
contradiction
staleness
access_frequency
success_contribution
failure_contribution
token_cost
scope_match
actionability
privacy_risk
```

第一版允许部分特征使用规则模拟。

验收标准：

```text
pytest tests/test_policy_features.py
```

---

### Phase 5：Rule-based Policy Controller

实现：

```text
WriteScore
RetrieveScore
UpdateScore
MergeScore
ArchiveScore
CompressScore
```

输出 operation decision：

```json
{
  "op": "ADD",
  "score": 0.78,
  "reason": "High importance, high novelty, low redundancy."
}
```

验收标准：

```text
pytest tests/test_policy_controller.py
```

---

### Phase 6：Operation Executor

实现：

```text
ADD
UPDATE
MERGE
ARCHIVE
COMPRESS
RETRIEVE
PIN
DREAM
IGNORE
```

第一版可以重点实现：

```text
ADD
UPDATE
MERGE
ARCHIVE
RETRIEVE
IGNORE
```

验收标准：

```text
python scripts/run_demo.py --backend hybrid
```

能够完成：

```text
1. 新记忆写入
2. 重复记忆合并
3. 冲突记忆更新
4. 过期记忆归档
5. 根据 query 召回
```

---

### Phase 7：Benchmark Runner

实现：

```text
1. benchmark sample loader
2. baseline runner
3. VMP runner
4. metrics calculator
5. result JSONL 保存
6. markdown report 导出
```

先实现自建 benchmark：

```text
data/benchmarks/memory_policy_toy.jsonl
```

样本字段：

```json
{
  "sample_id": "case_001",
  "events": [],
  "query": "用户现在的主要求职方向是什么？",
  "gold_answer": "Agent 开发和 LLM 应用开发",
  "gold_memory_ids": [],
  "expected_operations": ["UPDATE", "RETRIEVE"],
  "metadata": {
    "task_type": "preference_update"
  }
}
```

验收标准：

```text
python scripts/run_benchmark.py --config configs/benchmark.yaml
```

输出：

```text
outputs/runs/{run_id}/results.jsonl
outputs/reports/{run_id}.md
```

---

### Phase 8：Baselines

实现 baseline：

```text
NoMemoryBaseline
FullContextBaseline
SummaryMemoryBaseline
NaiveVectorRAGBaseline
VectorRAGRecencyBaseline
VectorRAGImportanceBaseline
VMPRuleBaseline
```

验收标准：

```text
python scripts/run_benchmark.py --baselines no_memory,vector_rag,vmp_rule
```

输出对比表：

```text
Accuracy
Evidence Precision
Memory Growth
Token Cost
Conflict Retrieval Rate
Stale Memory Usage Rate
```

---

### Phase 9：Learned Policy

实现：

```text
1. 从 benchmark 和 operation logs 构造训练数据
2. Logistic Regression / LightGBM / MLP policy classifier
3. 输入 policy feature vector
4. 输出 operation probabilities
```

第一版优先使用：

```text
Logistic Regression
```

因为可解释。

验收标准：

```text
python scripts/train_policy.py
python scripts/run_benchmark.py --policy learned
```

---

### Phase 10：消融实验与报告

实现：

```text
run_ablation.py
```

支持：

```text
--disable recency
--disable contradiction
--disable redundancy
--disable success_contribution
--disable token_cost
```

输出：

```text
outputs/reports/ablation.md
```

报告内容：

```text
1. 实验设置
2. baseline 对比
3. ablation 对比
4. 指标表格
5. 错误案例分析
6. memory operation 示例
```

---

## 15. Demo 场景

需要提供一个最小 demo，展示 VMP 的价值。

### 场景：职业方向更新

事件 1：

```text
用户之前考虑转 Java 后端。
```

系统写入：

```text
mem_old: 用户考虑 Java 后端方向。
```

事件 2：

```text
用户现在不想 all in Java，而是主攻 Agent 开发和 LLM 应用开发。
```

系统应该执行：

```text
UPDATE mem_old
PROMOTE new agent-dev preference
DEMOTE old Java-backend preference
```

查询：

```text
用户现在主要求职方向是什么？
```

正确召回：

```text
用户当前主要求职方向是 Agent 开发和 LLM 应用开发。
```

错误行为：

```text
召回旧 Java 后端偏好，并错误回答用户主攻 Java。
```

---

## 16. 代码风格要求

```text
1. Python 3.11+
2. 使用 type hints
3. 使用 pydantic 定义 schema
4. 所有核心逻辑必须有单元测试
5. 所有 operation 必须写入 JSONL log
6. 所有 benchmark 结果必须可复现
7. 不要把 API key 写进代码
8. 配置统一放在 configs/
9. 日志统一放在 outputs/runs/
10. workspace 数据统一放在 memory_workspace/
```

---

## 17. 依赖建议

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "numpy",
    "scikit-learn",
    "sentence-transformers",
    "pyyaml",
    "rich",
    "typer",
    "tqdm",
    "python-frontmatter",
    "markdown",
    "orjson",
    "pytest",
    "ruff",
    "mypy"
]
```

可选依赖：

```text
qdrant-client
chromadb
mem0ai
langgraph
langchain
openai
litellm
lightgbm
```

---

## 18. 研究贡献预期

项目完成后，论文贡献可以写成：

```text
1. 提出 Memory Policy Embedding，区别于传统 content embedding，用于表示记忆的生命周期管理状态。
2. 提出 backend-agnostic Memory Policy Layer，可以接入 vector DB、file memory、workspace memory 和现有 memory frameworks。
3. 统一建模 ADD、UPDATE、MERGE、ARCHIVE、COMPRESS、RETRIEVE 等记忆操作。
4. 在长期记忆 benchmark 和自建 memory lifecycle benchmark 上验证该策略能够减少冲突召回、冗余记忆和 token 成本，同时提升任务成功率。
```

---

## 19. 第一版 MVP 范围

第一版必须完成：

```text
1. FileMemoryBackend
2. VectorMemoryBackend
3. HybridMemoryBackend
4. MemoryItem schema
5. Policy feature builder
6. Rule-based Policy Controller
7. ADD / UPDATE / MERGE / ARCHIVE / RETRIEVE / IGNORE
8. operation logs
9. 自建 benchmark
10. NoMemory / NaiveVectorRAG / VMPRule 三个 baseline
11. benchmark report
```

第一版暂不做：

```text
1. 大规模 RL
2. 全量 LongMemEval
3. 全量 Mem0 adapter
4. 复杂图数据库
5. 多用户权限系统
6. Web UI
```

---

## 20. 最终验收命令

项目第一阶段完成后，应支持以下命令：

```bash
python scripts/init_workspace.py
python scripts/run_demo.py --backend hybrid
python scripts/run_benchmark.py --config configs/benchmark.yaml
python scripts/run_ablation.py --config configs/ablation.yaml
pytest
```

所有命令执行后，应生成：

```text
memory_workspace/logs/operations.jsonl
memory_workspace/logs/retrievals.jsonl
outputs/runs/{run_id}/results.jsonl
outputs/reports/{run_id}.md
outputs/reports/ablation.md
```
