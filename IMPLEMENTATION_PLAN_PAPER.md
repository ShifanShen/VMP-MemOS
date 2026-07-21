# VMP-MemOS 论文实验实施计划（改进版）

## 0. 总目标

VMP-MemOS 的论文目标是评估一个面向长程 LLM Agent 的可解释记忆管理框架。核心贡献不是再造向量数据库，也不是单纯会话摘要，而是在 memory backend 之上加入 **Memory Policy Layer**，用可解释的 **Memory Policy Features / Policy Embedding** 决定记忆的生命周期操作：

```text
ADD / UPDATE / MERGE / ARCHIVE / COMPRESS / RETRIEVE / IGNORE
```

核心区分：

```text
content_embedding：这条记忆在说什么
policy_embedding：这条记忆现在应该怎么被管理
```

论文要证明：

```text
1. VMP-MemOS 比普通向量检索更准确地召回长期记忆证据。
2. VMP-MemOS 能减少过期、冲突、冗余记忆的错误召回。
3. VMP-MemOS 能通过生命周期操作控制 memory growth 和 token cost。
4. VMP-MemOS 是 backend-agnostic policy layer，可以接到不同 memory backend 上。
5. VMP-MemOS 的提升来自 policy decision，而不是不公平地换用了更强 LLM 或 embedding。
```

## 1. 当前项目基础

当前 MVP 已完成：

```text
1. Python src/ 项目结构
2. Pydantic schema
3. FileMemoryBackend
4. VectorMemoryBackend
5. HybridMemoryBackend
6. PolicyFeatureBuilder
7. RuleBasedPolicyController
8. MemoryOperationExecutor
9. toy benchmark
10. NoMemory / FullContext / SummaryMemory / NaiveVectorRAG / Recency / Importance / VMPRule baseline
11. learned policy 第一版
12. toy ablation runner
13. vLLM OpenAI-compatible client
14. LLM memory candidate extractor smoke
```

从本计划开始，重点从 toy benchmark 转向 **公开权威 benchmark + 公平对比实验**。

## 2. 主 benchmark 选择

主实验使用：

```text
LongMemEval-cleaned
主文件：longmemeval_s_cleaned.json
```

选择理由：

```text
1. LongMemEval 是 ICLR 2025 长期对话记忆 benchmark。
2. 数据包含 500 个高质量问题。
3. 覆盖 information extraction、multi-session reasoning、knowledge update、temporal reasoning、abstention。
4. LongMemEval-S 每题约 40 个 history sessions，规模适合 4090D 单卡做多方法实验。
5. cleaned 版本已被官方推荐替代原始版本，减少 noisy sessions 对答案正确性的干扰。
```

数据使用顺序：

```text
1. longmemeval_oracle.json
   用于 smoke test、reader 上限测试、QA prompt 检查。

2. longmemeval_s_cleaned.json
   用于主实验，包括 retrieval、QA、ablation、case study。

3. longmemeval_m_cleaned.json
   只作为后续压力测试，不作为当前论文初稿主实验。
```

暂不使用 LongMemEval-V2 作为主 benchmark。LongMemEval-V2 更偏 web-agent / enterprise-agent trajectory memory，可作为 future work。

## 3. 最高优先级公平性约束

所有主实验必须遵守以下公平性约束。

### 3.1 所有 LLM 调用统一使用本地 vLLM

主实验中任何需要 LLM 的环节都必须走同一个本地 vLLM OpenAI-compatible server：

```text
QA Reader
Memory candidate extraction（如果启用）
Summary / compression（如果启用）
Importance / contradiction judge（如果启用）
Framework adapter 内部 LLM 调用（如果可控）
LLM-as-Judge（如果作为补充实验启用）
```

不允许主实验中出现：

```text
VMP 用 Qwen，本地 vLLM
Mem0 用 OpenAI
Letta 用 Claude
ReMe 用另一套 API
```

如果某个官方框架无法强制使用同一个本地 vLLM，则：

```text
1. 不进入主性能表。
2. 可以进入 secondary comparison。
3. 必须在 fairness table 中标记为 partially controlled 或 unavailable。
```

### 3.2 固定本地模型

主实验默认模型：

```text
Reader LLM = Qwen/Qwen2.5-7B-Instruct via vLLM
temperature = 0.0
```

可选补充实验：

```text
Qwen/Qwen2.5-14B-Instruct via vLLM
```

但是主表只报告一个固定 reader model 的结果，避免模型大小混入方法比较。

### 3.3 固定 embedding

主实验 embedding 固定：

```text
BAAI/bge-m3
```

如果 4090D 上资源不足，可先用轻量模型 smoke：

```text
sentence-transformers/all-MiniLM-L6-v2
```

但正式论文表格必须明确报告 embedding model。所有主表方法必须使用同一 embedding。

### 3.4 固定 ingestion granularity

主 retrieval 实验第一版不启用 LLM extraction，所有方法吃同样的原始 chunk：

```text
session-level chunk
或
turn-level chunk
```

推荐第一轮：

```text
session-level chunk
```

理由：

```text
1. LongMemEval 官方 evidence 是 session-level answer_session_ids。
2. session-level 更容易先打通主实验。
3. 后续再扩展 turn-level，用 has_answer 做更细 evidence evaluation。
```

LLM extraction 只作为 secondary experiment：

```text
VMP-Rule raw chunk
VMP-Rule + LLM extraction
```

这样可以区分“policy 贡献”和“LLM extraction 贡献”。

### 3.5 固定 top-k 与 prompt

主实验固定：

```text
top_k = 5
reader prompt = 同一模板
max_reader_tokens = 512
temperature = 0.0
```

所有方法必须输出统一格式的 retrieved evidence，才能计算 retrieval metrics 和 token cost。

## 4. 官方框架优先的公平性分级

每个方法必须标记 fairness level。

总原则：

```text
1. 对比实验优先接入官方开源框架，而不是先做 style reimplementation。
2. 只有当官方实现无法在本实验约束下公平控制时，才做 style fallback。
3. style fallback 必须明确写成 “-style”，不能冒充 official framework。
4. 主性能表只放 Level 1 或接近 Level 1 的方法。
5. 所有官方框架都必须先经过 controllability audit，再决定进入主表、附录或标记 unavailable。
```

### Level 1：Official OSS / Fully Controlled

满足：

```text
1. 同一 vLLM reader
2. 同一 embedding model
3. 同一 ingestion granularity
4. 同一 top-k
5. 可导出 retrieved evidence
6. 可统计 token cost / latency / memory size
7. 官方框架内部如需 LLM，也能强制走同一 vLLM endpoint
8. 官方框架内部如需 embedding，也能强制走同一 embedding provider
```

只有 Level 1 或接近 Level 1 的方法可以进入主性能表。

### Level 2：Official OSS / Partially Controlled

使用官方开源实现，但满足部分条件，不能完全控制内部 LLM、embedding、检索 evidence 或统计项。

用途：

```text
secondary comparison
appendix table
availability table
```

### Level 3：Style Reimplementation / Fallback Only

不是官方框架，而是复现其设计思想。只在以下情况使用：

```text
1. 官方实现不可用、无法安装、无法配置到本地 vLLM，或无法导出 evidence。
2. 官方实现依赖闭源服务，导致主实验不可复现。
3. 论文需要展示该思想类别的参考表现，但不能声称是官方框架结果。
```

命名必须使用：

```text
Mem0-style
Letta-style
LangMem-style
Graphiti-style
ReMe-style
memU-style
```

不得写成 official Mem0 / official Letta / official LangMem / official Graphiti / official ReMe / official memU 结果。

### 4.1 Framework controllability audit

每个外部框架必须导出一条审计记录：

```text
framework_name
official_repo
version_or_commit
license
install_status
local_vllm_supported
local_embedding_supported
evidence_export_supported
reset_workspace_supported
token_latency_stats_supported
fairness_level
main_table_eligible
notes
```

这张表不是装饰品，而是论文公平性的证据。若某框架不能进入主表，也要在这里解释原因。

## 5. 改进后的实施顺序

不要一次性实现全部论文蓝图。按以下阶段推进。

### P1：LongMemEval 数据接入

实现：

```text
scripts/download_longmemeval.py
scripts/inspect_longmemeval.py
src/vmp_memos/longmemeval/loader.py
src/vmp_memos/longmemeval/schema.py
src/vmp_memos/longmemeval/converter.py
```

验收：

```bash
python scripts/download_longmemeval.py --target data/longmemeval
python scripts/inspect_longmemeval.py --data data/longmemeval/longmemeval_s_cleaned.json --limit 3
```

### P2：统一实验 schema 与 adapter 基类

实现：

```text
RetrievedMemory
BaseMemoryFrameworkAdapter
FrameworkRegistry
LongMemEvalRunConfig
```

注意：新增 `EventType.BENCHMARK_QUERY`，或者 query 不转 Event。推荐新增。

### P3：主 retrieval baselines

第一批先实现完全可控的基础方法，用来打通 LongMemEval runner、metrics、表格与 QA pipeline：

```text
EmptyRetrieval / NoMemory
BM25
NaiveVectorRAG
VectorRAG + Recency
VectorRAG + Importance
VMP-Rule
```

这一批不是最终论文全部对比，而是统一实验平台的地基。外部官方框架不要在 runner 未稳定前强行接入。

### P3.5：官方 OSS framework audit

先为候选框架建立可控性审计，不急着写完整 adapter：

```text
Mem0 OSS
Letta OSS
LangMem OSS
Zep / Graphiti OSS
Text2Mem official implementation（如可用）
ReMe official implementation（如可用）
memU official implementation（如可用）
其他后续确认的开源 memory framework
```

审计输出：

```text
outputs/longmemeval/tables/table6_fairness.csv
outputs/longmemeval/audit/framework_controllability.json
```

审计结论决定后续：

```text
Level 1：实现 official adapter，允许进入主表。
Level 2：实现 official adapter，但只进入 secondary / appendix。
Level 3：如果论文确实需要，才实现 -style fallback。
unavailable：记录原因，不阻塞主实验。
```

### P4：Retrieval metrics 与 runner

实现：

```text
scripts/run_longmemeval_retrieval.py
src/vmp_memos/evaluation/retrieval_metrics.py
src/vmp_memos/longmemeval/retrieval_runner.py
```

主指标：

```text
Official Session Recall-All@5 / @10
Official Session Recall-Any@5 / @10
Supplementary Fractional Recall@1 / @3 / @5 / @10
Session Precision@5
Session MRR
Official Session NDCG@5 / @10
Retrieved Tokens
Retrieval Latency
Memory Count
Memory Storage Size
```

abstention 处理：

```text
Retrieval metrics 默认跳过 abstention questions。
QA metrics 单独计算 abstention accuracy。
官方 `_abs` question_id 是 abstention 的权威判定；全量 500 题中应有 30 条。
```

### P5：表格导出

实现：

```text
scripts/export_longmemeval_tables.py
outputs/longmemeval/tables/table1_retrieval_overall.csv
outputs/longmemeval/tables/table2_by_question_type.csv
```

导出格式：

```text
CSV
Markdown
LaTeX
```

### P6：vLLM QA reader

复用当前已有：

```text
src/vmp_memos/llm/
scripts/serve_vllm.sh
scripts/run_llm_smoke.py
```

不要再另起一套重复的 `readers/`，除非只做薄 wrapper。

新增：

```text
src/vmp_memos/llm/reader.py
src/vmp_memos/longmemeval/qa_runner.py
scripts/run_longmemeval_qa.py
```

QA prompt 固定：

```text
You are answering a LongMemEval question using retrieved long-term memory.

Question date:
{question_date}

Question:
{question}

Retrieved memory:
{memory_context}

Instructions:
- Answer using only the retrieved memory.
- Prefer newer evidence when memories conflict.
- If the answer is not supported by the retrieved memory, say "I don't know".
- Keep the answer concise.
```

### P7：QA metrics 与官方格式导出

主 QA metrics 使用本地可复现指标：

```text
Normalized Exact Match
Token F1
Contains Answer
Abstention Accuracy
Reader Input Tokens
Reader Output Tokens
End-to-End Latency
```

同时导出官方兼容 hypothesis：

```json
{"question_id": "...", "hypothesis": "..."}
```

官方 LongMemEval GPT-based evaluator 只作为 optional，不作为本地 pipeline 硬依赖。

### P8：VMP-Tuned

必须先做固定 dev/test split，避免调参污染测试集。

当前实现采用 VMP-v4 稳健安全链路：

```text
Dense Top-10 safety set
-> cached query-independent policy features
-> temporal-intent gated policy features
-> guarded Top-5 policy reranking (at least four dense-head items)
-> cached non-destructive lifecycle annotations
```

调参的第一个 trial 固定为纯 dense，所有 trial 同时检查确定性 fold 稳定性、
question-type 宏平均与最差类型召回；最终选择首先最大化官方
`Recall-All@5`。`ARCHIVE/MERGE` 在 retrieval 阶段不得删除 source session；
只有 Dev `Recall-All@5 >= 0.90` 且相对 dense 至少提升 0.02，才能运行 Test。

推荐：

```text
seed = 42
dev = 100 questions
test = 400 questions
split file = outputs/longmemeval/splits/dev_test_seed42.json
```

VMP-Tuned 只在 dev 上调参，在 test 上报告。

调优目标：

```text
Objective =
0.80 * Official Session Recall-All@5
+ 0.35 * MacroTypeRecall-All@5
+ 0.15 * WorstTypeRecall-All@5
+ 0.40 * MRR
- 0.20 * FoldRecallStdDev
- 0.05 * NormalizedTokenCost
- 0.10 * StaleRetrievalRate
- 0.10 * ConflictRetrievalRate
```

### P9：官方 OSS framework adapters

优先接入官方开源实现，而不是 style 复刻。候选 adapter：

```text
Mem0 official adapter
Letta official adapter
LangMem official adapter
Zep / Graphiti official adapter
Text2Mem official adapter（如有可运行开源实现）
ReMe official adapter（如有可运行开源实现）
memU official adapter（如有可运行开源实现）
```

每个 official adapter 只有满足以下条件才进入主表：

```text
1. 能使用同一 vLLM OpenAI-compatible endpoint
2. 能使用同一 embedding
3. 能导出 retrieved evidence
4. 能统计 token cost / latency
5. 能在每个 LongMemEval sample 前 reset workspace，避免样本间泄漏
6. 能接受统一的 session-level 或 turn-level ingestion granularity
```

否则：

```text
进入 secondary comparison、appendix，或标记 unavailable
```

实现原则：

```text
1. adapter 只负责 ingest / retrieve / stats。
2. QA reader 仍由统一的 vLLM QA runner 完成。
3. 不允许某个框架自带 reader 直接回答问题后进入主表。
4. 如果官方框架必须自带 reader，则只能进入 secondary，并明确标注。
5. 对官方框架的 config、commit、依赖版本、失败原因都要写入 audit。
```

### P10：LongMemEval ablation

在 LongMemEval 上实现：

```text
VMP-full
VMP w/o recency
VMP w/o contradiction
VMP w/o redundancy
VMP w/o importance
VMP w/o confidence
VMP w/o token_cost
VMP w/o scope_match
VMP w/o update_operation
VMP w/o merge_operation
VMP w/o archive_operation
```

### P11：case export

导出论文案例：

```text
Case 1: VMP 正确处理 knowledge update
Case 2: NaiveVectorRAG 召回冲突旧记忆
Case 3: VMP 通过 archive 抑制过期记忆
Case 4: VMP 错误案例
```

### P12：Fallback style baselines（仅在必要时）

style baseline 不作为默认对比路线。只有当官方实现经过 audit 后无法公平接入，且论文仍需要覆盖该类思想时，才实现 fallback。

可选 fallback：

```text
Mem0-style
Letta-style
LangMem-style
Graphiti-style
Text2Mem-style IR
ReMe-style
memU-style
```

论文写法必须区分：

```text
official Mem0 OSS adapter：可以说是 Mem0 官方实现结果。
Mem0-style fallback：只能说是参考 Mem0 记忆思想的本项目复现。
```

如果 official adapter 已经达到 Level 1，则不需要再实现同名 style fallback，除非用于附录中的实现差异分析。

## 6. LongMemEval 数据字段

loader 必须兼容官方字段：

```text
question_id
question_type
question
answer
question_date
haystack_session_ids
haystack_dates
haystack_sessions
answer_session_ids
```

`haystack_sessions` 结构：

```text
list[session]
session = list[turn]
turn = {"role": "user" | "assistant", "content": "...", "has_answer": optional bool}
```

需要保留：

```text
history_session_id
history_date
turn_idx
role
has_answer
```

这些字段用于 session-level 和 turn-level evidence evaluation。

## 7. LongMemEval → Event 转换

每个 session turn 转为 Event：

```json
{
  "event_id": "lme_{question_id}_{session_id}_{turn_idx}",
  "session_id": "{question_id}_{session_id}",
  "task_id": "{question_id}",
  "event_type": "user_message | assistant_message",
  "content": "...",
  "metadata": {
    "dataset": "longmemeval",
    "question_id": "...",
    "question_type": "...",
    "history_session_id": "...",
    "history_date": "...",
    "turn_idx": 0,
    "role": "user",
    "has_answer": true
  }
}
```

query event：

```json
{
  "event_id": "lme_{question_id}_query",
  "session_id": "lme_eval_{question_id}",
  "task_id": "{question_id}",
  "event_type": "benchmark_query",
  "content": "{question}",
  "metadata": {
    "dataset": "longmemeval",
    "question_id": "...",
    "question_type": "...",
    "question_date": "..."
  }
}
```

需要在 schema 中新增：

```text
EventType.BENCHMARK_QUERY = "benchmark_query"
```

## 8. Adapter 接口

新增：

```text
src/vmp_memos/frameworks/
├── __init__.py
├── base.py
├── registry.py
├── audit.py
├── bm25.py
├── naive_vector.py
├── vector_recency.py
├── vector_importance.py
├── vmp_memos.py
├── official/
│   ├── __init__.py
│   ├── mem0_adapter.py
│   ├── letta_adapter.py
│   ├── langmem_adapter.py
│   └── graphiti_adapter.py
└── style/
    ├── __init__.py
    └── optional fallback implementations
```

统一 schema：

```python
class RetrievedMemory(BaseModel):
    memory_id: str
    content: str
    score: float
    source_session_id: str | None = None
    source_turn_id: str | None = None
    source_date: str | None = None
    memory_type: str | None = None
    metadata: dict = {}
    token_count: int = 0
```

官方框架审计 schema：

```python
class FrameworkCapabilityReport(BaseModel):
    framework_name: str
    official_repo: str | None = None
    version_or_commit: str | None = None
    license: str | None = None
    install_status: str = "unknown"
    local_vllm_supported: bool = False
    local_embedding_supported: bool = False
    evidence_export_supported: bool = False
    reset_workspace_supported: bool = False
    token_latency_stats_supported: bool = False
    fairness_level: str = "unavailable"
    main_table_eligible: bool = False
    notes: dict = {}
```

统一 adapter：

```python
class BaseMemoryFrameworkAdapter:
    name: str
    fairness_level: str

    def reset(self, workspace_dir: str) -> None:
        ...

    def ingest_event(self, event: Event) -> None:
        ...

    def ingest_session(self, events: list[Event]) -> None:
        ...

    def retrieve(
        self,
        query: str,
        top_k: int,
        question_date: str | None = None,
        metadata: dict | None = None,
    ) -> list[RetrievedMemory]:
        ...

    def stats(self) -> dict:
        ...

    def close(self) -> None:
        ...
```

回答生成不放进 adapter，统一由 QA runner 调用同一个 vLLM reader。这样确保公平。

## 9. 主实验一：Retrieval

数据：

```text
longmemeval_s_cleaned.json
```

第一批方法：

```text
EmptyRetrieval
BM25
NaiveVectorRAG
VectorRAG + Recency
VectorRAG + Importance
VMP-Rule
```

第二批方法：

```text
VMP-Tuned
Mem0 official（如果 Level 1 可控）
Letta official（如果 Level 1 可控）
LangMem official（如果 Level 1 可控）
Graphiti official（如果 Level 1 可控）
其他 official OSS adapter（如果 Level 1 可控）
```

命令：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --top-k 5 \
  --limit 100
```

正式全量：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --top-k 5
```

官方框架加入主表前，先运行：

```bash
python scripts/audit_frameworks.py \
  --frameworks mem0,letta,langmem,graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --verification-dir outputs/longmemeval/audit
```

只有 audit 标记为 `main_table_eligible=true` 的方法才加入正式全量命令。

## 10. 主实验二：End-to-End QA

所有方法使用同一个 retrieval 输出和同一个 vLLM reader。

命令：

```bash
python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/{run_id} \
  --methods bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --reader vllm \
  --top-k 5 \
  --limit 100
```

输出：

```text
outputs/longmemeval/runs/{run_id}/qa/{method}.jsonl
outputs/longmemeval/runs/{run_id}/hypotheses/{method}.jsonl
```

## 11. 主实验三：Ablation

命令：

```bash
python scripts/run_longmemeval_ablation.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --top-k 5 \
  --limit 100
```

消融项：

```text
features:
  recency
  contradiction
  redundancy
  importance
  confidence
  token_cost
  scope_match

operations:
  update_operation
  merge_operation
  archive_operation
```

## 12. 主实验四：Cost and Efficiency

统一统计：

```text
Ingestion Latency
Retrieval Latency
Reader Latency
End-to-End Latency
Retrieved Tokens
Reader Input Tokens
Reader Output Tokens
Memory Count
Storage Size
Cost per Correct Answer
```

## 13. 表格导出

必须导出：

```text
Table 1: Overall LongMemEval Retrieval Results
Table 2: Retrieval Results by Question Type
Table 3: End-to-End QA Results
Table 4: Ablation Study
Table 5: Cost and Efficiency
Table 6: Framework Availability and Fairness Level
```

Table 6 必须至少包含：

```text
Framework
Official/Style
Repo or Reference
Version/Commit
Local vLLM
Local Embedding
Evidence Export
Workspace Reset
Stats Export
Fairness Level
Main Table Eligible
Reason if Excluded
```

导出路径：

```text
outputs/longmemeval/tables/
├── table1_retrieval_overall.csv
├── table2_by_question_type.csv
├── table3_qa.csv
├── table4_ablation.csv
├── table5_cost.csv
└── table6_fairness.csv
```

每张表同时导出：

```text
CSV
Markdown
LaTeX
```

## 14. 案例导出

导出路径：

```text
outputs/longmemeval/cases/
```

至少包含：

```text
1. VMP 正确处理 knowledge update
2. NaiveVectorRAG 召回冲突旧记忆
3. VMP 通过 archive 抑制过期记忆
4. VMP 错误案例
```

案例格式：

```markdown
## Case: Knowledge Update

Question ID:
...

Question Type:
...

Question:
...

Gold Answer:
...

NaiveVectorRAG Retrieved:
...

VMP Retrieved:
...

VMP Operations:
- UPDATE ...
- ARCHIVE ...
- RETRIEVE ...

Analysis:
...
```

## 15. 服务器运行命令

### 15.1 启动 vLLM

当前已有：

```bash
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_VLLM_GPU_MEMORY_UTILIZATION=0.90
bash scripts/serve_vllm.sh
```

后续可补：

```bash
bash scripts/serve_vllm_qwen7b.sh
bash scripts/serve_vllm_qwen14b.sh
```

### 15.2 下载数据

```bash
python scripts/download_longmemeval.py --target data/longmemeval
```

### 15.3 检查数据

```bash
python scripts/inspect_longmemeval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --limit 3
```

### 15.4 Retrieval smoke

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --methods empty,bm25,naive_vector,vmp_rule \
  --top-k 5 \
  --limit 20
```

### 15.5 官方框架可控性审计

```bash
python scripts/audit_frameworks.py \
  --frameworks mem0,letta,langmem,graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --verification-dir outputs/longmemeval/audit
```

如果某些框架还没有安装或 adapter 尚未实现，审计脚本应输出 `unavailable`，而不是让整条实验 pipeline 崩掉。

### 15.6 QA smoke

```bash
python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/{run_id} \
  --methods bm25,naive_vector,vmp_rule \
  --reader vllm \
  --top-k 5 \
  --limit 20
```

### 15.7 导出表格

```bash
python scripts/export_longmemeval_tables.py --latest
```

### 15.8 导出案例

```bash
python scripts/export_error_cases.py --latest
```

## 16. 第一轮最小论文可交付版本

如果时间有限，第一轮只交付：

```text
1. LongMemEval-cleaned loader / inspector
2. vLLM reader
3. BM25
4. NaiveVectorRAG
5. VectorRAG + Recency
6. VectorRAG + Importance
7. VMP-Rule
8. Retrieval runner
9. Retrieval metrics
10. QA runner
11. QA metrics
12. Ablation
13. Cost analysis
14. Table export
15. Case export
16. Framework controllability audit
```

VMP-Tuned、Level 1 official OSS adapters 作为第二轮增强。style fallback 只在 official adapter 无法公平接入时再考虑。

## 17. 当前不要做

暂时不要做：

```text
1. RL policy learning
2. 大规模 learned policy
3. LongMemEval-V2
4. Web UI
5. 多用户权限系统
6. 复杂图数据库
7. longmemeval_m_cleaned 全量重复实验
8. 70B 本地模型实验
9. 将 style baseline 冒充 official framework
10. 在主实验中混用云端 LLM
11. 因为官方框架接入麻烦，就直接跳过 audit 改写成 style baseline
12. 让某个框架自带 reader 回答问题后与统一 vLLM reader 的方法混入同一主表
```

## 18. Codex 实施提示

实施时按阶段推进。不要因为当前本机没有 GPU、vLLM、LongMemEval 数据或 Mem0 而停止。

当前优先顺序：

```text
P1. [done] LongMemEval downloader / loader / inspector
P2. [done] LongMemEval schema conversion + EventType.BENCHMARK_QUERY
P3. [done] RetrievedMemory + BaseFrameworkAdapter + registry
P4. [done] BM25 / NaiveVector / Recency / Importance / VMP-Rule adapters
P5. [done] Retrieval metrics + run_longmemeval_retrieval.py
P6. [done] Retrieval report/table export
P7. [done] vLLM QA reader + run_longmemeval_qa.py
P8. [done] QA metrics + official hypothesis export
P9. [done] Framework controllability audit
P10. [done: Mem0/LangMem/Graphiti/Letta] Official OSS adapters if fully controllable
P11. [done] VMP-Tuned with fixed dev/test split
P12. [done] LongMemEval ablation
P13. [done] Cost analysis
P14. [done] Case export
P15. Fallback style baselines only if needed
```

所有主实验必须统一使用本地 vLLM 模型；不能把云端 API 结果和本地 vLLM 结果混入同一主表。
