你是一个资深 Python 工程师和 LLM Agent 框架开发者。现在需要在当前项目目录中实现一个名为 **VMP-MemOS** 的 Agent Memory 框架。

项目目标不是再造一个普通向量数据库，而是实现一个面向长程 LLM Agent 的 **Memory Policy Layer**。该层通过可解释的 **Memory Policy Embedding** 判断每条记忆应该如何被管理，包括写入、更新、合并、归档、压缩、召回、置顶和后台整理。

请严格阅读并遵循项目根目录下的 `IMPLEMENTATION_PLAN.md`。不要一次性实现所有内容，必须按阶段推进，每次只完成当前指定 Phase，并保证代码可运行、可测试、结构清晰。

## 核心设计原则

1. 区分 `content_embedding` 和 `policy_embedding`。

   * `content_embedding` 表示记忆内容的语义。
   * `policy_embedding` 表示记忆当前的管理状态。

2. 不要把所有历史直接塞进 prompt。

   * 需要通过 policy score 决定哪些记忆值得写入、更新、合并、归档和召回。

3. 所有记忆操作必须结构化记录。

   * 每次 ADD / UPDATE / MERGE / ARCHIVE / RETRIEVE / IGNORE 都要写入 `memory_workspace/logs/operations.jsonl`。
   * 每次召回都要写入 `memory_workspace/logs/retrievals.jsonl`。

4. 框架必须 backend-agnostic。

   * 第一阶段至少实现 FileMemoryBackend、VectorMemoryBackend、HybridMemoryBackend。
   * 后续预留 Mem0Adapter、ReMeAdapter、memUAdapter、Letta-style adapter。

5. 第一版优先实现 rule-based VMP。

   * 不要一开始做 RL。
   * 不要一开始做复杂深度学习。
   * 先实现可解释规则、benchmark runner、baseline 对比和消融实验。

6. 项目必须适合后续论文实验。

   * 所有实验结果要可复现。
   * 所有指标要可导出。
   * 所有配置要放入 `configs/`。
   * 所有输出要放入 `outputs/`。

## 技术要求

使用 Python 3.11+。

优先使用以下技术：

* pydantic：定义 schema
* numpy：向量计算
* scikit-learn：后续 learned policy
* sentence-transformers：本地 embedding
* python-frontmatter：Markdown frontmatter 读写
* typer：命令行脚本
* rich：终端输出
* pytest：测试
* ruff：代码规范

不要把 API key 写进代码。所有外部服务配置必须通过 `.env` 或 `configs/` 注入。

## 推荐目录结构

请按以下结构实现：

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
│       ├── extraction/
│       ├── embeddings/
│       ├── policy/
│       ├── operations/
│       ├── backends/
│       ├── retrieval/
│       ├── consolidation/
│       ├── benchmark/
│       └── utils/
├── scripts/
├── tests/
└── outputs/
```

## 第一阶段 MVP 范围

第一阶段只需要实现以下内容：

1. 项目初始化
2. Pydantic schema
3. FileMemoryBackend
4. VectorMemoryBackend
5. HybridMemoryBackend
6. Policy Feature Builder
7. Rule-based Policy Controller
8. Memory Operation Executor
9. 自建 toy benchmark
10. NoMemory、NaiveVectorRAG、VMPRule 三个 baseline
11. benchmark report 导出
12. pytest 测试

第一阶段暂不实现：

1. Web UI
2. 复杂图数据库
3. RL policy learning
4. 全量 LongMemEval
5. 全量 Mem0 接入
6. 多用户权限系统

## 关键 schema

请实现以下核心对象：

```text
Event
MemoryCandidate
MemoryItem
PolicyFeatures
MemoryOperation
RetrievalResult
BenchmarkSample
BenchmarkResult
```

每个对象必须支持：

```text
1. Pydantic validation
2. JSON serialization
3. stable id
4. timestamp
```

## Memory Policy Features

请实现以下 policy features：

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

第一版允许部分特征使用启发式规则模拟，但必须保留接口，方便后续替换为 LLM-as-judge、NLI model 或 learned model。

## Rule-based Policy Controller

请实现以下 score：

```text
WriteScore
RetrieveScore
UpdateScore
MergeScore
ArchiveScore
CompressScore
```

初始公式如下：

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

```text
UpdateScore =
0.30 * semantic_similarity_to_existing
+ 0.30 * contradiction
+ 0.20 * recency
+ 0.15 * source_priority
+ 0.05 * confidence
```

```text
MergeScore =
0.35 * semantic_similarity
+ 0.30 * redundancy
+ 0.20 * scope_match
+ 0.15 * low_conflict
```

```text
ArchiveScore =
0.25 * staleness
+ 0.25 * redundancy
+ 0.20 * negative_contribution
+ 0.15 * low_importance
+ 0.15 * superseded
```

```text
CompressScore =
0.30 * token_cost
+ 0.25 * access_frequency
+ 0.20 * information_density
+ 0.15 * actionability
+ 0.10 * scope_match
```

所有 score 必须输出可解释 reason，例如：

```json
{
  "op": "ADD",
  "score": 0.78,
  "reason": "High importance, high novelty, low redundancy."
}
```

## 操作集合

第一阶段必须实现：

```text
ADD
UPDATE
MERGE
ARCHIVE
RETRIEVE
IGNORE
```

预留但可以暂不完整实现：

```text
SPLIT
DELETE
EXPIRE
LOCK
PROMOTE
DEMOTE
COMPRESS
PIN
DREAM
VERIFY
```

注意：第一阶段不要物理删除记忆，删除类操作先统一转为 `ARCHIVE`。

## Benchmark 要求

请先实现一个自建 toy benchmark，覆盖以下场景：

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

Benchmark sample 格式如下：

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

需要实现以下 baseline：

```text
NoMemoryBaseline
NaiveVectorRAGBaseline
VMPRuleBaseline
```

需要输出以下指标：

```text
Accuracy
Evidence Precision
Write Precision
Update Accuracy
Conflict Resolution Accuracy
Stale Memory Usage Rate
Memory Growth Rate
Token Cost
p50 Latency
p95 Latency
```

## 命令要求

第一阶段完成后，以下命令必须可以运行：

```bash
python scripts/init_workspace.py
python scripts/run_demo.py --backend hybrid
python scripts/run_benchmark.py --config configs/benchmark.yaml
pytest
```

运行后必须生成：

```text
memory_workspace/logs/operations.jsonl
memory_workspace/logs/retrievals.jsonl
outputs/runs/{run_id}/results.jsonl
outputs/reports/{run_id}.md
```

## 开发方式

请按以下方式工作：

1. 先检查当前目录结构。
2. 如果项目为空，创建标准项目结构。
3. 每次修改后运行相关测试。
4. 不要删除用户已有文件。
5. 不要把临时实验代码混入核心模块。
6. 每个 Phase 完成后更新 README 或对应文档。
7. 代码优先保证清晰、可测试、可扩展。
8. 所有复杂逻辑都要有简短注释。
9. 出现设计取舍时，优先选择简单、可运行、可复现的实现。
10. 不要提前实现大型训练、RL 或复杂 Web UI。

## 当前任务执行格式

当我要求你执行某个 Phase 时，请按以下格式回复并实施：

```text
计划：
- 本阶段要改哪些文件
- 要实现哪些功能
- 要新增哪些测试

实施：
- 直接修改代码

验证：
- 运行 pytest 或相关命令
- 报告通过情况

总结：
- 已完成内容
- 还未完成内容
- 下一阶段建议
```

现在请根据我指定的 Phase 开始实现。

## 运行环境约束

注意：当前开发环境不要求你在本地真正运行项目代码。

我会将你生成的代码通过 Git 提交到云服务器上运行，云服务器配置包括 4090D 单卡 GPU、Linux 环境和完整 Python 运行环境。因此，你在当前环境中的任务是：

1. 编写完整、结构清晰、可维护的项目代码。
2. 提供可以在云服务器上执行的运行脚本。
3. 提供清晰的启动命令、测试命令和 benchmark 命令。
4. 确保代码逻辑自洽、路径设计合理、配置文件完整。
5. 不要因为当前本地环境缺少依赖、GPU、数据库或模型而阻塞开发。

你不需要在本地完成真实运行，也不需要强行安装大型依赖。对于需要在服务器执行的内容，请提供对应脚本和命令即可。

如果某些功能依赖外部环境，例如 GPU、embedding 模型、向量数据库、Mem0、Qdrant、Chroma、OpenAI API 或本地 LLM，请做到：

```text
1. 代码中保留清晰接口
2. 配置项写入 configs/ 或 .env.example
3. 在 README 中说明如何在服务器上运行
4. 提供 scripts/ 下的可执行脚本
5. 不要把 API key、模型路径、服务器路径写死
```

你可以进行静态检查和基础代码审查，但不要把“无法在当前本地环境运行”作为停止实现的理由。

本项目最终将在云服务器上通过以下方式运行：

```bash
git pull
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/init_workspace.py
python scripts/run_demo.py --backend hybrid
python scripts/run_benchmark.py --config configs/benchmark.yaml
python scripts/run_ablation.py --config configs/ablation.yaml
pytest
```

因此，你需要重点保证：

```text
1. 脚本入口存在
2. 参数设计清晰
3. 配置文件完整
4. 目录结构自动创建
5. 日志和输出路径稳定
6. 错误提示清楚
7. 后续能在服务器上直接运行
```

如果某些地方暂时无法真实实现，例如 Mem0Adapter、QdrantBackend、LLM-as-Judge、LongMemEval 全量评测，请先实现 stub / interface / TODO，并保证不会影响第一阶段 MVP 的运行。
