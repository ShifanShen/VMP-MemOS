# VMP-MemOS

VMP-MemOS 是一个面向长期 LLM Agent 的可解释 Memory Policy Layer。项目把“记忆内容是什么”和“记忆应该如何被管理”分开建模：`content_embedding` 表示内容语义空间，`policy_embedding` / `PolicyFeatures` 表示写入、更新、合并、归档、召回等管理信号。

当前实现保持论文实验的可控边界：不实现 Web UI、RL 或大型模型训练；toy benchmark 可以在没有 GPU 的环境先跑通；LongMemEval 论文 benchmark 已开始接入；LLM 能力通过可选 vLLM OpenAI-compatible server 接入，只在服务器上启动模型。

## 当前实现范围

已完成 `IMPLEMENTATION_PLAN.md` 中 Phase 0 至 Phase 10，并补齐 MVP 需要的 Hybrid backend：

- 标准 Python 3.11+ `src/` 项目结构；
- `pyproject.toml`、`.env.example`、`configs/default.yaml`；
- 核心 Pydantic schema：`Event`、`MemoryCandidate`、`MemoryItem`、`PolicyFeatures`、`MemoryOperation`、`RetrievalResult`、`BenchmarkSample`、`BenchmarkResult`；
- 可重复执行且不覆盖已有数据的 `scripts/init_workspace.py`；
- Markdown + YAML frontmatter 的 `FileMemoryBackend`；
- SQLite + embedding cache + cosine search 的 `VectorMemoryBackend`；
- Markdown source-of-truth + vector retrieval index 的 `HybridMemoryBackend`；
- `BaseEmbedder`、`SentenceTransformerEmbedder`、`SQLiteEmbeddingCache`、`CachedEmbedder`；
- 规则化 `PolicyFeatureBuilder`，可生成 16 个可解释 policy features；
- `RuleBasedPolicyController`，可计算 Write/Retrieve/Update/Merge/Archive/Compress score 并输出 operation decision；
- `MemoryOperationExecutor`，可执行 `ADD / UPDATE / MERGE / ARCHIVE / RETRIEVE / IGNORE` decision；
- toy benchmark runner 和 Markdown report 导出；
- Phase 8 baseline：`no_memory`、`full_context`、`summary_memory`、`naive_vector_rag`、`vector_rag_recency`、`vector_rag_importance`、`vmp_rule`；
- Phase 9 learned policy：从 toy benchmark / operation logs 构造训练样本，训练纯 Python multiclass logistic regression，输出 operation probabilities；
- Phase 10 ablation runner：支持禁用 `recency / contradiction / redundancy / success_contribution / token_cost` 并导出 Markdown 报告；
- vLLM LLM client：通过 OpenAI-compatible `/v1/chat/completions` 调用本地 vLLM 服务，并提供可选 LLM memory candidate extractor；
- LongMemEval-cleaned 接入骨架：loader、inspector、Event 转换、session-level evidence chunk；
- 论文实验 adapter 基类：`RetrievedMemory`、`BaseMemoryFrameworkAdapter`、`FrameworkRegistry`、framework controllability audit；
- 第一批可控 retrieval adapter：`empty`、`bm25`、`naive_vector`、`vector_recency`、`vector_importance`、`vmp_rule`；
- VMP-Tuned：固定 SHA-256 question-level dev/test split，只在 dev 上搜索检索权重，冻结模型后强制只在 test 上报告；
- LongMemEval 消融：在同一冻结模型上运行 7 个 feature ablation 和 3 个 operation ablation，导出 retrieval/QA delta 表；
- Cost analysis：离线聚合 ingestion/retrieval/reader 延迟、token、active memory、storage 和每个正确答案成本；
- Case export：从 test 主实验与消融 run 中确定性导出四类可审计论文案例；
- LongMemEval retrieval runner：统一 BGE-M3 embedding、session/turn ingestion、Recall@1/3/5/10、Precision@5、MRR、NDCG@5、延迟与存储统计；
- LongMemEval QA runner：统一 vLLM reader、固定 prompt、断点续跑、本地 QA metrics 和官方兼容 hypothesis；
- retrieval 论文表格导出：CSV、Markdown、LaTeX；
- 官方 Mem0 OSS adapter：固定 `mem0ai==2.0.10`，统一 vLLM/BGE-M3，支持 evidence provenance、workspace reset 和 smoke 凭证；
- 官方 LangMem adapter：固定 `langmem==0.0.30`，直接使用 memory store manager、共享 BGE-M3 embedder 和本地 vLLM；
- 官方 Graphiti adapter：固定 `graphiti-core==0.29.2`，统一 vLLM/BGE-M3，通过专用 Neo4j 保存知识图谱并导出 episode provenance；
- 官方 Letta adapter：固定 `letta-client==1.12.1` 与 Letta Server `0.16.8`，使用 agent-managed core/archival memory，并导出带 provenance 的 evidence；
- 基础 pytest 测试，覆盖 schema、workspace 初始化、文件后端、embedding cache、向量后端、hybrid 后端、policy feature builder、rule-based controller、operation executor、benchmark runner、learned policy、ablation runner、vLLM client config 和 LLM extraction parsing。

尚未实现：Web UI、RL 和官方框架全量正式服务器实验。

## 项目结构

```text
.
├── configs/
│   ├── default.yaml
│   ├── benchmark.yaml
│   └── llm.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/
│       └── memory_policy_toy.jsonl
├── memory_workspace/
│   ├── INDEX.md
│   ├── MEMORY.md
│   ├── memories/
│   ├── archive/
│   ├── versions/
│   ├── vector/
│   ├── cache/
│   ├── logs/
│   └── {projects,skills,episodes,resources}/
├── outputs/
│   ├── runs/
│   ├── reports/
│   ├── figures/
│   └── models/
├── scripts/
│   ├── audit_frameworks.py
│   ├── create_longmemeval_split.py
│   ├── export_longmemeval_ablation.py
│   ├── export_longmemeval_cost.py
│   ├── export_longmemeval_cases.py
│   ├── download_longmemeval.py
│   ├── inspect_longmemeval.py
│   ├── init_workspace.py
│   ├── run_benchmark.py
│   ├── run_demo.py
│   ├── run_llm_smoke.py
│   ├── run_longmemeval_retrieval.py
│   ├── run_longmemeval_qa.py
│   ├── run_longmemeval_ablation.sh
│   ├── run_vmp_tuned_experiment.sh
│   ├── run_ablation.py
│   ├── serve_embeddings.py
│   ├── serve_graphiti_neo4j.sh
│   ├── serve_letta.sh
│   ├── serve_vllm.sh
│   ├── train_policy.py
│   ├── train_vmp_tuned.py
│   └── setup_server.sh
├── src/vmp_memos/
│   ├── backends/
│   ├── benchmark/
│   ├── extraction/
│   ├── frameworks/
│   ├── longmemeval/
│   ├── embeddings/
│   ├── llm/
│   ├── operations/
│   ├── policy/
│   └── schemas/
└── tests/
```

## 在 Linux / 4090D 服务器上运行

推荐一键 bootstrap：

```bash
git pull
bash scripts/setup_server.sh
```

等价手动命令：

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/init_workspace.py
python scripts/run_demo.py --backend file --workspace outputs/demo-workspace
python scripts/run_benchmark.py --config configs/benchmark.yaml
python scripts/train_policy.py --config configs/benchmark.yaml
python scripts/run_benchmark.py --config configs/benchmark.yaml --policy learned
python scripts/run_ablation.py --config configs/benchmark.yaml
python -m pytest
```

## vLLM LLM Integration

本项目的 LLM 接入方式是：在服务器上用 vLLM 启动 OpenAI-compatible API server，项目代码通过 HTTP 调用 `/v1/chat/completions`。本地开发不需要安装 vLLM，也不会自动下载模型。

推荐先在 4090D 服务器上安装基础依赖：

```bash
git pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

安装 vLLM：

```bash
python -m pip install vllm
```

启动 vLLM server。默认模型是 `Qwen/Qwen2.5-7B-Instruct`，你可以按显存情况替换：

```bash
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_VLLM_GPU_MEMORY_UTILIZATION=0.90
bash scripts/serve_vllm.sh
```

服务启动后，另开一个 shell：

```bash
source .venv/bin/activate
export VMP_LLM_API_KEY="local-vllm-key"
python scripts/run_llm_smoke.py \
  --config configs/llm.yaml \
  --prompt "用一句话说明 VMP-MemOS 的作用。"
```

如果没有设置 vLLM API key，也可以不导出 `VMP_LLM_API_KEY`，但 `scripts/serve_vllm.sh` 和 client 两边要保持一致。

运行 LLM memory candidate extraction smoke：

```bash
python scripts/run_llm_smoke.py \
  --config configs/llm.yaml \
  --extract-memory \
  --scope project/vmp-memos \
  --prompt "用户现在主攻 Agent 和 LLM 应用开发，不再 all in Java 后端。"
```

常用覆盖参数：

```bash
python scripts/run_llm_smoke.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-key local-vllm-key \
  --max-tokens 256 \
  --temperature 0
```

如果要运行向量检索或 hybrid demo，请安装 embedding extra：

```bash
python -m pip install -e ".[dev,embeddings]"

python scripts/run_demo.py \
  --backend vector \
  --workspace outputs/demo-vector-workspace \
  --device cuda

python scripts/run_demo.py \
  --backend hybrid \
  --workspace outputs/demo-hybrid-workspace \
  --device cuda
```

也可以让 bootstrap 脚本安装 embedding extra：

```bash
VMP_EXTRAS="dev,embeddings" bash scripts/setup_server.sh
```

如需指定 SentenceTransformers 模型缓存目录：

```bash
python scripts/run_demo.py \
  --backend vector \
  --workspace outputs/demo-vector-workspace \
  --device cuda \
  --model-cache-dir /path/to/model-cache
```

## LongMemEval 论文 benchmark 接入

当前已完成 LongMemEval-cleaned 的数据接入、schema 转换、可控 retrieval adapters、retrieval runner 与统一 vLLM QA runner。确定性检索 baseline 不调用 LLM；官方 memory frameworks 可在记忆抽取阶段调用同一个本地 vLLM，所有向量方法统一使用 BGE-M3。

下载 LongMemEval-cleaned：

```bash
source .venv/bin/activate
python scripts/download_longmemeval.py --target data/longmemeval
```

如果服务器网络无法直接访问 Hugging Face，也可以手动从 `xiaowu0162/longmemeval-cleaned` 下载以下文件并放入 `data/longmemeval/`：

```text
longmemeval_oracle.json
longmemeval_s_cleaned.json
longmemeval_m_cleaned.json
```

检查数据：

```bash
python scripts/inspect_longmemeval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --limit 3
```

安装检索实验依赖：

```bash
python -m pip install -e ".[dev,embeddings]"
```

先做不下载 embedding 模型的结构 smoke：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --methods empty,bm25,naive_vector,vmp_rule \
  --top-k 5 \
  --retrieval-depth 10 \
  --limit 20 \
  --no-embeddings \
  --run-id lme_retrieval_smoke
```

在 4090D 上运行正式 retrieval 实验：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --top-k 5 \
  --retrieval-depth 10 \
  --embedding-model BAAI/bge-m3 \
  --embedding-device cuda \
  --embedding-cache-dir /path/to/huggingface-cache \
  --run-id lme_s_bge_m3_main
```

`--top-k 5` 是后续 QA 使用的 evidence 数量；`--retrieval-depth 10` 会额外保存前 10 条结果，以便合法计算 Recall@10。每个 run 会输出：

```text
outputs/longmemeval/runs/{run_id}/manifest.json
outputs/longmemeval/runs/{run_id}/summary.json
outputs/longmemeval/runs/{run_id}/{method}/retrieval.jsonl
outputs/longmemeval/runs/{run_id}/{method}/summary.json
```

正式结果不要使用 `--no-embeddings`。该选项只用于验证数据、runner 和输出格式。

### VMP-Tuned：固定 dev/test 后调参

先生成一次固定 split。分配只使用 `question_id` 和 seed，不读取答案；manifest
同时记录原始数据 SHA-256，后续数据文件有任何字节变化都会拒绝运行：

```bash
python scripts/create_longmemeval_split.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --output outputs/longmemeval/splits/dev_test_seed42.json \
  --seed 42 \
  --dev-size 100 \
  --test-size 400
```

只用 dev 的 gold session 调优并冻结模型：

```bash
python scripts/train_vmp_tuned.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --split-manifest outputs/longmemeval/splits/dev_test_seed42.json \
  --output outputs/longmemeval/models/vmp_tuned_seed42.json \
  --report outputs/longmemeval/models/vmp_tuned_seed42_search.json \
  --embedding-model BAAI/bge-m3 \
  --embedding-device cuda \
  --embedding-cache-dir "${HOME}/.cache/huggingface" \
  --trials 64 \
  --tuning-seed 2025
```

模型工件记录 dataset SHA、split manifest SHA、embedding identifier、搜索 seed、
目标函数、dev 指标和 `test_labels_used=false`。正式评测必须显式指定 test；runner
会拒绝在训练 split 上运行 `vmp_tuned`，也会拒绝数据、split 或 embedding 不匹配：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --split-manifest outputs/longmemeval/splits/dev_test_seed42.json \
  --split test \
  --vmp-tuned-model outputs/longmemeval/models/vmp_tuned_seed42.json \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned \
  --top-k 5 \
  --retrieval-depth 10 \
  --embedding-model BAAI/bge-m3 \
  --embedding-device cuda \
  --embedding-cache-dir "${HOME}/.cache/huggingface" \
  --run-id lme_test_vmp_tuned_seed42
```

一条脚本可顺序执行 split、dev 调优、test retrieval 和表格导出：

```bash
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
RUN_ID=lme_test_vmp_tuned_seed42 \
bash scripts/run_vmp_tuned_experiment.sh
```

脚本默认不启动或调用 vLLM，避免单卡上 vLLM 预分配显存后 BGE-M3 无法加载。
retrieval 完成后再启动 vLLM 并执行下文 QA 命令。若你已为两者留出足够显存，
可以设置 `RUN_QA=1` 让脚本继续执行 QA。

### LongMemEval 消融实验

消融实验严格复用前一步生成的 split manifest、BGE-M3 和冻结
`vmp_tuned_seed42.json`，不会为任何消融变体重新调参。
本阶段将模型 schema 升级为 `1.1` 以记录 operation policy；拉取新代码后应先
重新执行 `run_vmp_tuned_experiment.sh`，旧 `1.0` 工件会被明确拒绝。变体包括：

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

operation ablation 具有独立语义：update 识别并提升新的冲突证据，archive
抑制被较新 update session 取代的旧证据，merge 去除近重复 session。
每道题的 operation counts、禁用组件和模型 split ID 都写入 retrieval record。

先在未启动 vLLM 时完成全部 test retrieval。脚本使用持久化 SQLite embedding
cache，避免 11 个变体重复计算相同 BGE-M3 向量：

```bash
RUN_ID=lme_test_ablation_seed42 \
bash scripts/run_longmemeval_ablation.sh
```

启动 vLLM 后，为同一 retrieval run 生成统一 QA：

```bash
ABLATION_METHODS="vmp_tuned,vmp_tuned__no_recency,vmp_tuned__no_contradiction,vmp_tuned__no_redundancy,vmp_tuned__no_importance,vmp_tuned__no_confidence,vmp_tuned__no_token_cost,vmp_tuned__no_scope_match,vmp_tuned__no_update_operation,vmp_tuned__no_merge_operation,vmp_tuned__no_archive_operation"

python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_ablation_seed42 \
  --methods "${ABLATION_METHODS}" \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128

python scripts/export_longmemeval_ablation.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_ablation_seed42
```

输出 `table4_ablation.{csv,md,tex}`，包含 Recall@5、MRR、NDCG@5、
retrieved tokens、Normalized EM、Token F1 及其相对 VMP-full 的差值。

### Cost and Efficiency

成本分析完全读取已有 retrieval/QA JSONL，不会重新运行 embedding 或 LLM。
默认要求 QA 已完整结束，并验证每个方法的 question ID 和顺序与 retrieval
严格一致：

```bash
python scripts/export_longmemeval_cost.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_tuned_seed42
```

输出：

```text
outputs/longmemeval/tables/cost_analysis.json
outputs/longmemeval/tables/table5_cost.csv
outputs/longmemeval/tables/table5_cost.md
outputs/longmemeval/tables/table5_cost.tex
```

Table 5 包含 mean/P95 ingestion、retrieval、reader、end-to-end latency，
retrieved/reader/framework tokens、active memory count、memory retention ratio、
storage size、operation counts，以及 observed tokens / milliseconds per correct
answer。本地 vLLM 不伪造美元价格；官方框架无法导出内部 LLM usage 时保留
`null` 并报告 coverage，绝不按零成本处理。

如果只想在 QA 前预览 retrieval 成本，可以显式运行：

```bash
python scripts/export_longmemeval_cost.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_tuned_seed42 \
  --allow-missing-qa
```

### 论文案例导出

案例导出需要一个包含 `naive_vector`、`vmp_tuned` 及完整 QA 的主实验 run，
以及一个包含 `vmp_tuned__no_archive_operation` 的消融 run：

```bash
python scripts/export_longmemeval_cases.py \
  --retrieval-run outputs/longmemeval/runs/lme_official_main \
  --ablation-run outputs/longmemeval/runs/lme_test_ablation_seed42 \
  --vmp-method vmp_tuned \
  --vector-method naive_vector
```

导出器会验证两个 run 使用相同 dataset SHA、test split 和冻结 VMP 模型，
然后确定性选择：

```text
1. VMP 正确处理 knowledge update
2. NaiveVectorRAG 召回与新证据相关但更旧的非 gold 证据
3. VMP archive 相对 no-archive 减少 active memory
4. VMP 的真实错误案例
```

输出：

```text
outputs/longmemeval/cases/manifest.json
outputs/longmemeval/cases/cases.json
outputs/longmemeval/cases/paper_cases.md
outputs/longmemeval/cases/case1_knowledge_update.json
outputs/longmemeval/cases/case2_stale_vector_retrieval.json
outputs/longmemeval/cases/case3_archive_suppression.json
outputs/longmemeval/cases/case4_vmp_error.json
```

JSON 保留 gold sessions、retrieved evidence、QA prediction、policy
features/contributions、operation counts 和源 manifest hash。Markdown
只使用确定性模板，不额外调用 LLM；若没有符合某类定义的样本会报错，不会用
“最接近”的样本冒充成功或失败案例。

导出 retrieval 论文表格：

```bash
python scripts/export_longmemeval_tables.py \
  --retrieval-run outputs/longmemeval/runs/lme_s_bge_m3_main
```

输出包括：

```text
outputs/longmemeval/tables/table1_retrieval_overall.{csv,md,tex}
outputs/longmemeval/tables/table2_by_question_type.{csv,md,tex}
```

启动统一 vLLM reader 后运行端到端 QA：

```bash
export VMP_LLM_API_KEY="local-vllm-key"

python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_s_bge_m3_main \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128
```

如果服务器任务中断，使用完全相同的参数并添加 `--resume`。runner 会跳过已经写入的 question，并检查整个 run 只能出现一个 reader provider/model：

```bash
python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_s_bge_m3_main \
  --methods empty,bm25,naive_vector,vector_recency,vector_importance,vmp_rule \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128 \
  --resume
```

QA 输出：

```text
outputs/longmemeval/runs/{run_id}/qa/manifest.json
outputs/longmemeval/runs/{run_id}/qa/{method}.jsonl
outputs/longmemeval/runs/{run_id}/qa/{method}.summary.json
outputs/longmemeval/runs/{run_id}/qa/summary.json
outputs/longmemeval/runs/{run_id}/hypotheses/{method}.jsonl
```

本地 QA 指标包括 Normalized Exact Match、Token F1、Contains Answer 和 Abstention Accuracy，同时记录 reader token usage 与端到端延迟。`hypotheses/{method}.jsonl` 使用 `{"question_id": "...", "hypothesis": "..."}` 格式，可交给官方 evaluator；官方 GPT-based evaluator 不属于本地 pipeline 的硬依赖。

审计外部官方框架是否能公平进入主表：

```bash
python scripts/audit_frameworks.py \
  --frameworks mem0,letta,langmem,graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024
```

审计输出：

```text
outputs/longmemeval/audit/framework_controllability.json
outputs/longmemeval/tables/table6_fairness.csv
```

当前策略是：官方 OSS adapter 优先；只有通过本地 vLLM、本地 embedding、evidence export、workspace reset 等可控性检查的方法，才允许进入论文主性能表。style baseline 只作为兜底和附录，不冒充官方框架结果。

### 官方 Mem0 / LangMem / Graphiti / Letta OSS 对比

Mem0 adapter 直接调用官方 `mem0.Memory`，没有复刻 Mem0 算法。为了避免框架版本漂移，论文环境固定使用 `mem0ai==2.0.10`：

```bash
python -m pip install -e \
  ".[dev,embeddings,official-mem0,official-langmem,official-graphiti,official-letta]"
```

Mem0 ingestion 会调用本地 vLLM 做官方 memory extraction。Qwen2.5 的工具调用需要 Hermes parser；`scripts/serve_vllm.sh` 已默认启用。由于 vLLM 与 BGE-M3 同时占用 4090D，建议 Mem0 实验降低 vLLM 显存预留：

```bash
export VMP_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VMP_VLLM_ENABLE_TOOL_CALLING=1
export VMP_VLLM_TOOL_CALL_PARSER=hermes
bash scripts/serve_vllm.sh
```

四个官方 adapter 共用 `official_llm_temperature=0.0` 和
`official_llm_max_tokens=512`；实际值会写入 retrieval manifest。若修改，
必须对所有官方框架使用同一组 CLI 参数或环境变量。

先在另一个 shell 运行官方 adapter smoke：

```bash
source .venv/bin/activate
export VMP_LLM_API_KEY="local-vllm-key"

python scripts/run_official_framework_smoke.py \
  --framework mem0 \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda
```

使用同一套模型验证 LangMem：

```bash
python scripts/run_official_framework_smoke.py \
  --framework langmem \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda
```

Graphiti 还需要一个只供本项目使用的 Neo4j。下面的脚本会创建带专用标签的容器；benchmark 每道题都会清空该数据库的全部节点，因此绝不能把 URI 指向已有业务数据库：

```bash
export VMP_GRAPHITI_NEO4J_PASSWORD="replace-with-a-strong-password"
export VMP_GRAPHITI_NEO4J_URI="bolt://127.0.0.1:7687"
export VMP_GRAPHITI_NEO4J_USER="neo4j"
bash scripts/serve_graphiti_neo4j.sh
```

使用同一套 vLLM 与 BGE-M3 验证 Graphiti。`--graphiti-allow-destructive-reset` 是有意设置的安全确认：

```bash
python scripts/run_official_framework_smoke.py \
  --framework graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda \
  --graphiti-neo4j-uri bolt://127.0.0.1:7687 \
  --graphiti-allow-destructive-reset
```

Letta 是独立的官方状态服务。先在新 shell 启动同一个 BGE-M3 的
OpenAI-compatible embedding endpoint：

```bash
source .venv/bin/activate
python scripts/serve_embeddings.py \
  --host 0.0.0.0 \
  --port 8001 \
  --model BAAI/bge-m3 \
  --device cuda
```

再启动固定为 `0.16.8` 的专用 Letta Server。该脚本使用 Linux host
network，使容器连接前面的 vLLM 和 embedding endpoint：

```bash
export VMP_LLM_BASE_URL=http://127.0.0.1:8000/v1
export VMP_LLM_API_KEY="local-vllm-key"
export VMP_LETTA_BASE_URL=http://127.0.0.1:8283
export VMP_LETTA_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
bash scripts/serve_letta.sh
```

最后验证 Letta agent-managed memory。Letta 会为每道题创建独立 agent，
并在下一题或退出时删除：

```bash
python scripts/run_official_framework_smoke.py \
  --framework letta \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --letta-base-url http://127.0.0.1:8283 \
  --letta-embedding-base-url http://127.0.0.1:8001/v1
```

Letta 的旧 `vllm` provider 会覆盖生成上限，因此本适配器使用其官方
OpenAI-compatible provider 连接同一个本地 vLLM。这样四个框架才能严格
共享温度、生成上限、API key 和模型；请求不会流向 OpenAI 云端。

smoke 成功后重新执行审计：

```bash
python scripts/audit_frameworks.py \
  --frameworks mem0,letta,langmem,graphiti \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --official-llm-max-tokens 512 \
  --official-llm-temperature 0 \
  --verification-dir outputs/longmemeval/audit
```

只有客户端/服务端版本、模型配置和 smoke 凭证都匹配时，`mem0` /
`langmem` / `graphiti` / `letta` 才会得到
`main_table_eligible=true`。先跑 20 条：

```bash
python scripts/run_longmemeval_retrieval.py \
  --data data/longmemeval/longmemeval_s_cleaned.json \
  --split-manifest outputs/longmemeval/splits/dev_test_seed42.json \
  --split test \
  --vmp-tuned-model outputs/longmemeval/models/vmp_tuned_seed42.json \
  --methods bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned,mem0,langmem,graphiti,letta \
  --top-k 5 \
  --retrieval-depth 10 \
  --limit 20 \
  --embedding-model BAAI/bge-m3 \
  --embedding-dimension 1024 \
  --embedding-device cuda \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen/Qwen2.5-7B-Instruct \
  --graphiti-neo4j-uri bolt://127.0.0.1:7687 \
  --graphiti-allow-destructive-reset \
  --letta-base-url http://127.0.0.1:8283 \
  --letta-embedding-base-url http://127.0.0.1:8001/v1 \
  --run-id lme_official_smoke20
```

确认 smoke 结果后移除 `--limit 20`，并更换新的 `--run-id`
运行完整 test split。正式官方框架实验必须保留相同 split 和冻结的
VMP-Tuned 模型，同时保留默认 `--official-memory-infer`；不能使用
`--no-embeddings` 或 `--no-official-memory-infer`。

retrieval 完成后，所有方法继续共用同一个 QA reader：

```bash
python scripts/run_longmemeval_qa.py \
  --retrieval-run outputs/longmemeval/runs/lme_official_main \
  --methods bm25,naive_vector,vector_recency,vector_importance,vmp_rule,vmp_tuned,mem0,langmem,graphiti,letta \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --top-k 5 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 128
```

## Workspace 初始化

默认命令会创建缺失目录和空 JSONL 日志，同时保留已有 Markdown 与日志内容：

```bash
python scripts/init_workspace.py
```

可指定独立实验目录：

```bash
python scripts/init_workspace.py --workspace /path/to/experiment/memory_workspace
```

只有明确需要刷新种子 Markdown 时才使用 `--force`；即使使用该参数，已有 JSONL 日志也不会被截断。

## 后端说明

`FileMemoryBackend` 将活动记忆写到 `memories/{memory_id}.md`，归档记忆写到 `archive/{memory_id}.md`，更新前版本写到 `versions/{memory_id}/v000001.md`。每次 ADD / UPDATE / ARCHIVE / RETRIEVE 都会写入 `logs/operations.jsonl` 或 `logs/retrievals.jsonl`。Phase 2 的 file search 是确定性词法检索，便于早期验证。

`VectorMemoryBackend` 将当前记忆、content embedding 和版本历史写入 `vector/memories.sqlite3`，embedding cache 写入 `cache/embeddings.sqlite3`。它会记录 embedding namespace 和维度，防止同一个向量库混用不同模型。检索使用 query embedding 与候选 memory embedding 的 cosine similarity。

`HybridMemoryBackend` 将 `FileMemoryBackend` 作为 source of truth，将 `VectorMemoryBackend` 作为 retrieval index。`get/list` 从 Markdown 文件读取；`search` 先由向量后端排序，再用 file 后端按 ID hydrate 可读 memory；`add/update/archive` 会同步两个组件。

## Policy Feature Builder

`PolicyFeatureBuilder` 会生成 16 个 policy features：

```text
semantic_relevance, importance, confidence, recency,
stability, novelty, redundancy, contradiction,
staleness, access_frequency, success_contribution, failure_contribution,
token_cost, scope_match, actionability, privacy_risk
```

第一版完全规则化：如果调用方已经提供 embedding，则使用 cosine similarity；否则使用词法相似度、时间半衰期、scope、访问次数和关键词启发式。它不会调用 LLM，也不会依赖 GPU。

## Rule-based Policy Controller

`RuleBasedPolicyController` 实现六个规则分数：

```text
WriteScore, RetrieveScore, UpdateScore,
MergeScore, ArchiveScore, CompressScore
```

每个 `score_*()` 会返回分数、阈值、是否通过、贡献项和可读 reason；每个 `decide_*()` 会返回 `ADD / UPDATE / MERGE / ARCHIVE / COMPRESS / RETRIEVE / IGNORE` decision。Decision 可以转换为 `MemoryOperation`，供 executor 写入 JSONL。

## Operation Executor

`MemoryOperationExecutor` 将 policy decision 执行到任意 `BaseMemoryBackend` 上。第一版实现：

- `ADD`：写入新 memory；
- `UPDATE`：对目标 memory 应用 patch；
- `MERGE`：更新主 memory，并可归档 source memories；
- `ARCHIVE`：归档目标 memory；
- `RETRIEVE`：调用 backend search；
- `IGNORE`：不修改 backend，但写入一条 `IGNORE` operation log。

## Toy Benchmark Runner

默认数据集：

```text
data/benchmarks/memory_policy_toy.jsonl
```

覆盖 8 类场景：

```text
preference update, fact conflict, project state change,
multi-session integration, stale memory archive,
duplicate memory merge, long tool log compression,
failure-to-procedural-memory
```

默认 baseline：

```text
no_memory
full_context
summary_memory
naive_vector_rag
vector_rag_recency
vector_rag_importance
vmp_rule
```

其中 `vector_rag` 是 `naive_vector_rag` 的 CLI 兼容别名，方便按计划中的验收命令运行。

运行完整 toy benchmark：

```bash
python scripts/run_benchmark.py --config configs/benchmark.yaml
```

只跑部分 baseline：

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark.yaml \
  --baselines no_memory,vector_rag,vmp_rule \
  --run-id local_smoke
```

输出：

```text
outputs/runs/{run_id}/results.jsonl
outputs/reports/{run_id}.md
```

报表包含：

```text
Accuracy
Evidence Precision / Recall
Operation Recall
Write Precision
Memory Growth
Token Cost
Conflict Retrieval Rate
Stale Memory Usage Rate
Latency
```

## Learned Policy

Phase 9 的第一版 learned policy 是一个无额外依赖的 multiclass logistic regression。它输入 `PolicyFeatures` 的 16 维向量，输出以下 operation probabilities：

```text
ADD, UPDATE, MERGE, ARCHIVE, RETRIEVE, IGNORE
```

训练命令：

```bash
python scripts/train_policy.py --config configs/benchmark.yaml
```

默认输出：

```text
outputs/models/learned_policy.json
outputs/models/learned_policy_examples.jsonl
```

可叠加 operation logs 作为训练数据：

```bash
python scripts/train_policy.py \
  --config configs/benchmark.yaml \
  --operation-log memory_workspace/logs/operations.jsonl
```

运行 learned policy benchmark：

```bash
python scripts/run_benchmark.py --config configs/benchmark.yaml --policy learned
```

指定模型路径：

```bash
python scripts/run_benchmark.py \
  --config configs/benchmark.yaml \
  --policy learned \
  --policy-model-path outputs/models/learned_policy.json
```

## Ablation Runner

Phase 10 提供 feature-level 消融实验，用来观察 policy feature 对 VMP rule baseline 的影响。默认会比较：

```text
no_memory
naive_vector_rag
vmp_rule
vmp_rule__no_recency
vmp_rule__no_contradiction
vmp_rule__no_redundancy
vmp_rule__no_success_contribution
vmp_rule__no_token_cost
```

运行默认消融：

```bash
python scripts/run_ablation.py --config configs/benchmark.yaml
```

只跑指定 feature 消融：

```bash
python scripts/run_ablation.py \
  --config configs/benchmark.yaml \
  --disable recency \
  --disable contradiction
```

默认输出：

```text
outputs/runs/{run_id}/results.jsonl
outputs/reports/ablation.md
```

报告包含实验设置、baseline 对比、ablation 对比、完整指标表、错误案例和 memory operation 示例。

## 验证命令

```bash
python -m pytest
python -m pytest tests/test_ablation_runner.py
python -m pytest tests/test_benchmark_runner.py
python -m pytest tests/test_learned_policy.py
python -m pytest tests/test_hybrid_backend.py
python -m pytest tests/test_operation_executor.py
python -m pytest tests/test_policy_controller.py
python -m pytest tests/test_policy_feature_builder.py
python -m pytest tests/test_file_backend.py tests/test_vector_backend.py
python -m pytest tests/test_retrieval_metrics.py tests/test_longmemeval_retrieval_runner.py
python -m pytest tests/test_longmemeval_splits.py tests/test_vmp_tuned.py
python -m pytest tests/test_longmemeval_ablation.py
python -m pytest tests/test_longmemeval_cost.py
python -m pytest tests/test_longmemeval_cases.py
python -m pytest tests/test_qa_metrics.py tests/test_longmemeval_qa_runner.py
python -m pytest tests/test_mem0_official_adapter.py tests/test_framework_audit.py
python -m pytest tests/test_langmem_official_adapter.py
python -m pytest tests/test_graphiti_official_adapter.py
python -m pytest tests/test_letta_official_adapter.py tests/test_embedding_server.py
ruff check src scripts tests
mypy src scripts
```

## 快速示例

```python
from vmp_memos.backends import VectorMemoryBackend
from vmp_memos.policy import PolicyFeatureBuilder
from vmp_memos.schemas import MemoryItem, MemorySource

backend = VectorMemoryBackend("memory_workspace")
memory = MemoryItem(
    type="semantic",
    scope="career/agent-dev",
    content="用户当前主攻 Agent 开发和 LLM 应用开发。",
    source=MemorySource(source_type="conversation"),
)

memory = PolicyFeatureBuilder().enrich_memory(memory)
backend.add(memory, reason="New stable preference.")
results = backend.search("Agent 长期记忆开发", top_k=5)
```

所有本地敏感配置都应通过 `.env` 或 `configs/` 注入。不要提交 `.env`、API key、模型缓存或服务器绝对路径。
