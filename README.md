# VMP-MemOS

[简体中文](README.md) | [English](README_EN.md)

VMP-MemOS 是一个面向长程 LLM Agent 的可解释记忆策略层。它将“记忆的内容”与“记忆应如何被管理”分开建模，并为写入、更新、合并、归档、召回和压缩提供可审计的策略信号。

> 项目目前处于论文实验阶段。代码、评测链路和复现脚本已经开放；最终论文结论应以冻结配置下的完整测试集结果为准。

## 核心思路

传统 RAG 通常只根据内容相似度召回。VMP-MemOS 在内容相关性之外维护 `PolicyFeatures`，显式描述重要性、置信度、时效性、矛盾、冗余、更新信号、token 成本等管理属性。

```text
Events / Conversations
        │
        ▼
Memory Candidates ──► Content Embedding
        │
        └────────────► Policy Features
                              │
                              ▼
                     Memory Policy Layer
                  ADD / UPDATE / MERGE / ARCHIVE
                              │
                              ▼
                 File / Vector / Hybrid Backend
                              │
                              ▼
              Retrieval + Evidence-set Selection
```

## 主要能力

- 严格的 Pydantic schema：`Event`、`MemoryCandidate`、`MemoryItem`、`PolicyFeatures`、`MemoryOperation`、`RetrievalResult`、`BenchmarkSample`、`BenchmarkResult`。
- File、Vector 和 Hybrid 三类持久化后端。
- 可解释的规则策略与轻量 learned policy。
- 结构化 ADD、UPDATE、MERGE、ARCHIVE、RETRIEVE、IGNORE 操作日志。
- BM25、向量、时效、重要性和 VMP 系列检索基线。
- LongMemEval 数据加载、固定 Dev/Test 切分、检索、QA、消融、成本分析和论文表格导出。
- 基于本地 vLLM 的匿名证据抽取与确定性 evidence-set coverage。
- Mem0、LangMem、Graphiti 和 Letta 官方 OSS adapter，主实验可统一使用本地模型与 embedding。
- 断点续跑、严格质量 gate、数据与模型 provenance 审计。

## 仓库结构

```text
configs/              可复现实验配置
data/                 toy benchmark 与本地数据目录
memory_workspace/     文件记忆、归档、版本和运行日志
scripts/              训练、评测、服务与导出入口
src/vmp_memos/        核心 Python 包
tests/                网络无关单元与集成测试
outputs/              本地实验产物（默认不提交）
```

## 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- 核心功能与 toy benchmark 可在 CPU 上运行
- LongMemEval 的 BGE-M3/vLLM 实验建议使用 Linux、CUDA 和 NVIDIA GPU

## 快速开始

```bash
git clone https://github.com/ShifanShen/VMP-MemOS.git
cd VMP-MemOS

uv sync --extra dev
uv run python scripts/init_workspace.py
uv run python scripts/run_benchmark.py
```

运行测试：

```bash
uv run python -m pytest -q
```

如需本地向量模型：

```bash
uv sync --extra dev --extra embeddings
```

## 本地 vLLM

论文主实验要求所有可比较方法使用同一个本地模型。安装与 CUDA 环境匹配的 vLLM 后，可通过 OpenAI-compatible server 启动模型：

```bash
export VMP_LLM_MODEL=/path/to/Qwen2.5-7B-Instruct
export VMP_LLM_API_KEY=local-vllm-key
export VMP_VLLM_ENABLE_TOOL_CALLING=0

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync bash scripts/serve_vllm.sh
```

检查服务端实际暴露的 model ID：

```bash
curl -s -H "Authorization: Bearer local-vllm-key" \
  http://127.0.0.1:8000/v1/models
```

客户端的 `VMP_LLM_MODEL` 必须使用返回结果中的 `id`，即使服务端从本地路径加载模型。

## LongMemEval 实验

下载官方 cleaned 数据集：

```bash
uv run --no-sync python scripts/download_longmemeval.py \
  --target data/longmemeval \
  --files longmemeval_s_cleaned.json
```

网络受限时可使用 Hugging Face 镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=600 \
uv run --no-sync python scripts/download_longmemeval.py \
  --target data/longmemeval \
  --files longmemeval_s_cleaned.json
```

当前研究入口是 VMP-v6.4。它恢复 V6.2 的高召回抽取指令模板，同时保留 V6.3 的数组事实规范化、证据坐标防污染、纯列表标记过滤和 V4 excerpt；候选池、Top-k 策略与覆盖权重均未改变。建议先执行四个诊断样本的 smoke：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_smoke \
uv run --no-sync bash scripts/run_vmp_v64_experiment.sh
```

再执行完整 Dev 实验：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v64_experiment.sh
```

只有 Dev gate 通过后，才应执行冻结的 Test candidate、rerank 和 QA 流程。中断后可设置 `RERANK_RESUME=1` 继续同一配置的 run。

### Hybrid QA Reader v2.1

QA-v2.1 同时使用 reranker 保存的 grounded facts，以及从完整 Top-5 session 中无标签、确定性提取的 query-centered evidence windows。窗口保留 lexical anchor、相邻句和相邻同角色 turn，解决纯 facts handoff 遗漏答案或计算操作数的问题；当前日期与问题仍放在 prompt 末尾。旧的 `qa/` 和 `qa_v2_*` 结果会保留，新结果分别写入 `qa_v21_smoke/`、`qa_v21_dev/` 和 `qa_v21_test/`。

在 vLLM 已启动后，先运行 10 个 Dev 样本：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_smoke \
uv run --no-sync bash scripts/run_vmp_v64_qa_experiment.sh
```

确认 smoke 输出合理后运行完整 Dev；脚本会在结束时执行拒答率、事实覆盖率和本地答案质量 gate：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev \
uv run --no-sync bash scripts/run_vmp_v64_qa_experiment.sh
```

只有 Dev QA gate 通过后再生成 Test 答案：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=test \
uv run --no-sync bash scripts/run_vmp_v64_qa_experiment.sh
```

成本表必须显式读取通过 gate 的 QA 目录，避免误用旧结果：

```bash
uv run --no-sync python scripts/export_longmemeval_cost.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_v64_rerank_seed42 \
  --qa-subdir qa_v21_test
```

### 官方 Prompt 兼容判分与统计

QA 目录中的 EM、Token F1 和 Contains Answer 是无需模型的本地诊断指标，不等同于 LongMemEval 论文使用的二元 QA accuracy。仓库现在复刻官方 `evaluate_qa.py` 的六类判分 prompt 与 `yes` 判定规则，并通过同一个本地 vLLM 模型公平判分所有方法。产物会显式标记为 `official_prompt_local_vllm_judge`；它适合仓库内横向比较，但不能伪装成官方固定 GPT-4o judge 的公开分数。

先用保存的 Test 答案运行 10 题 judge smoke；这一步不会重新生成答案：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=judge_smoke \
uv run --no-sync bash scripts/run_vmp_v64_paper_qa_eval.sh
```

确认 judge 仅输出 `yes`/`no` 后，完成全部判分并生成 CSV、Markdown、LaTeX 表格、成对 bootstrap 95% CI 与 exact McNemar 检验：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=all \
uv run --no-sync bash scripts/run_vmp_v64_paper_qa_eval.sh
```

默认产物位于：

```text
outputs/longmemeval/runs/lme_test_vmp_v64_rerank_seed42/
  qa_v21_test/official_judge_local_vllm_v1/
    manifest.json
    summary.json
    paper/qa_paper_report.json
    paper/table3_qa_official_prompt_overall.{csv,md,tex}
    paper/table4_qa_official_prompt_by_type.{csv,md,tex}
    paper/table5_qa_paired_significance.{csv,md,tex}
```

`qa_v21_test/hypotheses/*.jsonl` 同时保持官方上游要求的 `question_id`/`hypothesis` 格式，可用于独立复核。若要与使用固定 GPT-4o judge 的已发表数字直接比较，必须另外用 LongMemEval 上游 evaluator 和完全相同的 judge 版本复评；本地 Qwen judge 分数只能与使用同一 judge 的框架结果比较。

### 官方 OSS 框架公平对比

[`configs/official_frameworks.yaml`](configs/official_frameworks.yaml) 冻结了 Mem0、LangMem、Graphiti、Letta 的包版本和共享实验协议。[`scripts/run_official_framework_paper_experiment.sh`](scripts/run_official_framework_paper_experiment.sh) 每次只运行一个框架，并将 native memory retrieval、V6.4 共享 evidence selection、QA-v2.1 与本地 official-prompt judge 分成可恢复的独立阶段。官方检索现在每完成一个问题就持久化并 `fsync`；命令中断后使用完全相同的环境变量重跑即可从严格校验过的有序前缀继续。

每个官方框架使用独立 uv 环境，避免某个框架升级 `torch`、`transformers`
或 `torchvision` 后影响 vLLM。不要在同一环境混装四个 extra。Mem0 的
embedding 通过官方 OpenAI-compatible provider 调用本机 BGE-M3 服务，因此
Mem0 实验环境本身不安装 `sentence-transformers`：

```bash
UV_PROJECT_ENVIRONMENT=.venv-official-mem0 \
uv sync --extra dev --extra official-mem0

# Mem0 原生 BM25 需要 fastembed 模型，lemmatization 需要 spaCy 模型。
# 在切换到 offline 模式前各执行一次。
UV_PROJECT_ENVIRONMENT=.venv-official-mem0 \
uv run --no-sync python -m spacy download en_core_web_sm

HF_ENDPOINT=https://hf-mirror.com \
UV_PROJECT_ENVIRONMENT=.venv-official-mem0 \
uv run --no-sync python -c \
  "from fastembed import SparseTextEmbedding; list(SparseTextEmbedding(model_name='Qdrant/bm25').embed(['cache warmup']))"

# LangMem/Graphiti 需要进程内 BGE；分别换成自己的环境名和 extra。
# UV_PROJECT_ENVIRONMENT=.venv-official-langmem \
# uv sync --extra dev --extra embeddings --extra official-langmem
```

启动共享 vLLM 时建议给 BGE-M3 预留显存。以下设置已按单张 24 GB 4090D、vLLM 0.26.0 和 32K 上下文校准：`0.72` 只能提供约 1.51 GiB KV cache，不足以启动 32K；`0.75` 可覆盖约 1.75 GiB 的需求，并保留约 25% 显存给 BGE-M3。若两个服务同时运行仍 OOM，可暂时使用 `EMBEDDING_DEVICE=cpu`，但 CPU 延迟不得与 CUDA 延迟放进同一效率表：

```bash
VMP_LLM_MODEL=/home/shenshifan/models/Qwen2.5-7B-Instruct \
VMP_VLLM_SERVED_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
VMP_LLM_API_KEY=local-vllm-key \
VMP_VLLM_GPU_MEMORY_UTILIZATION=0.75 \
VMP_VLLM_MAX_MODEL_LEN=32768 \
VMP_VLLM_ENABLE_TOOL_CALLING=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync bash scripts/serve_vllm.sh
```

Mem0 和 Letta 共享一个独立的本地 BGE-M3 服务。首次创建其隔离环境，然后
在另一个 tmux 窗口启动服务；它与 vLLM 使用不同 Python 环境：

```bash
UV_PROJECT_ENVIRONMENT=.venv-embeddings \
uv sync --extra embeddings

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
UV_PROJECT_ENVIRONMENT=.venv-embeddings \
uv run --no-sync python scripts/serve_embeddings.py \
  --model BAAI/bge-m3 \
  --cache-folder "$HOME/.cache/huggingface" \
  --device cuda --port 8001 --batch-size 2

# `/v1/models` 只检查进程存活；以下请求还会验证模型可从缓存加载并实际编码。
curl -sS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-m3","input":["embedding readiness probe"]}'
```

以 Mem0 为例，按顺序运行以下阶段。`status` 不访问模型，可随时查看已有产物：

```bash
export FRAMEWORK=mem0
export VMP_LLM_API_KEY=local-vllm-key
export VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  # 必须等于 /v1/models 返回的 id
export VMP_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export UV_PROJECT_ENVIRONMENT=.venv-official-mem0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STAGE=smoke          uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=audit          uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=dev_protocol   uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_candidates uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_rerank    uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_qa        uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_judge     uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=status         uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
```

Mem0 的 `dev_protocol` 会先运行 20 个 Dev 样本，逐题记录 JSON 初次失败、重试、
失败类别与长度、最终失败、完整 memory 数量以及 BM25/spaCy 状态。初次请求仍固定
为 2048 tokens；仅在 JSON 无效时使用 8192-token 单次重试。只有最终失败率为 0、初次无效
JSON 比例不高于 2%，且原生 hybrid 依赖全部启用时，`test_candidates` 才会启动。
该门禁不读取 Test 标签。旧目录 `lme_test_mem0_official_candidates_seed42` 使用
512-token 旧协议，`official_v2`/`official_v3` 仍受旧 shell 覆盖值影响，两者仅
保留为诊断记录；新论文运行写入 `lme_test_mem0_official_v4_candidates_seed42`，
不要对旧目录使用 `--resume`。

Graphiti 还需要一个专用且允许清空的 Neo4j：

```bash
export FRAMEWORK=graphiti
export VMP_GRAPHITI_NEO4J_PASSWORD='仅用于该实验容器的密码'
uv run --no-sync bash scripts/serve_graphiti_neo4j.sh
```

Letta 使用上面同一个 BGE OpenAI-compatible endpoint，并额外需要固定版本的
Letta server：

```bash
uv run --no-sync bash scripts/serve_letta.sh
```

不得用 `FRAMEWORK=mem0` 的候选 run 搭配其他框架的后续阶段；run ID、方法名、数据哈希、split、模型、包版本及所有不可变参数都会写入 manifest 并在恢复时核验。原生检索与共享 rerank 应分别报告，以区分框架本身的 memory quality 和统一后处理带来的增益。

四个框架全部 judge 完成后，将 VMP 和官方框架的独立 rerank run 合并为一个严格可比、不可变的论文证据包。构建器不重新调用模型；它会沿着 rerank → QA → judge 的 manifest 哈希链验证数据、split、完整题序、prompt、generation 参数和实际模型，然后统一生成检索表、QA 表、bootstrap/McNemar 显著性表、成本表以及以 official-prompt 正确数为分母的效率表：

```bash
COMPARE_DIR=outputs/longmemeval/comparisons/official_frameworks_qwen_seed42_v1

uv run --no-sync python scripts/build_longmemeval_paper_comparison.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_mem0_official_v4_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_langmem_official_v4_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_graphiti_official_v4_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_letta_official_v4_v64_rerank_seed42 \
  --output "${COMPARE_DIR}" \
  --reference-method vmp_hierarchical__vllm_boundary \
  --bootstrap-samples 10000 \
  --seed 42
```

比较目录是不可变产物；若需要重新构建，请使用新的版本化目录名，以免覆盖论文证据。`token_accounting_complete=false` 表示某个官方框架没有为所有问题导出原生 LLM usage，此时 token/correct 只是明确标注的观测下界，绝不能把缺失 usage 当作 0。

## 公平比较原则

论文主表遵循以下约束：

- 相同 LongMemEval 数据、切分与 session-level evidence。
- 相同本地 reader LLM、temperature、top-p 和输出上限。
- 所有向量方法统一使用相同 embedding 模型。
- Dev 仅用于模型选择，Test 标签在架构冻结前保持封闭。
- 官方框架结果与 style reimplementation 明确区分。
- 同时报告检索质量、QA、token、延迟、存储增长和失败率。

## 配置与产物

- 基础配置：[`configs/default.yaml`](configs/default.yaml)
- LongMemEval：[`configs/longmemeval.yaml`](configs/longmemeval.yaml)
- VMP-v6.4：[`configs/vmp_v64.yaml`](configs/vmp_v64.yaml)
- 环境变量模板：[`.env.example`](.env.example)

默认情况下，数据缓存、模型、运行日志和 `outputs/longmemeval/` 不会提交到 Git。

## 研究状态

VMP-v6.4 已完成冻结的 LongMemEval Test 检索、QA-v2.1 和本地 official-prompt 判分。VMP-Hierarchical 的 `Recall-All@5 = 0.9388`、`MRR = 0.9555`；共享本地 Qwen judge accuracy 为 `0.5100`，VMP-Tuned 为 `0.4825`，差值 `+0.0275`，bootstrap 95% CI 为 `[-0.0025, 0.0575]`，exact McNemar `p = 0.0895`。因此当前改进尚未达到常用的 0.05 显著性阈值，也不能宣称 SOTA。下一阶段是运行上述官方 Mem0、LangMem、Graphiti、Letta 同模型对比，并增加第二 benchmark。

## License

本项目在 `pyproject.toml` 中声明为 MIT License。
