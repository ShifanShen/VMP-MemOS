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

VMP-v6.4 已完成冻结的 LongMemEval Test 检索与 QA-v2.1 生成。当前 Test 上 VMP-Hierarchical 的 `Recall-All@5 = 0.9388`、`MRR = 0.9555`；本地词法 QA 指标为 `Token F1 = 0.4007`、`Contains Answer = 0.3963`。这些 QA 数字是诊断指标，不是官方 judge accuracy。下一阶段是使用新增的统一 judge/report 链路完成 VMP 与官方 Mem0、LangMem、Graphiti、Letta 的同模型对比，并增加第二 benchmark；在这些实验完成前，本仓库不宣称 SOTA。

## License

本项目在 `pyproject.toml` 中声明为 MIT License。
