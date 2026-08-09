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

当前研究入口是 VMP-v6.2。建议先执行只处理三个诊断样本的 smoke：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_smoke \
uv run --no-sync bash scripts/run_vmp_v62_experiment.sh
```

再执行完整 Dev 实验：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v62_experiment.sh
```

只有 Dev gate 通过后，才应执行冻结的 Test candidate、rerank 和 QA 流程。中断后可设置 `RERANK_RESUME=1` 继续同一配置的 run。

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
- VMP-v6.2：[`configs/vmp_v62.yaml`](configs/vmp_v62.yaml)
- 环境变量模板：[`.env.example`](.env.example)

默认情况下，数据缓存、模型、运行日志和 `outputs/longmemeval/` 不会提交到 Git。

## 研究状态

当前仓库已经具备完整实验基础设施和 VMP-v6.2 Dev 验证入口。公开主结果、近期学术框架对比以及第二 benchmark 仍需在冻结配置上完成，因此本仓库暂不宣称 SOTA。

## License

本项目在 `pyproject.toml` 中声明为 MIT License。
