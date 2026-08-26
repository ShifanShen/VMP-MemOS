# VMP-MemOS

[简体中文](README.md) | [English](README_EN.md)

VMP-MemOS is an explainable memory policy layer for long-horizon LLM agents. It separates what a memory contains from how that memory should be managed, providing auditable policy signals for writing, updating, merging, archiving, retrieving, and compressing memory.

> The project is currently in the paper-experiment stage. The implementation, evaluation pipeline, and reproduction scripts are public; final claims must be based on the complete frozen-configuration test results.

## Core idea

Conventional RAG systems usually retrieve by content similarity alone. In addition to content relevance, VMP-MemOS maintains explicit `PolicyFeatures` describing importance, confidence, recency, staleness, contradiction, redundancy, update signals, token cost, and other management properties.

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

## Features

- Strict Pydantic schemas for `Event`, `MemoryCandidate`, `MemoryItem`, `PolicyFeatures`, `MemoryOperation`, `RetrievalResult`, `BenchmarkSample`, and `BenchmarkResult`.
- File, vector, and hybrid persistence backends.
- Explainable rule-based policy and a lightweight learned policy.
- Structured logs for ADD, UPDATE, MERGE, ARCHIVE, RETRIEVE, and IGNORE operations.
- BM25, vector, recency, importance, and VMP-family retrieval baselines.
- LongMemEval loading, fixed Dev/Test splits, retrieval, QA, ablations, cost analysis, and paper-table exports.
- Local-vLLM anonymous evidence extraction followed by deterministic evidence-set coverage.
- Official OSS adapters for Mem0, LangMem, Graphiti, and Letta with a shared local model and embedder for fair experiments.
- Resumable runs, strict quality gates, and dataset/model provenance auditing.

## Repository layout

```text
configs/              Reproducible experiment configurations
data/                 Toy benchmark and local dataset directories
memory_workspace/     File memories, archives, versions, and runtime logs
scripts/              Training, evaluation, serving, and export entry points
src/vmp_memos/        Core Python package
tests/                Network-free unit and integration tests
outputs/              Local experiment artifacts (ignored by default)
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Core functionality and the toy benchmark run on CPU
- Linux, CUDA, and an NVIDIA GPU are recommended for BGE-M3/vLLM LongMemEval experiments

## Quick start

```bash
git clone https://github.com/ShifanShen/VMP-MemOS.git
cd VMP-MemOS

uv sync --extra dev
uv run python scripts/init_workspace.py
uv run python scripts/run_benchmark.py
```

Run the test suite:

```bash
uv run python -m pytest -q
```

Install local embedding dependencies when needed:

```bash
uv sync --extra dev --extra embeddings
```

## Local vLLM server

All comparable paper experiments use the same local model. After installing a vLLM build compatible with the server CUDA environment, start its OpenAI-compatible endpoint:

```bash
export VMP_LLM_MODEL=/path/to/Qwen2.5-7B-Instruct
export VMP_LLM_API_KEY=local-vllm-key
export VMP_VLLM_ENABLE_TOOL_CALLING=0

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync bash scripts/serve_vllm.sh
```

Inspect the model ID exposed by the server:

```bash
curl -s -H "Authorization: Bearer local-vllm-key" \
  http://127.0.0.1:8000/v1/models
```

The client-side `VMP_LLM_MODEL` must match the returned `id`, even when vLLM loaded the weights from a local filesystem path.

## LongMemEval experiments

Download the official cleaned dataset:

```bash
uv run --no-sync python scripts/download_longmemeval.py \
  --target data/longmemeval \
  --files longmemeval_s_cleaned.json
```

For restricted networks, a Hugging Face mirror can be used:

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HUB_DOWNLOAD_TIMEOUT=600 \
uv run --no-sync python scripts/download_longmemeval.py \
  --target data/longmemeval \
  --files longmemeval_s_cleaned.json
```

The current research entry point is VMP-v6.3. It keeps the V6.2 candidate pool, Top-k policy, and coverage weights fixed while adding list-valued fact normalization, evidence-coordinate contamination guards, and bare-list-marker filtering. Start with the three-question diagnostic smoke run:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_smoke \
uv run --no-sync bash scripts/run_vmp_v63_experiment.sh
```

Then run the complete Dev experiment:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v63_experiment.sh
```

The frozen Test candidate, rerank, and QA stages should only be opened after the Dev quality gate passes. Set `RERANK_RESUME=1` only when continuing an interrupted run with identical settings.

## Fair-comparison policy

Main paper comparisons follow these constraints:

- Identical LongMemEval data, split, and session-level evidence.
- Identical local reader LLM, temperature, top-p, and output limit.
- The same embedding model for every vector-based method.
- Dev is used for model selection; Test labels remain sealed until the architecture is frozen.
- Official framework results are clearly separated from style reimplementations.
- Retrieval quality, QA, tokens, latency, storage growth, and failure rates are reported together.

## Configuration and artifacts

- Base configuration: [`configs/default.yaml`](configs/default.yaml)
- LongMemEval: [`configs/longmemeval.yaml`](configs/longmemeval.yaml)
- VMP-v6.3: [`configs/vmp_v63.yaml`](configs/vmp_v63.yaml)
- Environment template: [`.env.example`](.env.example)

Dataset caches, models, runtime logs, and `outputs/longmemeval/` are ignored by Git by default.

## Research status

The repository contains the complete experiment infrastructure and the VMP-v6.3 Dev validation workflow. V6.2 reached `Recall-All@5 = 0.9362` on Dev but failed the zero-regression gate because of one regressed question. The V6.3 corrections pass offline replay; formal results still require a fresh local-vLLM run and the unchanged strict gate. Public main results, comparisons with recent academic systems, and a second benchmark still require frozen-configuration experiments, so this repository does not currently claim state of the art.

## License

The project declares the MIT License in `pyproject.toml`.
