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

The current research entry point is VMP-v6.4. It restores the high-recall V6.2 extraction instruction templates while retaining V6.3 list-valued fact normalization, evidence-coordinate contamination guards, bare-list-marker filtering, and the V4 excerpt. The candidate pool, Top-k policy, and coverage weights remain fixed. Start with the four-question diagnostic smoke run:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_smoke \
uv run --no-sync bash scripts/run_vmp_v64_experiment.sh
```

Then run the complete Dev experiment:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_rerank \
uv run --no-sync bash scripts/run_vmp_v64_experiment.sh
```

The frozen Test candidate, rerank, and QA stages should only be opened after the Dev quality gate passes. Set `RERANK_RESUME=1` only when continuing an interrupted run with identical settings.

### Hybrid QA Reader v2.1

QA-v2.1 combines reranker-grounded facts with label-free, deterministic query-centered evidence windows extracted from the complete Top-5 sessions. Each window retains lexical anchors, adjacent sentences, and adjacent same-role turns, recovering answer values and reasoning operands lost by a facts-only handoff. The current date and question remain at the end of the prompt. Existing `qa/` and `qa_v2_*` artifacts remain untouched; new outputs are isolated under `qa_v21_smoke/`, `qa_v21_dev/`, and `qa_v21_test/`.

With vLLM already running, begin with ten Dev samples:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev_smoke \
uv run --no-sync bash scripts/run_vmp_v64_qa_experiment.sh
```

After inspecting the smoke answers, run all Dev questions. The script enforces refusal-rate, fact-coverage, and local-answer-quality gates at the end:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=dev \
uv run --no-sync bash scripts/run_vmp_v64_qa_experiment.sh
```

Generate Test answers only after the Dev QA gate passes:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=test \
uv run --no-sync bash scripts/run_vmp_v64_qa_experiment.sh
```

Cost export must explicitly select the gated QA directory so that stale results cannot be consumed accidentally:

```bash
uv run --no-sync python scripts/export_longmemeval_cost.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_v64_rerank_seed42 \
  --qa-subdir qa_v21_test
```

### Official-prompt-compatible judging and statistics

EM, Token F1, and Contains Answer in a QA directory are model-free diagnostics; they are not the binary QA accuracy used by the LongMemEval paper. The repository now mirrors the six task-specific prompts and `yes` decision rule from the official `evaluate_qa.py`, while using one shared local vLLM model to judge every method fairly. Artifacts are explicitly labeled `official_prompt_local_vllm_judge`: they support within-repository comparisons but must not be presented as scores from the official pinned GPT-4o judge.

First judge ten saved Test predictions. This does not regenerate any answer:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=judge_smoke \
uv run --no-sync bash scripts/run_vmp_v64_paper_qa_eval.sh
```

After confirming that the judge returns only `yes` or `no`, judge the complete run and export CSV, Markdown, and LaTeX tables, paired bootstrap 95% confidence intervals, and exact McNemar tests:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VMP_LLM_API_KEY=local-vllm-key \
VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
STAGE=all \
uv run --no-sync bash scripts/run_vmp_v64_paper_qa_eval.sh
```

Default artifacts are written under:

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

Files under `qa_v21_test/hypotheses/*.jsonl` retain the upstream `question_id`/`hypothesis` format for independent regrading. Direct comparison with published results obtained from the pinned GPT-4o judge requires a separate run of the upstream LongMemEval evaluator with that exact judge version. A local-Qwen judge score is comparable only with methods graded by the same local judge.

### Fair comparison with official OSS frameworks

[`configs/official_frameworks.yaml`](configs/official_frameworks.yaml) freezes the package versions and shared protocol for Mem0, LangMem, Graphiti, and Letta. [`scripts/run_official_framework_paper_experiment.sh`](scripts/run_official_framework_paper_experiment.sh) runs one framework at a time and separates native memory retrieval, shared V6.4 evidence selection, QA-v2.1, and the local official-prompt judge into resumable stages. Official retrieval now persists and `fsync`s every completed question. Re-running an interrupted command with identical environment variables resumes only after validating the existing ordered prefix and all immutable inputs.

Use a separate uv environment for each official framework so that one adapter
cannot upgrade `torch`, `transformers`, or `torchvision` underneath vLLM. Do not
combine all four extras in one environment. Mem0 calls the local BGE-M3 service
through its official OpenAI-compatible provider, so its experiment environment
does not install `sentence-transformers`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-official-mem0 \
uv sync --extra dev --extra official-mem0

# Mem0 native BM25 needs the fastembed model, while lemmatization needs a
# spaCy model. Run both commands once before switching to offline mode.
UV_PROJECT_ENVIRONMENT=.venv-official-mem0 \
uv run --no-sync python -m spacy download en_core_web_sm

HF_ENDPOINT=https://hf-mirror.com \
UV_PROJECT_ENVIRONMENT=.venv-official-mem0 \
uv run --no-sync python -c \
  "from fastembed import SparseTextEmbedding; list(SparseTextEmbedding(model_name='Qdrant/bm25').embed(['cache warmup']))"

# LangMem/Graphiti use in-process BGE; give each its own environment and extra.
# UV_PROJECT_ENVIRONMENT=.venv-official-langmem \
# uv sync --extra dev --extra embeddings --extra official-langmem
```

When serving the shared model, reserve GPU memory for BGE-M3. The values below are calibrated for one 24 GB 4090D, vLLM 0.26.0, and a 32K context: `0.72` exposes only about 1.51 GiB of KV cache and cannot start at 32K, while `0.75` covers the roughly 1.75 GiB requirement and leaves about 25% of the GPU for BGE-M3. If the two services still OOM when colocated, temporarily set `EMBEDDING_DEVICE=cpu`; CPU and CUDA latency must not be mixed in one efficiency table.

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

Mem0 and Letta share a separate local BGE-M3 service. Create its isolated
environment once, then start it in another tmux window. It does not share the
vLLM Python environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-embeddings \
uv sync --extra embeddings

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
UV_PROJECT_ENVIRONMENT=.venv-embeddings \
uv run --no-sync python scripts/serve_embeddings.py \
  --model BAAI/bge-m3 \
  --cache-folder "$HOME/.cache/huggingface" \
  --device cuda --port 8001 --batch-size 2

# `/v1/models` only checks liveness; this request also loads the cached model
# and performs a real encoding readiness probe.
curl -sS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-m3","input":["embedding readiness probe"]}'
```

For Mem0, run these stages in order. `status` is model-free and can be used at any time:

```bash
export FRAMEWORK=mem0
export VMP_LLM_API_KEY=local-vllm-key
export VMP_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  # must match the /v1/models id
export VMP_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export UV_PROJECT_ENVIRONMENT=.venv-official-mem0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

STAGE=smoke           uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=audit           uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=dev_protocol    uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_candidates uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_rerank     uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_qa         uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=test_judge      uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
STAGE=status          uv run --no-sync bash scripts/run_official_framework_paper_experiment.sh
```

For Mem0, `dev_protocol` first evaluates 20 Dev samples and records initial JSON
failures, retries, final failures, complete memory counts, and BM25/spaCy status
for every question. `test_candidates` starts only when the final failure rate is
zero, the initial-invalid rate is at most 2%, and the native hybrid dependencies
are active. This gate does not read Test labels. The old
`lme_test_mem0_official_candidates_seed42` directory used the 512-token pilot
protocol. Keep it as a diagnostic baseline; protocol-v2 paper runs use
`lme_test_mem0_official_v2_candidates_seed42` and must not resume the old run.

Graphiti also requires a dedicated, disposable Neo4j instance:

```bash
export FRAMEWORK=graphiti
export VMP_GRAPHITI_NEO4J_PASSWORD='password-for-this-experiment-container-only'
uv run --no-sync bash scripts/serve_graphiti_neo4j.sh
```

Letta uses the same BGE OpenAI-compatible endpoint above and additionally
requires the pinned Letta server:

```bash
uv run --no-sync bash scripts/serve_letta.sh
```

Never feed a candidate run created with `FRAMEWORK=mem0` into another framework's later stages. The manifest and resume gate bind run ID, method, dataset hash, split, model, package versions, and every immutable setting. Report native retrieval and shared reranking separately to distinguish framework memory quality from shared post-processing gains.

After all four framework judges complete, merge the VMP and official-framework rerank runs into one strict, immutable paper-evidence bundle. The builder never calls a model again. It validates the complete rerank → QA → judge manifest-hash chain, including dataset, split, ordered coverage, prompts, generation settings, and observed models, then emits unified retrieval, QA, bootstrap/McNemar significance, cost, and official-correctness efficiency tables:

```bash
COMPARE_DIR=outputs/longmemeval/comparisons/official_frameworks_qwen_seed42_v1

uv run --no-sync python scripts/build_longmemeval_paper_comparison.py \
  --retrieval-run outputs/longmemeval/runs/lme_test_vmp_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_mem0_official_v2_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_langmem_official_v2_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_graphiti_official_v2_v64_rerank_seed42 \
  --retrieval-run outputs/longmemeval/runs/lme_test_letta_official_v2_v64_rerank_seed42 \
  --output "${COMPARE_DIR}" \
  --reference-method vmp_hierarchical__vllm_boundary \
  --bootstrap-samples 10000 \
  --seed 42
```

The comparison directory is immutable. Use a new versioned directory name when rebuilding it so that paper evidence is never overwritten. `token_accounting_complete=false` means that an official framework did not export native LLM usage for every question; tokens per correct is then an explicitly marked observed lower bound, never a missing value imputed as zero.

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
- VMP-v6.4: [`configs/vmp_v64.yaml`](configs/vmp_v64.yaml)
- Environment template: [`.env.example`](.env.example)

Dataset caches, models, runtime logs, and `outputs/longmemeval/` are ignored by Git by default.

## Research status

VMP-v6.4 has completed frozen LongMemEval Test retrieval, QA-v2.1, and local official-prompt judging. VMP-Hierarchical reaches `Recall-All@5 = 0.9388` and `MRR = 0.9555`. Its shared local-Qwen judge accuracy is `0.5100`, versus `0.4825` for VMP-Tuned: a `+0.0275` difference with bootstrap 95% CI `[-0.0025, 0.0575]` and exact McNemar `p = 0.0895`. The current gain therefore does not pass the conventional 0.05 significance threshold and is not a SOTA claim. The next stage is the same-model official Mem0, LangMem, Graphiti, and Letta comparison above, followed by a second benchmark.

## License

The project declares the MIT License in `pyproject.toml`.
