# VMP-v3 experiment pipeline

## Purpose

VMP-v3 keeps retrieval quality anchored to the shared BGE-M3 baseline while
testing whether policy features can safely improve ordering. Lifecycle
operations are annotations, not destructive query-time mutations.

## Data and training boundary

1. Create the fixed `seed=42` split: 100 Dev and 400 Test questions.
2. Parse and validate every LongMemEval question/session date.
3. Build BGE-M3 embeddings once and reuse the SQLite embedding cache.
4. Tune only on answerable Dev questions.
5. Freeze a schema `1.3` model with dataset, split, embedding and search
   provenance.
6. Enforce the Dev quality gate before loading Test labels.

## Retrieval path

```text
sessions
  -> shared BGE-M3 semantic relevance
  -> normalized BM25 lexical relevance
  -> weighted hybrid anchor
  -> anchor Top-20 candidate pool
  -> query-text temporal intent gate
  -> bounded VMP policy delta
  -> soft lifecycle penalty
  -> Top-10 evidence export
  -> Top-5 shared reader context
```

The first tuning trial is always pure dense retrieval:

```text
semantic_anchor_weight = 1
lexical_anchor_weight = 0
policy_adjustment_limit = 0
archive_score_penalty = 0
```

The search chooses models lexicographically by official `Recall-All@5`, then
the composite objective and MRR. This prevents a lower-recall policy model from
winning merely because it stores or returns fewer tokens.

## Policy behavior

- `recency`, `staleness`, `contradiction`, `update_signal` and
  `action_signal` are disabled unless the query text contains temporal/update
  intent.
- Constant LongMemEval session features (`scope_match`, `confidence`,
  `importance`, `success_contribution`) are retained for schema/ablation
  compatibility but are not sampled by the V3 tuner.
- `policy_adjustment_limit` bounds how far policy can move the hybrid anchor
  score.
- A candidate must remain inside the anchor Top-20 pool before policy can
  rerank it.

## Lifecycle behavior

- Lifecycle relations are inferred from session content and chronology, not
  from gold labels or `question_type`.
- Supersession requires a newer session with an explicit update marker and
  conservative lexical overlap.
- Duplicate detection uses a high near-exact threshold.
- `ARCHIVE` and `MERGE` write `superseded` or `duplicate` status and a small
  score penalty.
- Original chunks, embeddings, source IDs and provenance are retained.
- Retrieval never mutates `adapter.chunks`; repeated retrieval is deterministic.

## Quality gate

Defaults:

```text
Dev Recall-All@5 >= 0.90
Dev delta versus dense safety trial >= +0.02
```

Failure exits before Test evaluation. Adjusting a gate is an experimental
protocol change and must be recorded, not silently bypassed.

## Server entry point

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
RUN_ID=lme_test_vmp_v3_seed42 \
DATA_PATH=data/longmemeval/longmemeval_s_cleaned.json \
uv run --no-sync bash scripts/run_vmp_tuned_experiment.sh
```

Keep `RUN_QA=0` until retrieval passes the gate and beats the frozen baselines.
