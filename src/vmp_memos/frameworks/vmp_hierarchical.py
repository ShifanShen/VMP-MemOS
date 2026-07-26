"""VMP-v5 hierarchical session/turn retrieval for LongMemEval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import Field, FiniteFloat, JsonValue, PositiveInt, model_validator

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks.base import MemoryChunk, RetrievedMemory, chunk_from_event
from vmp_memos.frameworks.text import (
    clamp01,
    dense_cosine,
    estimate_tokens,
    sparse_cosine,
    term_counts,
)
from vmp_memos.frameworks.vmp_tuned import (
    VMPTunedAblation,
    VMPTunedAdapter,
    VMPTunedModel,
    normalized_bm25_scores,
)
from vmp_memos.schemas import Event
from vmp_memos.schemas.base import NonEmptyStr, NonNegativeFloat, SchemaModel, Score


class VMPHierarchicalModel(SchemaModel):
    """Frozen Dev-only V5 fusion model wrapping a safe V4 policy model."""

    schema_version: NonEmptyStr = "2.0"
    model_type: NonEmptyStr = "vmp_v5_hierarchical_session_turn_ranker"
    base_model: VMPTunedModel
    session_semantic_weight: Score
    turn_semantic_weight: Score
    turn_lexical_weight: Score
    turn_pooling_top_n: PositiveInt = 1
    training_split: NonEmptyStr = "dev"
    split_id: NonEmptyStr
    split_manifest_sha256: NonEmptyStr
    split_assignment_sha256: NonEmptyStr
    dataset_sha256: NonEmptyStr
    embedding_identifier: str | None = None
    best_objective: FiniteFloat
    dev_metrics: dict[NonEmptyStr, NonNegativeFloat] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frozen_model(self) -> VMPHierarchicalModel:
        """Reject stale, test-trained, or provenance-inconsistent artifacts."""

        if self.schema_version != "2.0":
            raise ValueError("VMP-v5 model schema is obsolete; retrain the model")
        if self.training_split != "dev":
            raise ValueError("VMP-v5 artifacts must be trained only on the dev split")
        if self.metadata.get("test_labels_used") is not False:
            raise ValueError("VMP-v5 metadata must prove test labels were not used")
        if self.split_id != self.base_model.split_id:
            raise ValueError("VMP-v5 and base VMP model split IDs differ")
        if self.split_manifest_sha256 != self.base_model.split_manifest_sha256:
            raise ValueError("VMP-v5 and base VMP split manifests differ")
        if self.dataset_sha256 != self.base_model.dataset_sha256:
            raise ValueError("VMP-v5 and base VMP datasets differ")
        if self.embedding_identifier != self.base_model.embedding_identifier:
            raise ValueError("VMP-v5 and base VMP embedding identifiers differ")
        if self.base_model.promotion_ranker is not None:
            raise ValueError(
                "VMP-v5 must refit or disable the V4 promotion ranker after "
                "changing semantic-score geometry"
            )
        if self.base_model.protected_dense_count != self.base_model.safety_top_k:
            raise ValueError(
                "VMP-v5 must freeze Top-5 membership to the hierarchical dense head"
            )
        if self.fusion_weight_sum <= 0.0:
            raise ValueError("VMP-v5 fusion weights must have a positive sum")
        return self

    @property
    def fusion_weight_sum(self) -> float:
        """Return the denominator used by hierarchical score fusion."""

        return (
            float(self.session_semantic_weight)
            + float(self.turn_semantic_weight)
            + float(self.turn_lexical_weight)
        )

    def fuse(
        self,
        *,
        session_semantic: float,
        turn_semantic: float,
        turn_lexical: float,
    ) -> float:
        """Fuse session and turn signals using frozen Dev-only weights."""

        return hierarchical_fusion_score(
            session_semantic=session_semantic,
            turn_semantic=turn_semantic,
            turn_lexical=turn_lexical,
            session_semantic_weight=float(self.session_semantic_weight),
            turn_semantic_weight=float(self.turn_semantic_weight),
            turn_lexical_weight=float(self.turn_lexical_weight),
        )

    def save(self, path: str | Path) -> Path:
        """Persist the complete hierarchical model, including its base policy."""

        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> VMPHierarchicalModel:
        """Load and validate one frozen VMP-v5 artifact."""

        model_path = Path(path).expanduser().resolve()
        return cls.model_validate_json(model_path.read_text(encoding="utf-8"))


class VMPHierarchicalAdapter(VMPTunedAdapter):
    """Rank sessions with fused session/turn evidence and VMP policy ordering."""

    name = "vmp_hierarchical"

    def __init__(
        self,
        *,
        model: VMPHierarchicalModel,
        embedder: BaseEmbedder | None = None,
        ablation: VMPTunedAblation | None = None,
    ) -> None:
        super().__init__(
            model=model.base_model,
            embedder=embedder,
            ablation=ablation,
        )
        self.hierarchical_model = model
        self.name = "vmp_hierarchical"
        self.turn_chunks: list[MemoryChunk] = []
        self.skipped_empty_turn_count = 0
        self._hierarchical_components_by_session: dict[
            str,
            dict[str, float],
        ] = {}

    @property
    def memory_count(self) -> int:
        """Count both session evidence and the turn-level retrieval index."""

        return len(self.chunks) + len(self.turn_chunks)

    @property
    def total_tokens(self) -> int:
        """Report physical indexed tokens, including the auxiliary turn index."""

        return sum(
            chunk.token_count for chunk in [*self.chunks, *self.turn_chunks]
        )

    @property
    def storage_size_bytes(self) -> int:
        """Report logical content and vector storage for both hierarchy levels."""

        chunks = [*self.chunks, *self.turn_chunks]
        content_bytes = sum(len(chunk.content.encode("utf-8")) for chunk in chunks)
        vector_bytes = sum(4 * len(chunk.content_embedding) for chunk in chunks)
        return content_bytes + vector_bytes

    def _reset_impl(self) -> None:
        super()._reset_impl()
        self.turn_chunks = []
        self.skipped_empty_turn_count = 0
        self._hierarchical_components_by_session = {}

    def _ingest_event_impl(self, event: Event) -> None:
        raise ValueError(
            "vmp_hierarchical requires session ingestion so turns can be "
            "aggregated back to official session IDs"
        )

    def _ingest_session_impl(self, events: list[Event]) -> None:
        super()._ingest_session_impl(events)
        for event in events:
            turn_chunk = contextual_turn_chunk(event)
            if turn_chunk is not None:
                self.turn_chunks.append(turn_chunk)
            else:
                self.skipped_empty_turn_count += 1

    def _finalize_ingestion_impl(self) -> None:
        super()._finalize_ingestion_impl()
        self._embed_turn_chunks()

    def _semantic_relevance_scores(
        self,
        query: str,
        *,
        query_embedding: Sequence[float] | None,
    ) -> list[float]:
        session_scores = super()._semantic_relevance_scores(
            query,
            query_embedding=query_embedding,
        )
        turn_semantic_scores = self._turn_semantic_scores(
            query,
            query_embedding=query_embedding,
        )
        turn_lexical_scores = normalized_bm25_scores(
            query,
            [chunk.content for chunk in self.turn_chunks],
        )
        semantic_by_session: dict[str, list[float]] = {}
        lexical_by_session: dict[str, list[float]] = {}
        for chunk, semantic, lexical in zip(
            self.turn_chunks,
            turn_semantic_scores,
            turn_lexical_scores,
            strict=True,
        ):
            session_id = chunk.source_session_id or chunk.memory_id
            semantic_by_session.setdefault(session_id, []).append(semantic)
            lexical_by_session.setdefault(session_id, []).append(lexical)

        fused_scores: list[float] = []
        components: dict[str, dict[str, float]] = {}
        for chunk, session_semantic in zip(
            self.chunks,
            session_scores,
            strict=True,
        ):
            session_id = chunk.source_session_id or chunk.memory_id
            turn_semantic = pool_top_scores(
                semantic_by_session.get(session_id, ()),
                top_n=self.hierarchical_model.turn_pooling_top_n,
            )
            turn_lexical = pool_top_scores(
                lexical_by_session.get(session_id, ()),
                top_n=self.hierarchical_model.turn_pooling_top_n,
            )
            fused = self.hierarchical_model.fuse(
                session_semantic=session_semantic,
                turn_semantic=turn_semantic,
                turn_lexical=turn_lexical,
            )
            fused_scores.append(fused)
            components[session_id] = {
                "session_semantic_score": session_semantic,
                "turn_semantic_score": turn_semantic,
                "turn_lexical_score": turn_lexical,
                "hierarchical_fused_score": fused,
            }
        self._hierarchical_components_by_session = components
        return fused_scores

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        retrieved = super()._retrieve_impl(
            query,
            top_k=top_k,
            question_date=question_date,
            metadata=metadata,
        )
        enriched: list[RetrievedMemory] = []
        for memory in retrieved:
            session_id = memory.source_session_id or memory.memory_id
            item_metadata = dict(memory.metadata)
            item_metadata.update(
                cast(
                    dict[str, JsonValue],
                    self._hierarchical_components_by_session.get(session_id, {}),
                )
            )
            item_metadata.update(
                {
                    "retrieval_strategy": self.name,
                    "base_model_type": item_metadata.get("model_type"),
                    "model_type": self.hierarchical_model.model_type,
                    "hierarchical_model_type": self.hierarchical_model.model_type,
                    "hierarchical_schema_version": (
                        self.hierarchical_model.schema_version
                    ),
                    "turn_pooling_top_n": (
                        self.hierarchical_model.turn_pooling_top_n
                    ),
                }
            )
            enriched.append(memory.model_copy(update={"metadata": item_metadata}))
        return enriched

    def stats(self) -> dict[str, JsonValue]:
        """Add hierarchy-specific memory and ranking provenance."""

        stats = super().stats()
        stats["name"] = self.name
        stats["session_memory_count"] = len(self.chunks)
        stats["turn_memory_count"] = len(self.turn_chunks)
        stats["skipped_empty_turn_count"] = self.skipped_empty_turn_count
        stats["physical_memory_count"] = self.memory_count
        stats["model_type"] = self.hierarchical_model.model_type
        stats["model_schema_version"] = self.hierarchical_model.schema_version
        stats["ranking_pipeline"] = (
            "session_embedding + pooled_turn_embedding + turn_bm25 -> "
            "hierarchical_dense_top10 -> frozen_vmp_policy_ordering -> "
            "cached_non_destructive_lifecycle"
        )
        stats["hierarchical_fusion"] = cast(
            JsonValue,
            {
                "schema_version": self.hierarchical_model.schema_version,
                "session_semantic_weight": float(
                    self.hierarchical_model.session_semantic_weight
                ),
                "turn_semantic_weight": float(
                    self.hierarchical_model.turn_semantic_weight
                ),
                "turn_lexical_weight": float(
                    self.hierarchical_model.turn_lexical_weight
                ),
                "turn_pooling_top_n": (
                    self.hierarchical_model.turn_pooling_top_n
                ),
                "turn_representation": "role_prefixed_raw_turn",
            },
        )
        return stats

    def _embed_turn_chunks(self) -> None:
        if self.embedder is None:
            return
        pending = [
            chunk for chunk in self.turn_chunks if not chunk.content_embedding
        ]
        if not pending:
            return
        vectors = self.embedder.embed([chunk.content for chunk in pending])
        for chunk, vector in zip(pending, vectors, strict=True):
            chunk.content_embedding = list(vector)

    def _turn_semantic_scores(
        self,
        query: str,
        *,
        query_embedding: Sequence[float] | None,
    ) -> list[float]:
        query_counts = term_counts(query)
        return [
            (
                dense_cosine(query_embedding, chunk.content_embedding)
                if query_embedding is not None and chunk.content_embedding
                else sparse_cosine(query_counts, term_counts(chunk.content))
            )
            for chunk in self.turn_chunks
        ]


def contextual_turn_chunk(event: Event) -> MemoryChunk | None:
    """Create the role-prefixed turn representation shared by train and test."""

    content = str(event.content).strip()
    if not content:
        return None
    chunk = chunk_from_event(event)
    role = str(event.metadata.get("role") or event.event_type.value)
    contextual_content = f"{role}: {content}"
    return chunk.model_copy(
        update={
            "content": contextual_content,
            "memory_type": "turn",
            "token_count": estimate_tokens(contextual_content),
        }
    )


def pool_top_scores(scores: Sequence[float], *, top_n: int) -> float:
    """Mean-pool the strongest turns without depending on session length."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    selected = sorted((clamp01(score) for score in scores), reverse=True)[:top_n]
    return sum(selected) / len(selected) if selected else 0.0


def hierarchical_fusion_score(
    *,
    session_semantic: float,
    turn_semantic: float,
    turn_lexical: float,
    session_semantic_weight: float,
    turn_semantic_weight: float,
    turn_lexical_weight: float,
) -> float:
    """Return one normalized session score from the three hierarchy signals."""

    weights = (
        float(session_semantic_weight),
        float(turn_semantic_weight),
        float(turn_lexical_weight),
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("hierarchical fusion weights cannot be negative")
    denominator = sum(weights)
    if denominator <= 0.0:
        raise ValueError("hierarchical fusion weights must have a positive sum")
    return clamp01(
        (
            weights[0] * clamp01(session_semantic)
            + weights[1] * clamp01(turn_semantic)
            + weights[2] * clamp01(turn_lexical)
        )
        / denominator
    )
