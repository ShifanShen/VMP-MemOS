"""VMP-MemOS rule-based retrieval adapter for LongMemEval."""

from __future__ import annotations

from pydantic import JsonValue

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks.base import InMemorySessionAdapter, MemoryChunk, RetrievedMemory
from vmp_memos.frameworks.text import clamp01, heuristic_importance, parse_date, recency_score
from vmp_memos.policy import PolicyFeatureBuilder, PolicyFeatureContext, RuleBasedPolicyController
from vmp_memos.schemas import MemoryItem, MemorySource, MemoryType, PolicyFeatures


class VMPRuleAdapter(InMemorySessionAdapter):
    """Rule-based VMP adapter that scores retrieval with policy features."""

    name = "vmp_rule"

    def __init__(self, *, embedder: BaseEmbedder | None = None) -> None:
        super().__init__()
        self.embedder = embedder
        self.feature_builder = PolicyFeatureBuilder()
        self.controller = RuleBasedPolicyController()

    def _finalize_ingestion_impl(self) -> None:
        self._embed_new_chunks()

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[RetrievedMemory]:
        ranked: list[tuple[float, RetrievedMemory]] = []
        for chunk, features in self.feature_rows(
            query,
            question_date=question_date,
            metadata=metadata,
        ):
            decision = self.controller.decide_retrieve(features)
            final_score = _longmemeval_retrieve_score(decision.score, features)
            if decision.passed:
                ranked.append(
                    (
                        final_score,
                        chunk.to_retrieved(
                            score=final_score,
                            metadata={
                                "retrieval_strategy": self.name,
                                "controller_score": decision.score,
                                "decision_id": decision.decision_id,
                                "reason": decision.reason,
                                "policy_features": decision.feature_snapshot,
                                "policy_contributions": decision.contributions,
                            },
                        ),
                    )
                )
        ranked.sort(key=lambda pair: (-pair[0], pair[1].memory_id))
        return [result for _, result in ranked[:top_k]]

    def feature_rows(
        self,
        query: str,
        *,
        question_date: str | None,
        metadata: dict[str, JsonValue],
    ) -> list[tuple[MemoryChunk, PolicyFeatures]]:
        """Build query-dependent feature rows once for rule or tuned ranking."""

        self._embed_new_chunks()
        query_embedding = self.embedder.embed_one(query) if self.embedder else None
        memories = [
            self._chunk_to_memory_item(chunk, question_date=question_date)
            for chunk in self.chunks
        ]
        context = PolicyFeatureContext(
            query=query,
            target_scope=_target_scope(metadata),
            query_embedding=query_embedding,
            existing_memories=memories,
            now=parse_date(question_date),
            token_budget=int(metadata.get("token_budget", 2048))
            if isinstance(metadata.get("token_budget"), int)
            else None,
        )
        return [
            (chunk, self.feature_builder.build_for_memory(memory, context))
            for chunk, memory in zip(self.chunks, memories, strict=True)
        ]

    def _chunk_to_memory_item(
        self,
        chunk: MemoryChunk,
        *,
        question_date: str | None,
    ) -> MemoryItem:
        recency = recency_score(chunk.source_date, question_date)
        importance = heuristic_importance(chunk.content)
        confidence = 0.85
        source_datetime = parse_date(chunk.source_date)
        metadata = {
            "attributes": {
                **chunk.metadata,
                "source_session_id": chunk.source_session_id,
                "source_date": chunk.source_date,
            }
        }
        if source_datetime is not None:
            metadata["created_at"] = source_datetime
            metadata["updated_at"] = source_datetime
        return MemoryItem(
            id=chunk.memory_id,
            type=MemoryType.EPISODIC,
            scope=str(chunk.metadata.get("question_id") or "longmemeval"),
            content=chunk.content,
            content_embedding=list(chunk.content_embedding),
            source=MemorySource(
                source_type="longmemeval_session",
                uri=chunk.source_session_id,
            ),
            features=PolicyFeatures(
                importance=importance,
                confidence=confidence,
                recency=recency,
                token_cost=min(1.0, chunk.token_count / 2048.0),
            ),
            metadata=metadata,
        )

    def _embed_new_chunks(self) -> None:
        if self.embedder is None:
            return
        pending = [chunk for chunk in self.chunks if not chunk.content_embedding]
        if not pending:
            return
        vectors = self.embedder.embed([chunk.content for chunk in pending])
        for chunk, vector in zip(pending, vectors, strict=True):
            chunk.content_embedding = list(vector)


def _target_scope(metadata: dict[str, JsonValue]) -> str | None:
    question_id = metadata.get("question_id")
    if isinstance(question_id, str) and question_id:
        return question_id
    return "longmemeval"


def _longmemeval_retrieve_score(controller_score: float, features: PolicyFeatures) -> float:
    """Combine VMP retrieval score with update-aware signals for LME evidence.

    The generic controller penalizes contradiction during retrieval because in a
    normal memory store a contradictory memory can be risky. LongMemEval's
    knowledge-update questions often require exactly the newer contradictory
    evidence, so the adapter adds a bounded recency/actionability/conflict bonus
    without using gold labels.
    """

    update_signal = features.contradiction * features.recency
    action_signal = features.actionability * features.recency
    return clamp01(controller_score + 0.20 * update_signal + 0.10 * action_signal)
