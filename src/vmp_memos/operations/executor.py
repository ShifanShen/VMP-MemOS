"""Backend-agnostic execution of policy decisions as memory operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, JsonValue

from vmp_memos.backends.base import BaseMemoryBackend
from vmp_memos.policy import PolicyDecision
from vmp_memos.schemas import MemoryItem, MemoryOperation, OperationType
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel, TimestampedSchema, new_id


class OperationExecutionError(RuntimeError):
    """Raised when a policy decision cannot be executed safely."""


class OperationExecutionStatus(str, Enum):
    """Lifecycle status for one executor attempt."""

    APPLIED = "applied"
    IGNORED = "ignored"


class MergePlan(SchemaModel):
    """Deterministic merge instruction for duplicate or overlapping memories."""

    target_memory_id: NonEmptyStr
    source_memory_ids: list[NonEmptyStr]
    patch: dict[str, JsonValue] | None = None
    archive_sources: bool = True


class RetrievalPlan(SchemaModel):
    """Execution parameters for a RETRIEVE decision."""

    query: NonEmptyStr
    top_k: int = Field(default=20, ge=1)
    filters: dict[str, JsonValue] | None = None


class OperationExecutionResult(TimestampedSchema):
    """Structured result emitted by the operation executor."""

    execution_id: NonEmptyStr = Field(default_factory=lambda: new_id("exec"), frozen=True)
    decision_id: NonEmptyStr
    op: OperationType
    status: OperationExecutionStatus
    operation: MemoryOperation | None = None
    item_ids: list[NonEmptyStr] = Field(default_factory=list)
    items: list[MemoryItem] = Field(default_factory=list)
    query: NonEmptyStr | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def applied(self) -> bool:
        """Return whether the executor changed backend state or retrieved memories."""

        return self.status == OperationExecutionStatus.APPLIED


class MemoryOperationExecutor:
    """Execute Phase 5 policy decisions against any ``BaseMemoryBackend``."""

    def __init__(
        self,
        backend: BaseMemoryBackend,
        *,
        operation_log_path: str | Path | None = None,
    ) -> None:
        self.backend = backend
        inferred_log_path = getattr(backend, "operation_log_path", None)
        raw_path = operation_log_path or inferred_log_path
        self.operation_log_path = Path(raw_path) if raw_path is not None else None

    def execute(
        self,
        decision: PolicyDecision,
        *,
        memory_item: MemoryItem | None = None,
        target_memory_id: str | None = None,
        patch: Mapping[str, Any] | None = None,
        merge_plan: MergePlan | None = None,
        retrieval_plan: RetrievalPlan | None = None,
        source_event_id: str | None = None,
        source_memory_ids: Sequence[str] | None = None,
        scope: str | None = None,
    ) -> OperationExecutionResult:
        """Dispatch one decision to the matching executor method."""

        if decision.op == OperationType.IGNORE or not decision.passed:
            return self.ignore(
                decision,
                target_memory_id=target_memory_id,
                source_memory_ids=source_memory_ids,
                source_event_id=source_event_id,
                scope=scope,
            )
        if decision.op == OperationType.ADD:
            if memory_item is None:
                raise OperationExecutionError("ADD execution requires memory_item")
            return self.add(decision, memory_item, source_event_id=source_event_id)
        if decision.op == OperationType.UPDATE:
            if target_memory_id is None or patch is None:
                raise OperationExecutionError(
                    "UPDATE execution requires target_memory_id and patch"
                )
            return self.update(
                decision,
                target_memory_id,
                patch,
                source_event_id=source_event_id,
                scope=scope,
            )
        if decision.op == OperationType.MERGE:
            if merge_plan is None:
                if target_memory_id is None or not source_memory_ids:
                    raise OperationExecutionError(
                        "MERGE execution requires merge_plan or target/source IDs"
                    )
                merge_plan = MergePlan(
                    target_memory_id=target_memory_id,
                    source_memory_ids=list(source_memory_ids),
                )
            return self.merge(decision, merge_plan, source_event_id=source_event_id)
        if decision.op == OperationType.ARCHIVE:
            if target_memory_id is None:
                raise OperationExecutionError("ARCHIVE execution requires target_memory_id")
            return self.archive(
                decision,
                target_memory_id,
                source_event_id=source_event_id,
                scope=scope,
            )
        if decision.op == OperationType.RETRIEVE:
            if retrieval_plan is None:
                raise OperationExecutionError("RETRIEVE execution requires retrieval_plan")
            return self.retrieve(decision, retrieval_plan, source_event_id=source_event_id)
        raise OperationExecutionError(f"Unsupported operation for Phase 6: {decision.op.value}")

    def add(
        self,
        decision: PolicyDecision,
        memory_item: MemoryItem,
        *,
        source_event_id: str | None = None,
    ) -> OperationExecutionResult:
        """Execute an ADD decision by writing a new memory."""

        self._ensure_expected_op(decision, OperationType.ADD)
        stored = self.backend.add(
            memory_item,
            reason=decision.reason,
            policy_score=decision.score,
            confidence=decision.confidence,
        )
        operation = self._operation(
            decision,
            target_memory_id=stored.id,
            source_event_id=source_event_id or stored.source.event_id,
            scope=stored.scope,
        )
        return self._result(
            decision,
            status=OperationExecutionStatus.APPLIED,
            operation=operation,
            items=[stored],
            metadata={"backend_logged_operation": True},
        )

    def update(
        self,
        decision: PolicyDecision,
        target_memory_id: str,
        patch: Mapping[str, Any],
        *,
        source_event_id: str | None = None,
        scope: str | None = None,
    ) -> OperationExecutionResult:
        """Execute an UPDATE decision by applying a backend patch."""

        self._ensure_expected_op(decision, OperationType.UPDATE)
        updated = self.backend.update(
            target_memory_id,
            patch,
            reason=decision.reason,
            policy_score=decision.score,
            confidence=decision.confidence,
        )
        operation = self._operation(
            decision,
            target_memory_id=updated.id,
            source_event_id=source_event_id or updated.source.event_id,
            scope=scope or updated.scope,
            payload={"patch_fields": sorted(patch)},
        )
        return self._result(
            decision,
            status=OperationExecutionStatus.APPLIED,
            operation=operation,
            items=[updated],
            metadata={"backend_logged_operation": True},
        )

    def merge(
        self,
        decision: PolicyDecision,
        plan: MergePlan,
        *,
        source_event_id: str | None = None,
    ) -> OperationExecutionResult:
        """Execute MERGE as target update plus optional source archives."""

        self._ensure_expected_op(decision, OperationType.MERGE)
        target = self.backend.get(plan.target_memory_id)
        sources = [
            self.backend.get(memory_id)
            for memory_id in plan.source_memory_ids
            if memory_id != plan.target_memory_id
        ]
        if not sources:
            raise OperationExecutionError("MERGE execution requires at least one source memory")

        patch = plan.patch or self._build_merge_patch(target, sources, decision)
        updated = self.backend.update(
            target.id,
            patch,
            reason=f"{decision.reason} Merged memories: {', '.join(item.id for item in sources)}.",
            policy_score=decision.score,
            confidence=decision.confidence,
        )
        archived_sources: list[MemoryItem] = []
        if plan.archive_sources:
            for source in sources:
                archived_sources.append(
                    self.backend.archive(
                        source.id,
                        reason=f"Merged into {target.id}. {decision.reason}",
                        policy_score=decision.score,
                        confidence=decision.confidence,
                    )
                )

        operation = self._operation(
            decision,
            target_memory_id=target.id,
            source_memory_ids=[source.id for source in sources],
            source_event_id=source_event_id or target.source.event_id,
            scope=target.scope,
            payload={
                "archive_sources": plan.archive_sources,
                "patch_fields": sorted(patch),
                "archived_source_ids": [item.id for item in archived_sources],
            },
        )
        self._append_operation(operation)
        return self._result(
            decision,
            status=OperationExecutionStatus.APPLIED,
            operation=operation,
            items=[updated, *archived_sources],
            metadata={"executor_logged_operation": True},
        )

    def archive(
        self,
        decision: PolicyDecision,
        target_memory_id: str,
        *,
        source_event_id: str | None = None,
        scope: str | None = None,
    ) -> OperationExecutionResult:
        """Execute an ARCHIVE decision."""

        self._ensure_expected_op(decision, OperationType.ARCHIVE)
        archived = self.backend.archive(
            target_memory_id,
            reason=decision.reason,
            policy_score=decision.score,
            confidence=decision.confidence,
        )
        operation = self._operation(
            decision,
            target_memory_id=archived.id,
            source_event_id=source_event_id or archived.source.event_id,
            scope=scope or archived.scope,
        )
        return self._result(
            decision,
            status=OperationExecutionStatus.APPLIED,
            operation=operation,
            items=[archived],
            metadata={"backend_logged_operation": True},
        )

    def retrieve(
        self,
        decision: PolicyDecision,
        plan: RetrievalPlan,
        *,
        source_event_id: str | None = None,
    ) -> OperationExecutionResult:
        """Execute a RETRIEVE decision by delegating search to the backend."""

        self._ensure_expected_op(decision, OperationType.RETRIEVE)
        results = self.backend.search(
            plan.query,
            top_k=plan.top_k,
            filters=plan.filters,
        )
        operation = self._operation(
            decision,
            source_event_id=source_event_id,
            scope=_scope_from_filters(plan.filters),
            payload={
                "query": plan.query,
                "top_k": plan.top_k,
                "result_ids": [item.id for item in results],
            },
        )
        return self._result(
            decision,
            status=OperationExecutionStatus.APPLIED,
            operation=operation,
            items=results,
            query=plan.query,
            metadata={"backend_logged_operation": True},
        )

    def ignore(
        self,
        decision: PolicyDecision,
        *,
        target_memory_id: str | None = None,
        source_memory_ids: Sequence[str] | None = None,
        source_event_id: str | None = None,
        scope: str | None = None,
    ) -> OperationExecutionResult:
        """Record an IGNORE decision without mutating the backend."""

        operation = self._operation(
            decision,
            target_memory_id=target_memory_id,
            source_memory_ids=source_memory_ids,
            source_event_id=source_event_id,
            scope=scope or "global",
            op=OperationType.IGNORE,
        )
        self._append_operation(operation)
        return self._result(
            decision,
            status=OperationExecutionStatus.IGNORED,
            operation=operation,
            metadata={"executor_logged_operation": True},
        )

    def _operation(
        self,
        decision: PolicyDecision,
        *,
        op: OperationType | None = None,
        target_memory_id: str | None = None,
        source_memory_ids: Sequence[str] | None = None,
        source_event_id: str | None = None,
        scope: str = "global",
        payload: dict[str, JsonValue] | None = None,
    ) -> MemoryOperation:
        operation_payload: dict[str, JsonValue] = {
            "decision_id": decision.decision_id,
            "score_name": decision.score_name.value,
            "threshold": decision.threshold,
            "passed": decision.passed,
            "contributions": dict(decision.contributions),
            "feature_snapshot": dict(decision.feature_snapshot),
        }
        if payload:
            operation_payload.update(payload)
        return MemoryOperation(
            op=op or decision.op,
            target_memory_id=target_memory_id,
            source_memory_ids=list(source_memory_ids or []),
            source_event_id=source_event_id,
            reason=decision.reason,
            policy_score=decision.score,
            confidence=decision.confidence,
            scope=scope,
            backend=self._backend_name(),
            payload=operation_payload,
            metadata=dict(decision.metadata),
        )

    def _append_operation(self, operation: MemoryOperation) -> None:
        if self.operation_log_path is None:
            raise OperationExecutionError(
                "Cannot log executor-only operation: no operation_log_path configured"
            )
        operation.append_jsonl(self.operation_log_path)

    def _result(
        self,
        decision: PolicyDecision,
        *,
        status: OperationExecutionStatus,
        operation: MemoryOperation,
        items: Sequence[MemoryItem] = (),
        query: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> OperationExecutionResult:
        return OperationExecutionResult(
            decision_id=decision.decision_id,
            op=operation.op,
            status=status,
            operation=operation,
            item_ids=[item.id for item in items],
            items=list(items),
            query=query,
            metadata=metadata or {},
        )

    @staticmethod
    def _ensure_expected_op(decision: PolicyDecision, expected: OperationType) -> None:
        if decision.op != expected:
            raise OperationExecutionError(
                f"Expected {expected.value} decision, got {decision.op.value}"
            )
        if not decision.passed:
            raise OperationExecutionError(
                f"Cannot execute failed {expected.value} decision directly; use ignore()"
            )

    def _backend_name(self) -> str:
        return str(getattr(self.backend, "backend_name", self.backend.__class__.__name__))

    @staticmethod
    def _build_merge_patch(
        target: MemoryItem,
        sources: Sequence[MemoryItem],
        decision: PolicyDecision,
    ) -> dict[str, JsonValue]:
        merged_content = _merge_texts([target.content, *(item.content for item in sources)])
        summary_values = [
            value
            for value in [target.summary, *(item.summary for item in sources)]
            if value
        ]
        supersedes = sorted({*target.links.supersedes, *(item.id for item in sources)})
        related = sorted(
            {
                *target.links.related,
                *(memory_id for item in sources for memory_id in item.links.related),
            }
            - {target.id}
            - set(supersedes)
        )
        merged_from = sorted({*(item.id for item in sources)})
        return {
            "content": merged_content,
            "summary": _merge_summary(summary_values),
            "links": {
                "related": related,
                "supersedes": supersedes,
            },
            "metadata": {
                "attributes": {
                    "merged_from": merged_from,
                    "merge_decision_id": decision.decision_id,
                }
            },
        }


def _merge_texts(values: Sequence[str]) -> str:
    paragraphs: list[str] = []
    seen: set[str] = set()
    for value in values:
        for paragraph in value.split("\n\n"):
            normalized = " ".join(paragraph.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            paragraphs.append(paragraph.strip())
    return "\n\n".join(paragraphs)


def _merge_summary(values: Sequence[str]) -> str:
    if not values:
        return "Merged memory"
    seen: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.append(normalized)
    return "; ".join(seen[:3])


def _scope_from_filters(filters: Mapping[str, JsonValue] | None) -> str:
    if not filters:
        return "global"
    raw_scope = filters.get("scope")
    return raw_scope if isinstance(raw_scope, str) and raw_scope else "global"
