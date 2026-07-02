"""Registry for memory-framework adapters."""

from __future__ import annotations

from collections.abc import Callable

from vmp_memos.embeddings import BaseEmbedder
from vmp_memos.frameworks.base import BaseMemoryFrameworkAdapter
from vmp_memos.frameworks.bm25 import BM25Adapter
from vmp_memos.frameworks.naive_vector import NaiveVectorAdapter
from vmp_memos.frameworks.official import (
    GraphitiOfficialAdapter,
    LangMemOfficialAdapter,
    LettaOfficialAdapter,
    Mem0OfficialAdapter,
)
from vmp_memos.frameworks.runtime import FrameworkRuntimeConfig
from vmp_memos.frameworks.vector_importance import VectorImportanceAdapter
from vmp_memos.frameworks.vector_recency import VectorRecencyAdapter
from vmp_memos.frameworks.vmp_memos import VMPRuleAdapter
from vmp_memos.frameworks.vmp_tuned import (
    VMP_TUNED_METHODS,
    VMPTunedAdapter,
    VMPTunedModel,
    ablation_for_method,
)
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel

AdapterFactory = Callable[[], BaseMemoryFrameworkAdapter]


class FrameworkRegistry(SchemaModel):
    """Small explicit registry of adapter factories."""

    factories: dict[NonEmptyStr, AdapterFactory]

    def create(self, name: str) -> BaseMemoryFrameworkAdapter:
        """Create an adapter by CLI name."""

        normalized = normalize_adapter_name(name)
        if normalized not in self.factories:
            known = ", ".join(sorted(self.factories))
            raise ValueError(f"Unknown framework adapter {name!r}. Known adapters: {known}")
        return self.factories[normalized]()

    def names(self) -> list[str]:
        """Return registered adapter names."""

        return sorted(self.factories)


def default_registry(
    *,
    embedder: BaseEmbedder | None = None,
    runtime: FrameworkRuntimeConfig | None = None,
    vmp_tuned_model_path: str | None = None,
) -> FrameworkRegistry:
    """Return built-in and dependency-lazy official adapters."""

    official_runtime = runtime or FrameworkRuntimeConfig.from_env()
    factories: dict[NonEmptyStr, AdapterFactory] = {
        "empty": EmptyRetrievalAdapter,
        "no_memory": EmptyRetrievalAdapter,
        "bm25": BM25Adapter,
        "naive_vector": lambda: NaiveVectorAdapter(embedder=embedder),
        "naive_vector_rag": lambda: NaiveVectorAdapter(embedder=embedder),
        "vector_rag": lambda: NaiveVectorAdapter(embedder=embedder),
        "vector_recency": lambda: VectorRecencyAdapter(embedder=embedder),
        "vector_importance": lambda: VectorImportanceAdapter(embedder=embedder),
        "vmp_rule": lambda: VMPRuleAdapter(embedder=embedder),
        "mem0": lambda: Mem0OfficialAdapter(runtime=official_runtime),
        "mem0_official": lambda: Mem0OfficialAdapter(runtime=official_runtime),
        "langmem": lambda: LangMemOfficialAdapter(
            runtime=official_runtime,
            embedder=embedder,
        ),
        "langmem_official": lambda: LangMemOfficialAdapter(
            runtime=official_runtime,
            embedder=embedder,
        ),
        "graphiti": lambda: GraphitiOfficialAdapter(
            runtime=official_runtime,
            embedder=embedder,
        ),
        "graphiti_official": lambda: GraphitiOfficialAdapter(
            runtime=official_runtime,
            embedder=embedder,
        ),
        "letta": lambda: LettaOfficialAdapter(runtime=official_runtime),
        "letta_official": lambda: LettaOfficialAdapter(
            runtime=official_runtime,
        ),
    }
    for method_name in VMP_TUNED_METHODS:
        factories[method_name] = lambda method_name=method_name: VMPTunedAdapter(
            model=_load_vmp_tuned_model(vmp_tuned_model_path),
            embedder=embedder,
            ablation=ablation_for_method(method_name),
        )
    return FrameworkRegistry(factories=factories)


def adapter_for_name(
    name: str,
    *,
    embedder: BaseEmbedder | None = None,
    runtime: FrameworkRuntimeConfig | None = None,
    vmp_tuned_model_path: str | None = None,
) -> BaseMemoryFrameworkAdapter:
    """Create a built-in or official adapter by name."""

    return default_registry(
        embedder=embedder,
        runtime=runtime,
        vmp_tuned_model_path=vmp_tuned_model_path,
    ).create(name)


def normalize_adapter_name(name: str) -> str:
    """Normalize CLI aliases."""

    return name.strip().casefold().replace("-", "_")


class EmptyRetrievalAdapter(BaseMemoryFrameworkAdapter):
    """No-memory baseline returning no retrieved evidence."""

    name = "empty"

    @property
    def memory_count(self) -> int:
        return 0

    @property
    def total_tokens(self) -> int:
        return 0

    def _reset_impl(self) -> None:
        return None

    def _ingest_event_impl(self, event: object) -> None:
        return None

    def _ingest_session_impl(self, events: list[object]) -> None:
        return None

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int,
        question_date: str | None,
        metadata: dict,
    ) -> list:
        return []


def _load_vmp_tuned_model(path: str | None) -> VMPTunedModel:
    if not path:
        raise ValueError(
            "vmp_tuned requires a frozen model path; pass --vmp-tuned-model"
        )
    return VMPTunedModel.load(path)
