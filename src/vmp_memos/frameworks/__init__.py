"""Unified memory-framework adapter interface and built-in adapters."""

from vmp_memos.frameworks.audit import FrameworkCapabilityReport, audit_known_frameworks
from vmp_memos.frameworks.base import (
    BaseMemoryFrameworkAdapter,
    FairnessLevel,
    MemoryChunk,
    RetrievedMemory,
)
from vmp_memos.frameworks.bm25 import BM25Adapter
from vmp_memos.frameworks.naive_vector import NaiveVectorAdapter
from vmp_memos.frameworks.official import (
    GraphitiDependencyError,
    GraphitiOfficialAdapter,
    LangMemDependencyError,
    LangMemOfficialAdapter,
    LettaDependencyError,
    LettaOfficialAdapter,
    Mem0DependencyError,
    Mem0OfficialAdapter,
    build_letta_embedding_config,
    build_letta_llm_config,
    build_mem0_config,
)
from vmp_memos.frameworks.registry import FrameworkRegistry, adapter_for_name, default_registry
from vmp_memos.frameworks.runtime import FrameworkRuntimeConfig
from vmp_memos.frameworks.vector_importance import VectorImportanceAdapter
from vmp_memos.frameworks.vector_recency import VectorRecencyAdapter
from vmp_memos.frameworks.vmp_memos import VMPRuleAdapter
from vmp_memos.frameworks.vmp_tuned import (
    VMP_TUNED_ABLATIONS,
    VMPTunedAblation,
    VMPTunedAdapter,
    VMPTunedModel,
)

__all__ = [
    "BM25Adapter",
    "BaseMemoryFrameworkAdapter",
    "FairnessLevel",
    "FrameworkCapabilityReport",
    "FrameworkRegistry",
    "FrameworkRuntimeConfig",
    "GraphitiDependencyError",
    "GraphitiOfficialAdapter",
    "LangMemDependencyError",
    "LangMemOfficialAdapter",
    "LettaDependencyError",
    "LettaOfficialAdapter",
    "MemoryChunk",
    "Mem0DependencyError",
    "Mem0OfficialAdapter",
    "NaiveVectorAdapter",
    "RetrievedMemory",
    "VMPRuleAdapter",
    "VMP_TUNED_ABLATIONS",
    "VMPTunedAblation",
    "VMPTunedAdapter",
    "VMPTunedModel",
    "VectorImportanceAdapter",
    "VectorRecencyAdapter",
    "adapter_for_name",
    "audit_known_frameworks",
    "build_letta_embedding_config",
    "build_letta_llm_config",
    "build_mem0_config",
    "default_registry",
]
