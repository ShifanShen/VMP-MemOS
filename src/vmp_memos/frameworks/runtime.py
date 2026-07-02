"""Shared runtime settings for optional official framework adapters."""

from __future__ import annotations

import os

from pydantic import Field, JsonValue

from vmp_memos.schemas.base import NonEmptyStr, SchemaModel


class FrameworkRuntimeConfig(SchemaModel):
    """Fairness-critical model settings shared by official adapters."""

    vllm_base_url: NonEmptyStr = "http://127.0.0.1:8000/v1"
    llm_model: NonEmptyStr = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str | None = None
    embedding_model: NonEmptyStr = "BAAI/bge-m3"
    embedding_dimension: int = Field(default=1024, gt=0)
    embedding_device: NonEmptyStr = "cuda"
    official_memory_infer: bool = True
    official_llm_max_tokens: int = Field(default=512, gt=0)
    official_llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    graphiti_neo4j_uri: NonEmptyStr = "bolt://127.0.0.1:7687"
    graphiti_neo4j_user: NonEmptyStr = "neo4j"
    graphiti_neo4j_password: str | None = None
    graphiti_allow_destructive_reset: bool = False
    letta_base_url: NonEmptyStr = "http://127.0.0.1:8283"
    letta_api_key: str | None = None
    letta_server_version: NonEmptyStr = "0.16.8"
    letta_embedding_base_url: NonEmptyStr = "http://127.0.0.1:8001/v1"
    letta_context_window: int = Field(default=16_384, gt=0)

    @classmethod
    def from_env(cls) -> "FrameworkRuntimeConfig":
        """Load non-secret defaults and the optional vLLM key from environment."""

        return cls(
            vllm_base_url=os.getenv(
                "VMP_LLM_BASE_URL",
                "http://127.0.0.1:8000/v1",
            ),
            llm_model=os.getenv(
                "VMP_LLM_MODEL",
                "Qwen/Qwen2.5-7B-Instruct",
            ),
            vllm_api_key=os.getenv("VMP_LLM_API_KEY") or None,
            embedding_model=os.getenv("VMP_EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_dimension=int(os.getenv("VMP_EMBEDDING_DIMENSION", "1024")),
            embedding_device=os.getenv("VMP_EMBEDDING_DEVICE", "cuda"),
            official_memory_infer=_env_bool("VMP_OFFICIAL_MEMORY_INFER", default=True),
            official_llm_max_tokens=int(
                os.getenv("VMP_OFFICIAL_LLM_MAX_TOKENS", "512")
            ),
            official_llm_temperature=float(
                os.getenv("VMP_OFFICIAL_LLM_TEMPERATURE", "0.0")
            ),
            graphiti_neo4j_uri=os.getenv(
                "VMP_GRAPHITI_NEO4J_URI",
                "bolt://127.0.0.1:7687",
            ),
            graphiti_neo4j_user=os.getenv("VMP_GRAPHITI_NEO4J_USER", "neo4j"),
            graphiti_neo4j_password=(
                os.getenv("VMP_GRAPHITI_NEO4J_PASSWORD") or None
            ),
            graphiti_allow_destructive_reset=_env_bool(
                "VMP_GRAPHITI_ALLOW_DESTRUCTIVE_RESET",
                default=False,
            ),
            letta_base_url=os.getenv(
                "VMP_LETTA_BASE_URL",
                "http://127.0.0.1:8283",
            ),
            letta_api_key=os.getenv("VMP_LETTA_API_KEY") or None,
            letta_server_version=os.getenv("VMP_LETTA_SERVER_VERSION", "0.16.8"),
            letta_embedding_base_url=os.getenv(
                "VMP_LETTA_EMBEDDING_BASE_URL",
                "http://127.0.0.1:8001/v1",
            ),
            letta_context_window=int(
                os.getenv("VMP_LETTA_CONTEXT_WINDOW", "16384")
            ),
        )

    def public_metadata(self) -> dict[str, JsonValue]:
        """Return manifest-safe settings with no API key."""

        return {
            "vllm_base_url": self.vllm_base_url,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "embedding_device": self.embedding_device,
            "official_memory_infer": self.official_memory_infer,
            "official_llm_max_tokens": self.official_llm_max_tokens,
            "official_llm_temperature": self.official_llm_temperature,
            "graphiti_neo4j_uri": self.graphiti_neo4j_uri,
            "graphiti_neo4j_user": self.graphiti_neo4j_user,
            "graphiti_allow_destructive_reset": self.graphiti_allow_destructive_reset,
            "letta_base_url": self.letta_base_url,
            "letta_server_version": self.letta_server_version,
            "letta_embedding_base_url": self.letta_embedding_base_url,
            "letta_context_window": self.letta_context_window,
        }


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}
