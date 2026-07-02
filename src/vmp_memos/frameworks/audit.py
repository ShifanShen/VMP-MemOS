"""Capability audit records for external official framework adapters."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

from pydantic import Field, JsonValue

from vmp_memos.frameworks.base import FairnessLevel
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel


class FrameworkCapabilityReport(SchemaModel):
    """Audit row explaining whether a framework can enter the main table."""

    framework_name: NonEmptyStr
    official_repo: str | None = None
    version_or_commit: str | None = None
    license: str | None = None
    install_status: NonEmptyStr = "unknown"
    adapter_implemented: bool = False
    adapter_smoke_verified: bool = False
    local_vllm_supported: bool = False
    local_embedding_supported: bool = False
    evidence_export_supported: bool = False
    reset_workspace_supported: bool = False
    token_latency_stats_supported: bool = False
    fairness_level: FairnessLevel = FairnessLevel.UNAVAILABLE
    main_table_eligible: bool = False
    reason_if_excluded: str | None = None
    notes: dict[str, JsonValue] = Field(default_factory=dict)


_KNOWN_FRAMEWORKS: dict[str, dict[str, str]] = {
    "mem0": {
        "package": "mem0",
        "distribution": "mem0ai",
        "supported_version": "2.0.10",
        "license": "Apache-2.0",
        "repo": "https://github.com/mem0ai/mem0",
    },
    "letta": {
        "package": "letta_client",
        "distribution": "letta-client",
        "supported_version": "1.12.1",
        "supported_server_version": "0.16.8",
        "license": "Apache-2.0",
        "repo": "https://github.com/letta-ai/letta",
    },
    "langmem": {
        "package": "langmem",
        "distribution": "langmem",
        "supported_version": "0.0.30",
        "license": "MIT",
        "repo": "https://github.com/langchain-ai/langmem",
    },
    "graphiti": {
        "package": "graphiti_core",
        "distribution": "graphiti-core",
        "supported_version": "0.29.2",
        "license": "Apache-2.0",
        "repo": "https://github.com/getzep/graphiti",
    },
}


def audit_known_frameworks(
    names: list[str],
    *,
    vllm_base_url: str | None = None,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    official_llm_max_tokens: int = 512,
    official_llm_temperature: float = 0.0,
    verification_dir: str | Path | None = None,
) -> list[FrameworkCapabilityReport]:
    """Return conservative audit records for external framework candidates.

    Level 1 is granted only when the official adapter, pinned package version,
    shared model controls, and matching server smoke credential are all present.
    """

    return [
        _audit_one(
            name,
            vllm_base_url=vllm_base_url,
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            official_llm_max_tokens=official_llm_max_tokens,
            official_llm_temperature=official_llm_temperature,
            verification_dir=Path(verification_dir) if verification_dir else None,
        )
        for name in names
    ]


def _audit_one(
    name: str,
    *,
    vllm_base_url: str | None,
    llm_model: str | None,
    embedding_model: str | None,
    embedding_dimension: int | None,
    official_llm_max_tokens: int,
    official_llm_temperature: float,
    verification_dir: Path | None,
) -> FrameworkCapabilityReport:
    normalized = name.strip().casefold().replace("-", "_")
    known = _KNOWN_FRAMEWORKS.get(normalized, {})
    package = known.get("package", normalized)
    installed = find_spec(package) is not None
    installed_version = _installed_version(known.get("distribution", package))
    adapter_implemented = normalized in {"mem0", "langmem", "graphiti", "letta"}
    supported_version = known.get("supported_version")
    version_supported = bool(
        installed_version
        and supported_version
        and installed_version == supported_version
    )
    smoke_verified = _smoke_verified(
        normalized,
        verification_dir=verification_dir,
        installed_version=installed_version,
        vllm_base_url=vllm_base_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        official_llm_max_tokens=official_llm_max_tokens,
        official_llm_temperature=official_llm_temperature,
        server_version=known.get("supported_server_version"),
    )
    controls_configured = bool(
        vllm_base_url
        and llm_model
        and embedding_model
        and embedding_dimension
        and embedding_dimension > 0
        and official_llm_max_tokens > 0
        and 0.0 <= official_llm_temperature <= 2.0
    )
    eligible = bool(
        adapter_implemented
        and installed
        and version_supported
        and controls_configured
        and smoke_verified
    )
    if not adapter_implemented:
        reason = "official adapter not implemented"
    elif not installed:
        reason = "official package is not installed"
    elif not version_supported:
        reason = f"installed version {installed_version!r} is not pinned {supported_version!r}"
    elif not controls_configured:
        reason = "vLLM endpoint and embedding model must be specified"
    elif not smoke_verified:
        reason = "matching official-adapter smoke verification is missing"
    else:
        reason = None
    return FrameworkCapabilityReport(
        framework_name=normalized,
        official_repo=known.get("repo"),
        license=known.get("license"),
        version_or_commit=installed_version or supported_version,
        install_status=(
            "installed_supported_version"
            if version_supported
            else "installed_version_mismatch"
            if installed
            else "not_installed"
        ),
        adapter_implemented=adapter_implemented,
        adapter_smoke_verified=smoke_verified,
        local_vllm_supported=adapter_implemented,
        local_embedding_supported=adapter_implemented,
        evidence_export_supported=adapter_implemented,
        reset_workspace_supported=adapter_implemented,
        token_latency_stats_supported=adapter_implemented and normalized != "letta",
        fairness_level=(
            FairnessLevel.FULLY_CONTROLLED
            if eligible
            else FairnessLevel.PARTIALLY_CONTROLLED
            if adapter_implemented and installed
            else FairnessLevel.UNAVAILABLE
        ),
        main_table_eligible=eligible,
        reason_if_excluded=reason,
        notes={
            "python_package": package,
            "distribution": known.get("distribution", package),
            "supported_version": supported_version,
            "supported_server_version": known.get("supported_server_version"),
            "vllm_base_url": vllm_base_url,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "official_llm_max_tokens": official_llm_max_tokens,
            "official_llm_temperature": official_llm_temperature,
            "verification_dir": str(verification_dir) if verification_dir else None,
            "next_step": (
                "eligible for controlled main-table run"
                if eligible
                else "install pinned package and run official adapter smoke"
                if adapter_implemented
                else "implement and verify an official adapter"
            ),
        },
    )


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _smoke_verified(
    framework: str,
    *,
    verification_dir: Path | None,
    installed_version: str | None,
    vllm_base_url: str | None,
    llm_model: str | None,
    embedding_model: str | None,
    embedding_dimension: int | None,
    official_llm_max_tokens: int,
    official_llm_temperature: float,
    server_version: str | None,
) -> bool:
    if verification_dir is None:
        return False
    path = verification_dir / f"{framework}_smoke.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("status") == "passed"
        and payload.get("framework_version") == installed_version
        and payload.get("vllm_base_url") == vllm_base_url
        and payload.get("llm_model") == llm_model
        and payload.get("embedding_model") == embedding_model
        and payload.get("embedding_dimension") == embedding_dimension
        and payload.get("official_llm_max_tokens") == official_llm_max_tokens
        and payload.get("official_llm_temperature") == official_llm_temperature
        and (
            server_version is None
            or payload.get("server_version") == server_version
        )
    )
