"""Dependency-free tests for the OpenAI-compatible BGE-M3 bridge."""

from __future__ import annotations

import threading

from scripts.serve_embeddings import EmbeddingHTTPServer, EmbeddingRequestHandler
from vmp_memos.embeddings import BaseEmbedder, OpenAICompatibleEmbedder


class FakeEmbedder(BaseEmbedder):
    @property
    def identifier(self) -> str:
        return "fake-server-embedder"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def test_openai_compatible_embedding_bridge_round_trip() -> None:
    server = EmbeddingHTTPServer(
        ("127.0.0.1", 0),
        EmbeddingRequestHandler,
        embedder=FakeEmbedder(),
        model="BAAI/bge-m3",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        client = OpenAICompatibleEmbedder(
            base_url=f"http://{host}:{port}/v1",
            model="BAAI/bge-m3",
            dimension=2,
        )

        assert client.embed(["abc", "hello"]) == [[3.0, 1.0], [5.0, 1.0]]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
