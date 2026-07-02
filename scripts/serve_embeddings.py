"""Serve a local SentenceTransformer through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from vmp_memos.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from vmp_memos.frameworks.text import estimate_tokens


class EmbeddingHTTPServer(ThreadingHTTPServer):
    """Threaded server carrying one shared, lock-protected embedder."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        embedder: BaseEmbedder,
        model: str,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.embedder = embedder
        self.model = model
        self.embedding_lock = threading.Lock()


class EmbeddingRequestHandler(BaseHTTPRequestHandler):
    """Implement the small OpenAI API subset needed by Letta."""

    server: EmbeddingHTTPServer

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"/health", "/v1/health"}:
            self._write_json(HTTPStatus.OK, {"status": "ok", "model": self.server.model})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._write_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model,
                            "object": "model",
                            "owned_by": "vmp-memos",
                        }
                    ],
                },
            )
            return
        self._write_error(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/embeddings":
            self._write_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        try:
            payload = self._read_json()
            requested_model = payload.get("model")
            if requested_model not in {None, self.server.model}:
                raise ValueError(
                    f"server provides {self.server.model!r}, not {requested_model!r}"
                )
            texts = _embedding_inputs(payload.get("input"))
            if payload.get("encoding_format", "float") != "float":
                raise ValueError("only encoding_format='float' is supported")
            with self.server.embedding_lock:
                vectors = self.server.embedder.embed(texts)
            token_count = sum(estimate_tokens(text) for text in texts)
            self._write_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": vector,
                            "index": index,
                        }
                        for index, vector in enumerate(vectors)
                    ],
                    "model": self.server.model,
                    "usage": {
                        "prompt_tokens": token_count,
                        "total_tokens": token_count,
                    },
                },
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._write_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{type(exc).__name__}: {exc}",
            )

    def log_message(self, message_format: str, *args: object) -> None:
        print(f"embedding-server: {message_format % args}")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("request body is empty")
        value = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _write_error(self, status: HTTPStatus, message: str) -> None:
        self._write_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                }
            },
        )

    def _write_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-folder", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(
        args.model,
        device=args.device,
        cache_folder=args.cache_folder,
        batch_size=args.batch_size,
    )
    server = EmbeddingHTTPServer(
        (args.host, args.port),
        EmbeddingRequestHandler,
        embedder=embedder,
        model=args.model,
    )
    print(
        f"Serving {args.model} embeddings at "
        f"http://{args.host}:{args.port}/v1/embeddings"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        embedder.release()
    return 0


def _embedding_inputs(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(
        isinstance(item, str) for item in value
    ):
        return [str(item) for item in value]
    raise ValueError("input must be a non-empty string or list of strings")


if __name__ == "__main__":
    raise SystemExit(main())
