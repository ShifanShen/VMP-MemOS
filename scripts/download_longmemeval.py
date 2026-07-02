"""Download LongMemEval-cleaned files from Hugging Face.

This script is intentionally dependency-light. On the 4090D server you can run it
directly after installing the project; if Hugging Face changes access behavior,
manual downloads into ``data/longmemeval`` are also compatible with the loader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_REPO_ID = "xiaowu0162/longmemeval-cleaned"
DEFAULT_FILES = (
    "longmemeval_oracle.json",
    "longmemeval_s_cleaned.json",
    "longmemeval_m_cleaned.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="data/longmemeval", help="Download directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--files",
        default=",".join(DEFAULT_FILES),
        help="Comma-separated file names to download.",
    )
    parser.add_argument("--revision", default="main", help="Dataset revision.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    files = [item.strip() for item in args.files.split(",") if item.strip()]
    if not files:
        raise SystemExit("--files must contain at least one file name")

    for filename in files:
        output_path = target / filename
        if output_path.exists() and not args.overwrite:
            print(f"exists: {output_path}")
            continue
        url = _resolve_url(args.repo_id, args.revision, filename)
        print(f"downloading: {url}")
        try:
            _download(url, output_path, token=args.token)
        except (HTTPError, URLError) as exc:
            print(f"failed: {filename}: {exc}", file=sys.stderr)
            print(
                "Tip: manually download from "
                f"https://huggingface.co/datasets/{args.repo_id}/tree/{args.revision} "
                f"and place the file at {output_path}",
                file=sys.stderr,
            )
            return 1
        print(f"saved: {output_path}")
    return 0


def _resolve_url(repo_id: str, revision: str, filename: str) -> str:
    encoded_repo = "/".join(quote(part) for part in repo_id.split("/"))
    encoded_revision = quote(revision)
    encoded_filename = "/".join(quote(part) for part in filename.split("/"))
    return (
        "https://huggingface.co/datasets/"
        f"{encoded_repo}/resolve/{encoded_revision}/{encoded_filename}"
    )


def _download(url: str, output_path: Path, *, token: str | None = None) -> None:
    headers = {"User-Agent": "vmp-memos-longmemeval-downloader/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        output_path.write_bytes(response.read())


if __name__ == "__main__":
    raise SystemExit(main())
