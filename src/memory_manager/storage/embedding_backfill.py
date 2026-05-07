from __future__ import annotations

"""
Backfill `nodes.embedding_blob` for graph-memory retrieval.

API/runtime retrieval expects embeddings to already be present in SQLite so
it can avoid recomputing vectors on every request.

This script computes missing embeddings offline and stores them into
`nodes.embedding_blob` as float32 bytes.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.memory_manager.storage.memory_store import MemoryConfig, MemoryStore

EMBED_MODEL_DEFAULT = "all-MiniLM-L6-v2"


def _require_sentence_transformers() -> None:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Embedding backfill requires sentence-transformers. "
            "Install with: pip install -r requirements-seeding.txt"
        ) from exc


def _parse_types(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in (raw or "").split(",")]
    return [p for p in parts if p]


def backfill_node_embeddings(
    *,
    store: MemoryStore,
    types: Sequence[str],
    embedding_model_name: str = EMBED_MODEL_DEFAULT,
    batch_size: int = 64,
) -> int:
    _require_sentence_transformers()
    from sentence_transformers import SentenceTransformer  # type: ignore

    type_set = {t.strip().lower() for t in types if t and t.strip()}
    if not type_set:
        return 0

    rows = [
        r
        for r in store.fetch_all_nodes()
        if str(r["type"] or "").strip().lower() in type_set and r["embedding_blob"] is None
    ]
    if not rows:
        return 0

    model = SentenceTransformer(embedding_model_name)

    updated = 0
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        names = [str(r["name"] or "").strip().lower() for r in batch]
        vecs = model.encode(names)
        vecs = np.asarray(vecs, dtype=np.float32)

        for r, vec in zip(batch, vecs):
            name = str(r["name"] or "").strip().lower()
            node_type = str(r["type"] or "").strip().lower()
            summary = str(r["summary"] or "").strip()
            emb_bytes = np.asarray(vec, dtype=np.float32).tobytes()
            store.upsert_node(
                name=name,
                type_=node_type,
                summary=summary,
                embedding_blob=emb_bytes,
            )
            updated += 1

    return updated


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.memory_manager.storage.embedding_backfill",
        description="Backfill missing nodes.embedding_blob for graph retrieval.",
    )
    parser.add_argument(
        "--memory-db-dir",
        default=None,
        help="Override MEMORY_DB_DIR for this command. Database is <dir>/memory.sqlite3.",
    )
    parser.add_argument(
        "--types",
        default="base,cluster,concept",
        help="Comma-separated node types to backfill.",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBED_MODEL_DEFAULT,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.memory_db_dir:
        os.environ["MEMORY_DB_DIR"] = str(os.path.abspath(args.memory_db_dir))

    resolved = store_cfg_path_from_env()
    store = MemoryStore(MemoryConfig(db_path=resolved))
    updated = backfill_node_embeddings(
        store=store,
        types=_parse_types(args.types),
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(f"[embedding_backfill] updated_nodes={updated}")
    return updated


def store_cfg_path_from_env() -> Path:
    base_dir = os.getenv("MEMORY_DB_DIR", "data")
    db_path = Path(base_dir).resolve() / "memory.sqlite3"
    return db_path


if __name__ == "__main__":
    run()

