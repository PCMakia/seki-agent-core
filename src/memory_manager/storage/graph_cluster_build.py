from __future__ import annotations

"""
Offline graph builder for prototype/cluster nodes.

This script implements the "CLUSTER ≠ TOKEN" upgrade by converting existing
concept nodes (nodes.type='concept') into cluster prototype nodes
(nodes.type='cluster') and connecting them with `in_cluster` edges.

Notes:
- Cluster naming is deterministic: we choose the concept whose embedding is
  closest (cosine) to the cluster centroid embedding.
- The cluster builder is intentionally offline so it can use heavier
  embedding computations without impacting API latency.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.memory_manager.storage.memory_store import MemoryConfig, MemoryStore, _utc_now_iso


EMBED_MODEL_DEFAULT = "all-MiniLM-L6-v2"


def _require_sentence_transformers() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Cluster building requires sentence-transformers. "
            "Install with: pip install -r requirements-seeding.txt"
        ) from exc


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _centroid_phrase(
    *,
    member_indices: Sequence[int],
    concept_names: Sequence[str],
    vecs: np.ndarray,
) -> str:
    member_vecs = vecs[np.asarray(member_indices, dtype=int)]
    centroid = np.mean(member_vecs, axis=0)

    best_name = None
    best_score = -1.0
    best_lex = None

    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm <= 0.0:
        # Degenerate: fall back to first member.
        return concept_names[int(member_indices[0])]

    for idx in member_indices:
        name = concept_names[int(idx)]
        score = _cosine(vecs[int(idx)], centroid)
        lex = name.lower()
        if score > best_score or (score == best_score and (best_lex is None or lex < best_lex)):
            best_score = score
            best_name = name
            best_lex = lex

    assert best_name is not None
    return best_name


def _greedy_threshold_clusters(
    *,
    vecs: np.ndarray,
    similarity_threshold: float,
) -> list[list[int]]:
    """
    Deterministic greedy clustering:
    - Disjoint clusters
    - Seed cluster i, then add any j where cosine(vec_i, vec_j) >= threshold.
    """

    used: set[int] = set()
    clusters: list[list[int]] = []
    n = int(vecs.shape[0])

    for i in range(n):
        if i in used:
            continue
        cluster = [i]
        used.add(i)

        a = vecs[i]
        for j in range(i + 1, n):
            if j in used:
                continue
            sim = _cosine(a, vecs[j])
            if sim >= similarity_threshold:
                cluster.append(j)
                used.add(j)

        clusters.append(cluster)

    return clusters


@dataclass(frozen=True)
class ClusterBuildStats:
    concept_count: int
    cluster_count: int
    in_cluster_edges_written: int


def build_clusters_into_prototypes(
    *,
    store: MemoryStore,
    similarity_threshold: float,
    embedding_model_name: str = EMBED_MODEL_DEFAULT,
    concept_name_min_len: int = 3,
    in_cluster_relation_type: str = "in_cluster",
) -> ClusterBuildStats:
    _require_sentence_transformers()
    from sentence_transformers import SentenceTransformer  # type: ignore

    rows = [r for r in store.fetch_all_nodes() if str(r["type"] or "").lower() == "concept"]
    concept_rows = [
        r
        for r in rows
        if (str(r["name"] or "").strip()) and len(str(r["name"] or "").strip()) >= int(concept_name_min_len)
    ]
    concept_rows = sorted(concept_rows, key=lambda r: int(r["id"]))
    if not concept_rows:
        return ClusterBuildStats(concept_count=0, cluster_count=0, in_cluster_edges_written=0)

    concept_ids = [int(r["id"]) for r in concept_rows]
    concept_names = [str(r["name"] or "").strip().lower() for r in concept_rows]

    model = SentenceTransformer(embedding_model_name)
    vectors = model.encode(concept_names)
    vecs = np.asarray(vectors, dtype=np.float32)

    clusters = _greedy_threshold_clusters(vecs=vecs, similarity_threshold=float(similarity_threshold))
    ts = _utc_now_iso()

    cluster_count = 0
    edge_count = 0

    for member_indices in clusters:
        if len(member_indices) < 2:
            # Singletons rarely help; keep graph lighter.
            continue

        cluster_name = _centroid_phrase(
            member_indices=member_indices,
            concept_names=concept_names,
            vecs=vecs,
        )

        cluster_id = store.upsert_node(
            name=cluster_name,
            type_="cluster",
            summary=f"Prototype cluster (centroid phrase): {cluster_name}; members={len(member_indices)}",
        )
        cluster_count += 1

        for idx in member_indices:
            concept_id = concept_ids[int(idx)]
            if concept_id == cluster_id:
                continue

            store.upsert_edge(
                src_id=concept_id,
                dst_id=cluster_id,
                relation_type=in_cluster_relation_type,
                weight_delta=1.0,
                ts=ts,
            )
            store.upsert_edge(
                src_id=cluster_id,
                dst_id=concept_id,
                relation_type=in_cluster_relation_type,
                weight_delta=1.0,
                ts=ts,
            )
            edge_count += 2

    return ClusterBuildStats(
        concept_count=len(concept_ids),
        cluster_count=cluster_count,
        in_cluster_edges_written=edge_count,
    )


@dataclass(frozen=True)
class ClusterHeadLinkStats:
    head_count: int
    cluster_count: int
    cluster_of_edges_written: int


def link_clusters_to_heads(
    *,
    store: MemoryStore,
    embedding_model_name: str = EMBED_MODEL_DEFAULT,
    cluster_of_relation_type: str = "cluster_of",
) -> ClusterHeadLinkStats:
    """
    Connect each `nodes(type='cluster')` prototype to the most similar manual
    head (`nodes(type='base')`).
    """

    _require_sentence_transformers()
    from sentence_transformers import SentenceTransformer  # type: ignore

    head_rows = [r for r in store.fetch_all_nodes() if str(r["type"] or "").lower() == "base"]
    cluster_rows = [r for r in store.fetch_all_nodes() if str(r["type"] or "").lower() == "cluster"]

    if not head_rows or not cluster_rows:
        return ClusterHeadLinkStats(head_count=len(head_rows), cluster_count=len(cluster_rows), cluster_of_edges_written=0)

    # Sort to make tie-breaking deterministic.
    head_rows = sorted(head_rows, key=lambda r: (str(r["name"] or "").strip().lower(), int(r["id"])))
    cluster_rows = sorted(cluster_rows, key=lambda r: int(r["id"]))

    head_ids = [int(r["id"]) for r in head_rows]
    head_names = [str(r["name"] or "").strip().lower() for r in head_rows]

    cluster_ids = [int(r["id"]) for r in cluster_rows]
    cluster_names = [str(r["name"] or "").strip().lower() for r in cluster_rows]

    model = SentenceTransformer(embedding_model_name)
    head_vecs = np.asarray(model.encode(head_names), dtype=np.float32)
    cluster_vecs = np.asarray(model.encode(cluster_names), dtype=np.float32)

    # Normalize so dot product == cosine similarity.
    def _norm(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom <= 0.0] = 1.0
        return x / denom

    head_vecs = _norm(head_vecs)
    cluster_vecs = _norm(cluster_vecs)

    sims = cluster_vecs @ head_vecs.T  # [n_clusters, n_heads]
    best_head_idx = np.argmax(sims, axis=1)

    ts = _utc_now_iso()
    edge_count = 0
    for ci, hi_idx in enumerate(best_head_idx.tolist()):
        cluster_id = cluster_ids[ci]
        head_id = head_ids[int(hi_idx)]

        store.upsert_edge(
            src_id=head_id,
            dst_id=cluster_id,
            relation_type=cluster_of_relation_type,
            weight_delta=1.0,
            ts=ts,
        )
        store.upsert_edge(
            src_id=cluster_id,
            dst_id=head_id,
            relation_type=cluster_of_relation_type,
            weight_delta=1.0,
            ts=ts,
        )
        edge_count += 2

    return ClusterHeadLinkStats(
        head_count=len(head_ids),
        cluster_count=len(cluster_ids),
        cluster_of_edges_written=edge_count,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.memory_manager.storage.graph_cluster_build",
        description="Build prototype cluster nodes and connect them to concept nodes.",
    )
    parser.add_argument(
        "--memory-db-dir",
        default=None,
        help="Override MEMORY_DB_DIR for this command. Database is <dir>/memory.sqlite3.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for greedy clustering.",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBED_MODEL_DEFAULT,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--min-concept-name-len",
        type=int,
        default=3,
        help="Minimum concept name length to include in clustering.",
    )
    parser.add_argument(
        "--in-cluster-relation-type",
        default="in_cluster",
        help="Edge relation type for concept<->cluster connections.",
    )
    parser.add_argument(
        "--link-heads",
        action="store_true",
        help="After building clusters, link clusters to the best manual HEAD nodes (type='base').",
    )
    parser.add_argument(
        "--cluster-of-relation-type",
        default="cluster_of",
        help="Edge relation type for head<->cluster connections.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> ClusterBuildStats:
    args = _parse_args(argv)
    if args.memory_db_dir:
        os.environ["MEMORY_DB_DIR"] = str(os.path.abspath(args.memory_db_dir))
    store = MemoryStore(MemoryConfig(db_path=store_cfg_path_from_env()))
    stats = build_clusters_into_prototypes(
        store=store,
        similarity_threshold=args.similarity_threshold,
        embedding_model_name=args.embedding_model,
        concept_name_min_len=args.min_concept_name_len,
        in_cluster_relation_type=args.in_cluster_relation_type,
    )
    if args.link_heads:
        link_clusters_to_heads(
            store=store,
            embedding_model_name=args.embedding_model,
            cluster_of_relation_type=args.cluster_of_relation_type,
        )
    return stats


def store_cfg_path_from_env() -> Path:
    """
    Resolve db path consistently with MemoryStore.default_config.
    Kept as a helper to avoid circular imports and to keep this script standalone.
    """

    base_dir = os.getenv("MEMORY_DB_DIR", "data")
    db_path = Path(base_dir).resolve() / "memory.sqlite3"
    return db_path


if __name__ == "__main__":
    args = _parse_args()
    if args.memory_db_dir:
        os.environ["MEMORY_DB_DIR"] = str(os.path.abspath(args.memory_db_dir))

    # Build store using MemoryStore default_config behavior.
    resolved = store_cfg_path_from_env()
    store = MemoryStore(MemoryConfig(db_path=resolved))
    stats = build_clusters_into_prototypes(
        store=store,
        similarity_threshold=args.similarity_threshold,
        embedding_model_name=args.embedding_model,
        concept_name_min_len=args.min_concept_name_len,
        in_cluster_relation_type=args.in_cluster_relation_type,
    )
    print(
        f"[graph_cluster_build] concept_count={stats.concept_count} "
        f"cluster_count={stats.cluster_count} "
        f"in_cluster_edges_written={stats.in_cluster_edges_written}"
    )

    if args.link_heads:
        link_stats = link_clusters_to_heads(
            store=store,
            embedding_model_name=args.embedding_model,
            cluster_of_relation_type=args.cluster_of_relation_type,
        )
        print(
            f"[graph_cluster_build] head_count={link_stats.head_count} "
            f"cluster_count={link_stats.cluster_count} "
            f"cluster_of_edges_written={link_stats.cluster_of_edges_written}"
        )

