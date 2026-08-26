"""HTTP inspect/edit surface over the existing SQLite concept graph."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.memory_manager.storage.memory_store import MemoryStore
from src.time_utils import now_dt
from src.vault.last_chain import LastChainBuffer


def _since_iso(hours: float | None) -> str | None:
    if hours is None or hours <= 0:
        return None
    cutoff = now_dt() - timedelta(hours=float(hours))
    return cutoff.isoformat(timespec="seconds")


def _node_dict(row: Any, *, degree: int | None = None) -> dict[str, Any]:
    payload = {
        "id": int(row["id"]),
        "name": str(row["name"] or ""),
        "type": str(row["type"] or "concept"),
        "summary": str(row["summary"] or ""),
        "activation_weight": float(row["activation_weight"] or 0.0),
        "last_activation_ts": row["last_activation_ts"] or None,
    }
    if degree is not None:
        payload["degree"] = int(degree)
    return payload


def _edge_dict(row: Any) -> dict[str, Any]:
    edge_id = row["id"] if "id" in row.keys() else row["edge_id"]
    out = {
        "id": int(edge_id),
        "src_id": int(row["src_id"]),
        "src_name": str(row["src_name"] or ""),
        "dst_id": int(row["dst_id"]),
        "dst_name": str(row["dst_name"] or ""),
        "relation_type": str(row["relation_type"] or ""),
        "weight": float(row["weight"] or 0.0),
        "last_coactivation_ts": row["last_coactivation_ts"] if "last_coactivation_ts" in row.keys() else None,
    }
    if "direction" in row.keys():
        out["direction"] = str(row["direction"] or "")
    return out


def _episode_dict(row: Any, *, clip: bool = False) -> dict[str, Any]:
    user = str(row["user_text"] or "")
    assistant = str(row["assistant_text"] or "")
    if clip:
        user = user[:240]
        assistant = assistant[:240]
    return {
        "id": int(row["id"]),
        "session_id": str(row["session_id"] or ""),
        "ts": str(row["ts"] or ""),
        "user_text": user,
        "assistant_text": assistant,
        "topic": row["topic"],
        "importance": float(row["importance"] or 0.0),
    }


class NodePatch(BaseModel):
    summary: str = Field(..., max_length=8000)


class EdgeCreate(BaseModel):
    src_id: int
    dst_id: int
    relation_type: str = "co_occurs"
    weight: float = 1.0


def build_vault_router(
    get_store: Callable[[], MemoryStore],
    chains: LastChainBuffer,
) -> APIRouter:
    router = APIRouter(prefix="/agent/vault", tags=["vault"])

    @router.get("/stats")
    def vault_stats() -> dict[str, int]:
        return get_store().vault_stats()

    @router.get("/nodes")
    def list_nodes(
        q: str = Query(""),
        type: str = Query("", alias="type"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        rows, total = get_store().list_nodes(q=q, type_=type, limit=limit, offset=offset)
        return {"total": total, "items": [_node_dict(r) for r in rows]}

    @router.get("/nodes/{node_id}")
    def get_node(node_id: int) -> dict[str, Any]:
        store = get_store()
        row = store.get_node(node_id)
        if row is None:
            raise HTTPException(status_code=404, detail="node not found")
        return _node_dict(row, degree=store.node_degree(node_id))

    @router.patch("/nodes/{node_id}")
    def patch_node(node_id: int, body: NodePatch) -> dict[str, Any]:
        store = get_store()
        if not store.patch_node_summary(node_id, body.summary):
            raise HTTPException(status_code=404, detail="node not found")
        row = store.get_node(node_id)
        assert row is not None
        return _node_dict(row, degree=store.node_degree(node_id))

    @router.get("/nodes/{node_id}/neighbors")
    def node_neighbors(
        node_id: int,
        limit: int = Query(80, ge=1, le=200),
    ) -> dict[str, Any]:
        store = get_store()
        if store.get_node(node_id) is None:
            raise HTTPException(status_code=404, detail="node not found")
        rows = store.fetch_neighbors(node_id, limit=limit)
        return {"node_id": node_id, "items": [_edge_dict(r) for r in rows]}

    @router.get("/nodes/{node_id}/episodes")
    def node_episodes(
        node_id: int,
        limit: int = Query(20, ge=1, le=50),
    ) -> dict[str, Any]:
        store = get_store()
        if store.get_node(node_id) is None:
            raise HTTPException(status_code=404, detail="node not found")
        rows = store.fetch_episodes_for_node(node_id, limit=limit)
        return {"node_id": node_id, "items": [_episode_dict(r, clip=True) for r in rows]}

    @router.get("/episodes")
    def list_episodes(
        q: str = Query(""),
        session_id: str = Query(""),
        limit: int = Query(30, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        rows, total = get_store().list_episodes(
            q=q, session_id=session_id, limit=limit, offset=offset
        )
        return {"total": total, "items": [_episode_dict(r, clip=True) for r in rows]}

    @router.get("/episodes/{episode_id}")
    def get_episode(episode_id: int) -> dict[str, Any]:
        row = get_store().get_episode(episode_id)
        if row is None:
            raise HTTPException(status_code=404, detail="episode not found")
        return _episode_dict(row, clip=False)

    @router.post("/edges")
    def create_edge(body: EdgeCreate) -> dict[str, Any]:
        row = get_store().create_edge(
            src_id=body.src_id,
            dst_id=body.dst_id,
            relation_type=body.relation_type,
            weight=body.weight,
        )
        if row is None:
            raise HTTPException(status_code=400, detail="invalid edge (missing nodes or self-loop)")
        return _edge_dict(row)

    @router.delete("/edges/{edge_id}")
    def delete_edge(edge_id: int) -> dict[str, bool]:
        if not get_store().delete_edge(edge_id):
            raise HTTPException(status_code=404, detail="edge not found")
        return {"ok": True}

    @router.get("/hottest")
    def hottest(
        hours: float = Query(24, ge=0, le=24 * 30),
        limit: int = Query(30, ge=1, le=100),
    ) -> dict[str, Any]:
        since = _since_iso(hours)
        store = get_store()
        nodes = store.hottest_nodes(since_ts=since, limit=limit)
        if hours > 0 and len(nodes) < 4:
            nodes = store.hottest_nodes(since_ts=None, limit=limit)
        edges = store.hottest_edges(since_ts=since, limit=limit)
        if hours > 0 and len(edges) < 4:
            edges = store.hottest_edges(since_ts=None, limit=limit)
        return {
            "hours": hours,
            "since": since,
            "nodes": [_node_dict(r) for r in nodes],
            "edges": [_edge_dict(r) for r in edges],
        }

    @router.get("/graph")
    def graph(
        q: str = Query(""),
        hours: float = Query(24, ge=0, le=24 * 30),
        limit: int = Query(50, ge=2, le=120),
        center_id: int | None = Query(None),
    ) -> dict[str, Any]:
        since = None if center_id or q.strip() else _since_iso(hours)
        nodes, edges = get_store().graph_slice(
            center_id=center_id,
            since_ts=since,
            limit=limit,
            q=q,
        )
        return {
            "hours": hours,
            "center_id": center_id,
            "nodes": [_node_dict(r) for r in nodes],
            "edges": [_edge_dict(r) for r in edges],
        }

    @router.get("/chain")
    def last_chain(session_id: str = Query("default")) -> dict[str, Any]:
        payload = chains.get(session_id)
        if payload is None:
            return {"session_id": session_id, "chain": None}
        return {"session_id": session_id, "chain": payload}

    return router
