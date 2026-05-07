from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.time_utils import now_iso


def _utc_now_iso() -> str:
    # Kept for backward compatibility with existing callers, but uses the
    # configured app timezone (EST-default) per project plan.
    return now_iso()


def _safe_json(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return "{}"


_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-']{1,31}")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


@dataclass(frozen=True)
class MemoryConfig:
    db_path: Path


def default_config() -> MemoryConfig:
    db_dir = Path(os.getenv("MEMORY_DB_DIR", "data")).resolve()
    return MemoryConfig(db_path=(db_dir / "memory.sqlite3"))


SCHEMA_SQL = """
PRAGMA journal_mode=DELETE;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  user_text TEXT NOT NULL,
  assistant_text TEXT NOT NULL,
  topic TEXT,
  importance REAL NOT NULL DEFAULT 0.0,
  usage_json TEXT,
  consolidated_ts TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_session_ts
  ON episodes(session_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_episodes_consolidated_ts
  ON episodes(consolidated_ts, ts);

CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'unknown',
  UNIQUE(name, type)
);

CREATE TABLE IF NOT EXISTS episode_entities (
  episode_id INTEGER NOT NULL,
  entity_id INTEGER NOT NULL,
  PRIMARY KEY (episode_id, entity_id),
  FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
  FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episode_entities_entity
  ON episode_entities(entity_id);

CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  type TEXT NOT NULL DEFAULT 'concept',
  summary TEXT NOT NULL DEFAULT '',
  embedding_blob BLOB,
  activation_weight REAL NOT NULL DEFAULT 0.0,
  last_activation_ts TEXT,
  UNIQUE(name, type)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type_name
  ON nodes(type, name);

CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_id INTEGER NOT NULL,
  dst_id INTEGER NOT NULL,
  relation_type TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  last_coactivation_ts TEXT,
  UNIQUE(src_id, dst_id, relation_type),
  FOREIGN KEY (src_id) REFERENCES nodes(id) ON DELETE CASCADE,
  FOREIGN KEY (dst_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_src
  ON edges(src_id);

CREATE TABLE IF NOT EXISTS unresolved (
  task_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  instruction TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  event_start_ts TEXT NOT NULL,
  reminder_fire_at_ts TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_ts TEXT NOT NULL,
  updated_ts TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_unresolved_status_fire
  ON unresolved(status, reminder_fire_at_ts);
"""


class MemoryStore:
    def __init__(self, cfg: Optional[MemoryConfig] = None):
        self.cfg = cfg or default_config()
        self.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.cfg.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def save_episode(
        self,
        *,
        session_id: str,
        user_text: str,
        assistant_text: str,
        topic: str | None = None,
        importance: float = 0.0,
        usage: object | None = None,
        ts: str | None = None,
    ) -> int:
        ts = ts or _utc_now_iso()
        usage_json = _safe_json(usage) if usage is not None else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO episodes(session_id, ts, user_text, assistant_text, topic, importance, usage_json)
                VALUES(?,?,?,?,?,?,?)
                """,
                (session_id, ts, user_text, assistant_text, topic, float(importance), usage_json),
            )
            return int(cur.lastrowid)

    def upsert_entity(self, *, name: str, type_: str = "unknown") -> int:
        name = (name or "").strip()
        type_ = (type_ or "unknown").strip().lower()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO entities(name, type) VALUES(?,?)",
                (name, type_),
            )
            row = conn.execute(
                "SELECT id FROM entities WHERE name=? AND type=?",
                (name, type_),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to upsert entity")
            return int(row["id"])

    def link_episode_entity(self, *, episode_id: int, entity_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO episode_entities(episode_id, entity_id) VALUES(?,?)",
                (int(episode_id), int(entity_id)),
            )

    def upsert_node(
        self,
        *,
        name: str,
        type_: str = "concept",
        summary: str = "",
        embedding_blob: bytes | None = None,
    ) -> int:
        name = (name or "").strip()
        type_ = (type_ or "concept").strip().lower()
        summary = (summary or "").strip()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO nodes(name, type, summary, embedding_blob)
                VALUES(?,?,?,?)
                """,
                (name, type_, summary, embedding_blob),
            )
            # If row exists, optionally refresh missing fields.
            conn.execute(
                """
                UPDATE nodes
                SET summary = CASE WHEN summary = '' THEN ? ELSE summary END,
                    embedding_blob = COALESCE(embedding_blob, ?)
                WHERE name=? AND type=?
                """,
                (summary, embedding_blob, name, type_),
            )
            row = conn.execute(
                "SELECT id FROM nodes WHERE name=? AND type=?",
                (name, type_),
            ).fetchone()
            if row is None:
                raise RuntimeError("Failed to upsert node")
            return int(row["id"])

    def try_enrich_node_summary_web(
        self,
        *,
        node_id: int,
        new_summary: str,
        overwrite_seeded: bool = False,
        max_summary_len: int = 4000,
    ) -> bool:
        """Write ``[web] …`` summary when the node has no summary or a seeded placeholder.

        Does not overwrite hand-written summaries unless ``overwrite_seeded`` and the
        current text looks like episode/doc seeding.
        """
        new_summary = (new_summary or "").strip()
        if not new_summary:
            return False
        payload = f"[web] {new_summary}"[: int(max_summary_len)]
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM nodes WHERE id = ?",
                (int(node_id),),
            ).fetchone()
            if row is None:
                return False
            cur = (row["summary"] or "").strip()
            eligible = not cur
            if not eligible and overwrite_seeded:
                low = cur.lower()
                if low.startswith("seeded from") or low.startswith("(seeded"):
                    eligible = True
            if not eligible:
                return False
            conn.execute(
                "UPDATE nodes SET summary = ? WHERE id = ?",
                (payload, int(node_id)),
            )
        return True

    def upsert_edge(
        self,
        *,
        src_id: int,
        dst_id: int,
        relation_type: str,
        weight_delta: float = 1.0,
        ts: str | None = None,
    ) -> None:
        ts = ts or _utc_now_iso()
        relation_type = (relation_type or "co_occurs").strip().lower()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO edges(src_id, dst_id, relation_type, weight, last_coactivation_ts)
                VALUES(?,?,?,?,?)
                ON CONFLICT(src_id, dst_id, relation_type)
                DO UPDATE SET
                  weight = weight + excluded.weight,
                  last_coactivation_ts = excluded.last_coactivation_ts
                """,
                (int(src_id), int(dst_id), relation_type, float(weight_delta), ts),
            )

    # --- Unresolved task persistence -------------------------------------------

    def upsert_unresolved(
        self,
        *,
        task_id: str,
        session_id: str,
        instruction: str,
        payload_json: str,
        tags_json: str,
        event_start_ts: str,
        reminder_fire_at_ts: str,
        status: str = "active",
        ts: str | None = None,
    ) -> None:
        now_ts = ts or _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO unresolved(
                  task_id, session_id, instruction, payload_json, tags_json,
                  event_start_ts, reminder_fire_at_ts, status, created_ts, updated_ts
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                  session_id=excluded.session_id,
                  instruction=excluded.instruction,
                  payload_json=excluded.payload_json,
                  tags_json=excluded.tags_json,
                  event_start_ts=excluded.event_start_ts,
                  reminder_fire_at_ts=excluded.reminder_fire_at_ts,
                  status=excluded.status,
                  updated_ts=excluded.updated_ts
                """,
                (
                    str(task_id),
                    str(session_id),
                    str(instruction),
                    str(payload_json),
                    str(tags_json),
                    str(event_start_ts),
                    str(reminder_fire_at_ts),
                    str(status),
                    now_ts,
                    now_ts,
                ),
            )

    def list_unresolved_active(self, *, limit: int = 500) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT task_id, session_id, instruction, payload_json, tags_json,
                           event_start_ts, reminder_fire_at_ts, status, created_ts, updated_ts
                    FROM unresolved
                    WHERE status='active'
                    ORDER BY reminder_fire_at_ts ASC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            )

    def mark_unresolved_status(self, *, task_ids: Iterable[str], status: str, ts: str | None = None) -> None:
        ids = [str(t) for t in task_ids if t]
        if not ids:
            return
        now_ts = ts or _utc_now_iso()
        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE unresolved SET status=?, updated_ts=? WHERE task_id IN ({placeholders})",
                (str(status), now_ts, *ids),
            )

    def delete_unresolved(self, *, task_ids: Iterable[str]) -> None:
        ids = [str(t) for t in task_ids if t]
        if not ids:
            return
        placeholders = ",".join(["?"] * len(ids))
        with self._connect() as conn:
            conn.execute(f"DELETE FROM unresolved WHERE task_id IN ({placeholders})", ids)

    def fetch_nodes_for_keyword_seeding(self, tokens: Iterable[str], limit: int = 30) -> list[sqlite3.Row]:
        toks = [t.strip().lower() for t in tokens if t and t.strip()]
        if not toks:
            return []
        # Build (name LIKE ? OR summary LIKE ?) clauses.
        clauses: list[str] = []
        params: list[object] = []
        for t in toks[:12]:
            like = f"%{t}%"
            clauses.append("(LOWER(name) LIKE ? OR LOWER(summary) LIKE ?)")
            params.extend([like, like])
        where = " OR ".join(clauses)
        sql = f"""
        SELECT id, name, type, summary, embedding_blob
        FROM nodes
        WHERE {where}
        LIMIT ?
        """
        params.append(int(limit))
        with self._connect() as conn:
            return list(conn.execute(sql, params).fetchall())

    def fetch_all_nodes(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT id, name, type, summary, embedding_blob FROM nodes"
                ).fetchall()
            )

    def fetch_base_node_ids_for_token_names(self, tokens: Iterable[str]) -> list[int]:
        """Return ids of nodes with type 'base' whose names match given tokens (case-insensitive)."""
        toks = sorted({t.strip().lower() for t in tokens if t and str(t).strip()})
        if not toks:
            return []
        placeholders = ",".join("?" for _ in toks)
        sql = f"""
        SELECT id
        FROM nodes
        WHERE LOWER(type) = 'base' AND LOWER(name) IN ({placeholders})
        """
        with self._connect() as conn:
            rows = conn.execute(sql, toks).fetchall()
        return [int(r["id"]) for r in rows]

    def fetch_node_types_by_ids(self, ids: Iterable[int]) -> dict[int, str]:
        """Map node id -> type string for known ids."""
        id_list = [int(i) for i in ids if int(i) > 0]
        if not id_list:
            return {}
        placeholders = ",".join("?" for _ in id_list)
        sql = f"SELECT id, type FROM nodes WHERE id IN ({placeholders})"
        with self._connect() as conn:
            rows = conn.execute(sql, id_list).fetchall()
        return {int(r["id"]): str(r["type"] or "concept") for r in rows}

    def fetch_nodes_by_ids(self, ids: Iterable[int]) -> list[sqlite3.Row]:
        """Fetch nodes rows by id, including embeddings for graph retrieval."""
        id_list = [int(i) for i in ids if int(i) > 0]
        if not id_list:
            return []
        id_list = id_list[:2000]
        placeholders = ",".join("?" for _ in id_list)
        sql = f"""
        SELECT
            id,
            name,
            type,
            summary,
            embedding_blob,
            activation_weight
        FROM nodes
        WHERE id IN ({placeholders})
        """
        with self._connect() as conn:
            return list(conn.execute(sql, id_list).fetchall())

    def fetch_recent_episode_snippets_for_node_ids(
        self,
        *,
        session_id: str,
        node_ids: Iterable[int],
        limit: int = 6,
    ) -> list[str]:
        """
        Fetch recent episode snippets that are linked (via `episode_entities`)
        to any of the given concept node ids.

        Implementation detail:
        - `episode_entities` references `entities` (not `nodes`).
        - In consolidation, we mirror concept nodes into `entities(type='concept', name=<token>)`,
          so we can join `entities` -> `nodes` on (name,type).
        """
        ids = [int(i) for i in node_ids if int(i) > 0]
        if not ids:
            return []
        clean_limit = max(1, int(limit))

        ids = ids[:500]
        placeholders = ",".join("?" for _ in ids)

        # Deduplication happens in Python because a single episode may link to
        # multiple entities.
        sql = f"""
        SELECT
            ep.id AS episode_id,
            ep.ts,
            ep.user_text,
            ep.assistant_text
        FROM episodes ep
        JOIN episode_entities ee ON ee.episode_id = ep.id
        JOIN entities ent ON ent.id = ee.entity_id
        JOIN nodes n
          ON LOWER(n.name) = LOWER(ent.name)
         AND LOWER(n.type) = LOWER(ent.type)
        WHERE ep.session_id = ?
          AND n.id IN ({placeholders})
        ORDER BY ep.ts DESC
        LIMIT ?
        """

        params: list[object] = [session_id, *ids, int(clean_limit) * 5]
        with self._connect() as conn:
            rows = list(conn.execute(sql, params).fetchall())

        out: list[str] = []
        seen_episode_ids: set[int] = set()
        for r in rows:
            epid = int(r["episode_id"])
            if epid in seen_episode_ids:
                continue
            seen_episode_ids.add(epid)

            u = (r["user_text"] or "").strip().replace("\n", " ")
            a = (r["assistant_text"] or "").strip().replace("\n", " ")
            out.append(f"{r['ts']}: User: {u[:180]} | Assistant: {a[:180]}")
            if len(out) >= clean_limit:
                break
        return out

    def fetch_nodes_by_exact_names(self, names: Iterable[str], limit: int = 64) -> list[sqlite3.Row]:
        cleaned = [n.strip().lower() for n in names if n and n.strip()]
        if not cleaned:
            return []
        clauses = ",".join("?" for _ in cleaned[:64])
        sql = f"""
        SELECT id, name, type, summary, embedding_blob
        FROM nodes
        WHERE LOWER(name) IN ({clauses})
        LIMIT ?
        """
        params: list[object] = [*cleaned[:64], int(limit)]
        with self._connect() as conn:
            return list(conn.execute(sql, params).fetchall())

    def fetch_node_names_for_phrase_scan(self, limit: int = 400) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM nodes
                WHERE INSTR(name, ' ') > 0
                ORDER BY LENGTH(name) DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [str(r["name"] or "").strip() for r in rows if (r["name"] or "").strip()]

    def fetch_edges(self) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT src_id, dst_id, relation_type, weight FROM edges"
                ).fetchall()
            )

    def fetch_edges_for_node_ids(self, node_ids: Iterable[int], limit: int = 40) -> list[sqlite3.Row]:
        ids = [int(i) for i in node_ids if int(i) > 0]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        sql = f"""
        SELECT
            e.src_id,
            s.name AS src_name,
            e.dst_id,
            d.name AS dst_name,
            e.relation_type,
            e.weight
        FROM edges e
        JOIN nodes s ON s.id = e.src_id
        JOIN nodes d ON d.id = e.dst_id
        WHERE e.src_id IN ({placeholders})
           OR e.dst_id IN ({placeholders})
        ORDER BY e.weight DESC
        LIMIT ?
        """
        params: list[object] = [*ids, *ids, int(limit)]
        with self._connect() as conn:
            return list(conn.execute(sql, params).fetchall())

    def fetch_recent_episode_snippets_for_node_names(
        self, *, session_id: str, node_names: Iterable[str], limit: int = 6
    ) -> list[str]:
        names = [n.strip() for n in node_names if n and n.strip()]
        if not names:
            return []
        # Match episodes by substring in user_text or assistant_text.
        clauses: list[str] = []
        params: list[object] = [session_id]
        for n in names[:10]:
            like = f"%{n}%"
            clauses.append("(user_text LIKE ? OR assistant_text LIKE ?)")
            params.extend([like, like])
        where = " OR ".join(clauses)
        sql = f"""
        SELECT ts, user_text, assistant_text
        FROM episodes
        WHERE session_id = ?
          AND ({where})
        ORDER BY ts DESC
        LIMIT ?
        """
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        out: list[str] = []
        for r in rows:
            u = (r["user_text"] or "").strip().replace("\n", " ")
            a = (r["assistant_text"] or "").strip().replace("\n", " ")
            out.append(f"{r['ts']}: User: {u[:180]} | Assistant: {a[:180]}")
        return out

    def fetch_unconsolidated_episodes(self, *, limit: int = 50) -> list[sqlite3.Row]:
        """Return a small batch of episodes that have not yet been consolidated."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, ts, user_text, assistant_text, topic, importance
                FROM episodes
                WHERE consolidated_ts IS NULL
                ORDER BY ts ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return list(rows)

    def mark_episodes_consolidated(
        self, episode_ids: Iterable[int], ts: str | None = None
    ) -> None:
        """Mark a set of episodes as consolidated."""
        ids = [int(eid) for eid in episode_ids]
        if not ids:
            return
        ts = ts or _utc_now_iso()
        placeholders = ",".join("?" for _ in ids)
        sql = f"UPDATE episodes SET consolidated_ts=? WHERE id IN ({placeholders})"
        params: list[object] = [ts, *ids]
        with self._connect() as conn:
            conn.execute(sql, params)

    def decay_graph(self, *, decay: float = 0.98, min_weight: float = 0.05) -> None:
        """Apply simple decay-based forgetting to node activations and edge weights."""
        decay = float(decay)
        if not (0.0 < decay < 1.0):
            return
        min_weight = float(min_weight)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE nodes
                SET activation_weight = COALESCE(activation_weight, 0.0) * ?
                """,
                (decay,),
            )
            conn.execute(
                """
                UPDATE edges
                SET weight = weight * ?
                """,
                (decay,),
            )
            conn.execute(
                "DELETE FROM edges WHERE weight < ?",
                (min_weight,),
            )

