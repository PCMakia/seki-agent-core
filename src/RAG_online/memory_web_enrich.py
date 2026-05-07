"""Optional background enrichment of concept node summaries from the public web.

Requires ``MEMORY_WEB_ENRICH=1``. Uses :func:`src.internet_access.web_gloss_for_topic`
(search + scrape + extractive summary). Runs **after** each chat turn so the current
reply is not delayed.

Docker: install Playwright browsers in the image (``playwright install chromium``) or
scraping will no-op / fail silently.

Env:
  MEMORY_WEB_ENRICH=1|0           — enable (default off)
  MEMORY_WEB_ENRICH_MAX_NODES=n   — cap concept nodes per turn (default 2)
  MEMORY_WEB_SCRAPE_TOP_N=n       — search results to scrape per topic (default 3)
  MEMORY_WEB_ENRICH_SEEDED=1      — also replace "Seeded from …" placeholders
"""

from __future__ import annotations

import logging
import os

from src.memory_manager.storage.memory_store import MemoryStore
from src.memory_manager.retrieval.reasoning_chain import ReasoningChainResult

_LOG = logging.getLogger("personal_assistant.memory_web")


def memory_web_enrich_enabled() -> bool:
    return os.getenv("MEMORY_WEB_ENRICH", "").strip().lower() in ("1", "true", "yes", "on")


def enrich_nodes_from_web_after_turn(
    store: MemoryStore,
    reasoning_result: ReasoningChainResult | None,
    *,
    max_nodes: int | None = None,
    scrape_top_n: int | None = None,
    overwrite_seeded: bool | None = None,
) -> None:
    """For concept steps in the reasoning chain, fill empty (or seeded) summaries via web."""
    if not memory_web_enrich_enabled() or not reasoning_result:
        return

    from src.RAG_online.internet_access import web_gloss_for_topic

    max_n = max_nodes if max_nodes is not None else int(os.getenv("MEMORY_WEB_ENRICH_MAX_NODES", "2"))
    max_n = max(0, min(max_n, 8))
    if max_n == 0:
        return

    top_n = scrape_top_n if scrape_top_n is not None else int(os.getenv("MEMORY_WEB_SCRAPE_TOP_N", "3"))
    top_n = max(1, min(top_n, 10))

    if overwrite_seeded is None:
        overwrite_seeded = os.getenv("MEMORY_WEB_ENRICH_SEEDED", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    names: list[str] = []
    seen: set[str] = set()
    for step in reasoning_result.steps:
        if (step.step_type or "").strip().lower() != "concept":
            continue
        n = (step.name or "").strip()
        if not n:
            continue
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(n)

    if not names:
        return

    rows = store.fetch_nodes_by_exact_names(names[:48], limit=64)
    by_lower = {str(r["name"] or "").strip().lower(): r for r in rows}

    updated = 0
    for name in names:
        if updated >= max_n:
            break
        row = by_lower.get(name.lower())
        if row is None:
            continue
        if str(row["type"] or "").strip().lower() != "concept":
            continue
        node_id = int(row["id"])
        cur = str(row["summary"] or "").strip()
        if cur and not overwrite_seeded:
            continue
        if cur and overwrite_seeded:
            low = cur.lower()
            if not (low.startswith("seeded from") or low.startswith("(seeded")):
                continue

        try:
            gloss = web_gloss_for_topic(name, top_n=top_n, max_bullets=4)
        except Exception as exc:
            _LOG.warning("web gloss failed for %r: %s", name, exc)
            continue
        if not gloss:
            continue

        try:
            if store.try_enrich_node_summary_web(
                node_id=node_id,
                new_summary=gloss,
                overwrite_seeded=overwrite_seeded,
            ):
                updated += 1
                _LOG.info("memory web enrich updated node_id=%s name=%r", node_id, name)
        except Exception as exc:
            _LOG.warning("try_enrich_node_summary_web failed for %r: %s", name, exc)
