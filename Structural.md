# Structure

## `src/` package map

- `main.py` — Main FastAPI app and orchestration layer for chat, websocket, scheduling, TTS, and streamer mode.
- `chat_logger.py` — Persists chat logs in JSONL and text formats.
- `time_utils.py` — Single timezone source (`APP_TIMEZONE`) and helpers for aware timestamps.
- `task_scheduling.py` — Parses scheduling intent, computes event/reminder times, writes Outlook events, and stores unresolved reminders.
- `reminder_scheduler.py` — In-process reminder queue built on a heap + async wakeup loop.
- `streamer_mode.py` — Live streamer assistant runtime (OBS capture + optional STT + vision commentary + optional TTS).

### `src/GUI/`

- `gui_api.py` — HTTP client used by the desktop GUI to call backend endpoints.
- `gui_main.py` — CustomTkinter desktop chat app with async worker threads and status handling.

### `src/LLM_handler/`

- `llm_client.py` — Ollama client wrapper (sync/async, streaming, multimodal image input).
- `prompt_builder.py` — Builds layered secretary prompts and interaction-reaction prompts. Special code piece: `build_secretary_prompt_layers()` composes system policy + structured user context blocks (memory, reasoning, intent, events, time).
- `reply_sanitize.py` — Post-processes model output to remove scaffold leakage/noise. Special code piece: `sanitize_assistant_reply()` chains cleanup passes to strip next-action boilerplate, `--...--` artifacts, and internal prompt echoes.
- `conversation_intent.py` — Discourse resolver for follow-up messages like "explain?" using prior turns. Special code piece: `resolve_discourse_for_reasoning()` fuses prior user+assistant context into retrieval text when current input is underspecified.
- `intent_policy.py` — Rule-based intent classifier (`command` / `question` / `statement`). Special code piece: `classify_intent()` uses deterministic lexical/punctuation rules (including slash commands and definition-question gating).

### `src/RAG_online/`

- `internet_access.py` — Search/scrape/summarize web pipeline (DuckDuckGo + Playwright + BS4). Special code piece: `fetch_scraped_corpus()` orchestrates URL discovery + scraping + source-tagged corpus assembly for summarization.
- `summarizer.py` — Bullet summarization utilities (extractive embedding mode + LLM mode). Special code piece: `bullet_summary_extractive()` ranks sentences by centroid similarity and shortens bullets for compact summaries.
- `jit_web_knowledge.py` — Pre-prompt just-in-time web lookup for missing concepts in the current user message. Special code piece: `build_jit_web_knowledge_block()` gates lookups, persists web-derived summaries, links to detected graph head, and returns prompt text.
- `memory_web_enrich.py` — Post-turn background web enrichment for concept summaries. Special code piece: `enrich_nodes_from_web_after_turn()` updates only concept nodes referenced by reasoning steps, with overwrite guards for seeded placeholders.

### `src/memory_manager/retrieval/`

- `memory_manager.py` — Service wrapper over `MemoryStore` for retrieval, recording interactions, and efficiency metrics. Special code piece: `retrieve_context()` fail-opens from graph retrieval to keyword retrieval so chat keeps working if graph path fails.
- `graph_memory_retriever.py` — Graph-aware memory retrieval with head detection, bounded traversal, and reranking. Special code piece: `retrieve_context()` does HEAD detection + traversal + distance-aware scoring + evidence assembly for memory blocks.
- `reasoning_chain.py` — Deterministic (non-LLM) reasoning chain builder over retrieved concepts/relations/evidence. Special code piece: `ReasoningChainEngine.build_chain()` scores/re-ranks candidates and emits structured `ReasoningChainResult` for prompt creation.

### `src/memory_manager/storage/`

- `memory_store.py` — SQLite schema + data access layer for episodes, entities, graph nodes/edges, and unresolved tasks. Special code piece: `SCHEMA_SQL` defines the full persistent memory graph model and reminder tables used across the app.
- `memory_consolidation.py` — Background worker that converts episodes into graph updates and applies decay/forgetting. Special code piece: `run_once()` batches unconsolidated episodes into concept nodes, co-occurrence edges, anchoring edges, then marks progress.
- `memory_seeding.py` — Ingests seed documents/text into concept/base graph nodes and optional anchors/clusters. Special code piece: `seed_concepts_from_paths()` drives chunking, optional translation, token selection, and concept upsert flow.
- `embedding_backfill.py` — Offline utility to fill missing `nodes.embedding_blob` values. Special code piece: `backfill_node_embeddings()` batch-encodes node names and upserts embedding bytes without recomputing on each request.
- `graph_cluster_build.py` — Offline builder that creates cluster prototype nodes and cluster/head linking edges. Special code piece: `_greedy_threshold_clusters()` performs deterministic embedding-threshold clustering used to generate `cluster` nodes.
- `faiss_index.py` — FAISS helper for persistent cosine-similarity candidate retrieval over node embeddings. Special code piece: `build_faiss_flatip_index_from_db()` normalizes vectors and builds `IndexIDMap2(IndexFlatIP)` keyed by SQLite node IDs.

### `src/tools_external/windows/`

- `outlook_calendar.py` — Cross-environment Outlook calendar writer (native COM, PowerShell bridge, HTTP bridge).
- `notify_user.py` — Best-effort Windows toast notifications for reminders.
- `create_alarm.py` — Simple local alarm countdown + audible fallback signals.
- `calender_add_event.py` — Minimal standalone Outlook COM test script for creating an appointment item.

## `Execution_scripts/`

- `ollama_docker_entrypoint.sh` — Container entrypoint script for Ollama-related startup flow.
- `outlook_bridge_server.py` — Bridge server script used to expose Outlook integration endpoints.

## `external/`

### `external/qwen3-tts-api/`

- `main.py` — Top-level launch entrypoint.

### `external/qwen3-tts-api/app/`

- `config.py` — Runtime/configuration settings.
- `main.py` — Application bootstrap for the API service.
- `models.py` — Data models used by the app/API.
- `tts_model.py` — TTS model loading/inference integration.

### `external/qwen3-tts-api/app/api/`

- `router.py` — Aggregates and mounts API route modules.

### `external/qwen3-tts-api/app/api/endpoints/`

- `health.py` — Health/readiness API endpoint(s).
- `speech.py` — Speech generation API endpoint(s).
- `voices.py` — Voice listing/selection API endpoint(s).
