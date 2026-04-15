"""Entry point for the agent framework API."""
import asyncio
import base64
import functools
import io
import json
import logging
import os
import re
import time
import uuid
import wave
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.llm_client import LLMClient
from src.chat_logger import append_json_log, append_text_log
from src.prompt_builder import build_interaction_reaction_prompt, build_secretary_prompt
from src.memory_manager import MemoryManager
from src.memory_consolidation import ConsolidationWorker
from src.summarizer import summarize_round
from src.reasoning_chain import ReasoningChainEngine, ReasoningChainResult, format_reasoning_block_text
from src.intent_policy import classify_intent
from src.reminder_scheduler import ReminderItem, ReminderScheduler
from src.task_scheduling import schedule_from_user_input
from src.tools.windows.notify_user import notify_windows_reminder
from src.streamer_mode import StreamerModeService, StreamerRuntimeConfig
from src.reply_sanitize import sanitize_assistant_reply

_LOG = logging.getLogger("personal_assistant.main")

# Chunk long assistant replies for sequential TTS on mobile (see _split_reply_into_tts_chunks).
_TTS_CHUNK_MIN_WORDS = 50
_TTS_CHUNK_TARGET_WORDS = 75
_TTS_CHUNK_MAX_WORDS = 100
_TTS_CHUNK_REPLY_WORD_HARD = 150


def _word_count_text(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _should_split_reply_for_tts(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    wc = _word_count_text(t)
    if wc > _TTS_CHUNK_REPLY_WORD_HARD:
        return True
    if "\n\n" in t and wc > 40:
        return True
    if t.count("\n") >= 2 and wc > 60:
        return True
    return False


def _split_reply_into_tts_chunks(text: str) -> list[str]:
    """Split long replies into ~50–100 word chunks, preferring newlines and sentence ends."""
    t = (text or "").strip()
    if not t:
        return []
    if not _should_split_reply_for_tts(t):
        return [t]

    chunks: list[str] = []
    remaining = t
    while remaining:
        wc = _word_count_text(remaining)
        if wc <= _TTS_CHUNK_MAX_WORDS:
            chunks.append(remaining.strip())
            break

        spans = list(re.finditer(r"\S+", remaining))
        if not spans:
            chunks.append(remaining.strip())
            break
        n = len(spans)
        hi = min(_TTS_CHUNK_MAX_WORDS, n)
        lo = min(_TTS_CHUNK_MIN_WORDS, n)
        if lo < 1:
            lo = 1

        best: int | None = None
        for w in range(hi, lo - 1, -1):
            end_pos = spans[w - 1].end()
            after = remaining[end_pos:]
            if after.startswith("\n\n"):
                best = w
                break
            if after.startswith("\n") and w >= lo:
                best = w
                break
            prev_tok = remaining[spans[w - 1].start() : spans[w - 1].end()]
            if prev_tok and prev_tok[-1] in ".?!":
                best = w
                break
        cut_w = best if best is not None else hi

        piece = remaining[spans[0].start() : spans[cut_w - 1].end()].strip()
        if not piece:
            piece = remaining[spans[0].start() : spans[hi - 1].end()].strip()
            tail_start = spans[hi - 1].end()
        else:
            tail_start = spans[cut_w - 1].end()
        chunks.append(piece)
        remaining = remaining[tail_start:].lstrip()

    return [c for c in chunks if c]


llm = LLMClient()
memory = MemoryManager()
consolidation_worker = ConsolidationWorker(store=memory.store)
reasoning = ReasoningChainEngine(store=memory.store)


async def _on_reminder_fire(item: ReminderItem) -> None:
    # Keep terminal log for observability.
    print(f"[reminder] due: task_id={item.task_id} start={item.event_start_ts} instruction={item.instruction[:120]}")
    # Small local desktop reminder (best-effort, Windows only).
    notified = notify_windows_reminder(
        title="Scheduled Task Reminder",
        message=(item.instruction or "You have an upcoming scheduled task.")[:220],
    )
    _LOG.info(
        "[reminder] fire task_id=%s start=%s notified=%s",
        item.task_id,
        item.event_start_ts,
        bool(notified),
    )


reminder_scheduler = ReminderScheduler(on_fire=_on_reminder_fire)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background workers.
    await consolidation_worker.start()
    await reminder_scheduler.start()
    interaction_server: uvicorn.Server | None = None
    interaction_server_task: asyncio.Task | None = None

    if INTERACTION_WS_PORT > 0 and INTERACTION_WS_PORT != 8000:
        interaction_cfg = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=INTERACTION_WS_PORT,
            log_level="info",
            lifespan="off",
        )
        interaction_server = uvicorn.Server(interaction_cfg)
        interaction_server_task = asyncio.create_task(
            interaction_server.serve(),
            name="interaction-ws-server",
        )
        _LOG.info("[startup] interaction websocket server listening on %d", INTERACTION_WS_PORT)

    # Rehydrate unresolved tasks on startup, then clear them to avoid duplicates.
    try:
        rows = memory.store.list_unresolved_active(limit=500)
        loaded_ids: list[str] = []
        for r in rows:
            item = ReminderItem(
                task_id=str(r["task_id"]),
                session_id=str(r["session_id"]),
                instruction=str(r["instruction"]),
                payload_json=str(r["payload_json"]),
                tags_json=str(r["tags_json"]),
                event_start_ts=str(r["event_start_ts"]),
                reminder_fire_at_ts=str(r["reminder_fire_at_ts"]),
            )
            await reminder_scheduler.add(item)
            loaded_ids.append(item.task_id)
        if loaded_ids:
            # Clear loaded rows so old unresolved entries don't mix with new ones.
            memory.store.delete_unresolved(task_ids=loaded_ids)
    except Exception as exc:
        print(f"[startup] Failed to rehydrate unresolved tasks: {exc}")

    try:
        yield
    finally:
        # Flush remaining in-memory deferred tasks back to unresolved on shutdown.
        try:
            items = await reminder_scheduler.snapshot_items()
            for it in items:
                memory.store.upsert_unresolved(
                    task_id=it.task_id,
                    session_id=it.session_id,
                    instruction=it.instruction,
                    payload_json=it.payload_json,
                    tags_json=it.tags_json,
                    event_start_ts=it.event_start_ts,
                    reminder_fire_at_ts=it.reminder_fire_at_ts,
                    status="active",
                )
        except Exception as exc:
            print(f"[shutdown] Failed to persist unresolved tasks: {exc}")

        await reminder_scheduler.stop()
        await consolidation_worker.stop()
        await streamer_service.stop()
        if interaction_server is not None:
            interaction_server.should_exit = True
        if interaction_server_task is not None:
            await interaction_server_task


app = FastAPI(title="Agent Framework API", lifespan=lifespan)

# Simple stack memory (backend-only):
# - Stores the last 6 role/content turns (user + assistant)
# - Single global session (no session_id support yet)
HISTORY_LIMIT = 6
chat_history: list[dict[str, str]] = []

# Session-scoped round summaries (ephemeral, not persisted).
# Each round = one user message + one assistant reply. Deleted on app/docker down.
ROUND_SUMMARIES_LIMIT = 10
round_summaries: list[str] = []

TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://qwen3-tts:8880").rstrip("/")
TTS_MODEL = os.getenv("TTS_MODEL", "qwen3-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "Ono_Anna")
TTS_RESPONSE_FORMAT = os.getenv("TTS_RESPONSE_FORMAT", "wav")
TTS_TIMEOUT_S = float(os.getenv("TTS_TIMEOUT_SECONDS", "120"))
MOUTH_INTERVAL_MS = int(os.getenv("MOUTH_INTERVAL_MS", "150"))
MOUTH_OPEN_VALUE = float(os.getenv("MOUTH_OPEN_VALUE", "0.5"))
MOUTH_CLOSED_VALUE = float(os.getenv("MOUTH_CLOSED_VALUE", "0.0"))
INTERACTION_SCHEMA = os.getenv("INTERACTION_SCHEMA", "head_pat_v1")
INTERACTION_QUEUE_MAX = int(os.getenv("INTERACTION_QUEUE_MAX", "256"))
INTERACTION_AUTO_REACT = os.getenv("INTERACTION_AUTO_REACT", "1").strip() in {"1", "true", "TRUE", "yes"}
INTERACTION_WS_PORT = int(os.getenv("INTERACTION_WS_PORT", "0"))
# Immediate preset line on WebSocket so clients are not silent while prep/LLM run.
IN_TIME_RESPONSE = os.getenv("IN_TIME_RESPONSE", "1").strip() in {"1", "true", "TRUE", "yes", "on"}
IN_TIME_RESPONSE_TTS = os.getenv("IN_TIME_RESPONSE_TTS", "1").strip() in {"1", "true", "TRUE", "yes", "on"}
IN_TIME_MSG_THINK = "Hold on, I need to think for a bit"
IN_TIME_MSG_DATABASE = "Let me check my database"
IN_TIME_MSG_HEADPAT = "thanks for your love"


@dataclass
class InteractionEvent:
    event_id: str
    event_type: str
    schema: str
    session_id: str
    payload: dict[str, Any]
    client_event_ts_ms: int | None
    server_received_ts: str


@dataclass
class WSChatItem:
    message: str
    session_id: str
    tags: list[str]


@dataclass
class WSInteractionItem:
    event: InteractionEvent


def push_message(role: str, content: str) -> None:
    chat_history.append({"role": role, "content": content})
    while len(chat_history) > HISTORY_LIMIT:
        chat_history.pop(0)


def get_history() -> list[dict[str, str]]:
    return list(chat_history)


def get_conversation_summary() -> str:
    """Return accumulated round summaries for the current session (ephemeral)."""
    return "\n\n".join(round_summaries) if round_summaries else ""


def _append_round_summary(user_msg: str, assistant_reply: str) -> None:
    """Summarize a completed round and append to session-scoped round_summaries."""
    global round_summaries
    try:
        summary = summarize_round(user_msg, assistant_reply, max_bullets=5, mode="extractive", llm=llm)
        if summary:
            round_summaries.append(summary)
            while len(round_summaries) > ROUND_SUMMARIES_LIMIT:
                round_summaries.pop(0)
    except Exception as exc:
        print(f"[summarizer] Failed to summarize round: {exc}")


class ChatRequest(BaseModel):
    message: str
    # Optional session identifier; GUI/backend callers may omit it.
    session_id: str | None = None
    # Optional client tags (e.g. headpat / streaming); same normalization as WebSocket `tags`.
    tags: list[str] | None = Field(default=None)


class ScheduleRequest(BaseModel):
    message: str
    session_id: str | None = None


class ScheduleResponse(BaseModel):
    ok: bool
    note: str
    task_id: str | None = None
    outlook_entry_id: str | None = None
    start_ts: str | None = None

class ReasoningMeta(BaseModel):
    intent: str
    intent_source: str | None = None
    reasoning_steps_used: int = 0
    concept_hits: int = 0
    cache_hits: int = 0


class ChatResponse(BaseModel):
    reply: str
    prompt: str
    reasoning_meta: ReasoningMeta | None = None
    tts_audio_base64: str | None = None
    tts_format: str | None = None


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    response_format: str | None = None
    model: str | None = None


class TTSJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str


class TTSJobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    content_type: str | None = None
    audio_base64: str | None = None
    error: str | None = None


class PromptDebugResponse(BaseModel):
    prompt: str


class ReasoningChainRequest(BaseModel):
    text: str
    session_id: str | None = None


@dataclass
class ChatPreparedContext:
    session_id: str
    prompt: str
    reasoning_meta: ReasoningMeta
    clsm_memory_block: str
    reasoning_result: ReasoningChainResult | None = None
    web_knowledge_block: str = ""


class StreamerStartRequest(BaseModel):
    enable_mic: bool | None = None


class StreamerSegmentRequest(BaseModel):
    text: str


class StreamerControlResponse(BaseModel):
    ok: bool
    message: str
    status: dict[str, Any]


tts_jobs: dict[str, dict[str, str | None]] = {}
tts_jobs_lock = asyncio.Lock()
streamer_subscribers: set[WebSocket] = set()
streamer_subscribers_lock = asyncio.Lock()


async def _emit_streamer_event(payload: dict[str, Any]) -> None:
    async with streamer_subscribers_lock:
        subscribers = list(streamer_subscribers)
    if not subscribers:
        return
    stale: list[WebSocket] = []
    for ws in subscribers:
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    if stale:
        async with streamer_subscribers_lock:
            for ws in stale:
                streamer_subscribers.discard(ws)


streamer_service = StreamerModeService(
    llm=llm,
    emit_event=_emit_streamer_event,
    synthesize_tts=lambda text: synthesize_tts_audio(text),
    config=StreamerRuntimeConfig.from_env(),
)


@app.get("/")
async def root():
    return {"service": "agent-framework", "status": "ok"}

@app.get("/agent/health")
async def health():
    return {"status": "healthy"}


@app.post("/agent/schedule", response_model=ScheduleResponse)
async def schedule_task(req: ScheduleRequest):
    sid = (req.session_id or "default").strip() or "default"
    _LOG.info("[api] /agent/schedule request session_id=%s message=%r", sid, (req.message or "")[:220])
    try:
        res = await schedule_from_user_input(
            session_id=sid,
            user_text=req.message,
            llm=llm,
            store=memory.store,
            scheduler=reminder_scheduler,
        )
        _LOG.info(
            "[api] /agent/schedule result ok=%s task_id=%s outlook_entry_id=%s start_ts=%s",
            bool(res.created),
            res.task_id,
            res.outlook_entry_id,
            res.start_ts,
        )
        return ScheduleResponse(
            ok=bool(res.created),
            note=res.note,
            task_id=res.task_id,
            outlook_entry_id=res.outlook_entry_id,
            start_ts=res.start_ts,
        )
    except Exception as exc:
        _LOG.exception("[api] /agent/schedule failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


async def synthesize_tts_audio(
    text: str,
    voice: str | None = None,
    response_format: str | None = None,
    model: str | None = None,
) -> tuple[bytes, str]:
    payload = {
        "model": model or TTS_MODEL,
        "input": text,
        "voice": voice or TTS_VOICE,
        "response_format": response_format or TTS_RESPONSE_FORMAT,
    }

    async with httpx.AsyncClient(timeout=TTS_TIMEOUT_S) as client:
        resp = await client.post(f"{TTS_BASE_URL}/v1/audio/speech", json=payload)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, content_type


def _estimate_audio_duration_seconds(audio_bytes: bytes, content_type: str | None) -> float | None:
    if not audio_bytes:
        return None
    mime = (content_type or "").lower()
    # Keep this strict to WAV for now; other formats can be added later.
    if "wav" not in mime and "wave" not in mime:
        return None
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return float(frames) / float(frame_rate)
    except Exception:
        return None


def _lip_sync_mouth_control_message(estimated_duration_s: float | None = None) -> dict[str, Any]:
    """Same lip-sync parameters as the main chat TTS path (env: MOUTH_*)."""
    msg: dict[str, Any] = {
        "type": "mouth_control",
        "mode": "fixed_interval",
        "interval_ms": MOUTH_INTERVAL_MS,
        "open_value": MOUTH_OPEN_VALUE,
        "closed_value": MOUTH_CLOSED_VALUE,
    }
    if estimated_duration_s is not None:
        msg["estimated_duration_s"] = estimated_duration_s
    return msg


async def _lip_sync_tts_payloads(
    text: str,
    *,
    chunk_tag: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build mouth_control + tts_audio messages for ``text`` (shared by WS and SSE)."""
    tts_started_at = time.perf_counter()
    audio_bytes, content_type = await synthesize_tts_audio(text)
    duration_s = _estimate_audio_duration_seconds(audio_bytes, content_type)
    b64_audio = base64.b64encode(audio_bytes).decode("ascii")
    mouth = _lip_sync_mouth_control_message(duration_s)
    tts: dict[str, Any] = {
        "type": "tts_audio",
        "audio_base64": b64_audio,
        "content_type": content_type,
        "estimated_duration_s": duration_s,
        "generation_ms": int((time.perf_counter() - tts_started_at) * 1000),
    }
    if chunk_tag is not None:
        tts["chunk_tag"] = chunk_tag
    # Mobile streaming client syncs subtitles from this field if assistant_text is delayed or cleared.
    if (text or "").strip():
        tts["reply"] = text
    return mouth, tts


async def _ws_send_lip_sync_tts(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    text: str,
    *,
    chunk_tag: str | None = None,
) -> None:
    mouth, tts = await _lip_sync_tts_payloads(text, chunk_tag=chunk_tag)
    if not await _safe_ws_send(websocket, send_lock, mouth):
        return
    await _safe_ws_send(websocket, send_lock, tts)


def _utc_now_iso() -> str:
    # Kept for backward compatibility with existing response fields, but uses
    # the configured app timezone (EST-default) per project plan.
    from src.time_utils import now_iso

    return now_iso()


def _client_ts_to_iso(client_event_ts_ms: int | None) -> str:
    if client_event_ts_ms is None:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(client_event_ts_ms / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return "unknown"


def _build_interaction_context_block(events: list[InteractionEvent]) -> str:
    if not events:
        return ""
    processed_ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "These interaction events were queued while you were busy and are intentionally delayed.",
        "Acknowledge naturally that they happened earlier if relevant.",
    ]
    for ev in events:
        lag_note = "unknown"
        if ev.client_event_ts_ms is not None:
            try:
                lag_sec = max(0.0, (datetime.now(timezone.utc).timestamp() * 1000.0 - ev.client_event_ts_ms) / 1000.0)
                lag_note = f"{lag_sec:.2f}"
            except Exception:
                lag_note = "unknown"
        lines.append(
            (
                f"- id={ev.event_id} type={ev.event_type} client_ts={_client_ts_to_iso(ev.client_event_ts_ms)} "
                f"server_received={ev.server_received_ts} processed={processed_ts} lag_sec={lag_note} "
                f"payload={json.dumps(ev.payload, ensure_ascii=True)}"
            )
        )
    return "\n".join(lines)


def _normalize_client_tags(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("tags")
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            tag = str(item or "").strip()
            if not tag:
                continue
            out.append(tag.lower()[:64])
    elif raw is not None:
        tag = str(raw).strip()
        if tag:
            out.append(tag.lower()[:64])
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped[:12]


def _normalize_request_tags(raw: list[str] | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for item in raw:
        tag = str(item or "").strip().lower()
        if not tag:
            continue
        out.append(tag[:64])
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped[:12]


def _build_client_tags_context_block(tags: list[str]) -> str:
    if not tags:
        return ""
    return "Client source tags for this turn: " + ", ".join(tags)


def _tags_imply_headpat(tags: list[str]) -> bool:
    for t in tags:
        if "headpat" in t or t.startswith("head_pat") or "head pat" in t:
            return True
    return False


def _pick_in_time_response_text(*, tags: list[str], session_id: str, message: str) -> str:
    if _tags_imply_headpat(tags):
        return IN_TIME_MSG_HEADPAT
    salt = f"{session_id}\n{message}"
    h = hash(salt)
    return IN_TIME_MSG_THINK if (h % 2 == 0) else IN_TIME_MSG_DATABASE


async def _build_in_time_response_ws_payload(
    *,
    text: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "in_time_response", "text": text, "preset": True}
    if session_id is not None:
        payload["session_id"] = session_id
    if tags:
        payload["tags"] = list(tags)
    if IN_TIME_RESPONSE_TTS:
        try:
            tts_started_at = time.perf_counter()
            audio_bytes, content_type = await synthesize_tts_audio(text)
            duration_s = _estimate_audio_duration_seconds(audio_bytes, content_type)
            payload["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
            payload["content_type"] = content_type
            if duration_s is not None:
                payload["estimated_duration_s"] = duration_s
            payload["generation_ms"] = int((time.perf_counter() - tts_started_at) * 1000)
        except Exception as exc:
            payload["tts_error"] = str(exc)
            _LOG.warning("[tts] in_time_response synthesis failed: %s", exc)
    return payload


async def _send_in_time_response(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    *,
    text: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> None:
    payload = await _build_in_time_response_ws_payload(
        text=text,
        session_id=session_id,
        tags=tags,
    )
    if "audio_base64" in payload:
        await _safe_ws_send(
            websocket,
            send_lock,
            _lip_sync_mouth_control_message(payload.get("estimated_duration_s")),
        )
    await _safe_ws_send(websocket, send_lock, payload)


def _sse_data_line(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")


def _normalize_interaction_event(payload: dict[str, Any]) -> tuple[InteractionEvent | None, str | None]:
    event_type = str(payload.get("type", "")).strip()
    if event_type not in {"head_pat_long", "head_pat_end"}:
        return None, "Unsupported interaction event type."
    schema = str(payload.get("schema", "")).strip()
    if schema != INTERACTION_SCHEMA:
        return None, f"Unsupported interaction schema `{schema}`."
    session_id = (str(payload.get("session_id") or "default").strip() or "default")
    raw_client_ts = payload.get("client_event_ts_ms")
    client_event_ts_ms: int | None = None
    if raw_client_ts is not None:
        try:
            client_event_ts_ms = int(raw_client_ts)
        except Exception:
            return None, "Invalid `client_event_ts_ms`; expected integer milliseconds."
    copied = dict(payload)
    copied.pop("schema", None)
    copied.pop("session_id", None)
    copied.pop("type", None)
    return InteractionEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        schema=schema,
        session_id=session_id,
        payload=copied,
        client_event_ts_ms=client_event_ts_ms,
        server_received_ts=datetime.now(timezone.utc).isoformat(),
    ), None


async def _safe_ws_send(websocket: WebSocket, send_lock: asyncio.Lock, payload: dict[str, Any]) -> bool:
    try:
        async with send_lock:
            await websocket.send_json(payload)
        return True
    except Exception:
        return False


def _prepare_chat_context(
    message: str,
    session_id: str,
    history: list[dict[str, str]],
    external_event_context: str = "",
) -> ChatPreparedContext:
    reasoning_result: ReasoningChainResult | None = None
    reasoning_err: str | None = None
    try:
        reasoning_result = reasoning.build_chain(session_id=session_id, text=message)
    except Exception as exc:
        reasoning_err = str(exc)
        print(f"[reasoning] Failed to build chain: {exc}")

    intent_info = classify_intent(message, reasoning_result)
    reasoning_text = format_reasoning_block_text(
        reasoning_result, error=reasoning_err
    )
    intent_label = f"{intent_info['intent']} (source={intent_info['source']})"

    concept_hits = (
        sum(1 for s in reasoning_result.steps if s.step_type == "concept")
        if reasoning_result
        else 0
    )
    reasoning_steps_used = len(reasoning_result.steps) if reasoning_result else 0
    cache_hits = int(reasoning_result.cache_hits) if reasoning_result else 0
    reasoning_meta = ReasoningMeta(
        intent=intent_info["intent"],
        intent_source=intent_info["source"],
        reasoning_steps_used=reasoning_steps_used,
        concept_hits=concept_hits,
        cache_hits=cache_hits,
    )

    clsm_memory_block = ""
    try:
        clsm_memory_block = memory.retrieve_context(session_id=session_id, user_text=message).block
    except Exception as exc:
        print(f"[memory] Failed to retrieve context: {exc}")

    web_knowledge_block = ""
    try:
        from src.jit_web_knowledge import build_jit_web_knowledge_block, jit_web_enabled

        if jit_web_enabled():
            web_knowledge_block = build_jit_web_knowledge_block(
                user_text=message,
                store=memory.store,
                graph_retriever=reasoning.graph_retriever,
                reasoning_result=reasoning_result,
            )
    except Exception as exc:
        print(f"[jit_web] Failed to build web knowledge block: {exc}")

    prompt = build_secretary_prompt(
        user_input=message,
        recent_messages=history,
        clsm_memory=clsm_memory_block,
        conversation_summary=get_conversation_summary(),
        external_event_context=external_event_context,
        instruction=None,
        mode=None,
        reasoning_block=reasoning_text,
        intent_label=intent_label,
        web_knowledge_this_turn=web_knowledge_block,
    )
    return ChatPreparedContext(
        session_id=session_id,
        prompt=prompt,
        reasoning_meta=reasoning_meta,
        clsm_memory_block=clsm_memory_block,
        reasoning_result=reasoning_result,
        web_knowledge_block=web_knowledge_block,
    )


def _run_memory_web_enrich_safe(reasoning_result: ReasoningChainResult | None) -> None:
    try:
        from src.memory_web_enrich import enrich_nodes_from_web_after_turn, memory_web_enrich_enabled

        if not memory_web_enrich_enabled():
            return
        enrich_nodes_from_web_after_turn(memory.store, reasoning_result)
    except Exception as exc:
        _LOG.warning("memory web enrich background task failed: %s", exc)


def _persist_chat_side_effects(
    *,
    session_id: str,
    user_message: str,
    reply: str,
    usage: dict | None,
    clsm_memory_block: str,
    reasoning_meta: ReasoningMeta,
) -> None:
    try:
        memory.record_usage_metrics(
            session_id=session_id,
            user_text=user_message,
            clsm_block=clsm_memory_block,
            reply_text=reply,
        )
    except Exception as exc:
        print(f"[memory] Failed to record usage metrics: {exc}")

    push_message("user", user_message)
    push_message("assistant", reply)

    _append_round_summary(user_message, reply)

    reasoning_log = reasoning_meta.model_dump()
    try:
        append_json_log(user_message, reply, usage, reasoning_meta=reasoning_log)
        append_text_log(user_message, reply)
    except Exception as exc:
        print(f"[chat_logger] Failed to write chat log: {exc}")

    try:
        memory.record_interaction(
            session_id=session_id,
            user_text=user_message,
            assistant_text=reply,
            usage=usage,
        )
    except Exception as exc:
        print(f"[memory] Failed to record interaction: {exc}")


async def _run_tts_job(job_id: str, req: TTSRequest) -> None:
    async with tts_jobs_lock:
        job = tts_jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["updated_at"] = _utc_now_iso()
    try:
        audio, content_type = await synthesize_tts_audio(
            text=req.text,
            voice=req.voice,
            response_format=req.response_format,
            model=req.model,
        )
        async with tts_jobs_lock:
            job = tts_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "completed"
            job["content_type"] = content_type
            job["audio_base64"] = base64.b64encode(audio).decode("ascii")
            job["updated_at"] = _utc_now_iso()
    except Exception as exc:
        async with tts_jobs_lock:
            job = tts_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "failed"
            job["error"] = str(exc)
            job["updated_at"] = _utc_now_iso()


@app.post("/agent/tts")
async def generate_tts(req: TTSRequest):
    audio, content_type = await synthesize_tts_audio(
        text=req.text,
        voice=req.voice,
        response_format=req.response_format,
        model=req.model,
    )
    return Response(content=audio, media_type=content_type)


@app.post("/agent/tts/jobs", response_model=TTSJobResponse, status_code=201)
async def create_tts_job(req: TTSRequest):
    job_id = str(uuid.uuid4())
    now = _utc_now_iso()
    async with tts_jobs_lock:
        tts_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "content_type": None,
            "audio_base64": None,
            "error": None,
        }
    asyncio.create_task(_run_tts_job(job_id, req))
    return TTSJobResponse(job_id=job_id, status="queued", created_at=now)


@app.get("/agent/tts/jobs/{job_id}", response_model=TTSJobStatusResponse)
async def get_tts_job_status(job_id: str):
    async with tts_jobs_lock:
        job = tts_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="TTS job not found")
    return TTSJobStatusResponse(
        job_id=str(job["job_id"]),
        status=str(job["status"]),
        created_at=str(job["created_at"]),
        updated_at=str(job["updated_at"]),
        content_type=str(job["content_type"]) if job["content_type"] else None,
        audio_base64=str(job["audio_base64"]) if job["audio_base64"] else None,
        error=str(job["error"]) if job["error"] else None,
    )


async def _run_ws_chat_turn(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    message: str,
    session_id: str,
    tags: list[str],
    pending_interactions: dict[str, deque[InteractionEvent]],
    pending_interactions_lock: asyncio.Lock,
) -> None:
    started_at = time.perf_counter()
    if IN_TIME_RESPONSE:
        ack = _pick_in_time_response_text(tags=tags, session_id=session_id, message=message)
        await _send_in_time_response(
            websocket,
            send_lock,
            text=ack,
            session_id=session_id,
            tags=tags,
        )
    prep_started_at = time.perf_counter()
    async with pending_interactions_lock:
        queued_events = list(pending_interactions.get(session_id, []))
        pending_interactions[session_id].clear()
    event_block = _build_interaction_context_block(queued_events)
    tags_block = _build_client_tags_context_block(tags)
    external_parts = [p for p in (event_block, tags_block) if p]
    context = _prepare_chat_context(
        message=message,
        session_id=session_id,
        history=get_history(),
        external_event_context="\n\n".join(external_parts),
    )
    prep_elapsed_ms = int((time.perf_counter() - prep_started_at) * 1000)
    if not await _safe_ws_send(
        websocket,
        send_lock,
        {"type": "reasoning_meta", "reasoning_meta": context.reasoning_meta.model_dump()},
    ):
        return

    stream_started_at = time.perf_counter()
    first_token_ms: int | None = None
    pieces: list[str] = []
    try:
        async for piece in llm.async_stream_generate(context.prompt, history=[]):
            if piece:
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - stream_started_at) * 1000)
                pieces.append(piece)
                if not await _safe_ws_send(
                    websocket,
                    send_lock,
                    {"type": "assistant_token", "delta": piece},
                ):
                    return
    except Exception as exc:
        await _safe_ws_send(
            websocket,
            send_lock,
            {"type": "error", "message": f"LLM stream failed: {exc}"},
        )
        await _safe_ws_send(websocket, send_lock, {"type": "done"})
        return

    reply = sanitize_assistant_reply("".join(pieces))
    usage = None
    _persist_chat_side_effects(
        session_id=session_id,
        user_message=message,
        reply=reply,
        usage=usage,
        clsm_memory_block=context.clsm_memory_block,
        reasoning_meta=context.reasoning_meta,
    )

    try:
        from src.memory_web_enrich import memory_web_enrich_enabled

        if memory_web_enrich_enabled():
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                functools.partial(_run_memory_web_enrich_safe, context.reasoning_result),
            )
    except Exception as exc:
        _LOG.warning("memory web enrich schedule (ws) failed: %s", exc)

    chunks = _split_reply_into_tts_chunks(reply)
    if len(chunks) <= 1:
        if not await _safe_ws_send(
            websocket,
            send_lock,
            {
                "type": "assistant_text",
                "reply": reply,
                "reasoning_meta": context.reasoning_meta.model_dump(),
            },
        ):
            return
        try:
            await _ws_send_lip_sync_tts(websocket, send_lock, reply)
        except Exception as exc:
            print(f"[tts] ws synthesis failed: {exc}")
            await _safe_ws_send(
                websocket,
                send_lock,
                {"type": "tts_error", "message": str(exc)},
            )
    else:
        for i, chunk in enumerate(chunks):
            tag = "done" if i == len(chunks) - 1 else "continue"
            chunk_msg: dict[str, Any] = {
                "type": "assistant_text",
                "reply": chunk,
                "chunk_tag": tag,
            }
            if i == len(chunks) - 1:
                chunk_msg["reasoning_meta"] = context.reasoning_meta.model_dump()
            if not await _safe_ws_send(websocket, send_lock, chunk_msg):
                return
            try:
                await _ws_send_lip_sync_tts(
                    websocket,
                    send_lock,
                    chunk,
                    chunk_tag=tag,
                )
            except Exception as exc:
                print(f"[tts] ws chunk synthesis failed: {exc}")
                await _safe_ws_send(
                    websocket,
                    send_lock,
                    {"type": "tts_error", "message": str(exc)},
                )
                break

    total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    print(
        "[timing] ws_total_ms=%d prep_ms=%d first_token_ms=%s"
        % (total_elapsed_ms, prep_elapsed_ms, str(first_token_ms))
    )
    await _safe_ws_send(websocket, send_lock, {"type": "done"})


async def _emit_interaction_reaction(websocket: WebSocket, send_lock: asyncio.Lock, event: InteractionEvent) -> None:
    if not INTERACTION_AUTO_REACT:
        return
    try:
        prompt = build_interaction_reaction_prompt(
            event_type=event.event_type,
            payload=event.payload,
            client_ts_iso=_client_ts_to_iso(event.client_event_ts_ms),
            server_ts_iso=event.server_received_ts,
        )
        result = llm.generate(prompt, history=[])
        text = sanitize_assistant_reply(result.get("completion") or "")
        if text:
            await _safe_ws_send(
                websocket,
                send_lock,
                {
                    "type": "assistant_text",
                    "reply": text,
                    "subtype": "interaction_reaction",
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "delayed": True,
                },
            )
            try:
                await _ws_send_lip_sync_tts(websocket, send_lock, text)
            except Exception as tts_exc:
                _LOG.warning("[tts] interaction reaction synthesis failed: %s", tts_exc)
                await _safe_ws_send(
                    websocket,
                    send_lock,
                    {"type": "tts_error", "message": str(tts_exc)},
                )
    except Exception as exc:
        await _safe_ws_send(
            websocket,
            send_lock,
            {"type": "interaction_notice", "message": f"Queued interaction noted, reaction skipped: {exc}"},
        )


@app.websocket("/agent/ws")
@app.websocket("/agent/ws/interactions")
@app.websocket("/agent/ws/headpat")
async def agent_ws(websocket: WebSocket):
    await websocket.accept()
    send_lock = asyncio.Lock()
    await _safe_ws_send(
        websocket,
        send_lock,
        {
            "type": "connected",
            "message": "Send JSON payload: {\"message\": \"...\", \"session_id\": \"optional\"} or interaction events.",
        },
    )
    chat_queue: asyncio.Queue[WSChatItem] = asyncio.Queue(maxsize=INTERACTION_QUEUE_MAX)
    interaction_queue: asyncio.Queue[WSInteractionItem] = asyncio.Queue(maxsize=INTERACTION_QUEUE_MAX)
    pending_interactions: dict[str, deque[InteractionEvent]] = defaultdict(deque)
    pending_interactions_lock = asyncio.Lock()

    async def receiver_task() -> None:
        while True:
            raw_text = await websocket.receive_text()
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await _safe_ws_send(websocket, send_lock, {"type": "error", "message": "Invalid JSON payload."})
                continue
            if not isinstance(payload, dict):
                await _safe_ws_send(websocket, send_lock, {"type": "error", "message": "Payload must be a JSON object."})
                continue

            message = str(payload.get("message", "")).strip()
            event_type = str(payload.get("type", "")).strip()
            if message and not event_type:
                sid = (str(payload.get("session_id") or "default").strip() or "default")
                tags = _normalize_client_tags(payload)
                await chat_queue.put(WSChatItem(message=message, session_id=sid, tags=tags))
                continue

            if event_type:
                event, err = _normalize_interaction_event(payload)
                if err:
                    await _safe_ws_send(websocket, send_lock, {"type": "error", "message": err})
                    continue
                assert event is not None
                await interaction_queue.put(WSInteractionItem(event=event))
                await _safe_ws_send(
                    websocket,
                    send_lock,
                    {
                        "type": "interaction_ack",
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "session_id": event.session_id,
                        "server_received_ts": event.server_received_ts,
                    },
                )
                continue

            await _safe_ws_send(websocket, send_lock, {"type": "error", "message": "Missing non-empty `message` field."})

    async def chat_worker_task() -> None:
        while True:
            item = await chat_queue.get()
            await _run_ws_chat_turn(
                websocket=websocket,
                send_lock=send_lock,
                message=item.message,
                session_id=item.session_id,
                tags=item.tags,
                pending_interactions=pending_interactions,
                pending_interactions_lock=pending_interactions_lock,
            )

    async def interaction_worker_task() -> None:
        while True:
            item = await interaction_queue.get()
            event = item.event
            async with pending_interactions_lock:
                pending_interactions[event.session_id].append(event)
            if IN_TIME_RESPONSE and event.event_type in {"head_pat_long", "head_pat_end"}:
                await _send_in_time_response(
                    websocket,
                    send_lock,
                    text=IN_TIME_MSG_HEADPAT,
                    session_id=event.session_id,
                )
            await _emit_interaction_reaction(websocket, send_lock, event)

    receiver = asyncio.create_task(receiver_task(), name="agent-ws-receiver")
    chat_worker = asyncio.create_task(chat_worker_task(), name="agent-ws-chat-worker")
    interaction_worker = asyncio.create_task(interaction_worker_task(), name="agent-ws-interaction-worker")
    done, pending = await asyncio.wait(
        {receiver, chat_worker, interaction_worker},
        return_when=asyncio.FIRST_EXCEPTION,
    )
    for task in pending:
        task.cancel()
    for task in done:
        if task.exception() and not isinstance(task.exception(), WebSocketDisconnect):
            raise task.exception()


@app.post("/agent/streamer/start", response_model=StreamerControlResponse)
async def streamer_start(req: StreamerStartRequest):
    try:
        await streamer_service.start(enable_mic=req.enable_mic)
    except Exception as exc:
        return StreamerControlResponse(ok=False, message=str(exc), status=streamer_service.status())
    return StreamerControlResponse(ok=True, message="streamer mode started", status=streamer_service.status())


@app.post("/agent/streamer/stop", response_model=StreamerControlResponse)
async def streamer_stop():
    await streamer_service.stop()
    return StreamerControlResponse(ok=True, message="streamer mode stopped", status=streamer_service.status())


@app.get("/agent/streamer/status", response_model=StreamerControlResponse)
async def streamer_status():
    return StreamerControlResponse(ok=True, message="ok", status=streamer_service.status())


@app.post("/agent/streamer/segment", response_model=StreamerControlResponse)
async def streamer_segment(req: StreamerSegmentRequest):
    await streamer_service.enqueue_manual_utterance(req.text)
    return StreamerControlResponse(ok=True, message="segment queued", status=streamer_service.status())


@app.websocket("/agent/streamer/ws")
async def streamer_ws(websocket: WebSocket):
    await websocket.accept()
    async with streamer_subscribers_lock:
        streamer_subscribers.add(websocket)
    await websocket.send_json({"type": "connected", "message": "streamer event channel"})
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload."})
                continue
            if not isinstance(payload, dict):
                continue
            action = str(payload.get("action", "")).strip().lower()
            if action == "start":
                await streamer_service.start(enable_mic=payload.get("enable_mic"))
                await websocket.send_json({"type": "streamer_status", "status": streamer_service.status()})
            elif action == "stop":
                await streamer_service.stop()
                await websocket.send_json({"type": "streamer_status", "status": streamer_service.status()})
            elif action == "segment":
                text = str(payload.get("text", "")).strip()
                if text:
                    await streamer_service.enqueue_manual_utterance(text)
            elif action == "status":
                await websocket.send_json({"type": "streamer_status", "status": streamer_service.status()})
    except WebSocketDisconnect:
        pass
    finally:
        async with streamer_subscribers_lock:
            streamer_subscribers.discard(websocket)


@app.get("/agent/memory/debug")
async def memory_debug(limit: int = Query(20, ge=1, le=64)):
    """Return recent CLS-M memory usage metrics for debugging/inspection.

    This is intentionally lightweight and best-effort; failures are logged
    and an empty list is returned rather than impacting availability.
    """
    try:
        samples = memory.get_recent_metrics(limit=limit)
    except Exception as exc:
        print(f"[memory] Failed to read debug metrics: {exc}")
        samples = []
    return {"samples": samples}


@app.post("/agent/prompt-debug", response_model=PromptDebugResponse)
async def prompt_debug(req: ChatRequest):
    """Return the structured secretary prompt without calling the LLM.

    This is for GUI/debugging only and does not mutate chat history.
    If ``MEMORY_JIT_WEB`` is on, JIT web text is computed with ``persist=False`` (no DB writes).
    """
    history = get_history()
    session_id = (req.session_id or "default").strip() or "default"

    clsm_memory_block = ""
    try:
        clsm_memory_block = memory.retrieve_context(session_id=session_id, user_text=req.message).block
    except Exception as exc:
        print(f"[memory] Failed to retrieve context (prompt-debug): {exc}")

    reasoning_result: ReasoningChainResult | None = None
    reasoning_err: str | None = None
    try:
        reasoning_result = reasoning.build_chain(session_id=session_id, text=req.message)
    except Exception as exc:
        reasoning_err = str(exc)
        print(f"[reasoning] Failed to build chain (prompt-debug): {exc}")

    intent_info = classify_intent(req.message, reasoning_result)
    reasoning_text = format_reasoning_block_text(
        reasoning_result, error=reasoning_err
    )
    intent_label = f"{intent_info['intent']} (source={intent_info['source']})"

    web_knowledge_block = ""
    try:
        from src.jit_web_knowledge import build_jit_web_knowledge_block, jit_web_enabled

        if jit_web_enabled():
            web_knowledge_block = build_jit_web_knowledge_block(
                user_text=req.message,
                store=memory.store,
                graph_retriever=reasoning.graph_retriever,
                reasoning_result=reasoning_result,
                persist=False,
            )
    except Exception as exc:
        print(f"[jit_web] prompt-debug web block failed: {exc}")

    prompt = build_secretary_prompt(
        user_input=req.message,
        recent_messages=history,
        clsm_memory=clsm_memory_block,
        conversation_summary=get_conversation_summary(),
        instruction=None,
        mode=None,
        reasoning_block=reasoning_text,
        intent_label=intent_label,
        web_knowledge_this_turn=web_knowledge_block,
    )

    return PromptDebugResponse(prompt=prompt)


@app.get("/agent/reasoning-cache-debug")
async def reasoning_cache_debug(
    session_id: str | None = Query(None),
    limit: int = Query(10, ge=1, le=32),
):
    """Return recent topic-head cache entries for a session (process-local RAM).

    Used by the GUI / operators to inspect chain-of-thought head bias state
    after reasoning runs. Does not call the LLM.
    """
    sid = (session_id or "default").strip() or "default"
    recent = reasoning.cache.get_recent(session_id=sid, limit=int(limit))
    heads = [{"node_id": int(nid), "name": str(nm)} for nid, nm in recent]
    return {"session_id": sid, "limit": int(limit), "topic_heads": heads}


@app.post("/agent/reasoning-chain")
async def reasoning_chain(req: ReasoningChainRequest):
    """Build deterministic concept reasoning chain from text (no LLM call)."""
    session_id = (req.session_id or "default").strip() or "default"
    try:
        result = reasoning.build_chain(session_id=session_id, text=req.text)
        return result.to_dict()
    except Exception as exc:
        print(f"[reasoning] Failed to build chain: {exc}")
        return {
            "session_id": session_id,
            "input_text": req.text,
            "tokens": [],
            "steps": [],
            "relations": [],
            "evidence": [],
            "cache_hits": 0,
            "error": str(exc),
        }


@app.on_event("startup")
async def on_startup() -> None:
    # Start the CLS-M consolidation worker in the background.
    try:
        await consolidation_worker.start()
    except Exception as exc:
        # Fail-open: memory consolidation must never prevent startup.
        print(f"[memory] Failed to start consolidation worker: {exc}")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    try:
        await consolidation_worker.stop()
    except Exception as exc:
        print(f"[memory] Failed to stop consolidation worker: {exc}")

@app.post("/agent/chat/sse")
async def chat_sse(req: ChatRequest):
    """Server-Sent Events chat: immediate `in_time_response` (with optional TTS), then token stream and final reply."""

    async def event_stream():
        started_at = time.perf_counter()
        session_id = (req.session_id or "default").strip() or "default"
        tags = _normalize_request_tags(req.tags)
        reasoning_result_for_enrich: ReasoningChainResult | None = None
        try:
            if IN_TIME_RESPONSE:
                ack = _pick_in_time_response_text(
                    tags=tags,
                    session_id=session_id,
                    message=req.message,
                )
                payload = await _build_in_time_response_ws_payload(
                    text=ack,
                    session_id=session_id,
                    tags=tags or None,
                )
                if "audio_base64" in payload:
                    yield _sse_data_line(
                        _lip_sync_mouth_control_message(payload.get("estimated_duration_s")),
                    )
                yield _sse_data_line(payload)

            prep_started_at = time.perf_counter()
            tags_block = _build_client_tags_context_block(tags)
            context = _prepare_chat_context(
                message=req.message,
                session_id=session_id,
                history=get_history(),
                external_event_context=tags_block,
            )
            reasoning_result_for_enrich = context.reasoning_result
            prep_elapsed_ms = int((time.perf_counter() - prep_started_at) * 1000)

            yield _sse_data_line(
                {"type": "reasoning_meta", "reasoning_meta": context.reasoning_meta.model_dump()},
            )

            stream_started_at = time.perf_counter()
            first_token_ms: int | None = None
            pieces: list[str] = []
            try:
                async for piece in llm.async_stream_generate(context.prompt, history=[]):
                    if piece:
                        if first_token_ms is None:
                            first_token_ms = int((time.perf_counter() - stream_started_at) * 1000)
                        pieces.append(piece)
                        yield _sse_data_line({"type": "assistant_token", "delta": piece})
            except Exception as exc:
                yield _sse_data_line({"type": "error", "message": f"LLM stream failed: {exc}"})
                yield _sse_data_line({"type": "done"})
                return

            reply = sanitize_assistant_reply("".join(pieces))
            usage = None
            _persist_chat_side_effects(
                session_id=session_id,
                user_message=req.message,
                reply=reply,
                usage=usage,
                clsm_memory_block=context.clsm_memory_block,
                reasoning_meta=context.reasoning_meta,
            )

            chunks = _split_reply_into_tts_chunks(reply)
            if len(chunks) <= 1:
                yield _sse_data_line(
                    {
                        "type": "assistant_text",
                        "reply": reply,
                        "reasoning_meta": context.reasoning_meta.model_dump(),
                    },
                )
                try:
                    mouth_msg, tts_msg = await _lip_sync_tts_payloads(reply)
                    yield _sse_data_line(mouth_msg)
                    yield _sse_data_line(tts_msg)
                except Exception as exc:
                    _LOG.warning("[tts] sse main reply synthesis failed: %s", exc)
                    yield _sse_data_line({"type": "tts_error", "message": str(exc)})
            else:
                for i, chunk in enumerate(chunks):
                    tag = "done" if i == len(chunks) - 1 else "continue"
                    chunk_payload: dict[str, Any] = {
                        "type": "assistant_text",
                        "reply": chunk,
                        "chunk_tag": tag,
                    }
                    if i == len(chunks) - 1:
                        chunk_payload["reasoning_meta"] = context.reasoning_meta.model_dump()
                    yield _sse_data_line(chunk_payload)
                    try:
                        mouth_msg, tts_msg = await _lip_sync_tts_payloads(
                            chunk,
                            chunk_tag=tag,
                        )
                        yield _sse_data_line(mouth_msg)
                        yield _sse_data_line(tts_msg)
                    except Exception as exc:
                        _LOG.warning("[tts] sse chunk synthesis failed: %s", exc)
                        yield _sse_data_line({"type": "tts_error", "message": str(exc)})
                        break

            total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            print(
                "[timing] sse_chat_total_ms=%d prep_ms=%d first_token_ms=%s"
                % (total_elapsed_ms, prep_elapsed_ms, str(first_token_ms))
            )
            yield _sse_data_line({"type": "done"})
        except Exception as exc:
            _LOG.exception("chat_sse failed")
            yield _sse_data_line({"type": "error", "message": str(exc)})
            yield _sse_data_line({"type": "done"})
        finally:
            if reasoning_result_for_enrich is not None:
                try:
                    from src.memory_web_enrich import memory_web_enrich_enabled

                    if memory_web_enrich_enabled():
                        loop = asyncio.get_running_loop()
                        loop.run_in_executor(
                            None,
                            functools.partial(_run_memory_web_enrich_safe, reasoning_result_for_enrich),
                        )
                except Exception as exc:
                    _LOG.warning("memory web enrich schedule (sse) failed: %s", exc)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    started_at = time.perf_counter()
    prep_started_at = time.perf_counter()
    try:
        history = get_history()
        session_id = (req.session_id or "default").strip() or "default"
        tags = _normalize_request_tags(req.tags)
        tags_block = _build_client_tags_context_block(tags)
        context = _prepare_chat_context(
            message=req.message,
            session_id=session_id,
            history=history,
            external_event_context=tags_block,
        )
        prep_elapsed_ms = int((time.perf_counter() - prep_started_at) * 1000)

        llm_started_at = time.perf_counter()
        try:
            result = llm.generate(context.prompt, history=[])
        except Exception as exc:
            _LOG.exception("LLM generate failed (check Ollama is running and OLLAMA_MODEL is pulled)")
            raise HTTPException(
                status_code=503,
                detail=(
                    f"LLM backend error: {exc}. "
                    "Ensure Ollama is reachable (OLLAMA_BASE_URL), the model exists (`ollama pull <name>`), "
                    "and try again."
                ),
            ) from exc
        llm_elapsed_ms = int((time.perf_counter() - llm_started_at) * 1000)

        reply = sanitize_assistant_reply(result["completion"] or "")
        usage = result.get("usage")

        _persist_chat_side_effects(
            session_id=session_id,
            user_message=req.message,
            reply=reply,
            usage=usage,
            clsm_memory_block=context.clsm_memory_block,
            reasoning_meta=context.reasoning_meta,
        )

        background_tasks.add_task(_run_memory_web_enrich_safe, context.reasoning_result)

        total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        print(
            "[timing] chat_total_ms=%d prep_ms=%d llm_ms=%d"
            % (total_elapsed_ms, prep_elapsed_ms, llm_elapsed_ms)
        )

        return ChatResponse(
            reply=reply,
            prompt=context.prompt,
            reasoning_meta=context.reasoning_meta,
            tts_audio_base64=None,
            tts_format=None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _LOG.exception("/agent/chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)