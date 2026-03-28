"""Entry point for the agent framework API."""
import asyncio
import base64
import io
import json
import os
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.llm_client import LLMClient
from src.chat_logger import append_json_log, append_text_log
from src.prompt_builder import build_secretary_prompt
from src.memory_manager import MemoryManager
from src.memory_consolidation import ConsolidationWorker
from src.summarizer import summarize_round
from src.reasoning_chain import ReasoningChainEngine, ReasoningChainResult, format_reasoning_block_text
from src.intent_policy import classify_intent

app = FastAPI(title="Agent Framework API")
llm = LLMClient()
memory = MemoryManager()
consolidation_worker = ConsolidationWorker(store=memory.store)
reasoning = ReasoningChainEngine(store=memory.store)

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


tts_jobs: dict[str, dict[str, str | None]] = {}
tts_jobs_lock = asyncio.Lock()


@app.get("/")
async def root():
    return {"service": "agent-framework", "status": "ok"}

@app.get("/agent/health")
async def health():
    return {"status": "healthy"}


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prepare_chat_context(message: str, session_id: str, history: list[dict[str, str]]) -> ChatPreparedContext:
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

    prompt = build_secretary_prompt(
        user_input=message,
        recent_messages=history,
        clsm_memory=clsm_memory_block,
        conversation_summary=get_conversation_summary(),
        instruction=None,
        mode=None,
        reasoning_block=reasoning_text,
        intent_label=intent_label,
    )
    return ChatPreparedContext(
        session_id=session_id,
        prompt=prompt,
        reasoning_meta=reasoning_meta,
        clsm_memory_block=clsm_memory_block,
    )


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


@app.websocket("/agent/ws")
async def agent_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "connected",
            "message": "Send JSON payload: {\"message\": \"...\", \"session_id\": \"optional\"}",
        }
    )
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON payload."}
                )
                continue

            message = str(payload.get("message", "")).strip()
            session_id = payload.get("session_id")
            if not message:
                await websocket.send_json(
                    {"type": "error", "message": "Missing non-empty `message` field."}
                )
                continue

            sid = (str(session_id) if session_id is not None else "default").strip() or "default"
            started_at = time.perf_counter()
            prep_started_at = time.perf_counter()
            context = _prepare_chat_context(
                message=message,
                session_id=sid,
                history=get_history(),
            )
            prep_elapsed_ms = int((time.perf_counter() - prep_started_at) * 1000)

            await websocket.send_json(
                {"type": "reasoning_meta", "reasoning_meta": context.reasoning_meta.model_dump()}
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
                        await websocket.send_json(
                            {
                                "type": "assistant_token",
                                "delta": piece,
                            }
                        )
            except Exception as exc:
                await websocket.send_json(
                    {"type": "error", "message": f"LLM stream failed: {exc}"}
                )
                await websocket.send_json({"type": "done"})
                continue
            reply = "".join(pieces).strip()
            usage = None

            _persist_chat_side_effects(
                session_id=sid,
                user_message=message,
                reply=reply,
                usage=usage,
                clsm_memory_block=context.clsm_memory_block,
                reasoning_meta=context.reasoning_meta,
            )

            await websocket.send_json(
                {
                    "type": "assistant_text",
                    "reply": reply,
                    "reasoning_meta": context.reasoning_meta.model_dump(),
                }
            )

            tts_started_at = time.perf_counter()
            try:
                audio_bytes, content_type = await synthesize_tts_audio(reply)
                duration_s = _estimate_audio_duration_seconds(audio_bytes, content_type)
                b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                await websocket.send_json(
                    {
                        "type": "mouth_control",
                        "mode": "fixed_interval",
                        "interval_ms": MOUTH_INTERVAL_MS,
                        "open_value": MOUTH_OPEN_VALUE,
                        "closed_value": MOUTH_CLOSED_VALUE,
                        "estimated_duration_s": duration_s,
                    }
                )
                await websocket.send_json(
                    {
                        "type": "tts_audio",
                        "audio_base64": b64_audio,
                        "content_type": content_type,
                        "estimated_duration_s": duration_s,
                        "generation_ms": int((time.perf_counter() - tts_started_at) * 1000),
                    }
                )
            except Exception as exc:
                print(f"[tts] ws synthesis failed: {exc}")
                await websocket.send_json(
                    {
                        "type": "tts_error",
                        "message": str(exc),
                    }
                )

            total_elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            print(
                "[timing] ws_total_ms=%d prep_ms=%d first_token_ms=%s"
                % (total_elapsed_ms, prep_elapsed_ms, str(first_token_ms))
            )

            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        return


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

    This is for GUI/debugging only and does not mutate chat history or memory.
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

    prompt = build_secretary_prompt(
        user_input=req.message,
        recent_messages=history,
        clsm_memory=clsm_memory_block,
        conversation_summary=get_conversation_summary(),
        instruction=None,
        mode=None,
        reasoning_block=reasoning_text,
        intent_label=intent_label,
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

@app.post("/agent/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    started_at = time.perf_counter()
    prep_started_at = time.perf_counter()
    history = get_history()
    session_id = (req.session_id or "default").strip() or "default"
    context = _prepare_chat_context(
        message=req.message,
        session_id=session_id,
        history=history,
    )
    prep_elapsed_ms = int((time.perf_counter() - prep_started_at) * 1000)

    llm_started_at = time.perf_counter()
    result = llm.generate(context.prompt, history=[])
    llm_elapsed_ms = int((time.perf_counter() - llm_started_at) * 1000)

    reply = result["completion"]
    usage = result.get("usage")

    _persist_chat_side_effects(
        session_id=session_id,
        user_message=req.message,
        reply=reply,
        usage=usage,
        clsm_memory_block=context.clsm_memory_block,
        reasoning_meta=context.reasoning_meta,
    )

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)