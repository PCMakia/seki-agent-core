"""Entry point for the agent framework API."""
import uvicorn
from fastapi import FastAPI, Query
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
        summary = summarize_round(user_msg, assistant_reply, max_bullets=5, mode="llm", llm=llm)
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


class PromptDebugResponse(BaseModel):
    prompt: str


class ReasoningChainRequest(BaseModel):
    text: str
    session_id: str | None = None


@app.get("/")
async def root():
    return {"service": "agent-framework", "status": "ok"}

@app.get("/agent/health")
async def health():
    return {"status": "healthy"}


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
    history = get_history()

    session_id = (req.session_id or "default").strip() or "default"

    reasoning_result: ReasoningChainResult | None = None
    reasoning_err: str | None = None
    try:
        reasoning_result = reasoning.build_chain(session_id=session_id, text=req.message)
    except Exception as exc:
        reasoning_err = str(exc)
        print(f"[reasoning] Failed to build chain: {exc}")

    intent_info = classify_intent(req.message, reasoning_result)
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
    reasoning_log = reasoning_meta.model_dump()

    # CLS-M: retrieve context block before generation.
    # Fail-open: memory should never break the chat endpoint.
    clsm_memory_block = ""
    try:
        clsm_memory_block = memory.retrieve_context(session_id=session_id, user_text=req.message).block
    except Exception as exc:
        print(f"[memory] Failed to retrieve context: {exc}")

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

    # Use the structured secretary prompt as a single-turn input.
    result = llm.generate(prompt, history=[])
    reply = result["completion"]
    usage = result.get("usage")

    # Best-effort CLS-M context efficiency metrics.
    try:
        memory.record_usage_metrics(
            session_id=session_id,
            user_text=req.message,
            clsm_block=clsm_memory_block,
            reply_text=reply,
        )
    except Exception as exc:
        print(f"[memory] Failed to record usage metrics: {exc}")

    push_message("user", req.message)
    push_message("assistant", reply)

    _append_round_summary(req.message, reply)

    try:
        append_json_log(req.message, reply, usage, reasoning_meta=reasoning_log)
        append_text_log(req.message, reply)
    except Exception as exc:
        # Logging failures must not break the API
        print(f"[chat_logger] Failed to write chat log: {exc}")

    # CLS-M: persist each interaction after we have the assistant reply.
    try:
        memory.record_interaction(
            session_id=session_id,
            user_text=req.message,
            assistant_text=reply,
            usage=usage,
        )
    except Exception as exc:
        print(f"[memory] Failed to record interaction: {exc}")

    return ChatResponse(reply=reply, prompt=prompt, reasoning_meta=reasoning_meta)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)