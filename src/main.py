"""Entry point for the agent framework API."""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.llm_client import LLMClient
from src.chat_logger import append_json_log, append_text_log

app = FastAPI(title="Agent Framework API")
llm = LLMClient()

# Simple stack memory (backend-only):
# - Stores the last 5 role/content turns (user + assistant)
# - Single global session (no session_id support yet)
HISTORY_LIMIT = 5
chat_history: list[dict[str, str]] = []


def push_message(role: str, content: str) -> None:
    chat_history.append({"role": role, "content": content})
    while len(chat_history) > HISTORY_LIMIT:
        chat_history.pop(0)


def get_history() -> list[dict[str, str]]:
    return list(chat_history)


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
async def root():
    return {"service": "agent-framework", "status": "ok"}

@app.get("/agent/health")
async def health():
    return {"status": "healthy"}

@app.post("/agent/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    history = get_history()
    result = llm.generate(req.message, history=history)
    reply = result["completion"]
    usage = result.get("usage")

    push_message("user", req.message)
    push_message("assistant", reply)

    try:
        append_json_log(req.message, reply, usage)
        append_text_log(req.message, reply)
    except Exception as exc:
        # Logging failures must not break the API
        print(f"[chat_logger] Failed to write chat log: {exc}")

    return ChatResponse(reply=reply)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)