"""Entry point for the agent framework API."""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.llm_client import LLMClient
from src.chat_logger import append_json_log, append_text_log

app = FastAPI(title="Agent Framework API")
llm = LLMClient()

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
    result = llm.generate(req.message)
    reply = result["completion"]
    usage = result.get("usage")

    try:
        append_json_log(req.message, reply, usage)
        append_text_log(req.message, reply)
    except Exception as exc:
        # Logging failures must not break the API
        print(f"[chat_logger] Failed to write chat log: {exc}")

    return ChatResponse(reply=reply)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)