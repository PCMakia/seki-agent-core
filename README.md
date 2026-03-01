### *Personal Agent Assistant*

A flagship project to display my undergraduate knowledge about my favorite topic: machine learning

### Expectation
- PAA can run as an assistant to help with time scheduling on calendar and alarm
- Featuring a modular agent system that require minimal adjustment at the instruction portion when swapping LLM model (base model: qwen3:4B)

### Chat logging

Each call to the `/agent/chat` endpoint is logged under the `data/logs/` directory:
- `data/logs/chat_log.jsonl`: JSON Lines file with one object per exchange, including `timestamp`, `user_message`, `agent_reply`, and optional `usage` (token counts).
- `data/logs/chat_log.txt`: human-readable transcript.

Example text transcript entry:

```text
Chat log:
User: Hello
Agent: hello! how can I help you today?
```

By default logs are written to `data/logs/` inside the container, which is mounted from the host via `./data:/app/data` in `docker-compose.yml`. You can override the log directory by setting the `CHAT_LOG_DIR` environment variable (the default is `data/logs`)
