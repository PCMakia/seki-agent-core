### *Personal Agent Assistant*

A flagship project to display my undergraduate knowledge about my favorite topic: machine learning

### Expectation

- PAA can run as an assistant to help with time scheduling on calendar and alarm
- Featuring a modular agent system that require minimal adjustment at the instruction portion when swapping LLM model (base model: qwen3:4B)

### First Run

*(API)*

- Start the backend (agent API + Ollama) with Docker:

```bash
docker compose up --build
```

The API should respond on `http://localhost:8000/agent/health`.

*(GUI)*

After the backend is running, launch the desktop chat GUI:

```bash
python -m pip install -r requirements-gui.txt
python -m src.gui_main
```

Optional environment variable:

- `AGENT_BASE_URL`: defaults to `http://localhost:8000`

### Run (after first run)

- Run these command in terminal to start up the application after the first run

```bash
docker-compose up -d
python -m src.gui_main
```

### Terminate

- GUI can be turned off with the window's exit button
- Since Docker is started up manually, it must also be turned off manually:

```bash
docker-compose down
```

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

### Color theme .json

Each variable for example fg_color have two values: index 0 is light theme, index 1 is dark theme

```text
"fg_color": ["#176109", "#0b2404"]
```

