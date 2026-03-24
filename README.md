### *Personal Agent Assistant*

A flagship project to display my undergraduate knowledge about my favorite topic: machine learning.

### BIG UPDATE: Reasoning chain (first of a kind)
A chain of reasoning without LLM interference. User -> reasoning -> response concepts -> verbalize through LLM.

### Expectation

- PAA can run as an assistant to help with time scheduling on calendar and alarm.
- Featuring a modular agent system that requires minimal adjustment in instructions when swapping LLM model (base model: `qwen3:4b`).

### First Run

*(API)*

- Start backend services (agent API + Ollama + Qwen3-TTS):

```bash
docker compose up --build
```

The API should respond on `http://localhost:8000/agent/health`.

Notes:
- `qwen3-tts` starts on `http://localhost:8880` and downloads model weights on first run.
- `agent-framework` calls TTS at `http://qwen3-tts:8880` inside Docker.
- `ollama` and `qwen3-tts` can compete for GPU VRAM; if startup fails, reduce model size or switch one service to CPU.
- If TTS returns 400/500 with missing model components (for example `speech_tokenizer`), export a Hugging Face token before `docker compose up`:

```bash
# PowerShell
$env:HF_TOKEN="hf_xxx_your_token"
docker compose up --build
```

The token account must have access to the Qwen3-TTS model repository on Hugging Face.

*(GUI)*

After the backend is running, launch the desktop chat GUI:

```bash
python -m pip install -r requirements-gui.txt
python -m src.gui_main
```

Optional environment variables for `agent-framework`:
- `AGENT_BASE_URL`: defaults to `http://localhost:8000` (GUI side)
- `TTS_BASE_URL`: defaults to `http://qwen3-tts:8880` (container-to-container)
- `TTS_MODEL`: defaults to `qwen3-tts`
- `TTS_VOICE`: defaults to `Cherry`
- `TTS_RESPONSE_FORMAT`: defaults to `wav`

### Run (after first run)

Run these commands after initial setup:

```bash
docker compose up -d
python -m src.gui_main
```

### Terminate

- GUI can be turned off with the window's exit button.
- Since Docker is started manually, it must also be turned off manually:

```bash
docker compose down
```

### API responses

`POST /agent/chat` now includes text plus optional TTS payload:
- `reply`: assistant text
- `prompt`: built prompt for debugging
- `reasoning_meta`: reasoning metadata
- `tts_audio_base64`: base64 audio bytes (nullable when TTS fails)
- `tts_format`: response content type, for example `audio/wav`

There is also a dedicated endpoint:
- `POST /agent/tts` with `{ "text": "...", "voice": "...", "response_format": "wav", "model": "..." }`

### WebSocket endpoint (Android/real-time)

`/agent/ws` provides a simple real-time channel for Android clients.

- URL: `ws://<host>:8000/agent/ws`
- Client sends JSON:

```json
{ "message": "Hello", "session_id": "android-cubism" }
```

- Server event sequence:
  - `connected`: handshake/info message
  - `assistant_text`: assistant reply text + reasoning metadata
  - `mouth_control`: fixed-interval mouth mode (`interval_ms`, `open_value`, `closed_value`)
  - `tts_audio`: base64 encoded TTS audio + `content_type`
  - `done`: end of response cycle

Example `mouth_control` payload:

```json
{
  "type": "mouth_control",
  "mode": "fixed_interval",
  "interval_ms": 150,
  "open_value": 1.0,
  "closed_value": 0.0,
  "estimated_duration_s": 2.4
}
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

By default logs are written to `data/logs/` inside the container, which is mounted from the host via `./data:/app/data` in `docker-compose.yml`. You can override the log directory by setting the `CHAT_LOG_DIR` environment variable (default `data/logs`).

### Color theme .json

Each variable (for example `fg_color`) has two values: index 0 is light theme, index 1 is dark theme.

```text
"fg_color": ["#176109", "#0b2404"]
```
