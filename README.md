# *Private Assistant Agent*

A flagship project to display my undergraduate knowledge about my favorite topic: machine learning.

### BIG UPDATE(): Reasoning chain
A chain of reasoning without LLM interference. User -> reasoning -> response concepts -> verbalize through LLM.

### Expectation

- PAA can run as an assistant to help with time scheduling on calendar and alarm.
- Featuring a modular agent system that requires minimal adjustment in instructions when swapping LLM model (base model: `qwen3:4b`).

# First Run

Prerequisites to install manually on a fresh machine:
- Python 3.10+ ([Download Python](https://www.python.org/downloads/)) (must provide `python` command)
- Git (required to clone this repo and to get Git Bash on Windows) ([Git downloads](https://git-scm.com/downloads))
- Bash shell to run `install.sh` ([Git Bash](https://git-scm.com/download/win), [WSL install guide](https://learn.microsoft.com/windows/wsl/install))
- Docker Desktop (or Docker Engine) with Docker Compose v2 ([Docker Desktop](https://www.docker.com/products/docker-desktop/), [Docker Engine install docs](https://docs.docker.com/engine/install/), [Compose v2 docs](https://docs.docker.com/compose/))
- Ollama (required model runtime for chat/vision models used by this project) ([Ollama downloads](https://ollama.com/download))
- (For NVIDIA GPU) drivers + NVIDIA Container Toolkit for GPU containers ([NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx), [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- (Optional but recommended) Hugging Face token with access to Qwen3-TTS models ([Hugging Face tokens](https://huggingface.co/settings/tokens), [Qwen3-TTS 0.6B model page](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)) (`HF_TOKEN`)

### Ollama first-time setup (fresh install)

If Ollama is not yet installed on your machine:

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Start Ollama once so the service is running:
   - Windows/macOS: open the Ollama app.
   - Linux: run `ollama serve`.
3. Pull the models used by this repo (or let Docker pull them in-container later):

```bash
ollama pull phi3:mini
ollama pull llava:7b
ollama list
```

If you run only with Docker Compose, the repo also starts an `ollama` container and you can pull models with `docker compose exec ollama ...` in step 5 below.

## 0) Create your local env file from template

Copy `.env.example` to `.env` and edit values for your machine before first `docker compose up`:

```bash
cp .env.example .env
```

```powershell
Copy-Item .env.example .env
```

At minimum, set `HF_TOKEN` if your Qwen3-TTS model download requires authenticated Hugging Face access.

## 1) Install Python dependencies into the project venv


Optional flags:
- `INSTALL_SEEDING_DEPS=0 bash install.sh` to skip seeding extras
- `INSTALL_QWEN_TTS_PKG=1 bash install.sh` to also install `qwen-tts` into your venv

Example fresh install (enables all optional installer flags):

```bash
# Bash / Git Bash / WSL
export HF_TOKEN="hf_xxx_your_token"
INSTALL_SEEDING_DEPS=1 INSTALL_QWEN_TTS_PKG=1 bash install.sh
```

```powershell
# PowerShell
$env:HF_TOKEN="hf_xxx_your_token"
$env:INSTALL_SEEDING_DEPS="1"
$env:INSTALL_QWEN_TTS_PKG="1"
bash install.sh
```

What `install.sh` already installs for you:
- main backend Python deps (`requirements.txt`)
- GUI Python deps (`requirements-gui.txt`)
- seeding extras (`requirements-seeding.txt`, unless disabled)
- vendored external TTS API Python deps (`external/qwen3-tts-api/requirements.txt`)

You can add HF_TOKEN into the .env file, or into .env.example and redo step 0

Note:
- `requirements-summarizer-extractive.txt` is not required in the default app path because `qwen3-embed` is already listed in `requirements.txt`.

## 2) Activate the same venv before running GUI/host scripts (required)

Use one of these:

```bash
# Bash / Git Bash / WSL
source .venv/bin/activate
```

```powershell
# PowerShell
.venv\Scripts\Activate.ps1
```

Later step 7, if PowerShell policy blocks activation, use the venv Python directly:

```powershell
.venv\Scripts\python -m src.GUI.gui_main
```

## 3) Review `docker-compose.yml` host paths on Windows (required)

This repo currently mounts Ollama cache from line 15 of docker-compose.yml:

```docker-compose.yml
D:/data/ollama-data:/root/.ollama
```

If your machine does not have `D:` (or Docker Desktop file sharing for `D:` is disabled), update this path before step 4 (for example `C:/data/ollama-data:/root/.ollama`) and ensure the drive is shared in Docker Desktop settings.

## 4) Start backend services

- Start backend services (agent API + Ollama + Qwen3-TTS):

```bash
docker compose up --build
```

## 5) Fresh install: verify/pull Ollama models

On a new machine, Ollama may start without the model weights used by this repo.
Default backend model is `phi3:mini` (see `docker-compose.yml`), and streamer mode may use `llava:7b`.

After `docker compose up -d`, pull required models:

```bash
# Pull default chat model inside the Ollama container
docker compose exec ollama ollama pull phi3:mini

# Optional: pull vision model for streamer mode
docker compose exec ollama ollama pull llava:7b
```

Verify installed models:

```bash
docker compose exec ollama ollama list
```

If you change `OLLAMA_MODEL` or `OLLAMA_VISION_MODEL`, pull those model names instead.

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

## 6) (Optional Window only) Start Outlook bridge on host for calendar features

Scheduling/calendar calls from Docker expect a host bridge at `http://host.docker.internal:8765` (default `OUTLOOK_BRIDGE_URL` in compose).  
Run this on the Windows host if you plan to use Outlook calendar features:

```bash
python Execution_scripts/outlook_bridge_server.py
```

If you only use chat/TTS (mobile) and not Outlook calendar integration, this bridge is not required.

## 7) Start PC GUI

After the backend is running, launch the desktop chat GUI:

```bash
python -m src.GUI.gui_main
```

Optional environment variables for `agent-framework`:
- `AGENT_BASE_URL`: defaults to `http://localhost:8000` (GUI side)
- `TTS_BASE_URL`: defaults to `http://qwen3-tts:8880` (container-to-container)
- `TTS_MODEL`: defaults to `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` in `docker-compose.yml` (code fallback is `qwen3-tts` when env is unset)
- `TTS_VOICE`: defaults to `Ono_Anna`
- `TTS_RESPONSE_FORMAT`: defaults to `wav`
- `INTERACTION_WS_PORT`: optional second WebSocket listener port for interaction events (default `8001`)
- `INTERACTION_SCHEMA`: schema name for Android interaction payload validation (default `head_pat_v1`)
- `OLLAMA_VISION_MODEL`: multimodal model name for streamer mode (default `llava:7b`)
- `STREAMER_OBS_HOST` / `STREAMER_OBS_PORT` / `STREAMER_OBS_PASSWORD`: OBS WebSocket connection
- `STREAMER_STT_ENABLED`: set `1` to enable microphone VAD+STT loop
- `STREAMER_TTS_ENABLED`: set `1` to emit TTS in streamer mode

# Run (after first run)

Run these commands after initial setup:

```bash
docker compose up -d
python -m src.GUI.gui_main
```

### Terminate

- GUI can be turned off with the window's exit button.
- Since Docker is started manually, it must also be turned off manually:

```bash
docker compose down
```
# Details

### API responses

`POST /agent/chat` is text-first and returns:
- `reply`: assistant text
- `prompt`: built prompt for debugging
- `reasoning_meta`: reasoning metadata
- `tts_audio_base64`: currently `null` by default (TTS is decoupled from chat path)
- `tts_format`: currently `null` by default

There is also a dedicated endpoint:
- `POST /agent/tts` with `{ "text": "...", "voice": "...", "response_format": "wav", "model": "..." }`

Async TTS job endpoints:
- `POST /agent/tts/jobs` with the same payload as `/agent/tts` (returns `job_id` and `queued` status)
- `GET /agent/tts/jobs/{job_id}` to poll `queued | running | completed | failed`
- completed jobs include `audio_base64` and `content_type`

### WebSocket endpoint (Android/real-time + interaction queue)

`/agent/ws` (and alias `/agent/ws/interactions`) provide a concurrent real-time channel for Android clients.

- URL: `ws://<host>:8000/agent/ws`
- Optional second listener URL (same routes): `ws://<host>:8001/agent/ws` when `INTERACTION_WS_PORT` is enabled
- Client sends JSON:

```json
{ "message": "Hello", "session_id": "android-cubism" }
```

Android interaction events can be sent on the same socket while a chat turn is streaming:

```json
{
  "type": "head_pat_long",
  "schema": "head_pat_v1",
  "session_id": "android-cubism",
  "client_event_ts_ms": 1776000123456,
  "threshold_sec": 5.0,
  "elapsed_sec": 5.4
}
```

- Server event sequence:
  - `connected`: handshake/info message
  - `interaction_ack`: immediate ack for each accepted interaction event
  - `reasoning_meta`: metadata snapshot before generation
  - `assistant_token`: streamed text chunk (`delta`) from LLM
  - `assistant_text`: final assistant reply text + reasoning metadata (compatibility event)
  - `assistant_text` with `subtype=interaction_reaction`: short delayed-event reaction generated after current processing
  - `mouth_control`: fixed-interval mouth mode (`interval_ms`, `open_value`, `closed_value`)
  - `tts_audio`: base64 full TTS audio + `content_type`
  - `done`: end of response cycle

Queued interaction events are timestamped and injected into the next prompt context, so responses can acknowledge intentional delay naturally instead of feeling like backend lag.

Example `mouth_control` payload:

```json
{
  "type": "mouth_control",
  "mode": "fixed_interval",
  "interval_ms": 150,
  "open_value": 0.5,
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

### Streamer mode (voice + OBS screen commentary)

Streamer mode combines voice segments with OBS screen capture and a vision model:

- Start: `POST /agent/streamer/start` with optional body `{ "enable_mic": true }`
- Stop: `POST /agent/streamer/stop`
- Status: `GET /agent/streamer/status`
- Manual test segment: `POST /agent/streamer/segment` with `{ "text": "..." }`
- Event channel: `ws://<host>:8000/agent/streamer/ws`

Event channel emits:

- `streamer_input`: accepted voice segment
- `streamer_comment`: multimodal commentary reply (voice + OBS frame)
- `tts_audio`: optional when `STREAMER_TTS_ENABLED=1`
- `streamer_error`: runtime failures (OBS/STT/model)

OBS integration uses `GetCurrentProgramScene` + `GetSourceScreenshot`. If OBS Studio Mode differs between preview/program, screenshot semantics can differ as noted in [obs-websocket issue #1257](https://github.com/obsproject/obs-websocket/issues/1257).

### IMPORTANT: base anchoring nodes needs manual inserting.
Base-node are the higher node than concept-node, with the relationship to the concept-node using itself as the anchor. This forms chains of concepts node relating to the base node you created.

Example of adding base node group:
```bash
PC = store.upsert_node(name="personal_concepts", type_ = "base", summary="This pillar determines the unique identity an intelligence has. Elements in this pillar are emergent from the interactions with living-environment. Once a node is registered under this pillar, either by enough interactions-count or high affinity with lower interaction-count, the individual would have said concept as part of their identity. This means that the individual would more likely to react according to the properties of the concepts connecting to this pillar. The node with higher interaction-count has higher priority in decision making.")
secretary = store.upsert_node(name="secretary", type_="base", summary="One of personal concepts, which is part of your identity as an existence. This one in particular is a title for doing tasks like scheduling, organizing, and managing people's life.")


store.upsert_edge(
    src_id=secretary,
    dst_id=PC,
    relation_type="anchored_to",
)
```

### Seed memory.sqlite3 from DOCX/text

Use the production seeding pipeline:

```bash
python -m src.memory_manager.storage.memory_seeding --inputs docs/seed_source.docx --base-name thesis --memory-db-dir data
```

Common options:
- `--translate`: translate each chunk to English through Ollama (`LLMClient`) before token seeding.
- `--cluster-bases`: run embedding clustering to auto-create `type='base'` nodes and `anchored_to` edges (install extra deps first).
- `--dry-run`: parse and tokenize without writing to SQLite.

Dependency notes:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-seeding.txt
```

Operational notes:
- Seed the same DB used by the API (`MEMORY_DB_DIR` or `--memory-db-dir` path).
- Avoid concurrent DB writers while seeding (stop the server or use one process at a time).

### Color theme .json

Each variable (for example `fg_color`) has two values: index 0 is light theme, index 1 is dark theme.

```text
"fg_color": ["#176109", "#0b2404"]
```


### FAISS index (upgrade to graph database)
Build cluster
```bash
python -m src.memory_manager.storage.graph_cluster_build --memory-db-dir data --link-heads
```

Embedding
```bash
python -m src.memory_manager.storage.embedding_backfill --memory-db-dir data
```
Create index file
```bash
python -c "from src.memory_manager.storage.memory_store import MemoryStore; from src.memory_manager.storage.faiss_index import build_faiss_flatip_index_from_db; store=MemoryStore(); build_faiss_flatip_index_from_db(store=store); print('faiss index built')"
```


Ways to populate database:
- Chat with agent
- seed from docx
```bash
python -m src.memory_manager.storage.memory_seeding --inputs "path/to/file.docx" --memory-db-dir data
```

# Change log
### V1.0: IMPORTANT (need fresh install from beta)
- Structural subfolders in src
- Wired the outlook (legacy version) interaction scripts into the agent
- Adding Structure.md for keep track of script's functions 

### V0.18
- New feature: Immediate response
  - Preset responses to confirm the backend received the input
  - Currently only works on the mobile app Streaming mode
### V0.17
- Wired Internet search to agent
- Add sanitize function to clean up tag instructions for user in LLM result
- Exchange qwen3:4b into phi3:mini to increase response rate (experimental)
- In-time-response: preset dialog responses to be sent immediately after backend received user's input
- Mobile client backend features:
  - 2nd Websocket to accept interaction notifications from mobile client (ex. headpat)
  - synchronize function to hold result of llm if it is too long (>150 words), splits into chunks and converting by Text-to-Speech (TTS) and send the each chunk with its audio together to mobile client 


### V0.16
- Added tools to interact with window application: Calendar, Internet search, Notification system

### V0.15
- Added graph-aware memory retrieval and integrated it into memory/context selection with fail-open fallback.
- Added graph tooling pipeline: embedding backfill, cluster/head graph builder, and FAISS index utilities.
- Improved memory graph persistence by linking episodes to concept entities and writing bidirectional anchoring edges.


### V0.14
- Added network integration for Android avatar text output.
- [PATCH v0.14.1]
  - Detached text and voice generation; improved text generation speed by 50%.
- [PATCH v0.14.2]
  - Added DOCX-based concept-node seeding for memory and improved the reasoning chain.
- [PATCH v0.14.3]
  - Updated README format.

### V0.13
- Initial commit of the reasoning module.

### V0.12
- Added internet search and abstractive summary features.
- Updated application icon.

### V0.11
- Added summarizer function (not yet connected to agent flow).
- [PATCH v0.11.1]
  - Integrated summarizer into prompt builder.

### V0.10
- Added mode-specific instructions to specialize behavior.

### V0.9
- Integrated CLS-M memory with persistence through shutdown.

### V0.8
- Integrated prompt builder function.

### V0.7
- Updated stack memory with a limit of 5.
- [PATCH v0.7.1]
  - Added stack memory extraction to raw text.

### V0.6
- Updated GUI using CustomTkinter.

### V0.1
- Initialize framework.