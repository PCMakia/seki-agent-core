# seki-agent-core

Agent runtime: FastAPI orchestration, a **non-LLM reasoning chain** over a **SQLite concept graph**, retrieval, tools, and chat. Completions go to **seki-inference-engine** (`INFERENCE_URL`). This is the live memory vault Discord and the GUI write into — not the control-plane Postgres `AgentMemory` table.

## What this solves (and why it matters)

Most “agent + memory” demos dump chat logs into a vector store and hope. This service keeps:

- **Episodes** (user line + assistant line) as evidence
- **Nodes / edges** with activation, co-occurrence, and decay
- A **deterministic chain** built before the LLM speaks (the model translates the chain; it does not invent a second plan)
- An **Obsidian-like vault** at `/vault/` so you can see which notes are hot, edit summaries, and open the episode that produced a link

Give it a try if you care about **recall that is inspectable** (“what fired this turn”), Discord and a desktop GUI sharing one write path, or an agent that can hedge when a topic is off the working set.

It does **not** serve vLLM itself and does **not** replace the Next.js dashboard. Image messages on Discord are not wired; the default Instruct checkpoint is text-only.

Useful HTTP:

| Path | Role |
|---|---|
| `POST /agent/chat` | Chat (`phase`: `reply` or `hedge`; tags `discord` / `announce`) |
| `POST /agent/focus` | Cheap working-set probe (no LLM) |
| `GET /vault/` | Graph viewer (same origin as `/agent/vault/*`) |
| `GET /agent/vault/nodes`, `.../episodes`, `.../hottest`, `.../graph`, `.../chain` | Inspect / edit the SQLite graph |
| `GET /agent/health` | Liveness plus inference readiness |

Announce (ambient Discord) turns are **not** written back into the graph, so channel musings do not lock the hottest nodes onto one riff.

## Branch `agent-model` (companion identity)

On branch **`agent-model`**, default identity is **companion** (`AGENT_IDENTITY=companion`) instead of the v1 secretary workplace voice. Default chat mode is **BANTERING** (`AGENT_DEFAULT_MODE=BANTERING`). Set `AGENT_IDENTITY=secretary` to restore legacy prompts.

Pair with **seki-inference-engine** branch `agent-model` and run the A/B benchmark documented in `seki-inference-engine/docs/AGENT_MODEL_BENCHMARK.md`.

**Decision (2026-08-29):** Production chat model is **`seki-qwen-3b`** with companion identity. See `seki-inference-engine/docs/AGENT_MODEL_REPORT.md`.

**This process is CPU-side.** GPU belongs to `seki-inference-engine` / vLLM / Ollama.

- Python **3.10** (Docker image) or 3.10+ locally
- Disk for `data/memory.sqlite3` and optional FAISS indexes
- A reachable OpenAI-compatible gateway (`INFERENCE_URL`)
- Docker only if you use `Dockerfile.agent` (installs Playwright Chromium for optional JIT web)

**Install before first run**

- Working **seki-inference-engine** (`/ready` 200)
- Python 3.10+ and `pip` for a local venv
- Or Docker, and the parent mesh compose if you have `Production-grade/docker-compose.yml`

Optional: Outlook bridge and TTS URLs in `.env.example` for calendar / speech. Not required for chat + vault.

## Fresh install

```powershell
git clone https://github.com/PCMakia/seki-agent-core.git
cd seki-agent-core
Copy-Item .env.example .env
# INFERENCE_URL=http://localhost:9000/v1
# INFERENCE_API_KEY must match the gateway API_KEY
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.main
```

API listens on **8000** by default. The mesh maps host **9080** → container 8000.

**Docker (from this repo):**

```powershell
docker build -f Dockerfile.agent -t seki-agent-core .
docker run --rm -p 9080:8000 --env-file .env seki-agent-core
```

`INFERENCE_URL` inside Compose is `http://seki-v2-inference:8000/v1`. On the host GUI/bot use `http://127.0.0.1:9080`.

Confirm:

```powershell
curl.exe -s http://127.0.0.1:9080/agent/health
```

## How to use it

**Chat (same contract the GUI uses):**

```powershell
curl.exe -s http://127.0.0.1:9080/agent/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"hello\",\"session_id\":\"default\"}"
```

**Discord clients** should send `tags: ["discord"]` and the live `persona_prompt`. Ambient posts use `tags: ["discord","announce"]` so they skip graph writes.

**Vault:** open [http://127.0.0.1:9080/vault/](http://127.0.0.1:9080/vault/). Search notes, drag the graph, edit a summary, unlink an edge, click through to an episode. After a real (non-announce) reply, `GET /agent/vault/chain?session_id=default` shows the chain used that turn.

**Sessions:** GUI `default`; Discord `discord:{channel_id}`; DMs `discord:dm:{user_id}`. Graph nodes are global; episode evidence is session-scoped.

Point `seki-gui` and `seki-discord-bot` at this base URL. Leave control-plane `AgentMemory` as a separate catalog — the live graph is this SQLite file.
