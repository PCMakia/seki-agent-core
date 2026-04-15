from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from src.llm_client import LLMClient

try:
    import obsws_python as obsws
except Exception:  # pragma: no cover - optional dependency
    obsws = None

try:
    import numpy as np
    import sounddevice as sd
    import webrtcvad
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional dependency
    np = None
    sd = None
    webrtcvad = None
    WhisperModel = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

_LOG = logging.getLogger("personal_assistant.streamer_mode")


@dataclass
class StreamerRuntimeConfig:
    obs_host: str = "127.0.0.1"
    obs_port: int = 4455
    obs_password: str = ""
    obs_source_name: str = ""
    obs_capture_width: int = 960
    obs_capture_height: int = 540
    obs_capture_quality: int = 80
    obs_min_capture_interval_ms: int = 400
    streamer_session_id: str = "streamer-default"
    streamer_vision_model: str = ""
    streamer_comment_style: str = "short_reactive"
    stt_model_size: str = "small"
    stt_language: str = "en"
    stt_vad_aggressiveness: int = 2
    stt_min_speech_ms: int = 450
    stt_end_silence_ms: int = 700
    stt_enabled: bool = False
    tts_enabled: bool = False

    @staticmethod
    def from_env() -> "StreamerRuntimeConfig":
        return StreamerRuntimeConfig(
            obs_host=os.getenv("STREAMER_OBS_HOST", "127.0.0.1"),
            obs_port=int(os.getenv("STREAMER_OBS_PORT", "4455")),
            obs_password=os.getenv("STREAMER_OBS_PASSWORD", ""),
            obs_source_name=os.getenv("STREAMER_OBS_SOURCE_NAME", ""),
            obs_capture_width=int(os.getenv("STREAMER_OBS_WIDTH", "960")),
            obs_capture_height=int(os.getenv("STREAMER_OBS_HEIGHT", "540")),
            obs_capture_quality=int(os.getenv("STREAMER_OBS_QUALITY", "80")),
            obs_min_capture_interval_ms=int(os.getenv("STREAMER_OBS_MIN_INTERVAL_MS", "400")),
            streamer_session_id=os.getenv("STREAMER_SESSION_ID", "streamer-default"),
            streamer_vision_model=os.getenv("STREAMER_VISION_MODEL", ""),
            streamer_comment_style=os.getenv("STREAMER_COMMENT_STYLE", "short_reactive"),
            stt_model_size=os.getenv("STREAMER_STT_MODEL_SIZE", "small"),
            stt_language=os.getenv("STREAMER_STT_LANGUAGE", "en"),
            stt_vad_aggressiveness=int(os.getenv("STREAMER_STT_VAD_AGGRESSIVENESS", "2")),
            stt_min_speech_ms=int(os.getenv("STREAMER_STT_MIN_SPEECH_MS", "450")),
            stt_end_silence_ms=int(os.getenv("STREAMER_STT_END_SILENCE_MS", "700")),
            stt_enabled=os.getenv("STREAMER_STT_ENABLED", "0").strip() in {"1", "true", "TRUE", "yes"},
            tts_enabled=os.getenv("STREAMER_TTS_ENABLED", "0").strip() in {"1", "true", "TRUE", "yes"},
        )


class OBSFrameCapture:
    def __init__(self, config: StreamerRuntimeConfig):
        self.config = config

    async def capture_frame_base64_png(self) -> str:
        return await asyncio.to_thread(self._capture_sync)

    def _capture_sync(self) -> str:
        if obsws is None:
            raise RuntimeError("obsws-python is not installed")

        client = obsws.ReqClient(
            host=self.config.obs_host,
            port=self.config.obs_port,
            password=self.config.obs_password,
            timeout=5,
        )
        source_name = self.config.obs_source_name.strip()
        if not source_name:
            scene_resp = client.get_current_program_scene()
            source_name = getattr(scene_resp, "current_program_scene_name", "")
        if not source_name:
            raise RuntimeError("Could not resolve OBS source/scene name")

        shot = client.get_source_screenshot(
            source_name=source_name,
            image_format="png",
            image_width=self.config.obs_capture_width,
            image_height=self.config.obs_capture_height,
            image_compression_quality=max(0, min(100, self.config.obs_capture_quality)),
        )
        image_data = (
            getattr(shot, "image_data", "")
            or getattr(shot, "imageData", "")
            or getattr(shot, "img", "")
        )
        if not image_data:
            raise RuntimeError("OBS returned empty screenshot payload")
        if "base64," in image_data:
            image_data = image_data.split("base64,", 1)[1]

        # Optional post-resize if caller wants final enforcement and PIL exists.
        if Image is None:
            return image_data

        raw = base64.b64decode(image_data)
        with Image.open(io.BytesIO(raw)) as img:
            resized = img.convert("RGB").resize(
                (self.config.obs_capture_width, self.config.obs_capture_height),
                Image.Resampling.LANCZOS,
            )
            out = io.BytesIO()
            resized.save(out, format="PNG", optimize=True)
            return base64.b64encode(out.getvalue()).decode("ascii")


class DesktopSpeechPipeline:
    """
    Real-time microphone capture + VAD + local STT.

    Uses sounddevice RawInputStream + webrtcvad for utterance boundaries,
    and faster-whisper for text transcription.
    """

    def __init__(self, config: StreamerRuntimeConfig):
        self.config = config
        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000)
        self._utterance_queue: asyncio.Queue[str] = asyncio.Queue()
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=400)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def available(self) -> bool:
        return all([np is not None, sd is not None, webrtcvad is not None, WhisperModel is not None])

    async def start(self) -> None:
        if not self.available:
            raise RuntimeError("sounddevice/webrtcvad/faster-whisper dependencies are not available")
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        await asyncio.to_thread(self._start_sync)

    def _start_sync(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._worker_thread = threading.Thread(target=self._worker_main, name="streamer-stt-worker", daemon=True)
        self._worker_thread.start()

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
            callback=self._audio_callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_sync)

    def _stop_sync(self) -> None:
        self._stop_event.set()
        self._audio_queue.put(None)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - callback path
        del frames, time_info
        if status:
            _LOG.debug("[stt] input status: %s", status)
        try:
            self._audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            _LOG.warning("[stt] audio queue full; dropping frame")

    def _worker_main(self) -> None:  # pragma: no cover - callback path
        assert webrtcvad is not None
        assert np is not None
        assert WhisperModel is not None
        vad = webrtcvad.Vad(max(0, min(3, self.config.stt_vad_aggressiveness)))
        model = WhisperModel(self.config.stt_model_size, device="auto", compute_type="int8")

        speech_chunks = bytearray()
        in_speech = False
        speech_ms = 0
        silence_ms = 0

        while not self._stop_event.is_set():
            try:
                frame = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                break
            if len(frame) < self.frame_size * 2:
                continue

            is_speech = vad.is_speech(frame, self.sample_rate)
            if is_speech:
                in_speech = True
                speech_chunks.extend(frame)
                speech_ms += self.frame_ms
                silence_ms = 0
                continue

            if in_speech:
                speech_chunks.extend(frame)
                silence_ms += self.frame_ms
                if silence_ms < self.config.stt_end_silence_ms:
                    continue

                if speech_ms >= self.config.stt_min_speech_ms:
                    text = self._transcribe_bytes(model, bytes(speech_chunks))
                    if text and self._loop is not None:
                        self._loop.call_soon_threadsafe(self._utterance_queue.put_nowait, text)
                speech_chunks = bytearray()
                in_speech = False
                speech_ms = 0
                silence_ms = 0

    def _transcribe_bytes(self, model, pcm_bytes: bytes) -> str:
        assert np is not None
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio, language=self.config.stt_language, vad_filter=False)
        text = " ".join((seg.text or "").strip() for seg in segments).strip()
        return text

    async def get_next_utterance(self, timeout_s: float = 0.25) -> str | None:
        try:
            return await asyncio.wait_for(self._utterance_queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None


class StreamerModeService:
    def __init__(
        self,
        llm: LLMClient,
        emit_event: Callable[[dict], Awaitable[None]],
        synthesize_tts: Callable[[str], Awaitable[tuple[bytes, str]]],
        config: StreamerRuntimeConfig | None = None,
    ):
        self.llm = llm
        self._emit_event = emit_event
        self._synthesize_tts = synthesize_tts
        self.config = config or StreamerRuntimeConfig.from_env()
        self._capture = OBSFrameCapture(self.config)
        self._speech = DesktopSpeechPipeline(self.config)
        self._manual_text_queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._last_capture_ts = 0.0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, *, enable_mic: bool | None = None) -> None:
        if self.running:
            return
        self._stop_event.clear()
        use_mic = self.config.stt_enabled if enable_mic is None else bool(enable_mic)
        if use_mic:
            await self._speech.start()
        self._task = asyncio.create_task(self._run_loop(), name="streamer-mode-loop")

    async def stop(self) -> None:
        self._stop_event.set()
        await self._speech.stop()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def enqueue_manual_utterance(self, text: str) -> None:
        cleaned = (text or "").strip()
        if cleaned:
            await self._manual_text_queue.put(cleaned)

    def status(self) -> dict:
        return {
            "running": self.running,
            "stt_enabled": self.config.stt_enabled,
            "stt_available": self._speech.available,
            "obs_host": self.config.obs_host,
            "obs_port": self.config.obs_port,
            "obs_source_name": self.config.obs_source_name,
            "vision_model": self.config.streamer_vision_model or self.llm.vision_model,
        }

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            utterance = await self._next_utterance()
            if not utterance:
                continue
            try:
                await self._process_utterance(utterance)
            except Exception as exc:
                _LOG.exception("[streamer] failed to process utterance: %s", exc)
                await self._emit_event({"type": "streamer_error", "message": str(exc)})

    async def _next_utterance(self) -> str | None:
        try:
            return self._manual_text_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        return await self._speech.get_next_utterance(timeout_s=0.25)

    async def _process_utterance(self, utterance: str) -> None:
        now = time.time()
        min_gap_s = max(0.0, self.config.obs_min_capture_interval_ms / 1000.0)
        if now - self._last_capture_ts < min_gap_s:
            await asyncio.sleep(min_gap_s - (now - self._last_capture_ts))

        frame_b64 = await self._capture.capture_frame_base64_png()
        self._last_capture_ts = time.time()
        await self._emit_event(
            {
                "type": "streamer_input",
                "source": "voice_segment",
                "transcript": utterance,
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        prompt = self._build_commentator_prompt(utterance)
        model_name = (self.config.streamer_vision_model or "").strip() or None
        result = await self.llm.async_generate_with_images(
            prompt=prompt,
            image_base64_list=[frame_b64],
            model=model_name,
            history=[],
        )
        comment = (result.get("completion") or "").strip()
        usage = result.get("usage") or {}

        await self._emit_event(
            {
                "type": "streamer_comment",
                "reply": comment,
                "transcript": utterance,
                "usage": usage,
            }
        )

        if self.config.tts_enabled and comment:
            audio_bytes, content_type = await self._synthesize_tts(comment)
            await self._emit_event(
                {
                    "type": "tts_audio",
                    "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                    "content_type": content_type,
                }
            )

    def _build_commentator_prompt(self, transcript: str) -> str:
        style = self.config.streamer_comment_style.strip().lower()
        if style == "hype":
            instruction = "Give an energetic 1-2 sentence commentary like a livestream co-host."
        elif style == "analyst":
            instruction = "Give a tactical 1-2 sentence commentary focused on what is happening on screen."
        else:
            instruction = "Give a short natural 1-2 sentence commentary reacting to the voice and screen."

        return (
            "You are a live stream commentator for the user.\n"
            f"{instruction}\n"
            "Acknowledge uncertainty if the image is unclear.\n"
            "Do not mention implementation details.\n"
            f"Voice transcript: {transcript}"
        )
