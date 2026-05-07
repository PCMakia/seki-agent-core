"""
LLM client abstraction: talks to Ollama in Docker (http://ollama:11434)
and can be replaced later by a vLLM client.
"""
import os
import json
from typing import List, Dict, Any, Iterator, Optional, AsyncIterator

import httpx


def _ollama_think_compat_fields(model_name: str) -> Dict[str, Any]:
    """Qwen3 chat models may fill ``thinking`` and leave ``content`` empty unless think is off."""
    if (model_name or "").lower().lstrip().startswith("qwen3"):
        return {"think": False}
    return {}


class LLMClient:
    """
    Client for the local LLM backend (Ollama in Docker).
    Base URL is read from OLLAMA_BASE_URL (default http://localhost:11434 for local runs;
    Docker Compose sets http://ollama:11434 for container-to-container).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        # Explicit ``model`` wins; else OLLAMA_MODEL (e.g. Docker Compose); else default.
        self.model = (model or os.getenv("OLLAMA_MODEL") or "qwen3:4b").strip()
        self.vision_model = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return a single response.
        Returns dict with keys: completion (str), usage (dict with prompt_tokens, completion_tokens).
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **_ollama_think_compat_fields(self.model),
        }
        if params:
            payload["options"] = params

        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot reach Ollama at {self.base_url} ({exc}). "
                    "Start the Ollama service and check OLLAMA_BASE_URL."
                ) from exc
            except httpx.TimeoutException as exc:
                raise RuntimeError(
                    f"Ollama request timed out after {self.timeout}s (model={self.model!r})."
                ) from exc
            except httpx.HTTPStatusError as exc:
                body = (exc.response.text or "")[:800]
                raise RuntimeError(
                    f"Ollama HTTP {exc.response.status_code} for model {self.model!r}. "
                    f"Pull the model (`ollama pull {self.model}`) or fix OLLAMA_MODEL. Body: {body}"
                ) from exc
            data = resp.json()

        content = (data.get("message") or {}).get("content") or ""
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return {
            "completion": content,
            "usage": {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
            },
        }

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> Iterator[str]:
        """
        Send a chat request with stream=True and yield content chunks.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **_ollama_think_compat_fields(self.model),
        }
        if params:
            payload["options"] = params

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = (exc.response.read() or b"")[:800].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Ollama HTTP {exc.response.status_code} for model {self.model!r}. {body}"
                    ) from exc
                for line in response.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("error"):
                        raise RuntimeError(part["error"])
                    msg = (part.get("message") or {}).get("content") or ""
                    if msg:
                        yield msg
                    if part.get("done"):
                        break

    async def async_stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> AsyncIterator[str]:
        """
        Async variant of stream_generate suitable for FastAPI websocket handlers.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **_ollama_think_compat_fields(self.model),
        }
        if params:
            payload["options"] = params

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = (await exc.response.aread() or b"")[:800].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Ollama HTTP {exc.response.status_code} for model {self.model!r}. {body}"
                    ) from exc
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("error"):
                        raise RuntimeError(part["error"])
                    msg = (part.get("message") or {}).get("content") or ""
                    if msg:
                        yield msg
                    if part.get("done"):
                        break

    def generate_with_images(
        self,
        prompt: str,
        image_base64_list: List[str],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Send a multimodal chat request (text + base64 images) to Ollama and return one response.
        """
        content_parts: List[Dict[str, str]] = [{"type": "text", "text": prompt}]
        for image_b64 in image_base64_list:
            if image_b64:
                content_parts.append({"type": "image", "image": image_b64})

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content_parts})

        use_model = model or self.vision_model
        payload: Dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            **_ollama_think_compat_fields(use_model),
        }
        if params:
            payload["options"] = params

        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot reach Ollama at {self.base_url} ({exc})."
                ) from exc
            except httpx.HTTPStatusError as exc:
                body = (exc.response.text or "")[:800]
                raise RuntimeError(
                    f"Ollama HTTP {exc.response.status_code} for model {use_model!r}. {body}"
                ) from exc
            data = resp.json()

        content = (data.get("message") or {}).get("content") or ""
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return {
            "completion": content,
            "usage": {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
            },
        }

    async def async_generate_with_images(
        self,
        prompt: str,
        image_base64_list: List[str],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Async multimodal variant for streamer mode.
        """
        content_parts: List[Dict[str, str]] = [{"type": "text", "text": prompt}]
        for image_b64 in image_base64_list:
            if image_b64:
                content_parts.append({"type": "image", "image": image_b64})

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content_parts})

        use_model = model or self.vision_model
        payload: Dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            **_ollama_think_compat_fields(use_model),
        }
        if params:
            payload["options"] = params

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot reach Ollama at {self.base_url} ({exc})."
                ) from exc
            except httpx.HTTPStatusError as exc:
                body = (exc.response.text or "")[:800]
                raise RuntimeError(
                    f"Ollama HTTP {exc.response.status_code} for model {use_model!r}. {body}"
                ) from exc
            data = resp.json()

        content = (data.get("message") or {}).get("content") or ""
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        return {
            "completion": content,
            "usage": {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
            },
        }