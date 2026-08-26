"""
LLM client: OpenAI-compatible HTTP transport to seki-inference-engine.

Call sites keep the same generate / stream / vision methods. The gateway
owns vLLM primary and Ollama failover; this module does not talk to Ollama.
"""
from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAI


def _env_inference_url() -> str:
    return (
        os.getenv("INFERENCE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:8000/v1"
    ).strip()


def _env_inference_key() -> str:
    return (os.getenv("INFERENCE_API_KEY") or os.getenv("API_KEY") or "").strip()


def _env_model() -> str:
    return (
        os.getenv("INFERENCE_MODEL")
        or os.getenv("MODEL_NAME")
        or os.getenv("OLLAMA_MODEL")
        or "Qwen/Qwen2.5-3B-Instruct"
    ).strip()


def normalize_openai_base_url(url: str) -> str:
    """OpenAI SDK expects a base URL that already includes `/v1`."""
    base = (url or "").strip().rstrip("/")
    if not base:
        base = "http://localhost:8000/v1"
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def gateway_root_url(openai_base_url: str) -> str:
    """`http://host:8000/v1` → `http://host:8000` for `/health` and `/ready`."""
    base = (openai_base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return base[:-3].rstrip("/") or "http://localhost:8000"
    return base


def _openai_generation_kwargs(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    out: Dict[str, Any] = {}
    if params.get("temperature") is not None:
        out["temperature"] = params["temperature"]
    if params.get("top_p") is not None:
        out["top_p"] = params["top_p"]
    if params.get("max_tokens") is not None:
        out["max_tokens"] = params["max_tokens"]
    elif params.get("num_predict") is not None:
        out["max_tokens"] = params["num_predict"]
    if params.get("stop") is not None:
        out["stop"] = params["stop"]
    return out


def _messages(
    prompt: str,
    system_prompt: Optional[str],
    history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _usage_from_completion(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _content_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return ""


def _raise_inference_error(exc: Exception, *, model: str, base_url: str) -> None:
    if isinstance(exc, APIConnectionError):
        raise RuntimeError(
            f"Cannot reach inference gateway at {base_url} ({exc}). "
            "Start seki-inference-engine and check INFERENCE_URL."
        ) from exc
    if isinstance(exc, APITimeoutError):
        raise RuntimeError(
            f"Inference request timed out for model {model!r} at {base_url}."
        ) from exc
    if isinstance(exc, APIStatusError):
        body = ""
        try:
            body = (exc.response.text or "")[:800]
        except Exception:
            body = str(exc)[:800]
        raise RuntimeError(
            f"Inference HTTP {exc.status_code} for model {model!r}. "
            f"Check INFERENCE_API_KEY and MODEL_NAME. Body: {body}"
        ) from exc
    raise RuntimeError(f"Inference error for model {model!r}: {exc}") from exc


def _vision_user_content(prompt: str, image_base64_list: List[str]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_b64 in image_base64_list:
        if not image_b64:
            continue
        raw = image_b64.strip()
        url = raw if raw.startswith("data:") else f"data:image/jpeg;base64,{raw}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


class LLMClient:
    """
    Client for seki-inference-engine (OpenAI-compatible).

    Base URL: INFERENCE_URL (default http://localhost:8000/v1).
    API key: INFERENCE_API_KEY.
    Model: INFERENCE_MODEL or MODEL_NAME.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
    ):
        self.base_url = normalize_openai_base_url(base_url or _env_inference_url())
        self.api_key = (api_key if api_key is not None else _env_inference_key()) or ""
        self.model = (model or _env_model()).strip()
        self.vision_model = (
            os.getenv("INFERENCE_VISION_MODEL")
            or os.getenv("OLLAMA_VISION_MODEL")
            or self.model
        ).strip()
        self.timeout = timeout
        token = self.api_key or "EMPTY"
        self._sync = OpenAI(base_url=self.base_url, api_key=token, timeout=timeout)
        self._async = AsyncOpenAI(base_url=self.base_url, api_key=token, timeout=timeout)

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
        if stream:
            text = "".join(
                self.stream_generate(
                    prompt,
                    system_prompt=system_prompt,
                    history=history,
                    **params,
                )
            )
            return {
                "completion": text,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        kwargs = _openai_generation_kwargs(params)
        try:
            response = self._sync.chat.completions.create(
                model=self.model,
                messages=_messages(prompt, system_prompt, history),
                **kwargs,
            )
        except Exception as exc:
            _raise_inference_error(exc, model=self.model, base_url=self.base_url)
            raise

        choice = response.choices[0] if response.choices else None
        content = _content_text(choice.message if choice else None)
        if not content.strip():
            raise RuntimeError(f"Empty completion from inference model {self.model!r}")
        return {"completion": content, "usage": _usage_from_completion(response.usage)}

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> Iterator[str]:
        kwargs = _openai_generation_kwargs(params)
        try:
            stream = self._sync.chat.completions.create(
                model=self.model,
                messages=_messages(prompt, system_prompt, history),
                stream=True,
                **kwargs,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    yield text
        except Exception as exc:
            _raise_inference_error(exc, model=self.model, base_url=self.base_url)

    async def async_stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> AsyncIterator[str]:
        kwargs = _openai_generation_kwargs(params)
        try:
            stream = await self._async.chat.completions.create(
                model=self.model,
                messages=_messages(prompt, system_prompt, history),
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if text:
                    yield text
        except Exception as exc:
            _raise_inference_error(exc, model=self.model, base_url=self.base_url)
            return

    def generate_with_images(
        self,
        prompt: str,
        image_base64_list: List[str],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        use_model = model or self.vision_model
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append(
            {"role": "user", "content": _vision_user_content(prompt, image_base64_list)}
        )
        kwargs = _openai_generation_kwargs(params)
        try:
            response = self._sync.chat.completions.create(
                model=use_model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            _raise_inference_error(exc, model=use_model, base_url=self.base_url)
            raise
        choice = response.choices[0] if response.choices else None
        content = _content_text(choice.message if choice else None)
        return {"completion": content, "usage": _usage_from_completion(response.usage)}

    async def async_generate_with_images(
        self,
        prompt: str,
        image_base64_list: List[str],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        use_model = model or self.vision_model
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append(
            {"role": "user", "content": _vision_user_content(prompt, image_base64_list)}
        )
        kwargs = _openai_generation_kwargs(params)
        try:
            response = await self._async.chat.completions.create(
                model=use_model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            _raise_inference_error(exc, model=use_model, base_url=self.base_url)
            raise
        choice = response.choices[0] if response.choices else None
        content = _content_text(choice.message if choice else None)
        return {"completion": content, "usage": _usage_from_completion(response.usage)}
