"""Бэкенды VLM за единым интерфейсом.

Архитектура полностью повторяет проверенный бэкенд guide_robot_llm.lib.backends:
1. Базовый класс VllmBackend (он же OllamaVlmBackend / LlamaCppVlmBackend):
   работает со стандартным OpenAI-совместимым эндпоинтом /v1/chat/completions,
   который из коробки поддерживают vLLM, Ollama (/v1) и llama-server (llama.cpp).
   Поддерживает:
   - stream(): потоковая генерация текста через SSE для голосового TTS (с опциональным кадром камеры);
   - predict_action(): структурированный вызов со схемой JSON для принятия решений (Tool Calling);
   - мультимодальность: передача кадров через стандартный формат image_url (Data URL Base64);
   - защиту от искажения кириллицы при декодировании SSE-потока.
2. OpenAIVlmBackend(VllmBackend):
   наследует базовый класс и добавляет Bearer-авторизацию по токену из окружения.
3. MockVlmBackend / EchoBackend:
   детерминированный мок для тестов и CI без GPU и сети.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from guide_robot_vlm.lib.image_preprocessor import PreprocessedImage

_logger = logging.getLogger(__name__)

__all__ = [
    "Chunk",
    "EchoBackend",
    "LlamaCppVlmBackend",
    "MockVlmBackend",
    "OllamaVlmBackend",
    "OpenAIVlmBackend",
    "VllmBackend",
    "VlmBackend",
    "VlmResponse",
]

_CONNECT_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Chunk:
    """Один кусок потока генерации текста (для передачи в SentenceSplitter -> TTS)."""

    text: str
    done: bool


@dataclass(frozen=True)
class VlmResponse:
    """Сырой результат структурированного вызова VLM для валидатора."""

    raw_text: str
    latency_ms: float
    model_name: str


class VlmBackend(Protocol):
    """Общий интерфейс мультимодального бэкенда VLM."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        abort: threading.Event,
        image: PreprocessedImage | None = None,
    ) -> Iterator[Chunk]:
        """Потоковая генерация речевого ответа (SSE). Обязан проверять abort."""

    def predict_action(
        self,
        messages: list[dict[str, Any]],
        abort: threading.Event,
        schema: dict[str, Any] | None = None,
        image: PreprocessedImage | None = None,
    ) -> VlmResponse:
        """Разовый структурированный вызов VLM с JSON-схемой."""

    def health(self) -> tuple[bool, str]:
        """Проверить доступность эндпоинта инференса. Возвращает (ok, detail)."""


class VllmBackend:
    """HTTP-клиент OpenAI-совместимого эндпоинта /v1/chat/completions.

    Единый базовый класс для всех локальных серверов инференса:
    vLLM, Ollama (через /v1) и llama-server (llama.cpp).
    """

    def __init__(
        self,
        base_url: str,
        model: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
        request_timeout_s: float = 20.0,
    ) -> None:
        """Инициализировать параметры соединения."""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._stream = stream
        self._request_timeout_s = request_timeout_s

    def _endpoint(self, path: str) -> str:
        """Сформировать корректный URL с учетом возможного префикса /v1 в base_url."""
        base = self._base_url
        clean_path = path.lstrip("/")
        if base.endswith("/v1") and clean_path.startswith("v1/"):
            clean_path = clean_path[len("v1/") :]
        return f"{base}/{clean_path}"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _format_messages_with_image(
        self, messages: list[dict[str, Any]], image: PreprocessedImage | None
    ) -> list[dict[str, Any]]:
        """Встроить кадр камеры (Data URL) в последнее сообщение пользователя."""
        if image is None or not messages:
            return messages

        formatted: list[dict[str, Any]] = [dict(m) for m in messages]
        for i in reversed(range(len(formatted))):
            if formatted[i].get("role") == "user":
                content = formatted[i].get("content", "")
                parts: list[dict[str, Any]] = []
                if isinstance(content, str):
                    parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    parts.extend(content)
                parts.append({"type": "image_url", "image_url": {"url": image.data_url}})
                formatted[i]["content"] = parts
                break
        return formatted

    def _payload(
        self,
        messages: list[dict[str, Any]],
        image: PreprocessedImage | None = None,
        stream: bool | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Собрать JSON-полезную нагрузку для /v1/chat/completions."""
        use_stream = self._stream if stream is None else stream
        formatted_messages = self._format_messages_with_image(messages, image)

        payload: dict[str, Any] = {
            "messages": formatted_messages,
            "stream": use_stream,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._model:
            payload["model"] = self._model

        # Включение режима JSON при вызове действий
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def stream(
        self,
        messages: list[dict[str, Any]],
        abort: threading.Event,
        image: PreprocessedImage | None = None,
    ) -> Iterator[Chunk]:
        """Потоковая генерация текста для TTS (разбор SSE)."""
        response = requests.post(
            self._endpoint("v1/chat/completions"),
            json=self._payload(messages, image=image, stream=True),
            headers=self._headers(),
            stream=True,
            timeout=(_CONNECT_TIMEOUT_S, self._request_timeout_s),
        )
        try:
            response.raise_for_status()
            yield from self._iter_sse(response, abort)
        finally:
            response.close()

    def predict_action(
        self,
        messages: list[dict[str, Any]],
        abort: threading.Event,
        schema: dict[str, Any] | None = None,
        image: PreprocessedImage | None = None,
    ) -> VlmResponse:
        """Разовый вызов VLM со структурированным ответом."""
        if abort.is_set():
            return VlmResponse(raw_text="", latency_ms=0.0, model_name=self._model)

        start_time = time.monotonic()
        response = requests.post(
            self._endpoint("v1/chat/completions"),
            json=self._payload(messages, image=image, stream=False, schema=schema),
            headers=self._headers(),
            timeout=(_CONNECT_TIMEOUT_S, self._request_timeout_s),
        )
        latency_ms = (time.monotonic() - start_time) * 1000.0

        try:
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            raw_text = ""
            if choices:
                raw_text = (choices[0].get("message") or {}).get("content") or ""
            return VlmResponse(
                raw_text=raw_text,
                latency_ms=latency_ms,
                model_name=self._model,
            )
        finally:
            response.close()

    def _iter_sse(self, response: requests.Response, abort: threading.Event) -> Iterator[Chunk]:
        """Построчный разбор Server-Sent Events без потери UTF-8 кириллицы."""
        for raw_bytes in response.iter_lines(decode_unicode=False):
            if abort.is_set():
                return
            if not raw_bytes:
                continue
            raw_line = raw_bytes.decode("utf-8", errors="replace")
            if not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                _logger.warning("не удалось разобрать SSE-строку %r", raw_line)
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            text = (choices[0].get("delta") or {}).get("content") or ""
            if text:
                yield Chunk(text=text, done=False)
        yield Chunk(text="", done=True)

    def health(self) -> tuple[bool, str]:
        """Проверить доступность сервера: GET /health или GET /v1/models."""
        ok, detail = self._get_ok(self._endpoint("health"))
        if ok:
            return True, ""
        ok2, detail2 = self._get_ok(self._endpoint("v1/models"))
        if ok2:
            return True, ""
        return False, f"{detail}; {detail2}"

    def _get_ok(self, url: str) -> tuple[bool, str]:
        try:
            response = requests.get(url, headers=self._headers(), timeout=_CONNECT_TIMEOUT_S)
        except requests.RequestException as error:
            return False, str(error)
        if response.ok:
            return True, ""
        return False, f"HTTP {response.status_code}"


# Синонимы базового класса для совместимости с конфигурациями Ollama и llama.cpp
OllamaVlmBackend = VllmBackend
LlamaCppVlmBackend = VllmBackend


class OpenAIVlmBackend(VllmBackend):
    """Тот же протокол, что и VllmBackend, плюс Bearer-авторизация и обязательный model."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "VLM_API_KEY",
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
        request_timeout_s: float = 20.0,
    ) -> None:
        """Инициализировать клиент с чтением ключа из окружения."""
        if not model:
            raise ValueError("OpenAIVlmBackend требует непустой параметр model")
        super().__init__(
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            request_timeout_s=request_timeout_s,
        )
        self._api_key = os.environ.get(api_key_env, "")
        if not self._api_key:
            _logger.warning(
                "переменная окружения %s не задана -- запросы уйдут без авторизации",
                api_key_env,
            )

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


class MockVlmBackend:
    """Детерминированный мок для тестов и CI без GPU и сети (аналог EchoBackend)."""

    def __init__(
        self,
        reply_text: str = "Здравствуйте! Чем могу помочь?",
        action_json: str = '{"tool": "idle", "args": {}, "confidence": 1.0, "abstain": false}',
        word_delay_s: float = 0.0,
    ) -> None:
        """Задать предопределенный текстовый ответ и JSON действия."""
        self._reply_text = reply_text
        self._action_json = action_json
        self._word_delay_s = word_delay_s

    def stream(
        self,
        messages: list[dict[str, Any]],
        abort: threading.Event,
        image: PreprocessedImage | None = None,
    ) -> Iterator[Chunk]:
        """Отдать reply_text по словам с поддержкой abort."""
        del messages, image
        words = self._reply_text.split(" ")
        for index, word in enumerate(words):
            if abort.is_set():
                return
            text = word if index == 0 else " " + word
            yield Chunk(text=text, done=False)
            if self._word_delay_s > 0:
                time.sleep(self._word_delay_s)
        yield Chunk(text="", done=True)

    def predict_action(
        self,
        messages: list[dict[str, Any]],
        abort: threading.Event,
        schema: dict[str, Any] | None = None,
        image: PreprocessedImage | None = None,
    ) -> VlmResponse:
        """Вернуть мок-решение в зависимости от входного текста."""
        del schema, image
        if abort.is_set():
            return VlmResponse(raw_text="", latency_ms=0.0, model_name="mock")

        # Если в запросе есть ключевые слова прерывания / аудитории -> возвращаем interrupt
        user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_content = str(m.get("content", "")).lower()
                break

        if any(w in user_content for w in ("стоп", "стой", "прерви", "аудитор", "люд", "внимание")):
            raw = '{"tool": "interrupt", "args": {"reason": "no_audience", "people_count": 0, "looking_at_robot": false}, "confidence": 0.95, "abstain": false}'
        else:
            raw = self._action_json

        return VlmResponse(raw_text=raw, latency_ms=1.0, model_name="mock")

    def health(self) -> tuple[bool, str]:
        return True, "ok"


# Синоним для совместимости
EchoBackend = MockVlmBackend
