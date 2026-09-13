"""Юнит-тесты на бэкенды VLM: MockVlmBackend, VllmBackend, OpenAIVlmBackend, OllamaVlmBackend.

Тесты не требуют запущенных серверов и сокетов: requests мокаются фикстурами.
"""

from __future__ import annotations

import threading
import pytest
import requests

from guide_robot_vlm.lib.image_preprocessor import PreprocessedImage
from guide_robot_vlm.lib.vlm_backends import (
    Chunk,
    EchoBackend,
    LlamaCppVlmBackend,
    MockVlmBackend,
    OllamaVlmBackend,
    OpenAIVlmBackend,
    VllmBackend,
    VlmResponse,
)


# -- Вспомогательные фикстуры и моки ------------------------------------------


class FakeResponse:
    """Минимальная замена requests.Response для изоляции от сети."""

    def __init__(
        self,
        lines: list[bytes | str] | None = None,
        json_payload: dict | None = None,
        status_code: int = 200,
    ) -> None:
        self._lines = lines or []
        self._json_payload = json_payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode: bool = False) -> iter:
        for line in self._lines:
            if isinstance(line, str):
                yield line.encode("utf-8") if not decode_unicode else line
            else:
                yield line.decode("utf-8") if decode_unicode else line

    def json(self) -> dict:
        return self._json_payload or {}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_http(monkeypatch):
    calls: dict[str, object] = {}

    def make(post_response: FakeResponse | None = None, get_response: FakeResponse | None = None):
        def fake_post(url, json=None, headers=None, stream=None, timeout=None):  # noqa: ANN001
            calls["post_url"] = url
            calls["post_json"] = json
            calls["post_headers"] = headers
            calls["post_stream"] = stream
            calls["post_timeout"] = timeout
            return post_response or FakeResponse()

        def fake_get(url, headers=None, timeout=None):  # noqa: ANN001
            calls["get_url"] = url
            calls["get_headers"] = headers
            calls["get_timeout"] = timeout
            return get_response or FakeResponse()

        monkeypatch.setattr("guide_robot_vlm.lib.vlm_backends.requests.post", fake_post)
        monkeypatch.setattr("guide_robot_vlm.lib.vlm_backends.requests.get", fake_get)
        return calls

    return make


@pytest.fixture
def sample_image() -> PreprocessedImage:
    return PreprocessedImage(
        data_url="data:image/jpeg;base64,AAAA",
        width=320,
        height=240,
        capture_timestamp=100.0,
        size_bytes=3,
    )


# -- MockVlmBackend / EchoBackend --------------------------------------------


def test_mock_backend_streams_reply() -> None:
    backend = MockVlmBackend(reply_text="Экспонат номер один", word_delay_s=0.0)
    chunks = list(backend.stream([], threading.Event()))
    text = "".join(c.text for c in chunks)
    assert text == "Экспонат номер один"
    assert chunks[-1].done is True
    assert all(not c.done for c in chunks[:-1])


def test_mock_backend_stops_on_abort() -> None:
    backend = MockVlmBackend(reply_text="раз два три четыре пять", word_delay_s=0.0)
    abort = threading.Event()
    received: list[Chunk] = []
    for chunk in backend.stream([], abort):
        received.append(chunk)
        if len(received) == 2:
            abort.set()
    assert not any(c.done for c in received)
    assert len(received) < 5


def test_mock_backend_health() -> None:
    backend = MockVlmBackend()
    ok, detail = backend.health()
    assert ok is True
    assert detail == "ok"


def test_mock_backend_predict_action_default() -> None:
    backend = MockVlmBackend()
    resp = backend.predict_action([{"role": "user", "content": "что интересного?"}], threading.Event())
    assert resp.model_name == "mock"
    assert '"tool": "idle"' in resp.raw_text


def test_mock_backend_predict_action_interrupt_keywords() -> None:
    backend = MockVlmBackend()
    resp = backend.predict_action([{"role": "user", "content": "Стой, подожди"}], threading.Event())
    assert '"tool": "interrupt"' in resp.raw_text
    assert '"reason": "no_audience"' in resp.raw_text


def test_mock_backend_predict_action_aborted() -> None:
    backend = MockVlmBackend()
    abort = threading.Event()
    abort.set()
    resp = backend.predict_action([], abort)
    assert resp.raw_text == ""


# -- VllmBackend / OpenAI-Compatible (Ollama, vLLM, llama.cpp) ---------------


def test_vllm_backend_formats_image_in_payload(fake_http, sample_image) -> None:
    calls = fake_http(FakeResponse(lines=[b"data: [DONE]"]))
    backend = VllmBackend(base_url="http://127.0.0.1:8000", model="qwen-vl")

    messages = [{"role": "user", "content": "Что на картинке?"}]
    list(backend.stream(messages, threading.Event(), image=sample_image))

    post_json = calls["post_json"]
    sent_messages = post_json["messages"]
    assert len(sent_messages) == 1
    content = sent_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Что на картинке?"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}


def test_vllm_backend_endpoint_url_handling(fake_http) -> None:
    # Без /v1 в base_url
    b1 = VllmBackend(base_url="http://127.0.0.1:11434")
    assert b1._endpoint("v1/chat/completions") == "http://127.0.0.1:11434/v1/chat/completions"

    # С /v1 в base_url (не должно удваиваться)
    b2 = VllmBackend(base_url="http://127.0.0.1:11434/v1")
    assert b2._endpoint("v1/chat/completions") == "http://127.0.0.1:11434/v1/chat/completions"


def test_vllm_backend_parses_sse_stream(fake_http) -> None:
    lines = [
        b'data: {"choices":[{"delta":{"content":"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"}}]}',  # "Привет"
        b"",  # keep-alive пустая строка
        b'data: {"choices":[{"delta":{"content":" \xd0\xbc\xd0\xb8\xd1\x80"}}]}',  # " мир"
        b"data: [DONE]",
    ]
    fake_http(FakeResponse(lines=lines))
    backend = VllmBackend(base_url="http://127.0.0.1:8000")

    chunks = list(backend.stream([], threading.Event()))
    assert [c.text for c in chunks] == ["Привет", " мир", ""]
    assert chunks[-1].done is True


def test_vllm_backend_predict_action(fake_http, sample_image) -> None:
    json_payload = {
        "choices": [
            {
                "message": {
                    "content": '{"tool": "interrupt", "args": {"reason": "no_audience"}}'
                }
            }
        ]
    }
    calls = fake_http(FakeResponse(json_payload=json_payload))
    backend = VllmBackend(base_url="http://127.0.0.1:8000", model="vlm-model")

    schema = {"type": "object", "properties": {"tool": {"type": "string"}}}
    resp = backend.predict_action(
        [{"role": "user", "content": "проверь людей"}],
        threading.Event(),
        schema=schema,
        image=sample_image,
    )

    assert resp.raw_text == '{"tool": "interrupt", "args": {"reason": "no_audience"}}'
    assert resp.model_name == "vlm-model"
    assert resp.latency_ms >= 0.0
    assert calls["post_json"]["response_format"] == {"type": "json_object"}
    assert calls["post_json"]["stream"] is False


def test_vllm_backend_predict_action_aborted(fake_http) -> None:
    backend = VllmBackend(base_url="http://127.0.0.1:8000")
    abort = threading.Event()
    abort.set()
    resp = backend.predict_action([], abort)
    assert resp.raw_text == ""


def test_vllm_backend_health_check(fake_http) -> None:
    fake_http(get_response=FakeResponse(status_code=200))
    backend = VllmBackend(base_url="http://127.0.0.1:8000")
    ok, detail = backend.health()
    assert ok is True
    assert detail == ""


def test_vllm_backend_health_check_failure(fake_http) -> None:
    fake_http(get_response=FakeResponse(status_code=503))
    backend = VllmBackend(base_url="http://127.0.0.1:8000")
    ok, detail = backend.health()
    assert ok is False
    assert "HTTP 503" in detail


# -- OpenAIVlmBackend --------------------------------------------------------


def test_openai_backend_sets_auth_header(monkeypatch, fake_http) -> None:
    monkeypatch.setenv("TEST_VLM_KEY", "secret_token_123")
    calls = fake_http(FakeResponse(lines=[b"data: [DONE]"]))
    backend = OpenAIVlmBackend(
        base_url="https://api.openai.com",
        model="gpt-4o",
        api_key_env="TEST_VLM_KEY",
    )
    list(backend.stream([], threading.Event()))
    assert calls["post_headers"]["Authorization"] == "Bearer secret_token_123"


def test_openai_backend_requires_model() -> None:
    with pytest.raises(ValueError, match="модель|model"):
        OpenAIVlmBackend(base_url="https://api.openai.com", model="")


# -- Проверка псевдонимов классов ---------------------------------------------


def test_backend_aliases() -> None:
    assert OllamaVlmBackend is VllmBackend
    assert LlamaCppVlmBackend is VllmBackend
    assert EchoBackend is MockVlmBackend
