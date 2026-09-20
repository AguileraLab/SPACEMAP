import json
import stat

import pytest

from spacemap_assistant import model_client
from spacemap_assistant.model_client import (
    DEFAULT_NVIDIA_MODEL_ID,
    LocalModelClient,
    ModelUnavailable,
    NvidiaModelClient,
    ProviderSettings,
    completion_budget,
)


class _Response:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return b"".join(self._lines)


def test_model_client_rejects_non_loopback_connections() -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        LocalModelClient(base_url="https://example.com/v1")


def test_model_client_reads_stream_chunks(monkeypatch) -> None:
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    lines = [("data: " + json.dumps(item) + "\n\n").encode() for item in chunks]
    lines.append(b"data: [DONE]\n\n")
    client = LocalModelClient()
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: _Response(lines))
    assert "".join(client.stream_complete([{"role": "user", "content": "Hi"}])) == "Hello world"


def test_model_client_reports_stream_errors(monkeypatch) -> None:
    client = LocalModelClient()
    lines = [("data: " + json.dumps({"error": "GPU failed"}) + "\n\n").encode()]
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: _Response(lines))
    with pytest.raises(ModelUnavailable, match="GPU failed"):
        list(client.stream_complete([{"role": "user", "content": "Hi"}]))


def test_action_plans_receive_automatic_token_exceptions() -> None:
    assert completion_budget("Explain this tab", 500) == (500, False, "normal_chat")
    assert completion_budget("Run ROI extraction", 500) == (4096, True, "action_plan")
    assert completion_budget("Optimize Leiden parameters", 500) == (
        8192,
        True,
        "complex_action_plan",
    )
    assert completion_budget("Register these CODEX and H&E images", 500) == (
        4096,
        True,
        "action_plan",
    )
    assert completion_budget("Correct these batches and save the output", 500) == (
        4096,
        True,
        "action_plan",
    )
    assert completion_budget("What is segmentation?", 500) == (
        500,
        False,
        "normal_chat",
    )
    assert completion_budget("Explain how to run segmentation", 500) == (
        500,
        False,
        "normal_chat",
    )
    assert completion_budget("Register the images, then segment and phenotype them", 500) == (
        8192,
        True,
        "complex_action_plan",
    )


def test_provider_settings_fall_back_to_safe_defaults() -> None:
    settings = ProviderSettings(provider="unknown", nvidia_model_id="unknown/model").normalized()
    assert settings.provider == "local"
    assert settings.nvidia_model_id == DEFAULT_NVIDIA_MODEL_ID


def test_muse_glimmer_is_available_for_selection() -> None:
    metadata = model_client.NVIDIA_MODEL_CATALOG["meta/muse-glimmer-30b"]
    assert metadata["supports_images"] is True
    assert metadata["max_output_tokens"] == 8192


def test_nvidia_client_rejects_unapproved_model_and_endpoint() -> None:
    with pytest.raises(ValueError, match="approved model list"):
        NvidiaModelClient(model_id="unapproved/model")
    with pytest.raises(ValueError, match="must stay"):
        NvidiaModelClient(base_url="https://example.com/v1")


def test_nvidia_client_sends_selected_model_and_reads_answer(monkeypatch) -> None:
    captured = {}
    response = _Response(
        [json.dumps({"choices": [{"message": {"content": "Ready"}}]}).encode()]
    )
    client = NvidiaModelClient(model_id=DEFAULT_NVIDIA_MODEL_ID)

    def fake_request(*, payload):
        captured.update(payload)
        return response

    monkeypatch.setattr(client, "_request", fake_request)
    assert client.complete([{"role": "user", "content": "Hi"}]) == "Ready"
    assert captured["model"] == DEFAULT_NVIDIA_MODEL_ID
    assert captured["stream"] is False
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("client_class", [LocalModelClient, NvidiaModelClient])
def test_non_streaming_null_content_is_treated_as_empty(monkeypatch, client_class) -> None:
    """A provider JSON null must never become the visible word ``None``."""
    response = _Response(
        [json.dumps({"choices": [{"message": {"content": None}}]}).encode()]
    )
    client = client_class()
    if isinstance(client, LocalModelClient):
        monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    else:
        monkeypatch.setattr(client, "_request", lambda **kwargs: response)
    assert client.complete([{"role": "user", "content": "Finished?"}]) == ""


def test_muse_payload_uses_official_plain_openai_fields() -> None:
    client = NvidiaModelClient(model_id="meta/muse-glimmer-30b")
    payload = client._payload(
        [{"role": "user", "content": "Hi"}],
        temperature=1.0,
        max_tokens=8192,
        stream=False,
    )
    assert payload["model"] == "meta/muse-glimmer-30b"
    assert payload["max_tokens"] == 8192
    assert "chat_template_kwargs" not in payload
    assert "reasoning_budget" not in payload


def test_muse_gets_enough_generation_space_to_write_visible_text() -> None:
    client = NvidiaModelClient(model_id="meta/muse-glimmer-30b")
    payload = client._payload(
        [{"role": "user", "content": "Hi"}],
        temperature=0.0,
        max_tokens=512,
        stream=False,
        enable_thinking=False,
    )
    assert payload["max_tokens"] == 1024


def test_connection_probe_disables_gemma_thinking(monkeypatch) -> None:
    captured = {}

    def fake_stream_complete(self, messages, **kwargs):
        captured.update(kwargs)
        yield "SPACEMAP NVIDIA connection is ready"

    monkeypatch.setattr(NvidiaModelClient, "stream_complete", fake_stream_complete)
    answer = model_client.test_nvidia_connection(DEFAULT_NVIDIA_MODEL_ID)
    assert answer == "SPACEMAP NVIDIA connection is ready"
    assert captured["enable_thinking"] is False
    assert captured["max_tokens"] == 512


def test_nvidia_stream_hides_reasoning_and_returns_answer(monkeypatch) -> None:
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
        {"choices": [{"delta": {"content": "Public answer"}}]},
    ]
    lines = [("data: " + json.dumps(item) + "\n\n").encode() for item in chunks]
    lines.append(b"data: [DONE]\n\n")
    client = NvidiaModelClient()
    monkeypatch.setattr(client, "_request", lambda **kwargs: _Response(lines))
    assert "".join(client.stream_complete([{"role": "user", "content": "Hi"}])) == "Public answer"


def test_nvidia_key_is_saved_owner_only_and_can_be_removed(tmp_path, monkeypatch) -> None:
    key_path = tmp_path / "nvidia_api.key"
    monkeypatch.setattr(model_client, "NVIDIA_API_KEY_PATH", key_path)
    monkeypatch.setattr(model_client, "require_assistant_write_path", lambda path: path)

    model_client.save_nvidia_api_key("nvapi-test-key-that-is-long-enough")
    assert model_client.nvidia_api_key_saved()
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert model_client.delete_nvidia_api_key()
    assert not key_path.exists()
