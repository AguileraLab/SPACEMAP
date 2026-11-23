import base64

import pytest
from PIL import Image

# The model service is part of the optional AI assistant. Skip the whole module
# when the AI extras are not installed, rather than failing a default install.
pytest.importorskip("torch", reason="needs the 'ai' extra")
pytest.importorskip("transformers", reason="needs the 'ai' extra")
pytest.importorskip("fastapi", reason="needs the 'ai' extra")

from spacemap_assistant.model_service.server import (  # noqa: E402
    Message,
    _cast_layer_norm_input,
    _prepare_messages,
)


def _tiny_png_data_url() -> str:
    import io

    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def test_model_service_accepts_uploaded_data_images() -> None:
    messages = [
        Message(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": _tiny_png_data_url()}},
                {"type": "text", "text": "Check this preview."},
            ],
        )
    ]
    prepared = _prepare_messages(messages)
    assert prepared[0]["content"][0]["type"] == "image"
    assert prepared[0]["content"][0]["url"].size == (2, 2)


def test_model_service_blocks_remote_images() -> None:
    messages = [
        Message(
            role="user",
            content=[
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        )
    ]
    with pytest.raises(ValueError, match="remote URLs are blocked"):
        _prepare_messages(messages)


def test_vision_layer_norm_input_uses_the_layer_weight_dtype() -> None:
    import torch

    layer = torch.nn.LayerNorm(4, dtype=torch.float32)
    value = torch.ones((2, 4), dtype=torch.bfloat16)
    converted = _cast_layer_norm_input(layer, (value,))[0]
    assert converted.dtype == torch.float32


def _mock_stream(monkeypatch, generate):
    import queue
    from types import SimpleNamespace
    from spacemap_assistant.model_service import server
    class Streamer:
        def __init__(self, *a, **k):
            self.queue = queue.Queue()
        def __next__(self):
            value = self.queue.get(timeout=.01)
            if value is None:
                raise StopIteration
            return value
        def end(self):
            self.queue.put(None)
    monkeypatch.setattr(server, 'TextIteratorStreamer', Streamer)
    monkeypatch.setattr(server, '_input_handler', lambda: object())
    monkeypatch.setattr(server, '_generation_inputs', lambda request: ({}, 0))
    monkeypatch.setattr(server, '_generation_kwargs', lambda request, inputs: {})
    monkeypatch.setattr(server.STATE, 'model', SimpleNamespace(generate=generate))
    monkeypatch.setattr(server, 'STREAM_IDLE_SECONDS', .06)
    monkeypatch.setattr(server, 'STREAM_POLL_SECONDS', .01)
    return server


def test_stream_generation_failure_is_reported_without_queue_timeout(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError('synthetic failure')
    server = _mock_stream(monkeypatch, fail)
    answer = ''.join(server._stream(None, 'test', 1))
    assert 'synthetic failure' in answer and '"finish_reason": "stop"' not in answer
    assert answer.endswith('data: [DONE]\n\n')


def test_timed_out_generation_keeps_lock_until_it_really_exits(monkeypatch):
    import threading
    import time
    import torch
    exits = threading.Event()
    stop_requested = threading.Event()
    worker_done = threading.Event()
    def wait(**kwargs):
        try:
            assert torch.is_inference_mode_enabled()
            stop = kwargs['stopping_criteria'][-1]
            while not stop(None, None):
                time.sleep(.005)
            stop_requested.set()
            exits.wait(3)  # Simulate a GPU operation still winding down.
        finally:
            worker_done.set()
    server = _mock_stream(monkeypatch, wait)
    try:
        answer = ''.join(server._stream(None, 'test', 1))
        assert 'no new tokens' in answer and '"finish_reason": "stop"' not in answer
        assert stop_requested.wait(1)
        assert server.GENERATE_LOCK.locked()
    finally:
        exits.set()
        assert worker_done.wait(2)
        for _ in range(100):
            if not server.GENERATE_LOCK.locked():
                break
            time.sleep(.01)
        assert not server.GENERATE_LOCK.locked()


def test_closing_stream_requests_generation_cancellation(monkeypatch):
    import threading
    import time
    done = threading.Event()
    def generate(**kwargs):
        kwargs['streamer'].queue.put('Hello')
        try:
            stop = kwargs['stopping_criteria'][-1]
            while not stop(None, None):
                time.sleep(.005)
        finally:
            done.set()
    server = _mock_stream(monkeypatch, generate)
    stream = server._stream(None, 'test', 1)
    assert 'Hello' in next(stream)
    stream.close()
    assert done.wait(1)
