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
