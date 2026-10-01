"""Serve a local model through a small OpenAI-style API.

The server is deliberately narrow: it binds only to loopback, loads only the
model directory it is given on the command line, accepts no remote image URLs,
and exposes no filesystem or command-execution tool to the model.

It handles both shapes of model.  A multimodal model is loaded with
``AutoProcessor`` and can be shown images; a text-only model falls back to
``AutoTokenizer`` and ``AutoModelForCausalLM``.  When no GPU is present the
model is loaded on the CPU in float32 -- slow, but usable, and far better than
refusing to start.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

try:  # Present only in transformers builds that ship the unified multimodal API.
    from transformers import AutoModelForMultimodalLM
except ImportError:  # pragma: no cover - depends on the installed version
    AutoModelForMultimodalLM = None


#: Set from the command line by :func:`main`.  They are module globals rather
#: than constants because one server binary now serves any catalog entry.
MODEL_DIR = Path()
MODEL_ID = "local-model"
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_NEW_TOKENS = 8192
LOAD_LOCK = threading.Lock()
GENERATE_LOCK = threading.Lock()


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    messages: list[Message] = Field(min_length=1, max_length=80)
    max_tokens: int = Field(default=512, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stream: bool = False


class ModelState:
    processor: Any = None
    tokenizer: Any = None
    model: Any = None
    precision: str = "not_loaded"
    device: str = ""
    multimodal: bool = False
    gpu_name: str = ""
    loaded_at: float | None = None


STATE = ModelState()
API_KEY = ""


def _cast_layer_norm_input(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Keep Gemma 4 vision LayerNorm inputs aligned with their stored weights."""
    if not args or not isinstance(args[0], torch.Tensor):
        return args
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.dtype.is_floating_point:
        return args
    value = args[0]
    if value.dtype == weight.dtype:
        return args
    return (value.to(dtype=weight.dtype), *args[1:])


def _install_vision_dtype_guards(model: Any) -> None:
    """Work around mixed BF16/Float32 vision layers in the current Gemma build."""
    core = getattr(model, "model", None)
    vision = getattr(core, "embed_vision", None)
    if vision is None:
        return
    for name in ("patch_ln1", "patch_ln2", "pos_norm"):
        layer = getattr(vision, name, None)
        if layer is not None:
            layer.register_forward_pre_hook(_cast_layer_norm_input)


def _read_api_key(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Cannot read the local API key: {exc}") from exc
    if len(value) < 24:
        raise RuntimeError("The local API key is missing or too short.")
    return value


def _require_local_key(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {API_KEY}"
    if not API_KEY or authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid local model key.")


#: Names accepted from the runtime, plus the two legacy spellings.
_PRECISION_ALIASES = {"bf16": "bfloat16", "4bit": "int4", "8bit": "int8"}


def _select_precision(requested: str) -> str:
    """Choose a load precision that this machine can actually satisfy."""
    requested = _PRECISION_ALIASES.get(requested, requested)
    if not torch.cuda.is_available():
        # bitsandbytes has no usable CPU kernel and CPU bfloat16 is slower than
        # float32 on most hardware, so CPU always means float32.
        return "float32"
    if requested in {"float32", "bfloat16", "int8", "int4"}:
        return requested
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gib >= 40:
        return "bfloat16"
    return "int4" if total_gib < 20 else "int8"


def _has_weights(directory: Path) -> bool:
    """True when the directory holds at least one weight shard."""
    return any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))


def _load_input_handler() -> bool:
    """Load a processor if the model is multimodal, else a tokenizer.

    Returns True when a processor was loaded.  A text-only repository has no
    preprocessor configuration, and asking ``AutoProcessor`` for one raises.
    """
    try:
        STATE.processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    except (OSError, ValueError, KeyError):
        STATE.processor = None
    else:
        # A processor that is really just a tokenizer offers no image handling.
        if hasattr(STATE.processor, "image_processor") or hasattr(STATE.processor, "tokenizer"):
            return True
    STATE.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    return False


def _load_weights(model_kwargs: dict[str, Any]) -> Any:
    """Instantiate the model, preferring the multimodal class when available."""
    errors: list[str] = []
    if AutoModelForMultimodalLM is not None:
        try:
            return AutoModelForMultimodalLM.from_pretrained(MODEL_DIR, **model_kwargs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"multimodal loader: {exc}")
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL_DIR, **model_kwargs)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"causal loader: {exc}")
    raise RuntimeError(
        "The model could not be loaded by any supported loader.\n" + "\n".join(errors)
    )


def load_model(precision: str) -> None:
    with LOAD_LOCK:
        if STATE.model is not None:
            return
        if not MODEL_DIR.is_dir() or not _has_weights(MODEL_DIR):
            raise RuntimeError(f"Model weights are missing: {MODEL_DIR}")

        selected = _select_precision(precision)
        on_gpu = torch.cuda.is_available()
        STATE.device = "cuda" if on_gpu else "cpu"
        STATE.gpu_name = torch.cuda.get_device_name(0) if on_gpu else ""

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if on_gpu:
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["device_map"] = "auto"
            if selected == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif selected == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif selected == "bfloat16":
                if compute_dtype is not torch.bfloat16:
                    raise RuntimeError(
                        "This GPU does not support bfloat16. Restart with Auto or a "
                        "quantized precision."
                    )
                model_kwargs["dtype"] = torch.bfloat16
            else:
                model_kwargs["dtype"] = torch.float32
        else:
            model_kwargs["device_map"] = "cpu"
            model_kwargs["dtype"] = torch.float32

        STATE.multimodal = _load_input_handler()
        STATE.model = _load_weights(model_kwargs)
        _install_vision_dtype_guards(STATE.model)
        STATE.model.eval()
        STATE.precision = selected
        STATE.loaded_at = time.time()


def _decode_data_image(url: str) -> Image.Image:
    if not url.startswith("data:image/") or ";base64," not in url:
        raise ValueError("Only uploaded data:image/... images are allowed; remote URLs are blocked.")
    encoded = url.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("The uploaded image is not valid base64 data.") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Each uploaded image is limited to 20 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a supported image.") from exc


def _prepare_messages(messages: list[Message]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    image_count = 0
    for message in messages:
        if isinstance(message.content, str):
            content: str | list[dict[str, Any]] = message.content
        else:
            blocks: list[dict[str, Any]] = []
            for block in message.content:
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    blocks.append({"type": "text", "text": str(block.get("text") or "")})
                elif block_type == "image_url":
                    image_count += 1
                    if image_count > MAX_IMAGES:
                        raise ValueError(f"At most {MAX_IMAGES} images may be sent in one request.")
                    image_value = block.get("image_url")
                    url = image_value.get("url") if isinstance(image_value, dict) else image_value
                    blocks.append({"type": "image", "url": _decode_data_image(str(url or ""))})
                else:
                    raise ValueError(f"Unsupported message content type: {block_type or 'missing'}")
            content = blocks
        prepared.append({"role": message.role, "content": content})
    return prepared


def _input_handler() -> Any:
    """The processor or tokenizer that turns messages into tensors."""
    return STATE.processor if STATE.processor is not None else STATE.tokenizer


def _text_only(prepared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten image blocks away for a model that cannot see them."""
    flattened: list[dict[str, Any]] = []
    for message in prepared:
        content = message["content"]
        if isinstance(content, str):
            flattened.append(message)
            continue
        parts = [str(block.get("text") or "") for block in content if block.get("type") == "text"]
        dropped = len(content) - len(parts)
        if dropped:
            parts.append(f"[{dropped} image(s) omitted: this model cannot read images.]")
        flattened.append({"role": message["role"], "content": "\n".join(part for part in parts if part)})
    return flattened


def _generation_inputs(request: ChatRequest) -> tuple[dict[str, Any], int]:
    prepared = _prepare_messages(request.messages)
    if not STATE.multimodal:
        prepared = _text_only(prepared)
    handler = _input_handler()
    common: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    try:
        # Reasoning models emit hidden chain-of-thought unless this is off; it
        # is not accepted by every chat template, hence the fallback below.
        encoded = handler.apply_chat_template(prepared, enable_thinking=False, **common)
    except TypeError:
        encoded = handler.apply_chat_template(prepared, **common)
    inputs = encoded.to(STATE.model.device)
    return inputs, int(inputs["input_ids"].shape[-1])


def _generation_kwargs(request: ChatRequest, inputs: dict[str, Any]) -> dict[str, Any]:
    sampling = request.temperature > 0
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "do_sample": sampling,
        "use_cache": True,
    }
    if sampling:
        kwargs.update(temperature=request.temperature, top_p=request.top_p, top_k=64)
    return kwargs


# Gemma 4 can open an (empty) thinking channel even with thinking switched off:
# "<|channel>thought ... <channel|>answer". Decoding drops some of these markers
# as special tokens but not all, so answers came back as "thought <channel|>CD20".
_THINKING = re.compile(r"^\s*(?:<\|?channel\|?>\s*)?thought\b.*?<channel\|>\s*", re.DOTALL)
_MARKERS = re.compile(r"<\|?channel\|?>")


def clean_answer(text: str) -> str:
    """The visible answer: any leading thinking block and stray channel markers removed."""
    return _MARKERS.sub("", _THINKING.sub("", text or "", count=1)).strip()


class _AnswerFilter:
    """The same cleaning for a streamed answer: holds back a thinking block until it ends."""

    def __init__(self) -> None:
        self.buffer = ""
        self.open = True  # still deciding whether the answer starts with a thinking block

    def feed(self, piece: str) -> str:
        if not self.open:
            return _MARKERS.sub("", piece)
        self.buffer += piece
        head = self.buffer.lstrip()
        # Also true while only the first letters of a marker have arrived.
        maybe_thinking = any(head.startswith(start) or start.startswith(head)
                             for start in ("thought", "<|channel>", "<channel|>"))
        if maybe_thinking and "<channel|>" not in self.buffer and len(self.buffer) < 20000:
            return ""
        self.open = False
        return clean_answer(self.buffer) if maybe_thinking else _MARKERS.sub("", self.buffer)

    def flush(self) -> str:
        if not self.open:
            return ""
        self.open = False
        return clean_answer(self.buffer)


def _parsed_text(output: Any, prefix: Any) -> str:
    handler = _input_handler()
    try:
        parsed = handler.parse_response(output, prefix=prefix)
        if isinstance(parsed, dict):
            content = parsed.get("content", "")
            if isinstance(content, str):
                return clean_answer(content)
            if isinstance(content, list):
                return clean_answer("\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict)))
    except (AttributeError, TypeError, ValueError):
        # Only some processors implement parse_response; decoding below is the
        # portable path.
        pass
    try:
        prefix_len = int(prefix.shape[-1])
        output = output[prefix_len:]
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return clean_answer(handler.decode(output, skip_special_tokens=True))


def _complete(request: ChatRequest) -> tuple[str, int, int]:
    with GENERATE_LOCK, torch.inference_mode():
        inputs, input_len = _generation_inputs(request)
        output = STATE.model.generate(**_generation_kwargs(request, inputs))[0]
        text = _parsed_text(output, inputs["input_ids"])
        completion_tokens = max(0, int(output.shape[-1]) - input_len)
        return text, input_len, completion_tokens


def _stream(request: ChatRequest, completion_id: str, created: int) -> Iterator[str]:
    with GENERATE_LOCK, torch.inference_mode():
        inputs, _ = _generation_inputs(request)
        handler = _input_handler()
        streamer = TextIteratorStreamer(
            getattr(handler, "tokenizer", handler),
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=120.0,
        )
        kwargs = _generation_kwargs(request, inputs)
        kwargs["streamer"] = streamer
        error: list[BaseException] = []

        def generate() -> None:
            try:
                STATE.model.generate(**kwargs)
            except BaseException as exc:  # relayed through the stream
                error.append(exc)

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        answer = _AnswerFilter()

        def pieces() -> Iterator[str]:
            for piece in streamer:
                yield answer.feed(piece)
            yield answer.flush()

        for token in pieces():
            if not token:
                continue
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
            }
            yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
        thread.join()
        if error:
            yield "data: " + json.dumps({"error": str(error[0])}) + "\n\n"
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield "data: " + json.dumps(final) + "\n\n"
        yield "data: [DONE]\n\n"


def create_app(precision: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        load_model(precision)
        yield

    app = FastAPI(title="SPACEMAP Model Service", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/health", dependencies=[Depends(_require_local_key)])
    def health() -> dict[str, Any]:
        return {
            "status": "ready" if STATE.model is not None else "loading",
            "model": MODEL_ID,
            "precision": STATE.precision,
            "gpu": STATE.gpu_name,
            "device": STATE.device,
            "multimodal": STATE.multimodal,
            "loaded_at": STATE.loaded_at,
            "max_new_tokens": MAX_NEW_TOKENS,
        }

    @app.get("/v1/models", dependencies=[Depends(_require_local_key)])
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "google"}]}

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_local_key)])
    def chat(request: ChatRequest):
        if request.model != MODEL_ID:
            raise HTTPException(status_code=400, detail=f"Only {MODEL_ID} is installed.")
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        try:
            if request.stream:
                return StreamingResponse(
                    _stream(request, completion_id, created),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            text, prompt_tokens, completion_tokens = _complete(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise HTTPException(
                status_code=503,
                detail="The GPU ran out of memory. Restart the model in 4-bit mode or use a larger GPU.",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Local model generation failed: {str(exc)[:500]}",
            ) from exc
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return JSONResponse(payload)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only SPACEMAP model service")
    parser.add_argument(
        "--model-dir", required=True, type=Path, help="Directory holding the model weights."
    )
    parser.add_argument(
        "--model-id", default="local-model", help="Name echoed back in API responses."
    )
    # Loopback only: this server authenticates with a shared local key and is
    # not built to face a network.
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", default=8091, type=int)
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "float32", "bfloat16", "int8", "int4", "bf16", "4bit", "8bit"],
    )
    parser.add_argument("--api-key-file", required=True, type=Path)
    args = parser.parse_args()

    global API_KEY, MODEL_DIR, MODEL_ID
    MODEL_DIR = args.model_dir.expanduser().resolve()
    MODEL_ID = args.model_id
    if not MODEL_DIR.is_dir():
        parser.error(f"--model-dir does not exist: {MODEL_DIR}")
    API_KEY = _read_api_key(args.api_key_file)
    uvicorn.run(create_app(args.precision), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
