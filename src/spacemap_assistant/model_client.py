"""Model clients for the local model server and the optional hosted API.

The local endpoint is always loopback and always proxy-free; only its port is
configurable, so that two copies of SPACEMAP can run side by side.  The hosted
endpoint and its model IDs stay allowlisted here so that a chat message cannot
redirect credentials to an arbitrary server.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .paths import ASSISTANT_DATA_DIR, require_assistant_write_path


#: Fallback identifier used when no catalog model is selected.  The server
#: echoes whatever it is given, so this only affects the request body.
DEFAULT_LOCAL_MODEL_ID = "local-model"


def local_model_id() -> str:
    """The name the local model server was started with for the selected model.

    The runtime starts the server as ``--model-id <repo_id or id>`` and the
    server refuses any other name, so requests must use the same one (the
    catalog id alone was refused: "Only google/gemma-4-12B-it is installed").
    """
    from spacemap.ai_settings import load_settings

    selected = str(load_settings().get("active_model") or "").strip()
    if not selected:
        return DEFAULT_LOCAL_MODEL_ID
    try:
        from spacemap.model_catalog import get_model

        entry = get_model(selected)
    except Exception:
        return selected
    return entry.repo_id or entry.id


def local_port() -> int:
    """The loopback port the local model server listens on."""
    from spacemap.ai_settings import load_settings

    try:
        return int(load_settings().get("server_port") or 8091)
    except (TypeError, ValueError):
        return 8091


def local_base_url() -> str:
    """The OpenAI-style base URL of the local model server."""
    return f"http://127.0.0.1:{local_port()}/v1"


#: Retained for callers that display a name; prefer :func:`local_model_id`.
MODEL_ID = DEFAULT_LOCAL_MODEL_ID
MODEL_SETTINGS_PATH = ASSISTANT_DATA_DIR / "config" / "local_model_settings.json"
MODEL_KEY_PATH = ASSISTANT_DATA_DIR / "config" / "local_model.key"
PROVIDER_SETTINGS_PATH = ASSISTANT_DATA_DIR / "config" / "model_provider.json"
NVIDIA_API_KEY_PATH = ASSISTANT_DATA_DIR / "config" / "nvidia_api.key"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LOCAL_PROVIDER = "local"
NVIDIA_PROVIDER = "nvidia"
DEFAULT_NVIDIA_MODEL_ID = "google/gemma-4-31b-it"
NVIDIA_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "meta/muse-glimmer-30b": {
        "label": "Meta Muse Glimmer 30B (best SPACEMAP fit)",
        "description": (
            "Multimodal reasoning, tool calling, coding, and failure recovery. "
            "Recommended for future image QC after institutional approval."
        ),
        "max_output_tokens": 8_192,
        # Muse performs hidden reasoning before writing visible text. With a
        # 512-token cap it can finish by length with content=null, even for a
        # one-word answer. This is a generation allowance, not a reservation;
        # the service stops early when the answer is complete.
        "minimum_output_tokens": 1_024,
        "supports_images": True,
    },
    "google/gemma-4-31b-it": {
        "label": "Google Gemma 4 31B (scientific reasoning)",
        "description": "Strong general, scientific, coding, and multimodal reasoning.",
        "max_output_tokens": 16_384,
        "supports_images": True,
    },
    "google/diffusiongemma-26b-a4b-it": {
        "label": "Google DiffusionGemma 26B A4B",
        "description": "A faster mixture-of-experts model; useful when speed matters more than maximum quality.",
        "max_output_tokens": 8_192,
        "supports_images": True,
    },
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "label": "NVIDIA Nemotron 3 Ultra 550B A55B",
        "description": "Very large text reasoning and agent model. Hosted by NVIDIA; not an image model.",
        "max_output_tokens": 16_384,
        "supports_images": False,
    },
}
LOCAL_ONLY_OPENER = build_opener(ProxyHandler({}))
NETWORK_OPENER = build_opener()
ACTION_PLAN_MAX_TOKENS = 4096
COMPLEX_ACTION_PLAN_MAX_TOKENS = 8192

_EXECUTION_WORDS = (
    "go ahead",
    "can you do",
    "could you do",
    "would you do",
    "do the following",
    "proceed",
)
_EXECUTION_REQUEST_PATTERNS = (
    re.compile(
        r"^\s*(?:now\s+)?(?:please\s+)?(?:run|perform|process|apply|save|train|register|align|subtract|remove|extract|annotate|detect|segment|"
        r"correct|cluster|phenotype|classify|label|analyze|analyse|assess|calculate|generate|"
        r"create|draw|merge|export|test|optimize|optimise)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:run|perform|process|apply|save|train|register|align|subtract|remove|"
        r"extract|annotate|detect|segment|correct|cluster|phenotype|classify|label|analyze|"
        r"analyse|assess|calculate|generate|create|draw|merge|export|test|optimize|optimise)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi\s+(?:want|need|would like)\s+you\s+to\s+(?:run|perform|process|apply|save|train|register|align|subtract|remove|extract|"
        r"annotate|detect|segment|correct|cluster|phenotype|classify|label|analyze|analyse|"
        r"assess|calculate|generate|create|draw|merge|export|test|optimize|optimise)\b",
        re.IGNORECASE,
    ),
)
_COMPLEX_ACTION_WORDS = (
    "test several",
    "compare parameters",
    "optimize",
    "parameter sweep",
    "grid search",
    "hypotheses",
    "leiden",
    " then ",
    "from a to z",
    "workflow",
)


class ModelUnavailable(RuntimeError):
    """Raised when the selected model cannot answer."""


@dataclass(frozen=True)
class ModelSettings:
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_seconds: int = 600

    def normalized(self) -> "ModelSettings":
        return ModelSettings(
            temperature=min(2.0, max(0.0, float(self.temperature))),
            max_tokens=min(2048, max(32, int(self.max_tokens))),
            timeout_seconds=min(1800, max(30, int(self.timeout_seconds))),
        )


def load_model_settings() -> ModelSettings:
    try:
        payload = json.loads(MODEL_SETTINGS_PATH.read_text(encoding="utf-8"))
        return ModelSettings(
            temperature=payload.get("temperature", 0.2),
            max_tokens=payload.get("max_tokens", 512),
            timeout_seconds=payload.get("timeout_seconds", 600),
        ).normalized()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ModelSettings()


def save_model_settings(temperature: float, max_tokens: int, timeout_seconds: int) -> ModelSettings:
    settings = ModelSettings(temperature, max_tokens, timeout_seconds).normalized()
    target = require_assistant_write_path(MODEL_SETTINGS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return settings


@dataclass(frozen=True)
class ProviderSettings:
    """The selected provider and public model ID.  Secrets are stored separately."""

    provider: str = LOCAL_PROVIDER
    nvidia_model_id: str = DEFAULT_NVIDIA_MODEL_ID

    def normalized(self) -> "ProviderSettings":
        provider = str(self.provider).strip().lower()
        if provider not in {LOCAL_PROVIDER, NVIDIA_PROVIDER}:
            provider = LOCAL_PROVIDER
        model_id = str(self.nvidia_model_id).strip()
        if model_id not in NVIDIA_MODEL_CATALOG:
            model_id = DEFAULT_NVIDIA_MODEL_ID
        return ProviderSettings(provider=provider, nvidia_model_id=model_id)


def load_provider_settings() -> ProviderSettings:
    try:
        payload = json.loads(PROVIDER_SETTINGS_PATH.read_text(encoding="utf-8"))
        return ProviderSettings(
            provider=payload.get("provider", LOCAL_PROVIDER),
            nvidia_model_id=payload.get("nvidia_model_id", DEFAULT_NVIDIA_MODEL_ID),
        ).normalized()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return ProviderSettings()


def _atomic_private_text(path: Path, value: str) -> None:
    target = require_assistant_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            os.chmod(temp, 0o600)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        os.chmod(target, 0o600)
    finally:
        if temp.exists():
            temp.unlink()


def save_provider_settings(provider: str, nvidia_model_id: str) -> ProviderSettings:
    settings = ProviderSettings(provider=provider, nvidia_model_id=nvidia_model_id).normalized()
    _atomic_private_text(PROVIDER_SETTINGS_PATH, json.dumps(asdict(settings), indent=2) + "\n")
    return settings


def save_nvidia_api_key(value: str) -> None:
    key = str(value).strip()
    if len(key) < 24:
        raise ValueError("Enter a complete NVIDIA API key.")
    if any(character.isspace() for character in key):
        raise ValueError("The NVIDIA API key cannot contain spaces.")
    _atomic_private_text(NVIDIA_API_KEY_PATH, key + "\n")


def delete_nvidia_api_key() -> bool:
    target = require_assistant_write_path(NVIDIA_API_KEY_PATH)
    if not target.exists():
        return False
    target.unlink()
    return True


def nvidia_api_key_saved() -> bool:
    try:
        return NVIDIA_API_KEY_PATH.is_file() and len(
            NVIDIA_API_KEY_PATH.read_text(encoding="utf-8").strip()
        ) >= 24
    except OSError:
        return False


def _nvidia_api_key() -> str:
    try:
        value = NVIDIA_API_KEY_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ModelUnavailable("No NVIDIA API key is saved.") from exc
    if len(value) < 24 or any(character.isspace() for character in value):
        raise ModelUnavailable("The saved API key is invalid. Replace it in the assistant model settings.")
    return value


def completion_budget(prompt: str, normal_max_tokens: int) -> tuple[int, bool, str]:
    """Keep normal chat short while giving private action plans room to finish."""
    lowered = str(prompt).lower()
    normal = min(2048, max(32, int(normal_max_tokens)))
    is_execution = any(word in lowered for word in _EXECUTION_WORDS) or any(
        pattern.search(str(prompt)) for pattern in _EXECUTION_REQUEST_PATTERNS
    )
    if not is_execution:
        return normal, False, "normal_chat"
    if any(word in lowered for word in _COMPLEX_ACTION_WORDS):
        return max(normal, COMPLEX_ACTION_PLAN_MAX_TOKENS), True, "complex_action_plan"
    return max(normal, ACTION_PLAN_MAX_TOKENS), True, "action_plan"


def _local_api_key() -> str:
    try:
        value = MODEL_KEY_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ModelUnavailable("The local model server has not been started yet.") from exc
    if len(value) < 24:
        raise ModelUnavailable("The local model server key is invalid. Restart the model server.")
    return value


def _validate_base_url(base_url: str) -> str:
    """Accept only a loopback HTTP endpoint.

    The port is configurable but the host is not: pointing this client at a
    remote address would send the local API key, and the user's data, off the
    machine under the name "local model".
    """
    cleaned = str(base_url).rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "The SPACEMAP local model connection must stay on loopback, "
            f"for example http://127.0.0.1:{local_port()}/v1."
        )
    return cleaned


class LocalModelClient:
    def __init__(self, *, base_url: str | None = None, timeout_seconds: int = 600) -> None:
        self.base_url = _validate_base_url(base_url or local_base_url())
        self.timeout_seconds = min(1800, max(5, int(timeout_seconds)))

    def _request(self, path: str, *, payload: dict[str, Any] | None = None):
        url = self.base_url.removesuffix("/v1") + path if path.startswith("/") else self.base_url + "/" + path
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="GET" if payload is None else "POST",
            headers={
                "Authorization": f"Bearer {_local_api_key()}",
                "Content-Type": "application/json",
            },
        )
        try:
            # Some sites export a web proxy without a matching NO_PROXY.
            # Localhost model traffic, including the private key, must never be
            # sent through it.
            return LOCAL_ONLY_OPENER.open(request, timeout=self.timeout_seconds)
        except HTTPError as exc:
            detail = ""
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace"))
                detail = str(payload.get("detail") or payload.get("error") or "")
            except (ValueError, OSError):
                pass
            raise ModelUnavailable(detail or f"The local Gemma server returned HTTP {exc.code}.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ModelUnavailable(
                f"The local Gemma model is not available at {self.base_url}: {reason}"
            ) from exc

    def health(self) -> dict[str, Any]:
        with self._request("/health") as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": local_model_id(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        with self._request("chat/completions", payload=payload) as response:
            raw_result = response.read().decode("utf-8")
        try:
            result = json.loads(raw_result)
            content = result["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailable("The local model server returned an invalid response.") from exc
        # Some OpenAI-compatible servers return JSON null when generation ends
        # without a visible answer.  Converting null with str() produced the
        # misleading literal word "None" in Chat and prevented the normal
        # empty-response retry/fallback from running.
        return content.strip() if isinstance(content, str) else ""

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        payload = {
            "model": local_model_id(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with self._request("chat/completions", payload=payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise ModelUnavailable(str(chunk["error"]))
                try:
                    text = chunk["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, TypeError):
                    text = ""
                if text:
                    yield str(text)


def _nvidia_error_detail(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("detail") or "").strip()
        else:
            detail = str(error or payload.get("detail") or "").strip()
        if detail:
            return detail[:1_000]
    except (ValueError, OSError, AttributeError):
        pass
    return f"The NVIDIA API returned HTTP {exc.code}."


class NvidiaModelClient:
    """OpenAI-compatible client restricted to NVIDIA's hosted API catalog."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_NVIDIA_MODEL_ID,
        base_url: str = NVIDIA_BASE_URL,
        timeout_seconds: int = 600,
    ) -> None:
        if str(model_id) not in NVIDIA_MODEL_CATALOG:
            raise ValueError("That NVIDIA model is not in the SPACEMAP approved model list.")
        if str(base_url).rstrip("/") != NVIDIA_BASE_URL:
            raise ValueError(f"The NVIDIA connection must stay at {NVIDIA_BASE_URL}.")
        self.model_id = str(model_id)
        self.base_url = NVIDIA_BASE_URL
        self.timeout_seconds = min(1800, max(5, int(timeout_seconds)))

    def _request(self, *, payload: dict[str, Any]):
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {_nvidia_api_key()}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if payload.get("stream") else "application/json",
            },
        )
        try:
            return NETWORK_OPENER.open(request, timeout=self.timeout_seconds)
        except HTTPError as exc:
            raise ModelUnavailable(_nvidia_error_detail(exc)) from exc
        except RemoteDisconnected as exc:
            raise ModelUnavailable(
                "The remote service or institutional web proxy closed the connection before the "
                "model returned data. The selected NVIDIA prototype endpoint may be overloaded; "
                "try Muse Glimmer or retry later."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
                raise ModelUnavailable(
                    f"NVIDIA did not return model data within {self.timeout_seconds} seconds. "
                    "The network may still be working; the shared prototype model may be busy or starting. "
                    "This hosted request does not use a local GPU."
                ) from exc
            raise ModelUnavailable(f"The NVIDIA hosted model is not available: {reason}") from exc

    def health(self) -> dict[str, Any]:
        metadata = NVIDIA_MODEL_CATALOG[self.model_id]
        return {
            "status": "configured" if nvidia_api_key_saved() else "missing_api_key",
            "model": self.model_id,
            "max_new_tokens": int(metadata["max_output_tokens"]),
        }

    def _payload(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": min(1.5, max(0.0, float(temperature))),
            "top_p": 0.95,
            "max_tokens": min(
                int(NVIDIA_MODEL_CATALOG[self.model_id]["max_output_tokens"]),
                max(
                    int(NVIDIA_MODEL_CATALOG[self.model_id].get("minimum_output_tokens", 32)),
                    int(max_tokens),
                ),
            ),
            "stream": bool(stream),
        }
        if self.model_id == "nvidia/nemotron-3-ultra-550b-a55b":
            thinking = True if enable_thinking is None else bool(enable_thinking)
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
            if thinking:
                payload["reasoning_budget"] = min(4_096, payload["max_tokens"])
        elif self.model_id.startswith("google/"):
            payload["chat_template_kwargs"] = {
                "enable_thinking": True if enable_thinking is None else bool(enable_thinking)
            }
        # Muse Glimmer's hosted quickstart does not add model-specific fields.
        return payload

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        enable_thinking: bool | None = None,
    ) -> str:
        with self._request(
            payload=self._payload(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                enable_thinking=enable_thinking,
            )
        ) as response:
            raw_result = response.read().decode("utf-8")
        try:
            result = json.loads(raw_result)
            content = result["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailable("The NVIDIA API returned an invalid response.") from exc
        return content.strip() if isinstance(content, str) else ""

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        enable_thinking: bool | None = None,
    ) -> Iterator[str]:
        with self._request(
            payload=self._payload(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                enable_thinking=enable_thinking,
            )
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise ModelUnavailable(str(chunk["error"])[:1_000])
                try:
                    text = chunk["choices"][0]["delta"].get("content", "")
                except (KeyError, IndexError, TypeError):
                    text = ""
                if text:
                    yield str(text)


def test_nvidia_connection(model_id: str, *, timeout_seconds: int = 180) -> str:
    """Make one small streamed request from the explicit UI test button."""
    client = NvidiaModelClient(model_id=model_id, timeout_seconds=timeout_seconds)
    answer = "".join(
        client.stream_complete(
            [
                {
                    "role": "user",
                    "content": "Reply briefly: SPACEMAP NVIDIA connection is ready",
                },
            ],
            temperature=0.0,
            max_tokens=512,
            enable_thinking=False,
        )
    ).strip()
    if not answer:
        raise ModelUnavailable("The NVIDIA model connected but returned no final answer text.")
    return answer
