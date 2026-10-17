"""Studio's AI setup: the classic AI setup page's choices, in the Assistant tab.

Everything goes through the same functions as classic's page
(``components/ai_setup.py``): the assistant on or off, the local models
(download with progress, use, start, stop), the online model (NVIDIA, used
only when chosen; the key is saved, never shown), the GPU memory mode, whether
the model stops when SPACEMAP stops, and the model servers that earlier
sessions left running on this node.
"""

import json
import time
from datetime import datetime, timezone

from spacemap import ai_settings

_online_checks = {}


def check_online(model_id):
    """One explicit connection check, with synthetic data and no project actions."""
    from spacemap_assistant.model_client import ModelUnavailable, NvidiaModelClient

    client = NvidiaModelClient(model_id=model_id, timeout_seconds=20)
    client.require_complete = True
    client.max_response_seconds = 30
    start = time.monotonic()
    first = None
    result = {"checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        pieces = []
        for piece in client.stream_complete(
            [{"role":"user", "content":'Return only JSON {"status":"ready"}.'}],
            temperature=1, max_tokens=4096, enable_thinking=False, reasoning_strength="low",
        ):
            if first is None:
                first = time.monotonic() - start
            pieces.append(piece)
        raw = ''.join(pieces).strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        if json.loads(raw) != {"status":"ready"}:
            raise ValueError("Connected, but failed the expected reply check.")
        result.update(ok=True, message="Connection check passed")
    except (ModelUnavailable, ValueError) as error:
        result.update(ok=False, message=str(error))
    result.update(seconds=round(time.monotonic()-start, 1), first_answer_seconds=round(first, 1) if first is not None else None)
    _online_checks[model_id] = result
    return result

PRECISION_LABELS = {
    "auto": "Automatic",
    "float32": "Full precision (most memory)",
    "bfloat16": "Half precision",
    "int8": "8-bit (less memory)",
    "int4": "4-bit (least memory)",
}


def _runtime():
    try:
        from spacemap_assistant.model_runtime import runtime_status

        state = runtime_status()
        return {
            "running": state.running,
            "ready": state.ready,
            "message": state.message,
            "port": state.port,
            "model": state.model_id,
        }
    except Exception as error:  # noqa: BLE001 - the packages may be missing; the page says what to install
        return {"running": False, "ready": False, "message": str(error), "port": 0, "model": ""}


def state():
    """What the setup shows."""
    from spacemap.hardware import can_run, detect_hardware
    from spacemap.model_catalog import list_models
    from spacemap_assistant import model_client, model_installer, model_runtime

    report = detect_hardware()
    settings = ai_settings.load_settings()
    active = str(settings.get("active_model") or "")
    models = []
    for entry in list_models():
        status = model_installer.read_status(entry.id)
        fits, why = can_run(report, min_vram_gb=entry.min_vram_gb, cpu_capable=entry.cpu_capable)
        models.append(
            {
                "id": entry.id,
                "name": entry.name,
                "size_gb": entry.approx_size_gb,
                "installed": entry.is_installed(),
                "downloading": status.is_active,
                "percent": round(status.percent, 1),
                "failed": status.message if status.state == "error" else "",
                "fits": bool(fits),
                "why": why,
                "active": entry.id == active,
            }
        )
    provider = model_client.load_provider_settings()
    missing = [row for row in ai_settings.dependency_report() if row["required"] and not row["installed"]]
    return {
        "enabled": ai_settings.is_enabled(),
        "pinned": ai_settings.is_forced_on() or ai_settings.is_forced_off(),
        "hardware": report.summary(),
        "missing": [row["name"] for row in missing],
        "install": ai_settings.install_command([(row["name"], row["wanted"]) for row in missing]) if missing else "",
        "models": models,
        "runtime": _runtime(),
        "precision": str(settings.get("precision") or "auto"),
        "precisions": [[key, label] for key, label in PRECISION_LABELS.items()],
        "stop_with_spacemap": bool(settings.get("stop_with_spacemap", True)),
        "leftovers": [
            {key: item[key] for key in ("pid", "port", "model", "started")} for item in model_runtime.leftover_servers()
        ],
        "online": {
            "chosen": provider.provider == model_client.NVIDIA_PROVIDER,
            "model": provider.nvidia_model_id,
            "models": [[key, value["label"]] for key, value in model_client.NVIDIA_MODEL_CATALOG.items()],
            "details": {key:{"description":value["description"], "images":value["supports_images"]}
                        for key, value in model_client.NVIDIA_MODEL_CATALOG.items()},
            "checks": dict(_online_checks),
            "key_saved": model_client.nvidia_api_key_saved(),
        },
    }


def _entry(model_id):
    from spacemap.model_catalog import ModelCatalogError, get_model

    try:
        return get_model(str(model_id or ""))
    except ModelCatalogError as error:
        raise ValueError("Choose one of the listed models.") from error


def act(action, body):
    """Do one setup action; returns the new state."""
    from spacemap_assistant import model_client, model_installer, model_runtime

    if action == "enable":
        if ai_settings.is_forced_on() or ai_settings.is_forced_off():
            raise ValueError("The assistant is pinned on or off by SPACEMAP_ASSISTANT.")
        ai_settings.set_enabled(bool(body.get("on")))
    elif action == "download":
        try:
            model_installer.start_install(_entry(body.get("model")), token=ai_settings.read_token())
        except model_installer.ModelInstallError as error:
            raise ValueError(str(error)) from error
    elif action == "cancel":
        model_installer.cancel_install(_entry(body.get("model")).id)
    elif action == "use":
        entry = _entry(body.get("model"))
        if not entry.is_installed():
            raise ValueError(f"Download {entry.name} first.")
        if model_runtime.runtime_status().running and model_runtime.active_entry() != entry:
            model_runtime.stop_runtime()  # the model in use makes room for this one
        ai_settings.update_settings(active_model=entry.id, backend="local")
        model_client.save_provider_settings(
            model_client.LOCAL_PROVIDER, model_client.load_provider_settings().nvidia_model_id
        )
    elif action == "start":
        try:
            model_runtime.start_runtime(str(ai_settings.load_settings().get("precision") or "auto"))
        except model_runtime.ModelRuntimeError as error:
            raise ValueError(str(error)) from error
    elif action == "stop":
        try:
            model_runtime.stop_runtime()
        except model_runtime.ModelRuntimeError as error:
            raise ValueError(str(error)) from error
    elif action == "stop_leftovers":
        model_runtime.stop_leftover_servers()
    elif action == "settings":
        changes = {}
        if "precision" in body:
            if body["precision"] not in PRECISION_LABELS:
                raise ValueError("Choose a GPU memory mode from the list.")
            changes["precision"] = body["precision"]
        if "stop_with_spacemap" in body:
            changes["stop_with_spacemap"] = bool(body["stop_with_spacemap"])
        if changes:
            ai_settings.update_settings(**changes)
    elif action == "online":
        key = body.get("key")
        if key:
            model_client.save_nvidia_api_key(str(key))  # checked there; never returned or logged
        provider = model_client.NVIDIA_PROVIDER if body.get("use") else model_client.LOCAL_PROVIDER
        model_id = str(body.get("model") or model_client.load_provider_settings().nvidia_model_id)
        if provider == model_client.NVIDIA_PROVIDER and not model_client.nvidia_api_key_saved():
            raise ValueError("Save an NVIDIA API key first.")
        model_client.save_provider_settings(provider, model_id)
    elif action == "check_online":
        check_online(str(body.get("model") or model_client.load_provider_settings().nvidia_model_id))
    elif action == "forget_key":
        model_client.delete_nvidia_api_key()
        model_client.save_provider_settings(
            model_client.LOCAL_PROVIDER, model_client.load_provider_settings().nvidia_model_id
        )
    else:
        raise ValueError("Unknown AI setup action.")
    return state()
