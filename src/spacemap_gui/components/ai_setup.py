"""AI setup: one page for everything the assistant needs.

The same page is shown in Tab 0 (AI Assistant) and as the Setup tab of the
assistant page, so nothing has to be looked for in a second place:

1. **Status and next step.** What is done, and one button for what comes next
   (download, use, start, or a test question).
2. **Models.** Every model in one row with its state and its buttons
   (download with live progress, use, start, stop, remove), and a Hugging Face
   search to add any open chat model.
3. **Answer with an online model instead** (optional; off unless chosen).
4. **Storage and access:** models folder with free space, Hugging Face token.
5. **Advanced:** GPU memory mode, answer settings, cluster, folder access for
   AI tools, activity, the model server's log, turning the assistant off.

The chosen local model also starts by itself when a question is asked
(``model_runtime.ensure_running``), so starting it here is optional.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from spacemap import ai_settings
from spacemap.hardware import can_run, detect_hardware, free_disk_gb
from spacemap.model_catalog import ModelCatalogError, add_user_model, list_models, remove_user_model

PRECISION_LABELS = {
    "auto": "Auto (recommended): full quality when the GPU has room, else lighter",
    "bfloat16": "Full quality (bfloat16), needs about 1.2 x the model size in GPU memory",
    "int8": "8-bit, about half the GPU memory",
    "int4": "4-bit, about a third of the GPU memory",
    "float32": "float32 (CPU, slow)",
}


def _human_bytes(value: float) -> str:
    if value <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.1f} TB"


def _space(gigabytes: float) -> str:
    return f"{gigabytes / 1000:.0f} TB" if gigabytes >= 1000 else f"{gigabytes:.0f} GB"


@st.cache_resource(ttl=120, show_spinner=False)
def _hardware():
    return detect_hardware()


def _runtime_status():
    try:
        from spacemap_assistant.model_runtime import runtime_status

        return runtime_status()
    except Exception:  # the packages may be missing; the page still shows what to install
        return None


def _online_chosen() -> bool:
    try:
        from spacemap_assistant.model_client import NVIDIA_PROVIDER, load_provider_settings

        return load_provider_settings().provider == NVIDIA_PROVIDER
    except Exception:
        return False


def _log(event: str, message: str, **details) -> None:
    try:
        from spacemap_gui.components.assistant_panel import _store

        _store().log_activity(event, message, details=details or None)
    except Exception:
        pass


def _missing_packages() -> list[dict]:
    return [row for row in ai_settings.dependency_report() if row["required"] and not row["installed"]]


def _suggested(models, report):
    """The model to suggest downloading on this machine."""
    fitting = [m for m in models if m.backend == "transformers" and can_run(report, min_vram_gb=m.min_vram_gb,
                                                                            cpu_capable=m.cpu_capable)[0]]
    if report.has_gpu and fitting:
        return max(fitting, key=lambda m: m.approx_size_gb)  # the largest that fits
    small = [m for m in models if m.backend == "gguf"] or sorted(models, key=lambda m: m.approx_size_gb)
    return small[0] if small else None


# ---------------------------------------------------------------- 1. status

def _render_status(prefix: str) -> None:
    settings = ai_settings.load_settings()
    models = list_models()
    report = _hardware()
    missing = _missing_packages()
    online = _online_chosen()
    active = next((m for m in models if m.id == settings.get("active_model")), None)
    installed = [m for m in models if m.is_installed()]
    status = _runtime_status() if not missing else None

    if online:
        from spacemap_assistant.model_client import nvidia_api_key_saved

        steps = [("Assistant on", True), ("Online model chosen", True), ("API key saved", nvidia_api_key_saved())]
    else:
        steps = [("Assistant on", True), ("Packages installed", not missing), ("Model downloaded", bool(installed)),
                 ("Model chosen", bool(active and active.is_installed())),
                 ("Model running", bool(status and status.ready))]
    with st.container(border=True):
        columns = st.columns(len(steps))
        for column, (label, done) in zip(columns, steps):
            column.badge("Done" if done else "To do", color="green" if done else "orange")
            column.caption(label)
        _render_next_step(prefix, settings, models, report, missing, online, active, installed, status)


def _render_next_step(prefix, settings, models, report, missing, online, active, installed, status) -> None:
    if online:
        st.info("Questions go to the online model chosen below. Switch back to a local model under "
                "**Answer with an online model instead**.")
        return
    if missing:
        from spacemap_gui.components.assistant_setup import _render_dependency_status

        _render_dependency_status()
        return
    if not installed:
        suggestion = _suggested(models, report)
        if suggestion is None:
            st.info("Add a model below to start.")
            return
        st.markdown(f"**Next:** download a model. Suggested for this machine ({report.summary()}): "
                    f"**{suggestion.name}**, {suggestion.approx_size_gb:.0f} GB.")
        if suggestion.gated and not ai_settings.read_token():
            st.caption("This model needs a Hugging Face token first (see **Storage and access** below).")
        if st.button(f"Download {suggestion.name}", type="primary", key=f"{prefix}_next_download"):
            _start_download(suggestion)
        return
    if not active or not active.is_installed():
        choice = installed[0]
        st.markdown(f"**Next:** choose the model to use. **{choice.name}** is downloaded.")
        if st.button(f"Use {choice.name}", type="primary", key=f"{prefix}_next_use"):
            ai_settings.update_settings(active_model=choice.id, backend="local")
            st.rerun()
        return
    if status is not None and status.running and not status.ready:
        _starting_progress(prefix)
        return
    if status is None or not status.running:
        st.markdown(f"**Next:** start **{active.name}**, or just ask a question: it starts by itself "
                    "(about a minute on a GPU).")
        if st.button(f"Start {active.name}", type="primary", key=f"{prefix}_next_start"):
            _start(active)
        return
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.success(f"**{active.name}** is ready ({status.precision}{', ' + status.gpu if status.gpu else ''}). "
                 "Ask anything on the assistant page or in the assistant pop-up.")
    if right.button("Ask a test question", key=f"{prefix}_next_test"):
        _test_question()


def _start(entry) -> None:
    from spacemap_assistant.model_runtime import ModelRuntimeError, start_runtime

    try:
        start_runtime(str(ai_settings.load_settings().get("precision") or "auto"))
    except ModelRuntimeError as exc:
        st.error(str(exc))
        return
    _log("local_model_started", f"{entry.name} was started from AI setup.", model=entry.id)
    st.rerun()


@st.fragment(run_every=3)
def _starting_progress(prefix: str) -> None:
    """Loading progress that updates by itself; the page updates when the model is ready."""
    from spacemap_assistant.model_runtime import _log_tail

    status = _runtime_status()
    if status is None or status.ready or not status.running:
        st.rerun()
        return
    started = st.session_state.setdefault(f"{prefix}_start_seen", time.time())
    st.info(f"{status.message} {time.time() - started:.0f} s so far (usually about a minute on a GPU).")
    tail = [line for line in _log_tail(6).splitlines() if line.strip()]
    if tail:
        st.caption(tail[-1][-200:])


def _test_question() -> None:
    from spacemap_assistant.model_client import LocalModelClient, ModelUnavailable

    started = time.monotonic()
    try:
        answer = LocalModelClient(timeout_seconds=120).complete(
            [{"role": "user", "content": "In one short sentence: what does a CD8 T cell do?"}], max_tokens=60)
    except (ModelUnavailable, OSError, ValueError) as exc:
        st.error(f"No answer: {exc}")
        return
    st.success(f"Answered in {time.monotonic() - started:.1f} s: {answer}")


# ---------------------------------------------------------------- 2. models

def _start_download(entry) -> None:
    from spacemap_assistant import model_installer

    try:
        model_installer.start_install(entry, token=ai_settings.read_token())
    except model_installer.ModelInstallError as exc:
        st.error(str(exc))
        return
    _log("model_download_started", f"Download of {entry.name} started.", model=entry.id)
    st.rerun()


@st.fragment(run_every=2)
def _download_progress(model_id: str) -> None:
    """Download progress that updates by itself every 2 seconds (no Refresh button)."""
    from spacemap_assistant import model_installer

    status = model_installer.read_status(model_id)
    if not status.is_active:
        st.rerun()  # finished: the whole row shows the result
        return
    st.progress(min(1.0, status.percent / 100.0),
                text=f"{_human_bytes(status.downloaded_bytes)} of about {_human_bytes(status.total_bytes)} "
                     f"({status.percent:.0f}%) · updates by itself")


def _state_label(entry, status, active_id: str, runtime) -> str:
    if status.is_active:
        return "Downloading"
    if not entry.is_installed():
        return "Download failed" if status.state == "error" else "Not downloaded"
    if entry.id != active_id:
        return "Downloaded"
    if runtime is not None and runtime.ready:
        return "In use · ready"
    if runtime is not None and runtime.running:
        return "In use · starting"
    return "In use · stopped"


def _render_model_row(entry, report, prefix: str = "ai") -> None:
    """One model: what it is, its state, and its buttons, in one row."""
    from spacemap_assistant import model_installer

    status = model_installer.read_status(entry.id)
    installed = entry.is_installed()
    settings = ai_settings.load_settings()
    if installed and status.state == "done" and not str(settings.get("active_model") or "").strip():
        # With no model chosen yet, the one just downloaded is the one used.
        ai_settings.update_settings(active_model=entry.id, backend="local")
        settings = ai_settings.load_settings()
    active_id = str(settings.get("active_model") or "")
    runtime = _runtime_status() if entry.id == active_id and installed else None
    fits, why = can_run(report, min_vram_gb=entry.min_vram_gb, cpu_capable=entry.cpu_capable)
    state = _state_label(entry, status, active_id, runtime)

    with st.container(border=True):
        info, state_column, actions = st.columns([5, 2, 4], vertical_alignment="center")
        extras = " · reads images" if "image" in entry.modalities else ""
        info.markdown(f"**{entry.name}** · {entry.approx_size_gb:.1f} GB{extras}")
        details = [entry.publisher, entry.license or "license not recorded",
                   "fits this machine" if fits else why]
        if entry.gated:
            details.append("terms must be accepted on Hugging Face")
        info.caption(" · ".join(filter(None, details)))
        state_column.markdown(f"**{state}**")

        buttons = actions.columns(3)
        key = f"{prefix}_{entry.id}"
        if status.is_active:
            if buttons[0].button("Cancel", key=f"{key}_cancel"):
                model_installer.cancel_install(entry.id)
                st.rerun()
        elif not installed:
            needs_token = entry.gated and not ai_settings.read_token()
            if buttons[0].button("Download", key=f"{key}_download", type="primary",
                                 disabled=not entry.repo_id or needs_token,
                                 help="Save a Hugging Face token below first." if needs_token else None):
                _start_download(entry)
            if entry.source == "user" and buttons[1].button("Remove from list", key=f"{key}_forget"):
                remove_user_model(entry.id)
                st.rerun()
        elif entry.id != active_id:
            if buttons[0].button("Use", key=f"{key}_use", type="primary"):
                if runtime is None:
                    from spacemap_assistant.model_runtime import runtime_status, stop_runtime

                    if runtime_status().running:
                        stop_runtime()  # the model in use now makes room for this one
                ai_settings.update_settings(active_model=entry.id, backend="local")
                st.rerun()
        else:
            from spacemap_assistant.model_runtime import stop_runtime

            if runtime is not None and runtime.running:
                if buttons[0].button("Stop", key=f"{key}_stop"):
                    stop_runtime()
                    _log("local_model_stopped", f"{entry.name} was stopped from AI setup.", model=entry.id)
                    st.rerun()
            elif buttons[0].button("Start", key=f"{key}_start", type="primary"):
                _start(entry)
        if installed and not status.is_active:
            with buttons[2].popover("Remove files", key=f"{key}_remove_menu"):
                st.caption(f"Deletes {_human_bytes(entry.installed_size_bytes())} at `{entry.target_dir()}`. "
                           "It can be downloaded again.")
                if st.button("Delete the files", key=f"{key}_remove", type="primary"):
                    if entry.id == active_id:
                        from spacemap_assistant.model_runtime import stop_runtime

                        stop_runtime()
                    model_installer.remove_model(entry)
                    st.rerun()

        if status.is_active:
            _download_progress(entry.id)
        elif status.state == "error":
            st.error(status.message)
        elif status.state == "cancelled":
            st.caption("The download was cancelled; the part already downloaded is kept, and Download resumes it.")


def _render_search(prefix: str) -> None:
    from spacemap.model_catalog import get_model

    with st.expander("Find more models on Hugging Face"):
        st.caption("Any open chat model with safetensors weights can be added, then downloaded and used like "
                   "the ones above. Larger models answer better but need more GPU memory.")
        query_column, kind_column, button_column = st.columns([4, 2, 1], vertical_alignment="bottom")
        query = query_column.text_input("Search", key=f"{prefix}_hf_query",
                                        placeholder="for example: qwen3 instruct, llama 3.1 8b, medgemma")
        kind = kind_column.selectbox("Kind", ["Text", "Text and images"], key=f"{prefix}_hf_kind")
        results_key = f"{prefix}_hf_results"
        if button_column.button("Search", key=f"{prefix}_hf_search", width="stretch"):
            from spacemap_assistant.model_search import ModelSearchError, search_models

            try:
                with st.spinner("Searching Hugging Face..."):
                    st.session_state[results_key] = search_models(query, images=kind != "Text")
            except ModelSearchError as exc:
                st.session_state.pop(results_key, None)
                st.error(str(exc))
        results = st.session_state.get(results_key)
        if results == []:
            st.caption("Nothing found. Try fewer or other words.")
        for index, result in enumerate(results or []):
            from spacemap_assistant.model_search import catalog_record

            record = catalog_record(result)
            try:
                get_model(record["id"])
                present = True
            except Exception:
                present = False
            with st.container(border=True):
                text, button = st.columns([6, 1], vertical_alignment="center")
                text.markdown(f"**{result['repo_id']}** · about {result['size_gb']:.0f} GB "
                              f"({result['parameters_b']:.0f} B parameters)")
                text.caption(" · ".join(filter(None, [f"{result['downloads']:,} downloads", result["license"],
                                                      "needs its terms accepted on Hugging Face"
                                                      if result["gated"] else ""])))
                if button.button("Added" if present else "Add", key=f"{prefix}_hf_add_{index}", disabled=present):
                    try:
                        add_user_model(record)
                    except ModelCatalogError as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()


def _render_models(prefix: str) -> None:
    from spacemap_gui.components.assistant_setup import _render_custom_model, _render_local_import

    report = _hardware()
    st.markdown("#### Models")
    st.caption(f"Downloaded once into `{ai_settings.resolved_models_dir()}`. Nothing leaves this machine while a "
               f"local model answers. This machine: {report.summary()}.")
    models = list_models()
    for entry in models:
        _render_model_row(entry, report, prefix)
    _render_search(prefix)
    with st.expander("More ways to add a model"):
        _render_local_import(models)
        _render_custom_model()


# ---------------------------------------------------------------- 3-5

def _render_storage(prefix: str) -> None:
    from spacemap_gui.components.assistant_setup import _render_token

    with st.expander("Storage and access"):
        models = Path(ai_settings.resolved_models_dir())
        free = free_disk_gb(models if models.exists() else models.parent)
        st.markdown(f"**Models folder:** `{models}`" + (f" ({_space(free)} free)" if free else ""))
        if str(models).startswith(str(Path.home())):
            st.caption("This is inside your home folder. Home folders on clusters are often small (50 GB at "
                       "UTSW); a folder on /work is safer for models of 5-25 GB each.")
        folder = st.text_input("Change the models folder", value=str(ai_settings.load_settings().get("models_dir") or ""),
                               key=f"{prefix}_models_dir", placeholder=str(models),
                               help="Leave blank for the default. Models already downloaded elsewhere are "
                                    "found again when you point this at their folder.")
        if st.button("Save models folder", key=f"{prefix}_models_dir_save"):
            cleaned = folder.strip()
            if cleaned:
                try:
                    Path(cleaned).expanduser().mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    st.error(f"Could not use that folder: {exc}")
                    return
            ai_settings.update_settings(models_dir=cleaned)
            st.rerun()
        _render_token()


def _render_online() -> None:
    with st.expander("Answer with an online model instead (optional)"):
        st.caption("Off unless you choose it here. Online models run on NVIDIA's servers: your question and "
                   "recent chat leave this machine.")
        try:
            from spacemap_gui.components.assistant_panel import _render_provider_controls

            _render_provider_controls()
        except Exception as exc:
            st.warning(f"The online provider settings could not be shown: {exc}")


def _render_advanced(prefix: str) -> None:
    with st.expander("Advanced"):
        current = str(ai_settings.load_settings().get("precision") or "auto")
        options = list(PRECISION_LABELS)
        precision = st.selectbox("GPU memory mode for the local model", options,
                                 index=options.index(current) if current in options else 0,
                                 format_func=PRECISION_LABELS.get, key=f"{prefix}_precision",
                                 help="Used the next time the model starts.")
        if precision != current:
            ai_settings.update_settings(precision=precision)
        tabs = st.tabs(["Answers", "Cluster", "Folder access for AI tools", "Activity and model log"])
        try:
            from spacemap_gui.components import assistant_panel as panel
        except Exception as exc:
            st.warning(f"More settings could not be shown: {exc}")
            return
        with tabs[0]:
            panel._render_model_settings()
        with tabs[1]:
            from spacemap_gui.components.assistant_setup import _render_cluster

            settings = ai_settings.load_settings()
            _render_cluster(settings)
            if (settings.get("cluster") or {}).get("enabled"):
                panel._render_cluster_controls()
        with tabs[2]:
            panel._render_admin_access()
        with tabs[3]:
            panel._render_activity_console(expanded=False)
            from spacemap.userpaths import logs_dir

            log = logs_dir() / "model-server.log"
            if log.is_file():
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
                st.code("\n".join(lines), language="text", wrap_lines=True)
        if ai_settings.is_forced_on():
            st.caption("The assistant is pinned on by `SPACEMAP_ASSISTANT`.")
        elif st.button("Turn the assistant off", key=f"{prefix}_turn_off"):
            ai_settings.set_enabled(False)
            st.rerun()


def render_ai_setup(prefix: str = "ai") -> None:
    """The whole AI setup page (the assistant must be on; Tab 0 shows the switch when it is off)."""
    _render_status(prefix)
    _render_models(prefix)
    _render_online()
    _render_storage(prefix)
    _render_advanced(prefix)


__all__ = ["render_ai_setup"]
