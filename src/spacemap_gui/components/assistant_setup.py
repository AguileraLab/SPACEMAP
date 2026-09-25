"""Tab 0 controls for switching the AI assistant on and installing models.

This panel is the only place a user needs in order to go from a plain SPACEMAP
installation to a working offline assistant, and back again.  It is written so
that it renders correctly on a machine with no AI dependencies, no model files,
no GPU and no network: every capability is probed, and anything missing is
reported with the exact command or link that fixes it.

Only the standard library, Streamlit and the dependency-free ``spacemap``
helpers are imported at module level.  Heavier modules are imported inside the
callbacks that need them, so opening Tab 0 never pulls in torch.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import streamlit as st

from spacemap import ai_settings
from spacemap.hardware import can_run, detect_hardware, free_disk_gb, recommended_precision
from spacemap.model_catalog import ModelCatalogError, list_models, remove_user_model
from spacemap.userpaths import describe_layout

#: The extra that installs everything a local model needs.


def _human_bytes(value: float) -> str:
    """Format a byte count for display."""
    if value <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.1f} TB"


def _render_disabled_state() -> None:
    """What the user sees while the assistant is switched off."""
    st.info(
        "The AI assistant is **off**. SPACEMAP runs as a pipeline-only application: "
        "no AI dependencies are loaded, no model runs, and nothing leaves this machine."
    )
    st.markdown(
        "Switching it on adds an **AI Assistant** page that can answer questions about "
        "the pipeline, read your project layout, and propose parameters for steps such as "
        "registration and ROI extraction. Every action it proposes still needs your approval."
    )
    if st.button("Enable AI assistant", type="primary", key="ai_setup_enable"):
        ai_settings.set_enabled(True)
        st.rerun()


def _render_dependency_status() -> bool:
    """Show which packages the local model needs, which are installed, and how to add the rest.

    Returns True when everything required is installed.
    """
    report = ai_settings.dependency_report()
    missing = [item for item in report if item["required"] and not item["installed"]]
    optional = [item for item in report if not item["required"] and not item["installed"]]
    environment = ai_settings.environment_name()
    if not missing:
        st.success(f"Everything the local model needs is installed in `{environment}`.")
        if optional:
            st.caption(
                "Optional: `bitsandbytes` lets a large model fit a smaller NVIDIA GPU (8- or 4-bit). "
                "Install it only if a model does not fit: "
                f"`{ai_settings.install_command([(item['name'], item['wanted']) for item in optional])}`"
            )
        return True

    st.warning(f"The local model needs {len(missing)} more package(s) in the environment that runs SPACEMAP.")
    st.markdown(
        f"The assistant's model runs **inside the environment that already runs SPACEMAP** (`{environment}`), "
        "not in a separate one. Packages it has already are not installed again."
    )
    st.dataframe(
        [{"Package": item["name"], "Needed": "yes" if item["required"] else "optional",
          "Installed": item["installed"] or "missing"} for item in report],
        hide_index=True, width="stretch",
    )
    st.markdown(
        "Run this in a terminal on a machine with internet access (the cluster's login node is fine: "
        "the environment is on shared storage, so every node sees the new packages). It names SPACEMAP's "
        "own Python, so the packages go into the right environment whatever it is called:"
    )
    st.code(ai_settings.install_command([(item["name"], item["wanted"]) for item in missing]), language="bash")
    st.caption(
        f"The same with conda: `conda activate {environment}`, then "
        f"`python -m pip install {' '.join(repr(item['name'] + item['wanted']) for item in missing)}`. "
        "Then press **Re-check** below; no restart is needed."
    )
    st.caption(
        "For a small quantized model on a machine without a GPU, install "
        f"`{ai_settings.install_command([(n, v) for n, v, _, _ in ai_settings.GGUF_PACKAGES])}` instead. "
        "The assistant can also use a remote API provider without any of these packages."
    )
    if st.button("Re-check", key="ai_setup_recheck"):
        st.rerun()
    return False


def _render_hardware(report) -> None:
    """Report what this machine offers, so model choice is informed."""
    st.markdown(f"**This machine:** {report.summary()}")
    if not report.has_gpu:
        st.caption(
            "No GPU was detected. Models still run, using the CPU and system memory, but a "
            "reply can take minutes rather than seconds. The small quantized model in the "
            "library below is the practical choice here."
        )
    elif not report.torch_available:
        st.caption("PyTorch is not installed yet, so GPU details could not be read.")


def _render_model_row(entry, report, token: str) -> None:
    """One model in the library, with its install controls."""
    from spacemap_assistant import model_installer

    status = model_installer.read_status(entry.id)
    installed = entry.is_installed()
    runnable, reason = can_run(
        report, min_vram_gb=entry.min_vram_gb, cpu_capable=entry.cpu_capable
    )

    label = f"{entry.name} — {entry.approx_size_gb:.1f} GB"
    if installed:
        label = f"{label} (installed)"
    elif status.is_active:
        label = f"{label} (downloading)"

    with st.expander(label, expanded=status.is_active):
        if entry.publisher:
            st.caption(f"{entry.publisher} · {entry.license or 'license not recorded'} · {entry.backend}")
        if entry.notes:
            st.markdown(entry.notes)
        st.caption(("Ready: " if runnable else "Not ready: ") + reason)
        if entry.source == "user":
            st.caption("Added by you, in the user catalog.")

        if status.is_active:
            st.progress(
                min(1.0, status.percent / 100.0),
                text=f"{_human_bytes(status.downloaded_bytes)} of about "
                f"{_human_bytes(status.total_bytes)} ({status.percent:.0f}%)",
            )
            left, right = st.columns(2)
            if left.button("Refresh", key=f"ai_refresh_{entry.id}"):
                st.rerun()
            if right.button("Cancel", key=f"ai_cancel_{entry.id}"):
                model_installer.cancel_install(entry.id)
                st.rerun()
            return

        if status.state == "error":
            st.error(status.message)
        elif status.state == "cancelled":
            st.warning("The download was cancelled. Partly downloaded files were kept.")
        elif status.state == "done" and installed:
            st.success(status.message)

        if installed:
            size = entry.installed_size_bytes()
            st.caption(f"On disk: {_human_bytes(size)} at `{entry.target_dir()}`")
            left, right = st.columns(2)
            if left.button("Use this model", key=f"ai_use_{entry.id}", type="primary"):
                ai_settings.update_settings(active_model=entry.id, backend="local")
                st.success(f"{entry.name} is now the active model.")
                st.rerun()
            if right.button("Remove files", key=f"ai_remove_{entry.id}"):
                model_installer.remove_model(entry)
                st.rerun()
        else:
            required = entry.approx_size_gb * 1.1
            available = free_disk_gb(ai_settings.resolved_models_dir())
            if available and available < required:
                st.error(
                    f"Needs about {required:.0f} GB free and the models folder has "
                    f"{available:.0f} GB. Free space or change the models folder above."
                )
            if entry.gated:
                st.caption(
                    "This model is gated. Accept its terms while signed in to Hugging Face, "
                    "then save an access token below."
                    + (f" Terms: {entry.license_url}" if entry.license_url else "")
                )
            if st.button("Download", key=f"ai_install_{entry.id}", disabled=not entry.repo_id):
                try:
                    model_installer.start_install(entry, token=token)
                except model_installer.ModelInstallError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()

        if status.state in {"error", "cancelled", "done"}:
            if st.button("Clear message", key=f"ai_clear_{entry.id}"):
                model_installer.clear_status(entry.id)
                st.rerun()

        if entry.source == "user" and st.button("Remove from catalog", key=f"ai_forget_{entry.id}"):
            remove_user_model(entry.id)
            st.rerun()


def _render_local_import(models) -> None:
    """Install a model from a folder that is already on this machine."""
    with st.expander("Install from a folder on this machine (no network)"):
        st.markdown(
            "Use this on an offline machine, or when the weights were copied from "
            "elsewhere. The files are placed in the models folder so SPACEMAP can find them."
        )
        if not models:
            st.caption("Add a catalog entry first.")
            return
        choice = st.selectbox(
            "Catalog entry to fill",
            options=[entry.id for entry in models],
            format_func=lambda value: next(e.name for e in models if e.id == value),
            key="ai_import_target",
        )
        source = st.text_input(
            "Folder or file containing the weights",
            key="ai_import_source",
            placeholder="/path/to/downloaded/model",
        )
        move = st.checkbox(
            "Move instead of copy", key="ai_import_move", help="Faster, but empties the source."
        )
        if st.button("Install from folder", key="ai_import_run"):
            from spacemap_assistant import model_installer

            entry = next(item for item in models if item.id == choice)
            try:
                target = model_installer.import_local_model(entry, source, move=move)
            except model_installer.ModelInstallError as exc:
                st.error(str(exc))
            else:
                st.success(f"Installed into {target}")
                st.rerun()


def _render_custom_model() -> None:
    """Add a model the built-in catalog does not know about."""
    with st.expander("Add another model"):
        st.markdown(
            "Any Hugging Face model that works with `transformers`, or any GGUF file, "
            "can be added here. Entries are stored in your own catalog and survive upgrades."
        )
        with st.form("ai_add_model"):
            identifier = st.text_input("Short id", placeholder="my-model-8b")
            name = st.text_input("Display name", placeholder="My Model 8B Instruct")
            backend = st.selectbox("Backend", options=["transformers", "gguf"])
            repo_id = st.text_input("Hugging Face repository", placeholder="org/model-name")
            filename = st.text_input(
                "File name (GGUF only)", placeholder="model-Q4_K_M.gguf"
            )
            size = st.number_input("Approximate size (GB)", min_value=0.0, value=8.0, step=0.5)
            vram = st.number_input("Suggested GPU memory (GB, 0 for CPU)", min_value=0.0, value=0.0, step=1.0)
            gated = st.checkbox("Requires accepting terms on Hugging Face")
            submitted = st.form_submit_button("Add to catalog")

        if submitted:
            from spacemap.model_catalog import add_user_model

            try:
                add_user_model(
                    {
                        "id": identifier.strip(),
                        "name": name.strip() or identifier.strip(),
                        "backend": backend,
                        "repo_id": repo_id.strip(),
                        "filename": filename.strip(),
                        "local_dir": repo_id.strip() or identifier.strip(),
                        "approx_size_gb": float(size),
                        "min_vram_gb": float(vram),
                        "gated": bool(gated),
                    }
                )
            except ModelCatalogError as exc:
                st.error(str(exc))
            else:
                st.success(f"Added {identifier}.")
                st.rerun()


def _render_storage(settings) -> None:
    """Where models and assistant data are kept."""
    with st.expander("Storage locations"):
        layout = describe_layout()
        st.caption(
            "All writable state lives outside the installed package. Set `SPACEMAP_HOME` "
            "to move everything at once."
        )
        for key in ("home", "config", "models", "assistant_data", "logs"):
            st.text(f"{key:15} {layout[key]}")

        current = str(settings.get("models_dir") or "")
        new_dir = st.text_input(
            "Models folder (blank for the default)",
            value=current,
            key="ai_models_dir",
            help="Point this at a large disk or a shared read-only model store.",
        )
        if st.button("Save models folder", key="ai_models_dir_save"):
            cleaned = new_dir.strip()
            if cleaned:
                candidate = Path(cleaned).expanduser()
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    st.error(f"Could not use that folder: {exc}")
                    return
            ai_settings.update_settings(models_dir=cleaned)
            st.success("Saved.")
            st.rerun()


def _render_token() -> str:
    """Collect the Hugging Face token used for gated downloads."""
    token = ai_settings.read_token()
    with st.expander("Hugging Face access token" + (" (saved)" if token else "")):
        st.markdown(
            "Only needed to download a **gated** model. Create a read token at "
            "<https://huggingface.co/settings/tokens> after accepting the model's terms. "
            "It is stored in your configuration directory, readable only by you."
        )
        entered = st.text_input("Token", value="", type="password", key="ai_token_input")
        left, right = st.columns(2)
        if left.button("Save token", key="ai_token_save"):
            if entered.strip():
                ai_settings.write_token(entered)
                st.success("Saved.")
                st.rerun()
            else:
                st.warning("Enter a token first.")
        if right.button("Forget token", key="ai_token_clear", disabled=not token):
            ai_settings.clear_token()
            st.rerun()
    return token


def _render_cluster(settings) -> None:
    """Optional job-scheduler settings for sites that have one."""
    cluster = settings.get("cluster", {})
    with st.expander("Run model jobs on a cluster (optional)"):
        st.caption(
            "If this machine is a login node of an HPC cluster with Slurm, the assistant "
            "can submit the model server as a job instead of running it here. Leave this "
            "off on a workstation or laptop."
        )
        available = bool(shutil.which("sbatch"))
        if not available:
            st.caption("`sbatch` was not found on this machine, so this stays unavailable.")
        enabled = st.checkbox(
            "Submit model jobs with Slurm",
            value=bool(cluster.get("enabled")) and available,
            disabled=not available,
            key="ai_cluster_enabled",
        )
        partition = st.text_input("Partition", value=str(cluster.get("partition") or ""), key="ai_cluster_partition")
        account = st.text_input("Account (optional)", value=str(cluster.get("account") or ""), key="ai_cluster_account")
        time_limit = st.text_input("Time limit", value=str(cluster.get("time_limit") or "04:00:00"), key="ai_cluster_time")
        if st.button("Save cluster settings", key="ai_cluster_save"):
            ai_settings.update_settings(
                cluster={
                    "enabled": bool(enabled),
                    "scheduler": "slurm",
                    "partition": partition.strip(),
                    "account": account.strip(),
                    "time_limit": time_limit.strip() or "04:00:00",
                }
            )
            st.success("Saved.")
            st.rerun()


def render_assistant_setup() -> None:
    """Render the whole AI assistant section of Tab 0."""
    st.subheader("AI Assistant (optional)")

    if ai_settings.is_forced_off():
        st.info(
            "The AI assistant has been disabled for this installation by setting "
            "`SPACEMAP_ASSISTANT=off`. Unset that variable to make it available."
        )
        return

    if not ai_settings.is_enabled():
        _render_disabled_state()
        return

    settings = ai_settings.load_settings()
    active = str(settings.get("active_model") or "")

    header, control = st.columns([3, 1], vertical_alignment="center")
    with header:
        if active:
            st.success(f"The assistant is **on**. Active model: `{active}`.")
        else:
            st.warning(
                "The assistant is **on** but no model is selected yet. Install one below, "
                "or choose a remote API provider on the assistant page."
            )
    with control:
        if ai_settings.is_forced_on():
            st.caption("Pinned on by `SPACEMAP_ASSISTANT`.")
        elif st.button("Turn off", key="ai_setup_disable", width="stretch"):
            ai_settings.set_enabled(False)
            st.rerun()

    st.divider()
    dependencies_ready = _render_dependency_status()
    report = detect_hardware()
    _render_hardware(report)

    if dependencies_ready and report.has_gpu:
        st.caption(
            "Suggested precision for a 12 B model here: "
            f"`{recommended_precision(report, 23.9)}`."
        )

    token = _render_token()
    _render_storage(settings)

    st.markdown("#### Model library")
    st.caption(
        "Models are downloaded once and reused. Nothing is sent anywhere while a local "
        "model answers a question."
    )
    models = list_models()
    for entry in models:
        _render_model_row(entry, report, token)

    _render_local_import(models)
    _render_custom_model()
    _render_cluster(settings)


__all__ = ["render_assistant_setup"]
