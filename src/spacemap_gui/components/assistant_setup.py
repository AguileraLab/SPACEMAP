"""Tab 0 controls for switching the AI assistant on, and pieces of the AI setup page.

The page itself is spacemap_gui.components.ai_setup; this module keeps the
on/off switch, the package check, the Hugging Face token, adding a model by
hand or from a folder, and the cluster settings, which that page reuses.

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

import streamlit as st

from spacemap import ai_settings
from spacemap.model_catalog import ModelCatalogError

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
    """Tab 0's AI Assistant section: the on/off switch, then the whole AI setup page."""
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

    st.caption("Everything the assistant needs is on this page; the assistant page's Setup tab shows the same.")
    from spacemap_gui.components.ai_setup import render_ai_setup

    render_ai_setup("tab0")


__all__ = ["render_assistant_setup"]
