"""The classic chat uses the same checked viewer requests as Studio."""

import re

import streamlit as st

from . import assistant, commands

PENDING = "_studio_chat_pending"
LAST = "_studio_chat_last"


def answer(result):
    text = result.get("message", "")
    if result.get("corrections"):
        text += "\n\n" + "; ".join(item["requested"] + " → " + item["matched"] for item in result["corrections"])
    data = result.get("data") or {}
    if "counts" in data:
        text += "\n\n" + "\n".join(f"- {name}: {count:,}" for name, count in data["counts"].items())
    if "heatmap" in data:
        heat = data["heatmap"]
        text += "\n\n" + heat["note"] + "\n\n| Marker | Selected | Other cells |\n| --- | ---: | ---: |"
        for index, marker in enumerate(heat["markers"]):
            values = ["—" if row[index] is None else f"{row[index]:.3g}" for row in heat["values"]]
            text += "\n| " + marker.replace("|", "\\|") + " | " + " | ".join(values) + " |"
    return text


def present(result):
    if result.get("status") != "done":
        return
    st.session_state[LAST] = result
    if result["name"] not in {"count_cells", "save_thresholds", "selection_heatmap"}:
        from spacemap_gui.viewer import ui

        if not ui.viewer_window_open():
            st.session_state["_viewer_command_result"] = result
            st.session_state[ui._OPEN_KEY] = True


def submit(prompt, page_id, store, conversation_id):
    command = assistant.parse(prompt)
    explicit = bool(re.search(r"\b(?:in|using) (?:the )?(?:viewer|studio)\b", prompt, re.I))
    # Existing chat handles filesystem paths and scientific pipeline requests.
    if not command and not explicit:
        return False
    if command and command["name"] == "open_image" and re.search(r"[/\\]", command["args"]["image"]):
        return False
    from spacemap_gui.viewer import ui

    if not ui.studio_link(page_id):
        return False
    channel = ui._session_id()
    if command and not explicit:
        from spacemap_gui.viewer import registry
        from .names import normalise

        if command["name"] == "open_image" and not re.search(r"\bimage\b", prompt, re.I):
            wanted = normalise(command["args"]["image"])
            if not any(
                wanted in {normalise(item["name"]), normalise(item["sample"])} for item in ui.window_images(channel)
            ):
                return False
        if command["name"] == "show_markers" and not re.search(r"\bmarkers?\b", prompt, re.I):
            from spacemap.viewer.sources import image_info

            view = registry.session_view(channel, mark_seen=False)
            image = next((layer for layer in view["spec"].get("layers", []) if layer["type"] == "image"), None)
            names = (
                {normalise(name) for name in image_info(registry.resolve(image["token"])).channel_names}
                if image
                else set()
            )
            if not all(normalise(name) in names for name in command["args"]["markers"]):
                return False
    try:
        proposed = assistant.propose(channel, prompt, use_model=explicit)
        if proposed["status"] != "ready":
            result = proposed
        else:
            command = proposed["command"]
            result = commands.execute(channel, command)
            if result["status"] in {"needs_choice", "needs_confirmation"}:
                st.session_state[PENDING] = {
                    "channel": channel,
                    "command": command,
                    "result": result,
                    "conversation": conversation_id,
                }
            else:
                st.session_state.pop(PENDING, None)
            present(result)
        store.add_message(
            conversation_id,
            "assistant",
            answer(result),
            metadata={"source": "studio_command", "command_id": result.get("command_id")},
        )
    except (ValueError, RuntimeError, OSError) as exc:
        store.add_message(conversation_id, "assistant", str(exc), metadata={"source": "studio_command"})
    return True


def controls(store, conversation_id, key):
    pending = st.session_state.get(PENDING)
    if pending and pending["conversation"] == conversation_id:
        result = pending["result"]
        command = pending["command"]
        options = {}
        if result["status"] == "needs_choice":
            choices = result.get("choices") or []
            if not choices:
                return
            choice = st.selectbox(result["message"], choices, key=key + "_studio_choice")
            args = {**command["args"]}
            argument = {"marker set": "set"}.get(result["argument"], result["argument"])
            if argument == "marker" and command["name"] == "show_markers":
                args["markers"] = [choice if value == result["requested"] else value for value in args["markers"]]
            elif argument == "marker" and command["name"] == "select_cells":
                args["rule"] = {
                    **args["rule"],
                    "conditions": [
                        {**condition, "marker": choice} if condition["marker"] == result["requested"] else condition
                        for condition in args["rule"]["conditions"]
                    ],
                }
            elif argument == "rule":
                args["rule_name"] = choice
            elif argument == "phenotype" and command["name"] == "filter_cells":
                args["phenotypes"] = [choice if value == result["requested"] else value for value in args["phenotypes"]]
            else:
                args[argument] = choice
            command = {**command, "args": args}
        else:
            st.caption("This action will save changes to your project.")
            options = {"confirmed": True}
        if st.button("Apply", key=key + "_studio_apply"):
            result = commands.execute(pending["channel"], {**command, **options})
            if result["status"] in {"needs_choice", "needs_confirmation"}:
                st.session_state[PENDING] = {**pending, "command": command, "result": result}
            else:
                st.session_state.pop(PENDING, None)
            present(result)
            store.add_message(conversation_id, "assistant", answer(result), metadata={"source": "studio_command"})
            st.rerun()
    last = st.session_state.get(LAST)
    if last and last.get("undo") and st.button("Undo last viewer action", key=key + "_studio_undo"):
        from spacemap_gui.viewer import ui

        result = commands.execute(
            ui._session_id(), {"name": "undo", "args": {"command_id": last["command_id"]}, "confirmed": True}
        )
        present(result)
        store.add_message(conversation_id, "assistant", answer(result), metadata={"source": "studio_command"})
        st.rerun()
