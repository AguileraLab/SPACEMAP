"""The environment SPACEMAP runs in: its own modules must not break the libraries it shares."""

from __future__ import annotations

import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"


def test_spacemap_keeps_the_built_in_pairwise():
    built_in = itertools.pairwise
    import spacemap.utils  # noqa: F401  (used to replace it for palom)

    assert itertools.pairwise is built_in


def test_the_model_server_loads_after_spacemap():
    # Replacing itertools.pairwise made torch's compiler refuse to load, and with
    # it transformers 5 and the assistant's model server. Run in a fresh process,
    # in the order the assistant uses: SPACEMAP first, then the model libraries.
    script = ("import spacemap.utils, itertools; "
              "import spacemap_assistant.model_service.server; "
              "print(type(itertools.pairwise).__name__)")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
                          env={**__import__("os").environ, "PYTHONPATH": str(SOURCE)})
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "type"


def test_a_temporary_xdg_data_home_is_not_used(tmp_path, monkeypatch):
    # UTSW BioHPC sets XDG_DATA_HOME=/tmp/$USER for everyone; /tmp is per node.
    from spacemap import userpaths

    old = Path(tempfile.mkdtemp(prefix="spacemap-xdg-"))  # inside the system temporary folder
    (old / "spacemap" / "config").mkdir(parents=True)
    (old / "spacemap" / "config" / "assistant.json").write_text('{"enabled": true}')
    token = old / "spacemap" / "config" / "huggingface_token"
    token.write_text("hf_x")
    token.chmod(0o600)
    (old / "spacemap" / "state").mkdir()
    (old / "spacemap" / "state" / "interface_state.json").write_text("{}")
    (old / "spacemap" / "models").mkdir()
    home = tmp_path / "home"
    monkeypatch.delenv("SPACEMAP_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(old))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setitem(userpaths._CARRIED, "done", False)
    try:
        assert userpaths.spacemap_home() == home / ".spacemap"
        assert (home / ".spacemap" / "config" / "assistant.json").read_text() == '{"enabled": true}'
        assert oct((home / ".spacemap" / "config" / "huggingface_token").stat().st_mode & 0o777) == "0o600"
        assert not (home / ".spacemap" / "state").exists()  # the old interface cache is not brought back
        assert not (home / ".spacemap" / "models").exists()  # nor models
        (home / ".spacemap" / "config" / "assistant.json").write_text('{"enabled": false}')
        monkeypatch.setitem(userpaths._CARRIED, "done", False)
        userpaths.spacemap_home()
        assert (home / ".spacemap" / "config" / "assistant.json").read_text() == '{"enabled": false}'  # never overwritten
    finally:
        shutil.rmtree(old, ignore_errors=True)

    kept = tmp_path / "data"  # a lasting XDG_DATA_HOME is still honoured
    monkeypatch.setenv("XDG_DATA_HOME", str(kept))
    if not userpaths.is_temporary(kept):
        assert userpaths.spacemap_home() == kept / "spacemap"
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "explicit"))
    assert userpaths.spacemap_home() == tmp_path / "explicit"  # an explicit choice always wins


def test_the_chat_asks_for_the_model_under_the_name_the_server_was_started_with(monkeypatch):
    from spacemap import ai_settings
    from spacemap.model_catalog import list_models
    from spacemap_assistant import model_client, model_runtime

    entry = next(e for e in list_models() if e.backend != "gguf")
    monkeypatch.setattr(ai_settings, "load_settings", lambda: {"active_model": entry.id})
    command = model_runtime._build_command(entry, "bf16")
    assert model_client.local_model_id() == command[command.index("--model-id") + 1]


def test_thinking_markers_never_reach_the_answer():
    from spacemap_assistant.model_service.server import _AnswerFilter, clean_answer

    cases = {"thought\n<channel|>CD20": "CD20",  # seen from Gemma 4 12B on 2026-09-25
             "<|channel>thought\nlet me see\n<channel|>A CD8 T cell kills.": "A CD8 T cell kills.",
             "A CD8 T cell kills infected cells.": "A CD8 T cell kills infected cells.",
             "thoughtful answer": "thoughtful answer"}
    for raw, wanted in cases.items():
        assert clean_answer(raw) == wanted
        streamed = _AnswerFilter()
        assert ("".join(streamed.feed(ch) for ch in raw) + streamed.flush()).strip() == wanted
