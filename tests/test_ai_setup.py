"""The AI setup page: one place, the right next step, model search, and starting by itself."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spacemap import ai_settings


@pytest.fixture
def setup_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SPACEMAP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPACEMAP_ASSISTANT", "on")
    return tmp_path


def _page(prefix):
    import streamlit as st

    from spacemap_gui.components import ai_setup

    st.session_state.setdefault("ran", 0)
    ai_setup._render_status(prefix)


def test_the_next_step_follows_the_state(setup_home, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from spacemap.model_catalog import ModelEntry
    from spacemap_gui.components import ai_setup

    monkeypatch.setattr(ai_settings, "dependency_report", lambda packages=None: [
        {"name": "torch", "wanted": ">=2", "required": True, "installed": "2.12"}])
    monkeypatch.setattr(ai_setup, "_runtime_status", lambda: SimpleNamespace(ready=False, running=False))
    downloaded = {"ids": set()}
    monkeypatch.setattr(ModelEntry, "is_installed", lambda self: self.id in downloaded["ids"])

    app = AppTest.from_function(_page, args=("t",), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert any(b.label.startswith("Download ") for b in app.button)  # nothing downloaded: suggest one

    from spacemap.model_catalog import list_models

    first = list_models()[0]
    downloaded["ids"].add(first.id)
    app.run()
    assert [b.label for b in app.button] == [f"Use {first.name}"]
    app.button[0].click().run()
    assert ai_settings.load_settings()["active_model"] == first.id
    assert [b.label for b in app.button] == [f"Start {first.name}"]  # or just ask: it starts by itself
    assert any("starts by itself" in m.value for m in app.markdown)


def test_packages_missing_shows_the_one_command(setup_home, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ai_settings, "dependency_report", lambda packages=None: [
        {"name": "transformers", "wanted": ">=5.0", "required": True, "installed": None}])
    app = AppTest.from_function(_page, args=("t",), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert any('"transformers>=5.0"' in block.value for block in app.code)


def test_search_keeps_loadable_models_and_makes_catalog_entries():
    from spacemap_assistant.model_search import catalog_record, search_models

    class Api:
        def list_models(self, **kwargs):
            assert kwargs["pipeline_tag"] == "text-generation" and kwargs["sort"] == "downloads"
            return [
                SimpleNamespace(id="Qwen/Qwen3-8B", downloads=900_000, likes=10, gated=False,
                                safetensors=SimpleNamespace(total=8_190_000_000, parameters={"BF16": 8_190_000_000}),
                                card_data={"license": "apache-2.0"}, tags=[]),
                SimpleNamespace(id="someone/model-GGUF", downloads=5, likes=0, gated=False, safetensors=None,
                                card_data=None, tags=[]),  # no safetensors: the model server cannot load it
                SimpleNamespace(id="meta-llama/Llama-3.1-8B-Instruct", downloads=800_000, likes=5, gated="manual",
                                safetensors=SimpleNamespace(total=8_030_000_000, parameters={"BF16": 8_030_000_000}),
                                card_data=None, tags=["license:llama3.1"]),
            ]

    found = search_models("8b", api=Api())
    assert [r["repo_id"] for r in found] == ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"]
    assert found[0]["size_gb"] == 16.4 and found[0]["license"] == "apache-2.0"
    assert found[1]["gated"] and found[1]["license"] == "llama3.1"
    record = catalog_record(found[0])
    assert record["id"] == "qwen-qwen3-8b" and record["repo_id"] == record["local_dir"] == "Qwen/Qwen3-8B"
    assert record["backend"] == "transformers" and record["min_vram_gb"] == 18.9
    from spacemap.model_catalog import _coerce

    assert _coerce(record, source="user").name == "Qwen3-8B"  # a valid catalog entry


def test_asking_a_question_starts_the_chosen_model(monkeypatch):
    from spacemap_assistant import model_runtime

    states = iter([SimpleNamespace(ready=False, installed=True, running=False, message="stopped"),
                   SimpleNamespace(ready=False, installed=True, running=True, message="loading"),
                   SimpleNamespace(ready=True, installed=True, running=True, message="ready")])
    started, waited = [], []
    monkeypatch.setattr(model_runtime, "runtime_status", lambda: next(states))
    monkeypatch.setattr(model_runtime, "start_runtime", lambda precision: started.append(precision))
    monkeypatch.setattr(model_runtime.time, "sleep", lambda seconds: None)
    status = model_runtime.ensure_running(precision="auto", on_wait=lambda s, t: waited.append(s.message))
    assert status.ready and started == ["auto"] and waited == ["loading"]


def test_a_model_that_stops_while_loading_is_reported(monkeypatch):
    from spacemap_assistant import model_runtime

    states = iter([SimpleNamespace(ready=False, installed=True, running=False, message="stopped"),
                   SimpleNamespace(ready=False, installed=True, running=False, message="stopped")])
    monkeypatch.setattr(model_runtime, "runtime_status", lambda: next(states))
    monkeypatch.setattr(model_runtime, "start_runtime", lambda precision: None)
    monkeypatch.setattr(model_runtime, "_log_tail", lambda lines=12: "CUDA out of memory")
    with pytest.raises(model_runtime.ModelRuntimeError, match="out of memory"):
        model_runtime.ensure_running(precision="auto")
