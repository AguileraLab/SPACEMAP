import tempfile
from pathlib import Path

from spacemap_assistant.paths import ASSISTANT_DATA_DIR
from spacemap_assistant.storage import AssistantStore


def test_chat_memory_activity_and_exports_are_durable() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        store = AssistantStore(Path(temp_dir) / "assistant.sqlite3")
        conversation = store.create_conversation(project_root="example")
        assert store.get_conversation(conversation)["message_count"] == 0
        store.add_message(conversation, "user", "Explain Leiden")
        store.add_message(conversation, "assistant", "Leiden groups similar cells.")
        assert store.get_conversation(conversation)["message_count"] == 2
        assert store.list_conversations()[0]["id"] == conversation
        older_chat = store.create_conversation(project_root="example")
        store.add_message(older_chat, "user", "Use Leiden resolution 0.4 for panel alpha")
        related = store.relevant_prior_user_messages(
            "What Leiden setting did I use for alpha?",
            exclude_conversation_id=conversation,
        )
        assert related[0]["conversation_id"] == older_chat
        assert "resolution 0.4" in related[0]["content"]
        messages = store.get_messages(conversation)
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert "Explain Leiden" in store.conversation_markdown(conversation)
        assert '"role": "assistant"' in store.conversation_jsonl(conversation)

        store.log_activity("test", "Storage works")
        assert store.list_activity()[-1]["message"] == "Storage works"

        memory_id = store.add_memory(
            "CD3 is a T-cell marker for panel A.", status="proposed", source="user"
        )
        assert store.relevant_verified_memories("CD3 panel") == []
        store.set_memory_status(memory_id, "verified")
        memories = store.relevant_verified_memories("CD3 panel")
        assert memories[0]["id"] == memory_id
        assert store.add_memory_if_new(
            "  cd3 IS a t-cell marker for panel a.  ", status="proposed"
        ) is None
        store.update_memory(
            memory_id,
            "CD3 is the T-cell marker for panel A.",
            scope="lab",
            kind="marker_alias",
        )
        updated = next(item for item in store.list_memories() if item["id"] == memory_id)
        assert updated["scope"] == "lab"
        assert updated["kind"] == "marker_alias"

        portable = store.export_memories_jsonl()
        with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as second_dir:
            second_store = AssistantStore(Path(second_dir) / "assistant.sqlite3")
            assert second_store.import_memories_jsonl(portable) == 1
            imported = second_store.list_memories()[0]
            assert imported["status"] == "proposed"
            assert imported["content"] == updated["content"]

        artifact_path = Path(temp_dir) / "qc.png"
        artifact_path.write_bytes(b"test")
        artifact_id = store.register_artifact(
            artifact_path,
            kind="registration_qc_image",
            caption="QC preview",
            metadata={"action_id": "example"},
        )
        artifacts = store.list_artifacts(kind="registration_qc_image")
        assert artifacts[0]["id"] == artifact_id
        assert artifacts[0]["metadata"]["action_id"] == "example"


def test_chat_deletion_and_gallery_clearing_do_not_delete_output_files() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        store = AssistantStore(Path(temp_dir) / "assistant.sqlite3")
        chat_one = store.create_conversation()
        chat_two = store.create_conversation()
        store.add_message(chat_one, "user", "Show this image")
        store.add_message(chat_two, "user", "Keep this chat")
        output = Path(temp_dir) / "preview.png"
        output.write_bytes(b"preview")
        store.register_artifact(
            output,
            kind="assistant_image_preview",
            metadata={"conversation_id": chat_one},
        )

        assert store.delete_conversation_artifacts(
            chat_one, kind="assistant_image_preview"
        ) == 1
        assert output.is_file()

        store.register_artifact(
            output,
            kind="assistant_image_preview",
            metadata={"conversation_id": chat_one},
        )
        result = store.delete_conversation(chat_one)
        assert result == {"messages": 1, "artifacts": 1}
        assert not store.conversation_exists(chat_one)
        assert store.conversation_exists(chat_two)
        assert output.is_file()


def test_saved_chat_exports_can_be_deleted_separately() -> None:
    ASSISTANT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ASSISTANT_DATA_DIR) as temp_dir:
        store = AssistantStore(Path(temp_dir) / "assistant.sqlite3")
        conversation = store.create_conversation()
        store.add_message(conversation, "user", "Export me")
        paths = store.save_conversation_exports(conversation, "delete_test")
        assert all(path.is_file() for path in paths.values())
        removed = store.delete_conversation_exports(conversation)
        assert {path.suffix for path in removed} == {".md", ".jsonl"}
        assert all(not path.exists() for path in paths.values())
