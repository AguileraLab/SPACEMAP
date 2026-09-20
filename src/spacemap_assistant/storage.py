"""Durable, portable storage for conversations, activity, memory, and artifacts."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ASSISTANT_DATA_DIR, ensure_assistant_layout, require_assistant_write_path


DEFAULT_DATABASE_PATH = ASSISTANT_DATA_DIR / "database" / "assistant.sqlite3"

_SEARCH_STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "can",
    "could",
    "for",
    "from",
    "have",
    "how",
    "into",
    "just",
    "like",
    "need",
    "please",
    "should",
    "that",
    "the",
    "then",
    "this",
    "using",
    "want",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _decode(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return (cleaned or fallback)[:100]


def _search_terms(value: str) -> set[str]:
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+-]{1,}", str(value or ""))
    }
    return {term for term in terms if term not in _SEARCH_STOP_WORDS and len(term) >= 3}


class AssistantStore:
    """Small SQLite store. Large images and scientific tables stay as artifacts."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH):
        ensure_assistant_layout()
        self.database_path = require_assistant_write_path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 20000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    project_root TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_time
                ON messages(conversation_id, created_at);

                CREATE TABLE IF NOT EXISTS activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('proposed', 'verified', 'superseded')),
                    confidence REAL,
                    conditions_json TEXT NOT NULL DEFAULT '{}',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    caption TEXT NOT NULL DEFAULT '',
                    source_page TEXT NOT NULL DEFAULT '',
                    source_command TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_conversation(self, title: str = "New conversation", project_root: str = "") -> str:
        conversation_id = str(uuid.uuid4())
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at, project_root) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, title.strip() or "New conversation", now, now, project_root),
            )
        return conversation_id

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row is not None

    def list_conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT conversations.*, COUNT(messages.id) AS message_count
                FROM conversations
                LEFT JOIN messages ON messages.conversation_id = conversations.id
                GROUP BY conversations.id
                ORDER BY conversations.updated_at DESC LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT conversations.*, COUNT(messages.id) AS message_count
                FROM conversations
                LEFT JOIN messages ON messages.conversation_id = conversations.id
                WHERE conversations.id = ?
                GROUP BY conversations.id
                """,
                (conversation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def delete_conversation(self, conversation_id: str) -> dict[str, int]:
        """Delete one saved chat and its gallery links, but never delete output files."""
        target = str(conversation_id)
        if not self.conversation_exists(target):
            raise KeyError(target)
        artifact_ids = [
            item["id"]
            for item in self.list_artifacts(limit=100_000)
            if str((item.get("metadata") or {}).get("conversation_id") or "") == target
        ]
        with self._connect() as connection:
            message_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (target,)
                ).fetchone()[0]
            )
            if artifact_ids:
                placeholders = ",".join("?" for _item in artifact_ids)
                connection.execute(
                    f"DELETE FROM artifacts WHERE id IN ({placeholders})", artifact_ids
                )
            connection.execute("DELETE FROM conversations WHERE id = ?", (target,))
        return {"messages": message_count, "artifacts": len(artifact_ids)}

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported message role: {role}")
        if not self.conversation_exists(conversation_id):
            raise KeyError(f"Unknown conversation: {conversation_id}")
        message_id = str(uuid.uuid4())
        now = utc_now()
        clean_content = str(content).replace("\x00", "").strip()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(id, conversation_id, role, content, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, clean_content, now, _json(metadata)),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )
            first_user = connection.execute(
                "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' ORDER BY created_at LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if first_user:
                title = first_user["content"].replace("\n", " ").strip()[:80]
                connection.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (title or "New conversation", conversation_id),
                )
        return message_id

    def get_messages(self, conversation_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages WHERE conversation_id = ?
                    ORDER BY created_at DESC LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (conversation_id, max(1, int(limit))),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _decode(item.pop("metadata_json", "{}"), {})
            messages.append(item)
        return messages

    def list_user_messages(self, limit: int = 5_000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT messages.id, messages.conversation_id, messages.content,
                       messages.created_at, conversations.title
                FROM messages
                JOIN conversations ON conversations.id = messages.conversation_id
                WHERE messages.role = 'user'
                ORDER BY messages.created_at DESC LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def relevant_prior_user_messages(
        self,
        query: str,
        *,
        exclude_conversation_id: str,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Return a few matching old user messages without loading whole conversations."""
        query_terms = _search_terms(query)
        if not query_terms:
            return []
        ranked: list[dict[str, Any]] = []
        for item in self.list_user_messages(limit=2_000):
            if item["conversation_id"] == exclude_conversation_id:
                continue
            overlap = query_terms & _search_terms(item["content"])
            if not overlap:
                continue
            record = dict(item)
            record["score"] = len(overlap)
            record["matched_terms"] = sorted(overlap)
            record["content"] = str(record["content"])[:800]
            ranked.append(record)
        ranked.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)
        return ranked[: max(1, int(limit))]

    def log_activity(
        self,
        event: str,
        message: str,
        *,
        level: str = "info",
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO activity(created_at, level, event, message, details_json) VALUES (?, ?, ?, ?, ?)",
                (utc_now(), level, event, message, _json(details)),
            )

    def list_activity(self, limit: int = 250) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM activity ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in reversed(rows):
            item = dict(row)
            item["details"] = _decode(item.pop("details_json", "{}"), {})
            items.append(item)
        return items

    def add_memory(
        self,
        content: str,
        *,
        scope: str = "project",
        kind: str = "lesson",
        status: str = "proposed",
        confidence: float | None = None,
        conditions: dict[str, Any] | None = None,
        evidence: list[str] | None = None,
        source: str = "user",
    ) -> str:
        if status not in {"proposed", "verified", "superseded"}:
            raise ValueError(f"Unsupported memory status: {status}")
        memory_id = str(uuid.uuid4())
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories(
                    id, scope, kind, content, status, confidence, conditions_json,
                    evidence_json, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    scope,
                    kind,
                    str(content).strip(),
                    status,
                    confidence,
                    _json(conditions),
                    _json(evidence or []),
                    source,
                    now,
                    now,
                ),
            )
        return memory_id

    def list_memories(self, status: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        query = "SELECT * FROM memories"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["conditions"] = _decode(item.pop("conditions_json", "{}"), {})
            item["evidence"] = _decode(item.pop("evidence_json", "[]"), [])
            items.append(item)
        return items

    def set_memory_status(self, memory_id: str, status: str) -> None:
        if status not in {"proposed", "verified", "superseded"}:
            raise ValueError(f"Unsupported memory status: {status}")
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), memory_id),
            ).rowcount
        if not changed:
            raise KeyError(memory_id)

    def update_memory(
        self,
        memory_id: str,
        content: str,
        *,
        scope: str,
        kind: str,
    ) -> None:
        clean_content = str(content).replace("\x00", "").strip()
        if not clean_content:
            raise ValueError("Memory text cannot be empty.")
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE memories
                SET content = ?, scope = ?, kind = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean_content, scope, kind, utc_now(), memory_id),
            ).rowcount
        if not changed:
            raise KeyError(memory_id)

    def add_memory_if_new(
        self,
        content: str,
        *,
        scope: str = "project",
        kind: str = "lesson",
        status: str = "proposed",
        confidence: float | None = None,
        evidence: list[str] | None = None,
        source: str = "assistant_suggestion",
    ) -> str | None:
        clean_content = str(content).replace("\x00", "").strip()
        normalized = " ".join(clean_content.lower().split())
        if not normalized:
            return None
        with self._connect() as connection:
            rows = connection.execute("SELECT content FROM memories").fetchall()
        if any(" ".join(str(row["content"]).lower().split()) == normalized for row in rows):
            return None
        return self.add_memory(
            clean_content,
            scope=scope,
            kind=kind,
            status=status,
            confidence=confidence,
            evidence=evidence,
            source=source,
        )

    def relevant_verified_memories(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        memories = self.list_memories(status="verified", limit=500)
        terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_+-]{3,}", query)}
        for memory in memories:
            words = set(re.findall(r"[A-Za-z0-9_+-]{3,}", memory["content"].lower()))
            memory["_score"] = len(terms & words)
        ranked = sorted(memories, key=lambda item: (item["_score"], item["updated_at"]), reverse=True)
        if any(item["_score"] for item in ranked):
            ranked = [item for item in ranked if item["_score"] > 0]
        for item in ranked:
            item.pop("_score", None)
        return ranked[: max(1, int(limit))]

    def export_memories_jsonl(self) -> str:
        return "".join(_json(item) + "\n" for item in self.list_memories(limit=100_000))

    def import_memories_jsonl(self, payload: str) -> int:
        """Import portable memory as Proposed so a person reviews it on this agent."""
        text = str(payload or "")
        if len(text.encode("utf-8")) > 5 * 1024 * 1024:
            raise ValueError("Memory import is limited to 5 MB.")
        imported = 0
        allowed_scopes = {"project", "lab", "run"}
        allowed_kinds = {"lesson", "preference", "marker_alias", "recipe_note"}
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Memory line {line_number} is not valid JSON.") from error
            if not isinstance(item, dict) or not str(item.get("content") or "").strip():
                raise ValueError(f"Memory line {line_number} has no text.")
            scope = str(item.get("scope") or "project")
            kind = str(item.get("kind") or "lesson")
            memory_id = self.add_memory_if_new(
                str(item["content"]),
                scope=scope if scope in allowed_scopes else "project",
                kind=kind if kind in allowed_kinds else "lesson",
                status="proposed",
                confidence=item.get("confidence"),
                evidence=[f"imported_from:{item.get('id', 'unknown')}"],
                source="memory_import",
            )
            imported += int(memory_id is not None)
        return imported

    def conversation_markdown(self, conversation_id: str) -> str:
        messages = self.get_messages(conversation_id, limit=100_000)
        lines = ["# SPACEMAP AI Conversation", "", f"Conversation ID: `{conversation_id}`", ""]
        for message in messages:
            label = {"user": "User", "assistant": "Assistant", "system": "System"}[message["role"]]
            lines.extend([f"## {label}", "", f"*{message['created_at']}*", "", message["content"], ""])
        return "\n".join(lines).rstrip() + "\n"

    def conversation_jsonl(self, conversation_id: str) -> str:
        return "".join(_json(message) + "\n" for message in self.get_messages(conversation_id, limit=100_000))

    def save_conversation_exports(self, conversation_id: str, title: str = "conversation") -> dict[str, Path]:
        export_dir = require_assistant_write_path(ASSISTANT_DATA_DIR / "exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(title, "conversation") + "_" + conversation_id[:8]
        markdown_path = require_assistant_write_path(export_dir / f"{stem}.md")
        jsonl_path = require_assistant_write_path(export_dir / f"{stem}.jsonl")
        markdown_path.write_text(self.conversation_markdown(conversation_id), encoding="utf-8")
        jsonl_path.write_text(self.conversation_jsonl(conversation_id), encoding="utf-8")
        self.log_activity(
            "chat_export",
            f"Saved conversation exports: {markdown_path.name}, {jsonl_path.name}",
        )
        return {"markdown": markdown_path, "jsonl": jsonl_path}

    def delete_conversation_exports(self, conversation_id: str) -> list[Path]:
        """Delete explicit Markdown/JSONL exports belonging to one chat."""
        export_dir = require_assistant_write_path(ASSISTANT_DATA_DIR / "exports")
        if not export_dir.is_dir():
            return []
        suffix = f"_{str(conversation_id)[:8]}"
        removed: list[Path] = []
        for path in export_dir.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() in {".md", ".jsonl"}
                and path.stem.endswith(suffix)
            ):
                safe_path = require_assistant_write_path(path)
                safe_path.unlink()
                removed.append(safe_path)
        return removed

    def register_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        caption: str = "",
        source_page: str = "",
        source_command: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        safe_path = require_assistant_write_path(path)
        artifact_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(id, kind, path, caption, source_page, source_command, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    kind,
                    str(safe_path),
                    caption,
                    source_page,
                    source_command,
                    _json(metadata),
                    utc_now(),
                ),
            )
        return artifact_id

    def list_artifacts(self, *, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM artifacts"
        params: list[Any] = []
        if kind:
            query += " WHERE kind = ?"
            params.append(kind)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        artifacts: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _decode(item.pop("metadata_json", "{}"), {})
            artifacts.append(item)
        return artifacts

    def delete_conversation_artifacts(
        self,
        conversation_id: str,
        *,
        kind: str | None = None,
    ) -> int:
        """Remove gallery/database links for one chat without deleting the files."""
        target = str(conversation_id)
        artifact_ids = []
        for item in self.list_artifacts(kind=kind, limit=100_000):
            metadata = item.get("metadata") or {}
            if str(metadata.get("conversation_id") or "") == target:
                artifact_ids.append(str(item["id"]))
        if not artifact_ids:
            return 0
        placeholders = ",".join("?" for _item in artifact_ids)
        with self._connect() as connection:
            changed = connection.execute(
                f"DELETE FROM artifacts WHERE id IN ({placeholders})", artifact_ids
            ).rowcount
        return int(changed)
