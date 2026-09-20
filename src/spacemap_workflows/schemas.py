"""Small, model-neutral contracts for SPACEMAP tools and agent plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PERMISSION_CLASSES = {"read", "temporary_write", "artifact_write", "replace_output"}
TOOL_STATUSES = {"ready", "protected_backend", "manual_only"}


@dataclass(frozen=True)
class ResourceProfile:
    cpu: str = "standard"
    ram: str = "standard"
    gpu: str = "none"
    disk: str = "standard"
    long_running: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolSpec:
    """Description the planner sees; execution remains ordinary Python code."""

    name: str
    version: str
    page: str
    summary: str
    callable_path: str
    permission: str
    status: str
    inputs: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    qc: tuple[str, ...] = ()
    resources: ResourceProfile = field(default_factory=ResourceProfile)
    keywords: tuple[str, ...] = ()

    def validate(self) -> "ToolSpec":
        if not self.name or "." not in self.name:
            raise ValueError("A tool name must use a group.name form.")
        if self.permission not in PERMISSION_CLASSES:
            raise ValueError(f"Unknown tool permission: {self.permission}")
        if self.status not in TOOL_STATUSES:
            raise ValueError(f"Unknown tool status: {self.status}")
        return self

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resources"] = self.resources.as_dict()
        return payload


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    tool: str
    arguments: dict[str, Any]
    needs: tuple[str, ...] = ()
    approval: str = "inherit"
    on_failure: str = "stop"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["needs"] = list(self.needs)
        return payload


@dataclass(frozen=True)
class WorkflowPlan:
    goal: str
    steps: tuple[WorkflowStep, ...]
    execution_mode: str = "ask"
    notes: tuple[str, ...] = ()

    def validate(self, available_tools: set[str]) -> "WorkflowPlan":
        if self.execution_mode not in {"ask", "auto"}:
            raise ValueError("Execution mode must be ask or auto.")
        if not self.steps:
            raise ValueError("A workflow needs at least one step.")
        identifiers = [step.id for step in self.steps]
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Every workflow step needs a unique ID.")
        known: set[str] = set()
        for step in self.steps:
            if step.tool not in available_tools:
                raise ValueError(f"Workflow requested an unknown tool: {step.tool}")
            missing = set(step.needs) - known
            if missing:
                raise ValueError(
                    f"Step {step.id} depends on missing or later step(s): {', '.join(sorted(missing))}"
                )
            known.add(step.id)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "execution_mode": self.execution_mode,
            "steps": [step.as_dict() for step in self.steps],
            "notes": list(self.notes),
        }
