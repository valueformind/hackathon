"""OpenEnv-style data models for this environment."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CodingAction(BaseModel):
    """Action payload accepted by the HTTP step endpoint."""

    action_type: str = Field(..., description="Type of action to apply.")
    payload: dict[str, Any] = Field(default_factory=dict)


class CodingObservation(BaseModel):
    """Observation returned to clients."""

    code: str = ""
    diff: str = ""
    coverage: float = 0.0
    tests: list[str] = Field(default_factory=list)
    step_number: int = 0


class CodingStepResult(BaseModel):
    """Normalized step response shape."""

    observation: CodingObservation
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
