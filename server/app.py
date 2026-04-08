"""FastAPI entrypoint exposing reset/step for OpenEnv-style validation."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from env.openenv_env import OpenEnv, make_env, list_tasks

app = FastAPI(title="rl-coding-env")

# Default env uses the original YourTask; replaced per-request via /reset
_env: OpenEnv = make_env("rl_test_generation")
_state: dict = {}


# ── Request / response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = None   # e.g. "null_pointer"; None → default task


class StepRequest(BaseModel):
    action_type: str
    payload: dict = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    """Root route — satisfies HF Spaces health probes on GET /."""
    return {"status": "ok", "service": "rl-coding-env"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks() -> dict:
    """List all available task names."""
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()) -> dict:
    """
    Start a new episode, optionally for a specific task.

    Body (JSON, all optional):
        { "task_name": "null_pointer" }

    If task_name is omitted the default task (rl_test_generation) is used.
    """
    global _env, _state

    task_name = body.task_name or "rl_test_generation"
    try:
        _env = make_env(task_name)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Available: {list_tasks()}",
        )

    _state = _env.reset()
    return {
        "task": task_name,
        "observation": _state,
        "reward": 0.0,
        "done": False,
        "info": {},
    }


@app.post("/step")
def step(body: StepRequest) -> dict:
    """
    Take one step in the current episode.

    Body:
        { "action_type": "generate_tests", "payload": { "tests": [...] } }
    """
    global _state
    action = {"action_type": body.action_type, "payload": body.payload}
    result = _env.step(action)
    _state = result.get("state", _state)
    return {
        "observation": _state,
        "reward": float(result.get("reward", 0.0)),
        "done": bool(result.get("done", False)),
        "info": result.get("info", {}),
    }


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
