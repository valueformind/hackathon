"""FastAPI entrypoint exposing reset/step for OpenEnv-style validation."""

from __future__ import annotations

import os

from fastapi import FastAPI
import uvicorn

from env.openenv_env import OpenEnv
from tasks.task import YourTask

app = FastAPI(title="rl-coding-env")

_task = YourTask()
_env = OpenEnv(task=_task)
_state: dict = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> dict:
    global _state
    _state = _env.reset()
    return {
        "observation": _state,
        "reward": 0.0,
        "done": False,
        "info": {},
    }


@app.post("/step")
def step(action: dict) -> dict:
    global _state
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
