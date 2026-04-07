"""
my_env_v4.py — Local stub for the hackathon judge's environment package.

ON THE JUDGE'S MACHINE:
    The real `my_env_v4` package is installed by the organizers.
    It connects to a Docker container running the actual benchmark.

LOCALLY (this file):
    Wraps OpenEnv + YourTask so inference.py can be tested without
    the judge's infrastructure.  The async interface mirrors the real
    package so no changes to inference.py are needed.

Interface matched:
    MyEnvV4Action(message: str)
    MyEnvV4Env.from_docker_image(image_name) -> MyEnvV4Env
    await env.reset()   -> StepResult
    await env.step(action) -> StepResult
    await env.close()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from env.openenv_env import OpenEnv
from tasks.task import YourTask


# ── Data classes that mirror the real my_env_v4 interface ────────────────

@dataclass
class Observation:
    """The observation the agent sees each step."""
    echoed_message: str = ""
    code:           str = ""
    diff:           str = ""
    coverage:       float = 0.0
    tests:          list  = field(default_factory=list)
    step_number:    int   = 0


@dataclass
class StepResult:
    """Return type of reset() and step()."""
    observation: Observation
    reward:      float = 0.0
    done:        bool  = False
    info:        dict  = field(default_factory=dict)


@dataclass
class MyEnvV4Action:
    """Action sent to the environment each step."""
    message: str = ""


# ── Environment wrapper ───────────────────────────────────────────────────

class MyEnvV4Env:
    """
    Async wrapper around OpenEnv that matches the my_env_v4 interface.

    The real my_env_v4 spins up a Docker container; this stub runs
    the RL environment in-process for local testing.
    """

    def __init__(self) -> None:
        self._task = YourTask()
        self._env  = OpenEnv(task=self._task)
        self._state: dict = {}
        self._step_count: int = 0

    # ── Factory (mirrors MyEnvV4Env.from_docker_image) ───────────────────
    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None) -> "MyEnvV4Env":
        """
        Real version: pulls and starts a Docker container.
        Stub version: ignores image_name and returns an in-process env.
        """
        if image_name:
            print(
                f"[my_env_v4 stub] Ignoring IMAGE_NAME={image_name!r} — "
                "running local OpenEnv instead.",
                flush=True,
            )
        instance = cls()
        return instance

    # ── Episode lifecycle ─────────────────────────────────────────────────
    async def reset(self) -> StepResult:
        """Reset the environment and return the first observation."""
        self._state      = self._env.reset()
        self._step_count = 0
        return StepResult(
            observation=self._make_obs(""),
            reward=0.0,
            done=False,
        )

    async def step(self, action: MyEnvV4Action) -> StepResult:
        """
        Translate the LLM's free-text message into a structured RL action,
        step the environment, and return the result.

        Mapping strategy:
          - No tests yet        → generate_tests
          - Tests exist, no run → run_tests (simulated)
          - Bug found           → modify_code
          - Coverage >= 0.8     → finish
          - Otherwise           → run_tests
        """
        rl_action = self._message_to_action(action.message)
        result    = self._env.step(rl_action)

        self._state      = result["state"]
        self._step_count = result["state"].get("step_number", self._step_count + 1)

        # Reward: combine env reward with message-length bonus (echoes real env)
        reward = float(result["reward"]) + len(action.message) * 0.01

        return StepResult(
            observation=self._make_obs(action.message),
            reward=reward,
            done=bool(result["done"]),
            info=result.get("info", {}),
        )

    async def close(self) -> None:
        """Release resources. Real version stops the Docker container."""
        self._env.close()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _make_obs(self, last_message: str) -> Observation:
        return Observation(
            echoed_message=last_message,
            code=self._state.get("code", ""),
            diff=self._state.get("diff", ""),
            coverage=float(self._state.get("coverage", 0.0)),
            tests=list(self._state.get("tests", [])),
            step_number=int(self._state.get("step_number", 0)),
        )

    def _message_to_action(self, message: str) -> dict[str, Any]:
        """
        Convert a free-text LLM message into a structured RL action.
        Simple heuristic used only in the local stub.
        """
        state    = self._state
        tests    = state.get("tests", [])
        coverage = float(state.get("coverage", 0.0))
        last_result = state.get("last_test_result")

        # No tests yet → generate them from the message
        if not tests:
            return {
                "action_type": "generate_tests",
                "payload": {
                    "tests": [
                        f"def test_from_agent():\n    # {message[:80]}\n    assert True"
                    ]
                },
            }

        # Tests exist but never run → run them
        if last_result is None:
            return {
                "action_type": "run_tests",
                "payload": {
                    "passed":   max(1, len(tests)),
                    "failed":   0,
                    "coverage": 0.65,
                    "found_bug": True,
                },
            }

        # Coverage target met → finish
        if coverage >= 0.8:
            return {"action_type": "finish", "payload": {}}

        # Otherwise improve coverage with another run
        return {
            "action_type": "run_tests",
            "payload": {
                "passed":   max(1, len(tests)),
                "failed":   0,
                "coverage": min(1.0, coverage + 0.2),
                "found_bug": True,
            },
        }
