"""OpenEnv-compatible RL environment for code-diff test generation."""

from typing import Any, Dict, List

from tasks.task import YourTask


class OpenEnv:
    """
    RL environment implementing the standard (reset, step, done) interface.

    ┌─────────────────────────────────────────────────────────┐
    │  AGENT INPUT (observation space)                        │
    │  code        str   — current source code                │
    │  diff        str   — unified diff of the change         │
    │  tests       list  — test strings accumulated so far    │
    │  coverage    float — [0.0, 1.0] after last run          │
    │  last_test_result  dict | None                          │
    │  step_number int   — steps taken in this episode        │
    │  history     list  — (action, reward, status) per step  │
    ├─────────────────────────────────────────────────────────┤
    │  ACTION SPACE  →  task.action_space()                   │
    │  TASK RULES    →  task.task_rules()                     │
    │  SUCCESS/FAIL  →  task.success_conditions()             │
    └─────────────────────────────────────────────────────────┘
    """

    # Maximum steps per episode; episode hard-terminates after this.
    MAX_STEPS: int = 20

    def __init__(self, task: YourTask) -> None:
        self.task = task
        self.state: Dict[str, Any] = {}
        self.is_done: bool = False
        self.step_number: int = 0

    # ------------------------------------------------------------------
    # Observation space definition
    # ------------------------------------------------------------------
    @property
    def observation_space(self) -> Dict[str, Any]:
        """
        Schema of what the agent observes at each step.
        Keys match self.state exactly.
        """
        return {
            "code":             {"type": "str",         "description": "Current source code under test"},
            "diff":             {"type": "str",         "description": "Unified diff of the code change"},
            "tests":            {"type": "List[str]",   "description": "Test strings accumulated so far"},
            "coverage":         {"type": "float",       "range": [0.0, 1.0], "description": "Coverage after last run"},
            "last_test_result": {"type": "dict|None",   "description": "Full result of last run_tests call"},
            "step_number":      {"type": "int",         "description": "Steps taken in the current episode"},
            "history":          {"type": "List[dict]",  "description": "Per-step (action, reward, status) log"},
        }

    # ------------------------------------------------------------------
    # Episode start
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        """
        Episode flow — step 0:
        Load task scenario, reset all state, return initial observation.
        """
        task_data = self.task.get_task()
        self.step_number = 0
        self.is_done = False

        self.state = {
            "code":             task_data["buggy_code"],
            "diff":             task_data["diff"],
            "tests":            list(task_data["tests"]),
            "coverage":         0.0,
            "last_test_result": None,
            "step_number":      self.step_number,
            "history":          [],
        }
        return dict(self.state)

    # ------------------------------------------------------------------
    # Episode step
    # ------------------------------------------------------------------
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Episode flow — step N:
        1. Guard against post-done calls.
        2. Hard-terminate if MAX_STEPS exceeded (penalise).
        3. Delegate transition + reward to task.evaluate_action().
        4. Apply state updates, append history, advance step counter.
        5. Return standard RL tuple: {state, reward, done, info}.

        Action schema (see task.action_space() for full detail):
        {
            "action_type": "generate_tests" | "modify_code"
                           | "run_tests"    | "finish",
            "payload":     {...}
        }
        """
        # Guard: already done
        if self.is_done:
            return {
                "state":  dict(self.state),
                "reward": 0.0,
                "done":   True,
                "info":   {"status": "already_done"},
            }

        # Guard: step budget exceeded  →  failure condition
        if self.step_number >= self.MAX_STEPS:
            self.is_done = True
            self.state["step_number"] = self.step_number
            return {
                "state":  dict(self.state),
                "reward": -3.0,
                "done":   True,
                "info":   {
                    "status":  "max_steps_exceeded",
                    "details": {"max_steps": self.MAX_STEPS},
                },
            }

        # Evaluate action via task rules
        result = self.task.evaluate_action(action=action, state=self.state)

        # Apply state updates
        self.state.update(result.get("state_updates", {}))

        # Advance step counter
        self.step_number += 1
        self.state["step_number"] = self.step_number

        # Track history
        self.state.setdefault("history", []).append({
            "action": action,
            "reward": result.get("reward", 0.0),
            "status": result.get("status", ""),
        })

        self.is_done = bool(result.get("done", False))

        return {
            "state":  dict(self.state),
            "reward": result.get("reward", 0.0),
            "done":   self.is_done,
            "info": {
                "status":  result.get("status", ""),
                "details": result.get("details", {}),
            },
        }

    # ------------------------------------------------------------------
    # Episode teardown
    # ------------------------------------------------------------------
    def close(self) -> None:
        """
        Release any held resources and mark the episode as done.
        No-op for the local in-process env; mirrors the OpenEnv close()
        contract required by inference.py and demo scripts.
        """
        self.is_done = True

    # ------------------------------------------------------------------
    # Convenience introspection (useful for agents/trainers)
    # ------------------------------------------------------------------
    @property
    def action_space(self) -> Dict[str, Any]:
        """Proxy to task.action_space() for trainer convenience."""
        return self.task.action_space()

    @property
    def task_rules(self) -> List[str]:
        """Proxy to task.task_rules() for trainer convenience."""
        return self.task.task_rules()

    @property
    def success_conditions(self) -> Dict[str, Any]:
        """Proxy to task.success_conditions() for trainer convenience."""
        return self.task.success_conditions()

