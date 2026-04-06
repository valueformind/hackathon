"""Task definitions — RL problem spec, action space, rules, success conditions."""

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass(frozen=True)
class YourTask:
    """
    RL task: given a code change (diff), generate/improve tests that
    maximise coverage, find bugs, and remain stable (non-flaky).

    Three explicit contracts exposed to the environment and trainers:
        action_space()       — what actions exist and their schemas
        task_rules()         — reward shaping rules in plain English
        success_conditions() — what constitutes pass / fail
    """

    name: str = "rl_test_generation"
    description: str = "Generate/improve tests from code+diff and maximise bug-finding quality."
    target_coverage: float = 0.8

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------
    def valid_actions(self) -> Set[str]:
        return {"generate_tests", "modify_code", "run_tests", "finish"}

    def action_space(self) -> Dict[str, Any]:
        """
        Full schema of every valid action the agent may produce.

        Structure:
            { action_type: { description, payload_schema, reward_hint } }
        """
        return {
            "generate_tests": {
                "description": "Propose new test strings to add to the test suite.",
                "payload_schema": {
                    "tests": "List[str]  — list of test function source strings",
                },
                "reward_hint": "+0.2 per new unique test, +0.1 per assertion-like test",
            },
            "modify_code": {
                "description": "Replace the current code under test with a fixed version.",
                "payload_schema": {
                    "code": "str  — full source code string",
                },
                "reward_hint": "+0.3 if code changed, -0.2 if no change",
            },
            "run_tests": {
                "description": "Execute the current test suite and report results.",
                "payload_schema": {
                    "passed":            "int    — number of tests that passed",
                    "failed":            "int    — number of tests that failed",
                    "coverage":          "float  — [0.0, 1.0] measured coverage",
                    "found_bug":         "bool   — whether a bug was detected (optional)",
                    "deadlock_detected": "bool   — whether a deadlock/hang occurred (optional)",
                    "timeout_count":     "int    — tests that timed out (optional)",
                    "timeout_source":    "str    — 'downstream'|'config'|'env' triggers hard-fail (optional)",
                    "flaky_rate":        "float  — [0.0, 1.0] ratio of non-deterministic tests (optional)",
                },
                "reward_hint": (
                    "+0.2×passed, -0.5×failed, +5.0×coverage_gain, "
                    "+3.0 if bug found, +4.0 if deadlock, "
                    "-0.7×timeouts, -2.0×flaky_rate; "
                    "infra timeout → hard fail -5.0"
                ),
            },
            "finish": {
                "description": "End the episode; success gates are checked.",
                "payload_schema": {},
                "reward_hint": "+5.0 on success, -1.0 on incomplete",
            },
        }

    # ------------------------------------------------------------------
    # Task rules
    # ------------------------------------------------------------------
    def task_rules(self) -> List[str]:
        """
        Plain-English reward shaping rules for this task.
        Agents and trainers may inspect these to understand the objective.
        """
        return [
            "Rule 1: Each new unique test string added earns +0.2 reward.",
            "Rule 2: Each assertion-like test earns an additional +0.1 quality bonus.",
            "Rule 3: Modifying code to a different value earns +0.3; no-op costs -0.2.",
            "Rule 4: Each passing test earns +0.2; each failing test costs -0.5.",
            "Rule 5: Coverage gain above previous level earns 5.0 × gain.",
            "Rule 6: Detecting a bug earns +3.0; detecting a deadlock earns +4.0.",
            "Rule 7: Each timed-out test costs -0.7.",
            "Rule 8: Flaky tests cost -2.0 × flaky_rate.",
            "Rule 9: Calling finish with all gates met earns +5.0.",
            "Rule 10: Calling finish prematurely costs -1.0.",
            "Rule 11: Infra/env/config timeouts hard-fail the episode with -5.0.",
            "Rule 12: Exceeding MAX_STEPS hard-fails the episode with -3.0.",
            "Rule 13: Invalid action format or type costs -1.0.",
            "Rule 14: Invalid payload for generate_tests or modify_code costs -0.5.",
        ]

    # ------------------------------------------------------------------
    # Success / failure conditions
    # ------------------------------------------------------------------
    def success_conditions(self) -> Dict[str, Any]:
        """
        Explicit conditions that determine episode outcome.
        Both task.evaluate_action('finish') and grader.grade_trace()
        use these gates.
        """
        return {
            "success": {
                "coverage_gte":   self.target_coverage,
                "bug_found":      True,
                "has_tests":      True,
                "episode_complete": True,
            },
            "failure_reasons": {
                "premature_finish":  "finish called before all gates met",
                "env_setup_issue":   "infra/config/env timeout hard-failed the episode",
                "max_steps_exceeded":"agent used all steps without finishing",
                "low_coverage":      f"coverage < {self.target_coverage}",
                "no_bug_found":      "no bug or deadlock was detected",
                "no_tests":          "test list was empty at finish",
            },
            "reward_on_success":  5.0,
            "reward_on_failure": -1.0,
        }

    # ------------------------------------------------------------------
    # Initial task scenario
    # ------------------------------------------------------------------
    def get_task(self) -> Dict[str, Any]:
        """
        Return the initial scenario loaded by env.reset().
        Fully offline and deterministic.
        """
        buggy_code = (
            "def divide(a, b):\n"
            "    # BUG: no zero-division guard\n"
            "    return a / b\n"
        )
        diff = (
            "--- a/math_utils.py\n"
            "+++ b/math_utils.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def divide(a, b):\n"
            "+    # recently changed logic around division handling\n"
            "     return a / b\n"
        )
        baseline_tests = [
            "def test_divide_basic(): assert divide(6, 2) == 3",
        ]
        return {
            "buggy_code": buggy_code,
            "diff":       diff,
            "tests":      baseline_tests,
        }

    # ------------------------------------------------------------------
    # Core transition + reward function
    # ------------------------------------------------------------------
    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply task rules to (action, state) and return:
        {
            state_updates: dict,   # merged into env.state
            reward:        float,
            done:          bool,
            status:        str,
            details:       dict,
        }
        """
        # --- validation ---
        if not isinstance(action, dict):
            return {
                "state_updates": {},
                "reward": -1.0,
                "done":   False,
                "status": "invalid_action_format",
                "details": {"reason": "action must be a dict"},
            }

        action_type = action.get("action_type")
        payload     = action.get("payload", {}) or {}

        if action_type not in self.valid_actions():
            return {
                "state_updates": {},
                "reward": -1.0,
                "done":   False,
                "status": "invalid_action_type",
                "details": {"action_type": action_type},
            }

        # ── generate_tests ─────────────────────────────────────────────
        if action_type == "generate_tests":
            current_tests  = list(state.get("tests", []))
            proposed_tests = payload.get("tests", [])

            if not isinstance(proposed_tests, list):
                return {
                    "state_updates": {},
                    "reward": -0.5,
                    "done":   False,
                    "status": "invalid_payload",
                    "details": {"reason": "payload.tests must be a list"},
                }

            merged        = current_tests[:]
            new_count     = 0
            assertion_cnt = 0

            for t in proposed_tests:
                if not isinstance(t, str):
                    continue
                if t not in merged:
                    merged.append(t)
                    new_count += 1
                if "assert" in t.lower():
                    assertion_cnt += 1

            reward = 0.2 * new_count + 0.1 * assertion_cnt
            return {
                "state_updates": {"tests": merged},
                "reward": reward,
                "done":   False,
                "status": "tests_generated",
                "details": {
                    "new_tests_added":       new_count,
                    "total_tests":           len(merged),
                    "assertion_like_tests":  assertion_cnt,
                },
            }

        # ── modify_code ────────────────────────────────────────────────
        if action_type == "modify_code":
            new_code = payload.get("code")
            if not isinstance(new_code, str):
                return {
                    "state_updates": {},
                    "reward": -0.5,
                    "done":   False,
                    "status": "invalid_payload",
                    "details": {"reason": "payload.code must be a string"},
                }

            changed = new_code != state.get("code", "")
            return {
                "state_updates": {"code": new_code},
                "reward": 0.3 if changed else -0.2,
                "done":   False,
                "status": "code_modified" if changed else "no_code_change",
                "details": {"changed": changed},
            }

        # ── run_tests ──────────────────────────────────────────────────
        if action_type == "run_tests":
            passed            = int(payload.get("passed", 0))
            failed            = int(payload.get("failed", 0))
            found_bug         = bool(payload.get("found_bug", False))
            deadlock_detected = bool(payload.get("deadlock_detected", False))
            timeout_count     = max(0, int(payload.get("timeout_count", 0)))
            timeout_source    = str(payload.get("timeout_source", "")).strip().lower()

            try:
                flaky_rate = float(payload.get("flaky_rate", 0.0))
            except (TypeError, ValueError):
                flaky_rate = 0.0
            flaky_rate = max(0.0, min(1.0, flaky_rate))

            # Failure condition: infra/env timeout
            if timeout_source in {"downstream", "config", "env", "environment"} and timeout_count > 0:
                last_result = {
                    "passed": passed, "failed": failed,
                    "coverage":          float(state.get("coverage", 0.0)),
                    "found_bug":         False,
                    "deadlock_detected": False,
                    "timeout_count":     timeout_count,
                    "flaky_rate":        flaky_rate,
                    "failure_reason":    "Env setup issues",
                }
                return {
                    "state_updates": {"last_test_result": last_result},
                    "reward": -5.0,
                    "done":   True,
                    "status": "env_setup_issue",
                    "details": {
                        "reason":         "Env setup issues",
                        "timeout_source": timeout_source,
                        "timeout_count":  timeout_count,
                    },
                }

            # Normal test run
            try:
                coverage = float(payload.get("coverage", state.get("coverage", 0.0)))
            except (TypeError, ValueError):
                coverage = float(state.get("coverage", 0.0))

            coverage      = max(0.0, min(1.0, coverage))
            prev_coverage = float(state.get("coverage", 0.0))
            coverage_gain = max(0.0, coverage - prev_coverage)

            reward = (
                0.2  * passed
                - 0.5  * failed
                + 5.0  * coverage_gain
                + (3.0 if found_bug         else 0.0)
                + (4.0 if deadlock_detected else 0.0)
                - 0.7  * timeout_count
                - 2.0  * flaky_rate
            )

            last_result = {
                "passed":            passed,
                "failed":            failed,
                "coverage":          coverage,
                "found_bug":         found_bug or deadlock_detected,
                "deadlock_detected": deadlock_detected,
                "timeout_count":     timeout_count,
                "flaky_rate":        flaky_rate,
            }

            return {
                "state_updates": {"coverage": coverage, "last_test_result": last_result},
                "reward": reward,
                "done":   False,
                "status": "tests_ran",
                "details": {
                    "coverage_gain": coverage_gain,
                    "reward_breakdown": {
                        "passed_component":   0.2  * passed,
                        "failed_component":  -0.5  * failed,
                        "coverage_component": 5.0  * coverage_gain,
                        "bug_component":      3.0  if found_bug         else 0.0,
                        "deadlock_component": 4.0  if deadlock_detected else 0.0,
                        "timeout_penalty":   -0.7  * timeout_count,
                        "flaky_penalty":     -2.0  * flaky_rate,
                    },
                },
            }

        # ── finish ─────────────────────────────────────────────────────
        last      = state.get("last_test_result") or {}
        coverage  = float(state.get("coverage", 0.0))
        found_bug = bool(last.get("found_bug", False)) or bool(last.get("deadlock_detected", False))
        has_tests = len(state.get("tests", [])) > 0

        # Success condition check (mirrors success_conditions())
        success = coverage >= self.target_coverage and found_bug and has_tests
        return {
            "state_updates": {},
            "reward": 5.0 if success else -1.0,
            "done":   True,
            "status": "success" if success else "incomplete",
            "details": {
                "coverage":  coverage,
                "found_bug": found_bug,
                "has_tests": has_tests,
                "gates": {
                    "coverage_ok": coverage >= self.target_coverage,
                    "bug_found":   found_bug,
                    "tests_ok":    has_tests,
                },
            },
        }
