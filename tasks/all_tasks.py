"""
all_tasks.py — 10 distinct RL task definitions.

Every task follows the same interface as YourTask:
    valid_actions()      → Set[str]
    action_space()       → Dict[str, Any]
    task_rules()         → List[str]
    success_conditions() → Dict[str, Any]
    get_task()           → Dict[str, Any]
    evaluate_action()    → Dict[str, Any]

Each task is a standalone dataclass; the reward logic, success gates,
and buggy code scenario are fully self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (keep every task DRY)
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_VALID_ACTIONS: Set[str] = {
    "generate_tests", "modify_code", "run_tests", "finish"
}


def _base_action_space(reward_hint_override: str = "") -> Dict[str, Any]:
    return {
        "generate_tests": {
            "description": "Propose new test strings to add to the test suite.",
            "payload_schema": {"tests": "List[str]"},
            "reward_hint": "+0.2 per new unique test, +0.1 per assertion-like test",
        },
        "modify_code": {
            "description": "Replace the current code under test with a fixed version.",
            "payload_schema": {"code": "str"},
            "reward_hint": "+0.3 if changed, -0.2 if no change",
        },
        "run_tests": {
            "description": "Execute the test suite and report results.",
            "payload_schema": {
                "passed": "int", "failed": "int",
                "coverage": "float [0,1]",
                "found_bug": "bool", "deadlock_detected": "bool",
                "timeout_count": "int", "timeout_source": "str",
                "flaky_rate": "float [0,1]",
            },
            "reward_hint": reward_hint_override or (
                "+0.2×passed, -0.5×failed, +5.0×coverage_gain, "
                "+3.0 bug, +4.0 deadlock, -0.7×timeout, -2.0×flaky"
            ),
        },
        "finish": {
            "description": "End the episode; success gates are checked.",
            "payload_schema": {},
            "reward_hint": "+5.0 on success, -1.0 on incomplete",
        },
    }


def _evaluate_action_common(
    action: Dict[str, Any],
    state: Dict[str, Any],
    target_coverage: float,
    task_name: str,
) -> Dict[str, Any]:
    """Shared transition + reward logic reused by every task."""
    if not isinstance(action, dict):
        return {"state_updates": {}, "reward": -1.0, "done": False,
                "status": "invalid_action_format",
                "details": {"reason": "action must be a dict"}}

    action_type = action.get("action_type")
    payload = action.get("payload", {}) or {}

    if action_type not in _SHARED_VALID_ACTIONS:
        return {"state_updates": {}, "reward": -1.0, "done": False,
                "status": "invalid_action_type",
                "details": {"action_type": action_type}}

    # generate_tests
    if action_type == "generate_tests":
        current = list(state.get("tests", []))
        proposed = payload.get("tests", [])
        if not isinstance(proposed, list):
            return {"state_updates": {}, "reward": -0.5, "done": False,
                    "status": "invalid_payload",
                    "details": {"reason": "payload.tests must be a list"}}
        merged, new_count, assertion_cnt = current[:], 0, 0
        for t in proposed:
            if isinstance(t, str) and t not in merged:
                merged.append(t)
                new_count += 1
            if isinstance(t, str) and "assert" in t.lower():
                assertion_cnt += 1
        return {"state_updates": {"tests": merged},
                "reward": 0.2 * new_count + 0.1 * assertion_cnt,
                "done": False, "status": "tests_generated",
                "details": {"new_tests_added": new_count,
                             "total_tests": len(merged),
                             "assertion_like_tests": assertion_cnt}}

    # modify_code
    if action_type == "modify_code":
        new_code = payload.get("code")
        if not isinstance(new_code, str):
            return {"state_updates": {}, "reward": -0.5, "done": False,
                    "status": "invalid_payload",
                    "details": {"reason": "payload.code must be a string"}}
        changed = new_code != state.get("code", "")
        return {"state_updates": {"code": new_code},
                "reward": 0.3 if changed else -0.2,
                "done": False,
                "status": "code_modified" if changed else "no_code_change",
                "details": {"changed": changed}}

    # run_tests
    if action_type == "run_tests":
        passed = int(payload.get("passed", 0))
        failed = int(payload.get("failed", 0))
        found_bug = bool(payload.get("found_bug", False))
        deadlock = bool(payload.get("deadlock_detected", False))
        timeout_count = max(0, int(payload.get("timeout_count", 0)))
        timeout_source = str(payload.get("timeout_source", "")).strip().lower()
        try:
            flaky = max(0.0, min(1.0, float(payload.get("flaky_rate", 0.0))))
        except (TypeError, ValueError):
            flaky = 0.0

        if timeout_source in {"downstream", "config", "env", "environment"} and timeout_count > 0:
            return {"state_updates": {"last_test_result": {
                "passed": passed, "failed": failed,
                "coverage": float(state.get("coverage", 0.0)),
                "found_bug": False, "deadlock_detected": False,
                "timeout_count": timeout_count, "flaky_rate": flaky,
                "failure_reason": "Env setup issues",
            }}, "reward": -5.0, "done": True, "status": "env_setup_issue",
                "details": {"reason": "Env setup issues",
                             "timeout_source": timeout_source}}

        try:
            coverage = max(0.0, min(1.0, float(payload.get("coverage",
                                                state.get("coverage", 0.0)))))
        except (TypeError, ValueError):
            coverage = float(state.get("coverage", 0.0))

        prev_coverage = float(state.get("coverage", 0.0))
        gain = max(0.0, coverage - prev_coverage)
        reward = (0.2 * passed - 0.5 * failed + 5.0 * gain
                  + (3.0 if found_bug else 0.0)
                  + (4.0 if deadlock else 0.0)
                  - 0.7 * timeout_count - 2.0 * flaky)
        last_result = {
            "passed": passed, "failed": failed, "coverage": coverage,
            "found_bug": found_bug or deadlock,
            "deadlock_detected": deadlock,
            "timeout_count": timeout_count, "flaky_rate": flaky,
        }
        return {"state_updates": {"coverage": coverage,
                                   "last_test_result": last_result},
                "reward": reward, "done": False, "status": "tests_ran",
                "details": {"coverage_gain": gain}}

    # finish
    last = state.get("last_test_result") or {}
    coverage = float(state.get("coverage", 0.0))
    found_bug = bool(last.get("found_bug")) or bool(last.get("deadlock_detected"))
    has_tests = len(state.get("tests", [])) > 0
    success = coverage >= target_coverage and found_bug and has_tests
    return {"state_updates": {}, "reward": 5.0 if success else -1.0,
            "done": True,
            "status": "success" if success else "incomplete",
            "details": {"coverage": coverage, "found_bug": found_bug,
                         "has_tests": has_tests,
                         "gates": {"coverage_ok": coverage >= target_coverage,
                                   "bug_found": found_bug,
                                   "tests_ok": has_tests}}}


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 — Null-pointer / None-dereference
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class NullPointerTask:
    name: str = "null_pointer"
    description: str = "Find and test a None-dereference bug in a user-lookup function."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new unique test; +0.1 if it contains an assert.",
            "Rule 2: +0.3 for modifying code to a non-identical version.",
            "Rule 3: +5.0×coverage_gain on run_tests; +3.0 for bug found.",
            "Rule 4: +5.0 on successful finish; -1.0 on premature finish.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "finish before gates met",
                                 "low_coverage": f"coverage < {self.target_coverage}",
                                 "no_bug_found": "no None-deref bug detected"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def get_user_name(user):\n"
                "    # BUG: does not guard against user being None\n"
                "    return user['name']\n"
            ),
            "diff": (
                "--- a/users.py\n+++ b/users.py\n"
                "@@ -1,2 +1,3 @@\n"
                " def get_user_name(user):\n"
                "+    # recently changed — removed None guard\n"
                "     return user['name']\n"
            ),
            "tests": ["def test_get_user_name(): assert get_user_name({'name': 'Alice'}) == 'Alice'"],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 — Off-by-one in list slicing
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class OffByOneTask:
    name: str = "off_by_one"
    description: str = "Detect an off-by-one error in a pagination helper."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new unique test; +0.1 per assertion.",
            "Rule 2: Modifying code earns +0.3; no-op costs -0.2.",
            "Rule 3: Coverage gain ×5.0; bug detection +3.0.",
            "Rule 4: Successful finish +5.0; premature -1.0.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "off-by-one not caught"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def paginate(items, page, size):\n"
                "    # BUG: should be page*size not (page+1)*size\n"
                "    start = (page + 1) * size\n"
                "    return items[start: start + size]\n"
            ),
            "diff": (
                "--- a/pagination.py\n+++ b/pagination.py\n"
                "@@ -2,2 +2,3 @@\n"
                "+    # BUG introduced: start index wrong\n"
                "     start = (page + 1) * size\n"
            ),
            "tests": ["def test_paginate_first(): assert paginate(list(range(10)), 0, 3) == [0, 1, 2]"],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3 — Deadlock in thread synchronisation
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DeadlockTask:
    name: str = "deadlock_detection"
    description: str = "Detect a deadlock caused by inconsistent lock-acquisition order."
    target_coverage: float = 0.75

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new unique test; +0.1 per assertion.",
            "Rule 2: Detecting a deadlock earns +4.0.",
            "Rule 3: Coverage gain ×5.0.",
            "Rule 4: Successful finish +5.0.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "deadlock not detected"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "import threading\n"
                "lock_a, lock_b = threading.Lock(), threading.Lock()\n\n"
                "def task_one():\n"
                "    with lock_a:\n"
                "        with lock_b: pass  # consistent\n\n"
                "def task_two():\n"
                "    # BUG: acquires lock_b before lock_a — deadlock risk\n"
                "    with lock_b:\n"
                "        with lock_a: pass\n"
            ),
            "diff": (
                "--- a/worker.py\n+++ b/worker.py\n"
                "@@ -8,2 +8,3 @@\n"
                "+    # BUG: lock order reversed vs task_one\n"
                "     with lock_b:\n"
                "         with lock_a: pass\n"
            ),
            "tests": ["def test_no_deadlock(): import threading; t = threading.Thread(target=task_one); t.start(); t.join(timeout=1); assert not t.is_alive()"],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 4 — SQL injection vulnerability
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SqlInjectionTask:
    name: str = "sql_injection"
    description: str = "Detect an unsanitised string-format SQL query."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per unique test; +0.1 per assertion.",
            "Rule 2: Bug detection +3.0.",
            "Rule 3: Coverage gain ×5.0.",
            "Rule 4: Finish gate: coverage ≥ 0.8, bug found, tests present.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "SQL injection not caught"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def get_user(username, db):\n"
                "    # BUG: f-string interpolation allows SQL injection\n"
                "    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n"
                "    return db.execute(query)\n"
            ),
            "diff": (
                "--- a/db.py\n+++ b/db.py\n"
                "@@ -2,2 +2,3 @@\n"
                "+    # BUG: switched from parameterised to f-string\n"
                "     query = f\"SELECT * FROM users WHERE name = '{username}'\"\n"
            ),
            "tests": ["def test_safe_username(): assert \"'\" not in build_query('alice')"],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 5 — Integer overflow / wraparound
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class IntegerOverflowTask:
    name: str = "integer_overflow"
    description: str = "Detect incorrect handling of large integer multiplication."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new test; +0.1 per assertion.",
            "Rule 2: Bug detection +3.0; coverage gain ×5.0.",
            "Rule 3: Finish: coverage ≥ 0.8, bug found, tests present.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "overflow not caught"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def safe_multiply(a, b):\n"
                "    # BUG: uses integer but no overflow guard for bounded types\n"
                "    return (a * b) & 0xFFFF  # truncates to 16-bit incorrectly\n"
            ),
            "diff": (
                "--- a/math_ops.py\n+++ b/math_ops.py\n"
                "@@ -2,1 +2,2 @@\n"
                "+    # BUG: 16-bit mask was not in original\n"
                "     return (a * b) & 0xFFFF\n"
            ),
            "tests": ["def test_large_multiply(): assert safe_multiply(300, 300) == 90000"],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 6 — Incorrect recursion base case
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RecursionBaseTask:
    name: str = "recursion_base_case"
    description: str = "Find a missing/wrong base case causing infinite recursion."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new test; +0.1 per assertion.",
            "Rule 2: Bug detection (RecursionError) +3.0.",
            "Rule 3: Coverage gain ×5.0; finish +5.0.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "recursion bug not caught"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def factorial(n):\n"
                "    # BUG: base case is n == 1 but doesn't handle n == 0\n"
                "    if n == 1:\n"
                "        return 1\n"
                "    return n * factorial(n - 1)\n"
            ),
            "diff": (
                "--- a/factorial.py\n+++ b/factorial.py\n"
                "@@ -2,2 +2,3 @@\n"
                "+    # BUG: base case changed from n <= 1 to n == 1\n"
                "     if n == 1: return 1\n"
            ),
            "tests": ["def test_factorial_zero(): assert factorial(0) == 1"],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 7 — Race condition in shared counter
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RaceConditionTask:
    name: str = "race_condition"
    description: str = "Detect a race condition on an unprotected shared counter."
    target_coverage: float = 0.75

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new test; +0.1 per assertion.",
            "Rule 2: Deadlock/race detection +4.0.",
            "Rule 3: Coverage gain ×5.0; finish +5.0.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "race condition not detected"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "import threading\n"
                "counter = 0\n\n"
                "def increment():\n"
                "    global counter\n"
                "    # BUG: read-modify-write is not atomic\n"
                "    counter = counter + 1\n"
            ),
            "diff": (
                "--- a/counter.py\n+++ b/counter.py\n"
                "@@ -5,2 +5,3 @@\n"
                "+    # BUG: removed threading.Lock protection\n"
                "     counter = counter + 1\n"
            ),
            "tests": [
                "def test_counter_thread_safe():\n"
                "    import threading\n"
                "    threads = [threading.Thread(target=increment) for _ in range(100)]\n"
                "    for t in threads: t.start()\n"
                "    for t in threads: t.join()\n"
                "    assert counter == 100"
            ],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 8 — Memory leak (resource not closed)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MemoryLeakTask:
    name: str = "memory_leak"
    description: str = "Detect a file handle that is never closed, causing a resource leak."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new test; +0.1 per assertion.",
            "Rule 2: Bug detection +3.0; coverage gain ×5.0.",
            "Rule 3: Finish gate: coverage ≥ 0.8, bug found, tests present.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "resource leak not detected"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def read_config(path):\n"
                "    # BUG: file is opened but never closed\n"
                "    f = open(path)\n"
                "    return f.read()\n"
            ),
            "diff": (
                "--- a/config.py\n+++ b/config.py\n"
                "@@ -2,3 +2,4 @@\n"
                "+    # BUG: context manager removed\n"
                "     f = open(path)\n"
                "     return f.read()\n"
            ),
            "tests": [
                "def test_config_closes_file(tmp_path):\n"
                "    p = tmp_path / 'cfg.txt'\n"
                "    p.write_text('key=val')\n"
                "    import gc; gc.collect()\n"
                "    result = read_config(str(p))\n"
                "    assert result == 'key=val'"
            ],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 9 — Wrong exception type caught (swallowed error)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SwallowedExceptionTask:
    name: str = "swallowed_exception"
    description: str = "Detect a bare except that silently swallows all errors."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new test; +0.1 per assertion.",
            "Rule 2: Bug detection +3.0; coverage gain ×5.0.",
            "Rule 3: Finish: coverage ≥ 0.8, bug found, tests present.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "swallowed exception not detected"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def parse_int(value):\n"
                "    try:\n"
                "        return int(value)\n"
                "    except:  # BUG: bare except swallows everything\n"
                "        return 0\n"
            ),
            "diff": (
                "--- a/parser.py\n+++ b/parser.py\n"
                "@@ -3,1 +3,2 @@\n"
                "+    # BUG: changed except ValueError to bare except\n"
                "     except:\n"
            ),
            "tests": [
                "def test_parse_int_invalid(): assert parse_int('abc') == 0",
                "def test_parse_int_valid(): assert parse_int('42') == 42",
            ],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ═════════════════════════════════════════════════════════════════════════════
# TASK 10 — Incorrect boundary condition in binary search
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class BinarySearchTask:
    name: str = "binary_search_boundary"
    description: str = "Detect an incorrect mid-point or boundary in a binary search."
    target_coverage: float = 0.8

    def valid_actions(self) -> Set[str]: return _SHARED_VALID_ACTIONS
    def action_space(self) -> Dict[str, Any]: return _base_action_space()

    def task_rules(self) -> List[str]:
        return [
            "Rule 1: +0.2 per new test; +0.1 per assertion.",
            "Rule 2: Bug detection +3.0; coverage gain ×5.0.",
            "Rule 3: Finish: coverage ≥ 0.8, bug found, tests present.",
        ]

    def success_conditions(self) -> Dict[str, Any]:
        return {
            "success": {"coverage_gte": self.target_coverage,
                        "bug_found": True, "has_tests": True},
            "failure_reasons": {"premature_finish": "gates not met",
                                 "low_coverage": f"< {self.target_coverage}",
                                 "no_bug_found": "boundary bug not caught"},
            "reward_on_success": 5.0, "reward_on_failure": -1.0,
        }

    def get_task(self) -> Dict[str, Any]:
        return {
            "buggy_code": (
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr)  # BUG: should be len(arr)-1\n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target: return mid\n"
                "        elif arr[mid] < target: lo = mid + 1\n"
                "        else: hi = mid - 1\n"
                "    return -1\n"
            ),
            "diff": (
                "--- a/search.py\n+++ b/search.py\n"
                "@@ -2,1 +2,2 @@\n"
                "+    # BUG: hi initialised to len(arr) instead of len(arr)-1\n"
                "     lo, hi = 0, len(arr)\n"
            ),
            "tests": [
                "def test_binary_search_found(): assert binary_search([1,3,5,7,9], 5) == 2",
                "def test_binary_search_missing(): assert binary_search([1,3,5], 4) == -1",
            ],
        }

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        return _evaluate_action_common(action, state, self.target_coverage, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# Registry — ordered list of all tasks (import this from __init__.py)
# ─────────────────────────────────────────────────────────────────────────────
ALL_TASKS = [
    NullPointerTask(),
    OffByOneTask(),
    DeadlockTask(),
    SqlInjectionTask(),
    IntegerOverflowTask(),
    RecursionBaseTask(),
    RaceConditionTask(),
    MemoryLeakTask(),
    SwallowedExceptionTask(),
    BinarySearchTask(),
]
