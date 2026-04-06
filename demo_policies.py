"""
demo_policies.py — Validation of reward logic and policy differences.

Runs three scripted policies and emits the same [START]/[STEP]/[END] format
as inference.py for each episode, then prints a comparison summary table.

Policies:
    bad       — invalid payload, weak run, premature finish  → passed: False
    improving — full cycle, bug found, coverage 86%          → passed: True
    env-fail  — infra timeout hard-fail                      → passed: False

Run:
    python3 demo_policies.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from env.openenv_env import OpenEnv
from grader.grader import grade_trace
from tasks.task import YourTask
from utils.logging import log_end, log_start, log_step

BENCHMARK = "openenv"


def run_episode(
    policy_name: str,
    actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run one RL episode with a fixed action sequence.

    Emits [START] / [STEP]... / [END] to stdout — identical format to
    inference.py.  Returns the grader report for the summary table.
    """
    task  = YourTask()
    env   = OpenEnv(task=task)
    env.reset()

    model = f"scripted-{policy_name.replace(' ', '-').lower()}"

    trace:   List[Dict[str, Any]] = []
    rewards: List[float]          = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task.name, env=BENCHMARK, model=model)

    try:
        for i, action in enumerate(actions, start=1):
            error: Optional[str] = None
            reward = 0.0
            done   = False

            try:
                result = env.step(action)
                reward = result["reward"]
                done   = result["done"]

                status = result["info"].get("status", "")
                if status not in ("", "ok", "tests_ran", "finished",
                                  "code_modified", "tests_generated"):
                    error = status

                trace.append(result)
            except Exception as exc:
                error = f"env_error:{exc}"
                done  = True

            rewards.append(reward)
            steps_taken = i
            log_step(
                step=i,
                action=action["action_type"],
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        report  = grade_trace(trace=trace, task=task)
        score   = float(report.get("composite_score", 0.0))
        success = bool(report.get("passed", False))

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    total_reward = sum(rewards)
    return {
        "name":         policy_name,
        "report":       report,
        "total_reward": total_reward,
        "trace":        trace,
        "score":        score,
    }


def main() -> None:
    # ── 1) Bad policy — invalid payload, low coverage, premature finish ───
    bad_policy_actions: List[Dict[str, Any]] = [
        {"action_type": "generate_tests", "payload": {"tests": "not-a-list"}},
        {"action_type": "run_tests",       "payload": {"passed": 0, "failed": 2, "coverage": 0.05, "found_bug": False}},
        {"action_type": "finish",           "payload": {}},
    ]

    # ── 2) Improving policy — full cycle, bug found, coverage 86% ─────────
    improving_policy_actions: List[Dict[str, Any]] = [
        {
            "action_type": "generate_tests",
            "payload": {
                "tests": [
                    "def test_divide_zero():\n"
                    "    try:\n"
                    "        divide(1, 0)\n"
                    "        assert False\n"
                    "    except ZeroDivisionError:\n"
                    "        assert True",
                    "def test_divide_basic(): assert divide(6, 2) == 3",
                ]
            },
        },
        {
            "action_type": "run_tests",
            "payload": {"passed": 2, "failed": 1, "coverage": 0.62, "found_bug": True},
        },
        {
            "action_type": "modify_code",
            "payload": {
                "code": (
                    "def divide(a, b):\n"
                    "    if b == 0:\n"
                    "        raise ZeroDivisionError('b must not be zero')\n"
                    "    return a / b\n"
                )
            },
        },
        {
            "action_type": "run_tests",
            "payload": {"passed": 4, "failed": 0, "coverage": 0.86, "found_bug": True},
        },
        {"action_type": "finish", "payload": {}},
    ]

    # ── 3) Env / infra timeout — hard-fail with "Env setup issues" ─────────
    env_failure_actions: List[Dict[str, Any]] = [
        {
            "action_type": "run_tests",
            "payload": {"timeout_count": 1, "timeout_source": "downstream"},
        }
    ]

    print("=" * 60)
    print("  Policy Comparison — RL Coding Environment")
    print("=" * 60)
    print()

    bad_result  = run_episode("bad",       bad_policy_actions)
    print()
    good_result = run_episode("improving", improving_policy_actions)
    print()
    env_result  = run_episode("env-fail",  env_failure_actions)

    # ── Summary table ──────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  {'Policy':<20}  {'Score':>6}  {'Total Reward':>12}  {'Passed':>6}")
    print("-" * 65)
    for r in [bad_result, good_result, env_result]:
        print(
            f"  {r['name'][:20]:<20}  "
            f"{r['score']:>6.3f}  "
            f"{r['total_reward']:>12.2f}  "
            f"{str(r['report']['passed']):>6}"
        )
    print("=" * 65)

    # ── Automated checks — verify reward signal is meaningful ─────────────
    env_last_status = (
        env_result["trace"][-1].get("info", {}).get("status", "")
        if env_result["trace"] else ""
    )
    env_failure_reason = (
        env_result["trace"][-1]["state"]["last_test_result"].get("failure_reason")
        if env_result["trace"] else None
    )

    checks = [
        ("bad_policy_is_incomplete",   bad_result["report"]["passed"]  is False),
        ("improving_policy_passes",    good_result["report"]["passed"] is True),
        ("improving_beats_bad_reward", good_result["total_reward"] > bad_result["total_reward"]),
        ("improving_beats_bad_score",  good_result["score"]        > bad_result["score"]),
        ("env_timeout_hard_fails",     env_last_status == "env_setup_issue"),
        ("env_timeout_has_reason",     env_failure_reason == "Env setup issues"),
    ]

    print()
    print("=== Checks ===")
    failures = 0
    for label, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures += 1

    print(f"\n  check_failures={failures}")
    if failures:
        raise SystemExit(f"\n{failures} check(s) failed.")


if __name__ == "__main__":
    main()
