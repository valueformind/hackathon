"""
demo.py — Main run: single happy-path episode for the RL Coding Environment.

Follows the same [START] / [STEP] / [END] output format as inference.py.
No API key required — actions are scripted to showcase the full agent loop:

    reset → generate_tests → run_tests → modify_code → run_tests → finish

Run:
    python3 demo.py
"""

from env.openenv_env import OpenEnv
from grader.grader import grade_trace
from tasks.task import YourTask
from utils.logging import log_end, log_start, log_step

BENCHMARK = "openenv"
MODEL     = "scripted-demo"


def main() -> None:
    task  = YourTask()
    env   = OpenEnv(task=task)
    state = env.reset()

    # ── Scripted actions (replace with a policy network in real RL) ───────
    actions = [
        {
            "action_type": "generate_tests",
            "payload": {
                "tests": [
                    "def test_divide_zero_raises():\n"
                    "    try:\n"
                    "        divide(1, 0)\n"
                    "        assert False, 'expected ZeroDivisionError'\n"
                    "    except ZeroDivisionError:\n"
                    "        assert True",
                    "def test_divide_negative(): assert divide(-6, 2) == -3",
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

    rewards: list[float] = []
    trace:   list[dict]  = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task.name, env=BENCHMARK, model=MODEL)

    try:
        for i, action in enumerate(actions, start=1):
            error: str | None = None
            reward = 0.0
            done   = False

            try:
                result = env.step(action)
                state  = result["state"]
                reward = result["reward"]
                done   = result["done"]

                # Surface non-nominal statuses as the error field
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

        # Grade the full episode — composite_score is already in [0, 1]
        report  = grade_trace(trace=trace, task=task)
        score   = float(report.get("composite_score", 0.0))
        success = bool(report.get("passed", False))

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    # ── Human-readable grader summary (printed after [END]) ───────────────
    print()
    print("=" * 60)
    print("  Grader Report")
    print("=" * 60)
    print(f"  passed          : {report['passed']}")
    print(f"  composite_score : {report['composite_score']:.3f}")
    print(f"  total_reward    : {report['total_reward']:.2f}")
    print(f"  steps           : {report['steps']}")

    print("\n  Dimensions:")
    for dim, val in report["dimensions"].items():
        bar = "█" * int(val["score"] * 20)
        print(f"    {dim:<15} {val['score']:.2f}  {bar}")

    if report["failure_classifications"]:
        print(f"\n  Failures: {report['failure_classifications']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
