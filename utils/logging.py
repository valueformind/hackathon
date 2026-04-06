"""
utils/logging.py — Shared stdout logging helpers.

Emits the three mandatory line types required by the hackathon spec
and used consistently across demo.py, demo_policies.py, and inference.py.

Output contract:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Rules (from spec):
    - One [START] line at episode begin, before env.reset().
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward / rewards formatted to 2 decimal places.
    - score formatted to 3 decimal places, clamped to [0, 1].
    - done and success are lowercase booleans: true or false.
    - error is the error string, or null if none.
    - All fields on a single line.
"""

from typing import List, Optional


def log_start(task: str, env: str, model: str) -> None:
    """Print the [START] line. Call once, before env.reset()."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """Print one [STEP] line. Call immediately after each env.step() returns."""
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Print the [END] line. Call inside finally, after env.close()."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_clamped = min(max(float(score), 0.0), 1.0)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score_clamped:.3f} rewards={rewards_str}",
        flush=True,
    )
