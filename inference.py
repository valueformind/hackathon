"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env
from grader.task_graders import grade_by_task_name, TASK_GRADERS

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image

# ── Credentials & endpoint (hackathon required pattern) ──────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# MY_ENV_V4_TASK selects which of the 10 tasks to run.
# Valid values: rl_test_generation, null_pointer, off_by_one, deadlock_detection,
#               sql_injection, integer_overflow, recursion_base_case, race_condition,
#               memory_leak, swallowed_exception, binary_search_boundary
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "rl_test_generation")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5   # grader composite_score must be >= this

# Normalisation denominator: max possible raw reward across all steps
_MAX_REWARD_PER_STEP = 15.0     # rough upper bound per step
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert software testing agent operating in an RL environment.

    At every step you receive the current state (buggy code, diff, existing tests,
    coverage, last test result) and must return ONE JSON action to improve test quality.

    Available action_types:
      - generate_tests  payload: {"tests": ["def test_...: assert ..."]}
      - run_tests       payload: {"passed": int, "failed": int, "coverage": float,
                                  "found_bug": bool, "deadlock_detected": bool,
                                  "timeout_count": int, "flaky_rate": float}
      - modify_code     payload: {"code": "<full fixed source>"}
      - finish          payload: {}  — only call when coverage>=0.8 AND bug found

    Respond with ONLY valid JSON, no markdown, no explanation. Example:
    {"action_type": "generate_tests", "payload": {"tests": ["def test_zero_div():\\n    try:\\n        divide(1, 0)\\n    except ZeroDivisionError:\\n        pass\\n    else:\\n        assert False, 'expected ZeroDivisionError'"]}}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score_clamped = min(max(float(score), 0.0), 1.0)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score_clamped:.2f} rewards={rewards_str}",
        flush=True,
    )

def build_user_prompt(
    step: int,
    obs: Any,
    last_reward: float,
    history: List[str],
) -> str:
    """Build a task-aware prompt from the current observation."""
    # obs is a MyEnvV4 Observation object; pull relevant fields gracefully
    code      = getattr(obs, "code",     "") or ""
    diff      = getattr(obs, "diff",     "") or ""
    tests     = getattr(obs, "tests",    []) or []
    coverage  = getattr(obs, "coverage", 0.0)
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}  |  Last reward: {last_reward:.2f}  |  Coverage so far: {coverage:.0%}
        Task: {TASK_NAME}

        === Buggy code ===
        {code[:600]}

        === Diff ===
        {diff[:400]}

        === Tests so far ({len(tests)}) ===
        {chr(10).join(tests[-3:]) if tests else "None"}

        === Recent history ===
        {history_block}

        Return ONE JSON action (generate_tests / run_tests / modify_code / finish).
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    step: int,
    obs: Any,
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and parse its JSON action. Falls back to generate_tests on error."""
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw)
        if "action_type" not in action:
            raise ValueError("missing action_type")
        return action
    except Exception:
        # Safe fallback: generate a minimal test
        return {
            "action_type": "generate_tests",
            "payload": {"tests": [
                f"def test_task_{TASK_NAME}_step{step}():\n    assert True"
            ]},
        }


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env: Optional[MyEnvV4Env] = None

    # Accumulated episode state
    history: List[str] = []
    rewards: List[float] = []
    episode_trace: List[Dict[str, Any]] = []   # fed to grader at the end
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)
        result = await env.reset()
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # ── Ask the LLM for a structured JSON action ──────────────────
            action = get_model_action(client, step, obs, last_reward, history)
            action_str = json.dumps(action, separators=(",", ":"))

            # ── Execute in env ─────────────────────────────────────────────
            my_action = MyEnvV4Action(message=action_str)
            result = await env.step(my_action)
            obs = result.observation

            reward  = result.reward or 0.0
            done    = result.done
            error   = getattr(result, "last_action_error", None)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: action_type={action.get('action_type')} -> reward {reward:+.2f}")

            # Record trace entry for grader
            episode_trace.append({
                "step": step,
                "action": action,
                "reward": reward,
                "done": done,
            })

            if done:
                break

        # ── Raw reward normalisation ───────────────────────────────────────
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)

        # ── Task-specific grading ──────────────────────────────────────────
        if TASK_NAME in TASK_GRADERS:
            grade_report = grade_by_task_name(TASK_NAME, episode_trace)
            composite    = grade_report.get("composite_score", score)
            # Use grader composite score if it gives a higher-resolution signal
            score   = float(composite)
            success = bool(grade_report.get("passed", score >= SUCCESS_SCORE_THRESHOLD))
        else:
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception:
        pass

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())