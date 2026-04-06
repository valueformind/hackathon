# RL-Powered Coding Environment

> An offline-first Reinforcement Learning environment where an AI agent automatically writes and improves tests based on code changes — and learns to do it better over time.

---

## What This Project Does

Given a **code change (diff)**, the agent:

1. Reads the buggy code and the diff
2. Generates test cases that probe the changed behaviour
3. Runs the tests and observes results (coverage, bug detection, flakiness)
4. Optionally modifies the code to fix bugs
5. Receives a **shaped reward** based on test quality, coverage gain, and bug discovery
6. Improves its policy over repeated episodes

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── inference.py                   # LLM-driven agent — hackathon entry point
├── demo.py                        # Single happy-path episode (scripted)
├── demo_policies.py               # Bad vs improving vs env-failure comparison
├── Dockerfile
├── .dockerignore
├── .gitignore
├── env/
│   ├── __init__.py
│   └── openenv_env.py             # RL environment (reset / step / done)
├── tasks/
│   ├── __init__.py
│   └── task.py                    # Task spec, action space, reward, rules
├── grader/
│   ├── __init__.py
│   └── grader.py                  # 4-dimension episode grader
└── tests/
    └── test_negative_scenarios.py # Offline unit tests (negative paths)
```

---

## RL Design

### Observation Space (agent input)

| Field | Type | Description |
|---|---|---|
| `code` | `str` | Current source code under test |
| `diff` | `str` | Unified diff of the code change |
| `tests` | `List[str]` | Test strings accumulated so far |
| `coverage` | `float [0,1]` | Coverage measured after last run |
| `last_test_result` | `dict\|None` | Full result of last `run_tests` call |
| `step_number` | `int` | Steps taken in the current episode |
| `history` | `List[dict]` | Per-step (action, reward, status) log |

### Action Space

| Action | Payload | Reward signal |
|---|---|---|
| `generate_tests` | `{tests: List[str]}` | +0.2/new test, +0.1/assertion |
| `modify_code` | `{code: str}` | +0.3 if changed, −0.2 if no-op |
| `run_tests` | `{passed, failed, coverage, found_bug, deadlock_detected, timeout_count, flaky_rate}` | +0.2×pass, −0.5×fail, +5×coverage_gain, +3 bug, +4 deadlock, −0.7×timeout, −2×flaky |
| `finish` | `{}` | +5.0 on success, −1.0 if incomplete |

### Reward Shaping Rules

1. Each unique new test earns +0.2
2. Each assertion-like test earns +0.1 quality bonus
3. Code change earns +0.3; no-op costs −0.2
4. Passing tests +0.2 each; failing −0.5 each
5. Coverage gain × 5.0
6. Bug detected +3.0; deadlock detected +4.0
7. Timeout per test −0.7
8. Flaky rate penalty −2.0 × rate
9. Successful finish +5.0; premature finish −1.0
10. Infra/env/config timeout → hard fail −5.0
11. Exceeding MAX_STEPS (20) → hard fail −3.0

### Success Conditions (all must hold at `finish`)

- `coverage >= 0.8`
- `bug_found == True` (or `deadlock_detected == True`)
- `tests` list is non-empty

### Episode Failure Reasons

| Reason | Cause |
|---|---|
| `premature_finish` | `finish` called before gates met |
| `env_setup_issue` | infra/config/downstream timeout |
| `max_steps_exceeded` | 20-step budget exhausted |
| `low_coverage` | coverage < 0.8 at finish |
| `no_bug_found` | no bug or deadlock detected |
| `no_tests` | empty test list at finish |

### Grader Dimensions

The grader scores each episode across four dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Correctness | 35% | Bug/deadlock found, test pass rate, flakiness |
| Completeness | 25% | Finish reached, coverage target met, required actions used |
| Task quality | 25% | Test count, assertion density, coverage gain |
| Env adherence | 15% | Invalid actions, infra aborts, timeouts |

Pass threshold: `composite_score >= 0.7`

---

## Prerequisites

- Python 3.11+
- pip
- Docker (optional)

---

## Local Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo>

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running

> **Quick reference**
> - `demo.py` — **main run**: shows the full RL environment working end-to-end in one successful episode
> - `demo_policies.py` — **validation of reward logic and policy differences**: proves the reward function correctly separates bad, improving, and env-failure behaviours

### `demo.py` — Main run (single happy-path episode)

```bash
python3 demo.py
```

Expected output:
- 5 steps: generate → run → fix code → run → finish
- `composite_score: 0.95`, `passed: True`
- Use this to show judges what the system does

### `demo_policies.py` — Validation of reward logic and policy differences

```bash
python3 demo_policies.py
```

Runs three scripted policies and compares their outcomes side-by-side:
- **Bad policy** — invalid actions, low coverage, premature finish → reward `-2.25`, `passed: False`
- **Improving policy** — full cycle, bug found, coverage 86% → reward `+16.70`, `passed: True`
- **Env timeout policy** — infra hard-fail → `env_setup_issue`, reward `-5.00`, `passed: False`

Includes 5 automated checks that assert reward ordering and failure classification are correct.
Use this to prove the RL reward signal is meaningful and not random.

### `inference.py` — LLM-driven agent (hackathon submission entry point)

```bash
export HF_TOKEN=<your-api-key>
export API_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
export MODEL_NAME=gpt-4.1-mini                  # optional, default shown

python3 inference.py
```

The LLM observes the full environment state at every step and returns a structured JSON
action. The environment evaluates it and returns a reward. Output follows the required
hackathon format:

```
[START] task=buggy-divide-function env=openenv model=gpt-4.1-mini
[STEP]  step=1 action=generate_tests reward=0.60 done=false error=null
[STEP]  step=2 action=run_tests      reward=8.10 done=false error=null
[STEP]  step=3 action=modify_code    reward=0.30 done=false error=null
[STEP]  step=4 action=run_tests      reward=9.20 done=false error=null
[STEP]  step=5 action=finish         reward=5.00 done=true  error=null
[END]   success=true steps=5 rewards=0.60,8.10,0.30,9.20,5.00
```

| Env var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(required)* | API key for the LLM provider |
| `API_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible base URL |
| `MODEL_NAME` | `gpt-4.1-mini` | Model to use for action generation |

### Negative-scenario unit tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

All 7 tests cover invalid formats, bad payloads, premature finish, deadlock signals, and env timeouts.

---

## Docker

### Option A — docker compose (recommended)

```bash
# 1. Build the image once
docker compose build

# 2. Offline runs — no credentials needed
docker compose run --rm demo             # happy-path episode
docker compose run --rm demo-policies    # policy comparison + checks
docker compose run --rm tests            # unit test suite

# 3. LLM-driven inference — needs HF_TOKEN + Docker socket
#    First create your credentials file (never commit this):
cp .env.example .env                     # then edit .env with your real token
docker compose run --rm inference
```

### Option B — plain docker commands

```bash
# Build
docker build -t rl-coding-env:latest .

# Offline: demo / policies / tests
docker run --rm rl-coding-env:latest
docker run --rm rl-coding-env:latest python demo_policies.py
docker run --rm rl-coding-env:latest python -m unittest discover -s tests -p 'test_*.py' -v

# Online: inference.py (pass token via -e, never hardcode it)
docker run --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e IMAGE_NAME=<your-env-image> \
  -v /var/run/docker.sock:/var/run/docker.sock \
  rl-coding-env:latest python inference.py
```

### Credentials setup (`.env` file)

Create a `.env` file from the template — it is git-ignored and Docker-ignored:

```bash
# Required
HF_TOKEN=hf_your_token_here

# LLM endpoint (defaults shown)
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# my_env_v4 Docker image (provided by hackathon judges)
IMAGE_NAME=your-hackathon-env-image-name
MY_ENV_V4_TASK=echo
MY_ENV_V4_BENCHMARK=my_env_v4
```

### Why the Docker socket is mounted for `inference`

`inference.py` uses `MyEnvV4Env.from_docker_image(IMAGE_NAME)` which pulls and starts
the hackathon's benchmark container at runtime. Mounting `/var/run/docker.sock` gives
the agent container access to the host Docker daemon so it can manage those sub-containers.

---

## What Is and Is Not Real RL

| Component | Status |
|---|---|
| Environment (`reset`, `step`, `done`, `reward`) | ✅ Real |
| Observation space | ✅ Real |
| Action space with shaped reward | ✅ Real |
| Episode tracking, history, grading | ✅ Real |
| 4-dimension grader | ✅ Real |
| Negative/edge case handling | ✅ Real |
| Agent (policy network) | 🔧 Scripted for demo — plug in PPO/DQN here |
| Test runner | 🔧 Simulated payloads — replace with `subprocess` + `pytest` |
| Training loop | 🔧 Not yet — environment is ready to accept one |

---

## Roadmap

- [ ] Real subprocess test runner (`pytest` + `coverage.py`) replacing simulated payloads
- [ ] Actual git diff parser for real PR inputs
- [ ] Random / tabular RL baseline training loop
- [ ] PPO/DQN agent via `stable-baselines3`
- [ ] Multi-task benchmark dataset from open-source repos
- [ ] Reward hacking detection and kill switch

---

## Key Design Decisions

- **Offline-first**: no cloud services, no external APIs, runs fully in Docker
- **Modular**: env / task / grader are independently replaceable
- **Honest**: simulated parts are clearly labelled; the RL interface is real and stable
- **Robust**: invalid actions, infra failures, deadlocks, and step limits all handled
