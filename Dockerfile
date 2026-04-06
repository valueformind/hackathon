# ═══════════════════════════════════════════════════════════════
# Stage 1 — dependency layer (cached separately from source code)
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS deps

WORKDIR /app

# System deps needed by the Python docker SDK (for my_env_v4 container mgmt)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ═══════════════════════════════════════════════════════════════
# Stage 2 — final runtime image
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

LABEL description="RL Coding Environment — offline-first test generation agent"
LABEL python="3.11"

# Never write .pyc files; always flush stdout/stderr immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime system deps (ca-certs needed for HTTPS calls to HF / OpenAI)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from deps stage (keeps this layer small)
COPY --from=deps /usr/local/lib/python3.11/site-packages \
                 /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy source code (changes here don't bust the dependency layer)
COPY . /app

# ── Runtime env vars (all overridable at docker run / compose) ──────────
# HF_TOKEN and API_KEY are intentionally NOT set here.
# Pass them at runtime:  docker run -e HF_TOKEN=$HF_TOKEN ...
# Setting secrets in ENV bakes them into image layers — never do that.
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# my_env_v4 task settings (used by inference.py)
ENV IMAGE_NAME=""
ENV MY_ENV_V4_TASK="echo"
ENV MY_ENV_V4_BENCHMARK="my_env_v4"

# ── Default entry point: offline demo (no API key required) ─────────────
# Override at runtime:
#   docker run --rm rl-coding-env python demo_policies.py
#   docker run --rm rl-coding-env python -m unittest discover -s tests -p 'test_*.py' -v
#   docker run --rm --env-file .env \
#              -v /var/run/docker.sock:/var/run/docker.sock \
#              rl-coding-env python inference.py
CMD ["python", "demo.py"]
