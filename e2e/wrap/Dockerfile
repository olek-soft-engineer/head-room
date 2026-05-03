# ─── Stage 1: build the headroom-ai wheel in manylinux ─────────────────────
# Building from source inside `node:22-bookworm` produced a `_core.so` that
# referenced `__isoc23_strtoll` (a glibc 2.38+ symbol). The runtime in the
# same image couldn't resolve it at import time because something in the
# build chain — gcc version, libc6-dev backport, or cc-rs compiling
# transitive C deps against newer headers — emitted C23-era symbols that
# the runtime libc.so.6 doesn't have. Building inside the manylinux_2_28
# container (AlmaLinux 8, glibc 2.28 baseline) guarantees the wheel works
# on any glibc 2.28+ runtime, including bookworm.
FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

# `openssl-sys` (transitive via fastembed/hf-hub/ureq → native-tls) needs
# OpenSSL dev headers. perl-IPC-Cmd is required by openssl-sys's autotools
# probe.
RUN yum install -y openssl-devel pkgconfig perl-IPC-Cmd

# Install rust toolchain into the manylinux container. Match the
# rust-toolchain.toml at repo root (1.95.0 + rustfmt + clippy) so cargo
# doesn't try to mutate the toolchain on first invocation.
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/opt/python/cp311-cp311/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal -c rustfmt -c clippy --default-toolchain 1.95.0

WORKDIR /build

COPY pyproject.toml uv.lock README.md ./
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY headroom/ headroom/

# Build the wheel with maturin. PYO3_USE_ABI3_FORWARD_COMPATIBILITY allows
# building against Python 3.14+ until we bump pyo3 past 0.22.
ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
RUN /opt/python/cp311-cp311/bin/pip install 'maturin>=1.5,<2.0' && \
    /opt/python/cp311-cp311/bin/maturin build --release --out /dist --interpreter python3.11

# ─── Stage 2: node-based e2e runtime ────────────────────────────────────────
# Wraps Aider, Codex, OpenClaw on top of the prebuilt headroom-ai wheel.
# No rust/maturin needed at this stage — we just `pip install` the wheel
# from stage 1.
FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    AIDER_CHAT_VERSION=0.86.2 \
    CODEX_VERSION=0.118.0 \
    OPENCLAW_VERSION=2026.4.7 \
    PATH="/opt/headroom-venv/bin:/opt/aider-venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      python3 \
      python3-pip \
      python3-venv && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Bring in the prebuilt manylinux wheel from stage 1.
COPY --from=builder /dist/*.whl /tmp/wheels/

# `sdk/typescript` and `plugins/openclaw` are wired into the wrap-e2e
# harness (e2e/wrap/run.py invokes `npx openclaw`). DO NOT copy
# `headroom/` or `pyproject.toml` from the workspace — they would shadow
# the installed wheel's `headroom/` package via cwd and Python would
# import the source-only `headroom/` (which has no `_core.so`),
# triggering the same `ModuleNotFoundError` the wheel install fixes.
COPY sdk/typescript ./sdk/typescript
COPY plugins/openclaw ./plugins/openclaw

# Install the prebuilt manylinux wheel with [proxy] extras. Pip resolves
# extras from the wheel's metadata and pulls deps from public PyPI. The
# wheel's `_core.so` was compiled against glibc 2.28, so it loads cleanly
# against bookworm's glibc 2.36 at runtime.
RUN python -m venv /opt/headroom-venv && \
    /opt/headroom-venv/bin/python -m pip install --upgrade pip && \
    /opt/headroom-venv/bin/python -m pip install "$(ls /tmp/wheels/headroom_ai-*.whl)[proxy]" && \
    /opt/headroom-venv/bin/python -c "from headroom._core import DiffCompressor; print('headroom._core OK')" && \
    python -m venv /opt/aider-venv && \
    /opt/aider-venv/bin/python -m pip install --upgrade pip && \
    /opt/aider-venv/bin/python -m pip install "aider-chat==${AIDER_CHAT_VERSION}" && \
    npm install -g --no-fund --no-audit "@openai/codex@${CODEX_VERSION}" "openclaw@${OPENCLAW_VERSION}"

COPY e2e/wrap ./e2e/wrap

CMD ["python", "e2e/wrap/run.py"]
