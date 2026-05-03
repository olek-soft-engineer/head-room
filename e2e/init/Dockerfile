# ─── Stage 1: build the headroom-ai wheel in manylinux ─────────────────────
# See e2e/wrap/Dockerfile for the full rationale. tl;dr: building from
# source inside `node:22-bookworm` produced a `_core.so` referencing
# `__isoc23_strtoll` (glibc 2.38+) that the same image's runtime libc
# couldn't resolve. Building inside manylinux_2_28 (AlmaLinux 8, glibc
# 2.28 baseline) yields a wheel portable to any glibc 2.28+ runtime.
FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

RUN yum install -y openssl-devel pkgconfig perl-IPC-Cmd

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

ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
RUN /opt/python/cp311-cp311/bin/pip install 'maturin>=1.5,<2.0' && \
    /opt/python/cp311-cp311/bin/maturin build --release --out /dist --interpreter python3.11

# ─── Stage 2: node-based init e2e runtime ──────────────────────────────────
FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/headroom-venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      python3 \
      python3-pip \
      python3-venv && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /dist/*.whl /tmp/wheels/

# Init e2e harness imports from e2e._lib + .claude-plugin assets.
# DO NOT copy `headroom/` or `pyproject.toml` from the workspace —
# the installed wheel must be authoritative. The harness packages
# (e2e/_lib/, .claude-plugin/, plugins/) don't shadow `headroom`.
COPY .claude-plugin ./.claude-plugin
COPY .github/plugin ./.github/plugin
COPY plugins/headroom-agent-hooks ./plugins/headroom-agent-hooks
COPY e2e/__init__.py ./e2e/__init__.py
COPY e2e/_lib ./e2e/_lib
COPY e2e/init ./e2e/init

RUN python -m venv /opt/headroom-venv && \
    /opt/headroom-venv/bin/python -m pip install --upgrade "pip<25" && \
    /opt/headroom-venv/bin/python -m pip install "$(ls /tmp/wheels/headroom_ai-*.whl)[proxy]" && \
    /opt/headroom-venv/bin/python -c "from headroom._core import DiffCompressor; print('headroom._core OK')"

CMD ["python", "e2e/init/run.py"]
