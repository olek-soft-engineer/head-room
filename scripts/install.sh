#!/usr/bin/env bash

set -euo pipefail

IMAGE_DEFAULT="ghcr.io/chopratejas/headroom:latest"
INSTALL_DIR="${HOME}/.local/bin"
if [[ ! -d "${HOME}/.local" ]]; then
  INSTALL_DIR="${HOME}/bin"
fi

info() {
  printf '==> %s\n' "$*"
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

append_path_block() {
  local target_file="$1"
  local marker_start="# >>> headroom docker-native >>>"
  local marker_end="# <<< headroom docker-native <<<"
  local block="${marker_start}
export PATH=\"${INSTALL_DIR}:\$PATH\"
${marker_end}"

  touch "${target_file}"
  if grep -Fq "${marker_start}" "${target_file}"; then
    return
  fi

  {
    printf '\n%s\n' "${block}"
  } >>"${target_file}"
}

write_wrapper() {
  local wrapper_path="${INSTALL_DIR}/headroom"

  cat >"${wrapper_path}" <<'WRAPPER'
#!/usr/bin/env bash

set -euo pipefail

HEADROOM_IMAGE="${HEADROOM_DOCKER_IMAGE:-ghcr.io/chopratejas/headroom:latest}"
HEADROOM_CONTAINER_HOME="${HEADROOM_CONTAINER_HOME:-/tmp/headroom-home}"
HEADROOM_HOST_HOME="${HOME:?}"

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

detect_rtk_target() {
  local system
  local machine
  system="$(uname -s)"
  machine="$(uname -m)"

  case "${system}" in
    Darwin)
      if [[ "${machine}" == "arm64" ]]; then
        printf 'aarch64-apple-darwin'
      else
        printf 'x86_64-apple-darwin'
      fi
      ;;
    Linux)
      if [[ "${machine}" == "aarch64" ]]; then
        printf 'aarch64-unknown-linux-gnu'
      else
        printf 'x86_64-unknown-linux-musl'
      fi
      ;;
    *)
      die "Unsupported host platform for Docker-native wrapper: ${system}/${machine}"
      ;;
  esac
}

ensure_host_dirs() {
  mkdir -p \
    "${HEADROOM_HOST_HOME}/.headroom" \
    "${HEADROOM_HOST_HOME}/.claude" \
    "${HEADROOM_HOST_HOME}/.codex" \
    "${HEADROOM_HOST_HOME}/.gemini"
}

append_passthrough_envs() {
  local -n ref=$1
  local name

  for name in $(compgen -e); do
    case "${name}" in
      HEADROOM_*|ANTHROPIC_*|OPENAI_*|GEMINI_*|AWS_*|AZURE_*|VERTEX_*|GOOGLE_*|GOOGLE_CLOUD_*|MISTRAL_*|GROQ_*|OPENROUTER_*|XAI_*|TOGETHER_*|COHERE_*|OLLAMA_*|LITELLM_*|OTEL_*|SUPABASE_*|QDRANT_*|NEO4J_*|LANGSMITH_*)
        ref+=(--env "${name}")
        ;;
    esac
  done
}

append_common_container_args() {
  local -n ref=$1

  ensure_host_dirs
  ref+=(-w /workspace)
  ref+=(--env "HOME=${HEADROOM_CONTAINER_HOME}")
  ref+=(--env "PYTHONUNBUFFERED=1")
  ref+=(-v "${PWD}:/workspace")
  ref+=(-v "${HEADROOM_HOST_HOME}/.headroom:${HEADROOM_CONTAINER_HOME}/.headroom")
  ref+=(-v "${HEADROOM_HOST_HOME}/.claude:${HEADROOM_CONTAINER_HOME}/.claude")
  ref+=(-v "${HEADROOM_HOST_HOME}/.codex:${HEADROOM_CONTAINER_HOME}/.codex")
  ref+=(-v "${HEADROOM_HOST_HOME}/.gemini:${HEADROOM_CONTAINER_HOME}/.gemini")

  if command -v id >/dev/null 2>&1; then
    ref+=(--user "$(id -u):$(id -g)")
  fi

  append_passthrough_envs ref
}

append_tty_args() {
  local -n ref=$1

  if [[ -t 0 && -t 1 ]]; then
    ref+=(-it)
  elif [[ -t 0 ]]; then
    ref+=(-i)
  elif [[ -t 1 ]]; then
    ref+=(-t)
  fi
}

run_headroom() {
  local args=()
  args=(docker run --rm)
  append_tty_args args
  append_common_container_args args
  args+=(--entrypoint headroom "${HEADROOM_IMAGE}" "$@")
  "${args[@]}"
}

docker_container_exists() {
  local name="$1"
  docker ps --format '{{.Names}}' | grep -Fxq "${name}"
}

wait_for_proxy() {
  local container_name="$1"
  local port="$2"
  local attempt

  for attempt in $(seq 1 45); do
    if (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1; then
      return 0
    fi

    if ! docker_container_exists "${container_name}"; then
      break
    fi

    sleep 1
  done

  docker logs "${container_name}" >&2 || true
  return 1
}

start_proxy_container() {
  local port="$1"
  shift

  local container_name="headroom-proxy-${port}-$$"
  local args=()
  args=(docker run -d --rm --name "${container_name}" -p "${port}:${port}")
  append_common_container_args args
  args+=("${HEADROOM_IMAGE}" --host 0.0.0.0 --port "${port}" "$@")
  "${args[@]}" >/dev/null

  if ! wait_for_proxy "${container_name}" "${port}"; then
    docker stop "${container_name}" >/dev/null 2>&1 || true
    die "Headroom proxy failed to start on port ${port}"
  fi

  printf '%s\n' "${container_name}"
}

stop_proxy_container() {
  local container_name="${1:-}"
  if [[ -n "${container_name}" ]]; then
    docker stop "${container_name}" >/dev/null 2>&1 || true
  fi
}

run_claude_rtk_init() {
  local rtk_bin="${HEADROOM_HOST_HOME}/.headroom/bin/rtk"
  if [[ ! -x "${rtk_bin}" ]]; then
    warn "rtk was not installed at ${rtk_bin}; Claude hooks were not registered"
    return
  fi

  if ! "${rtk_bin}" init --global --auto-patch >/dev/null 2>&1; then
    warn "Failed to register Claude hooks with rtk; continuing without hook registration"
  fi
}

parse_wrap_args() {
  local -n out_known=$1
  local -n out_host=$2
  local -n out_port=$3
  local -n out_no_rtk=$4
  local -n out_no_proxy=$5
  local -n out_learn=$6
  local -n out_backend=$7
  local -n out_anyllm=$8
  local -n out_region=$9
  shift 9

  out_known=()
  out_host=()
  out_port=8787
  out_no_rtk=0
  out_no_proxy=0
  out_learn=0
  out_backend=""
  out_anyllm=""
  out_region=""

  while (($#)); do
    case "$1" in
      --)
        shift
        out_host+=("$@")
        break
        ;;
      --port|-p)
        out_port="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --port=*)
        out_port="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      --no-rtk)
        out_no_rtk=1
        out_known+=("$1")
        shift
        ;;
      --no-proxy)
        out_no_proxy=1
        out_known+=("$1")
        shift
        ;;
      --learn)
        out_learn=1
        out_known+=("$1")
        shift
        ;;
      --verbose|-v)
        out_known+=("$1")
        shift
        ;;
      --backend)
        out_backend="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --backend=*)
        out_backend="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      --anyllm-provider)
        out_anyllm="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --anyllm-provider=*)
        out_anyllm="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      --region)
        out_region="$2"
        out_known+=("$1" "$2")
        shift 2
        ;;
      --region=*)
        out_region="${1#*=}"
        out_known+=("$1")
        shift
        ;;
      *)
        out_host+=("$@")
        break
        ;;
    esac
  done
}

run_prepare_only() {
  local tool="$1"
  shift

  local args=()
  args=(docker run --rm)
  append_tty_args args
  append_common_container_args args
  args+=(--env "HEADROOM_RTK_TARGET=$(detect_rtk_target)")
  args+=(--entrypoint headroom "${HEADROOM_IMAGE}" wrap "${tool}" --prepare-only "$@")
  "${args[@]}"
}

run_host_tool() {
  local binary="$1"
  shift

  command -v "${binary}" >/dev/null 2>&1 || die "'${binary}' not found in PATH"
  "${binary}" "$@"
}

main() {
  require_cmd docker

  if (($# == 0)); then
    run_headroom --help
    return
  fi

  case "$1" in
    wrap)
      (($# >= 2)) || die "Usage: headroom wrap <claude|codex|aider|cursor> [...]"
      local tool="$2"
      shift 2

      local known_args host_args port no_rtk no_proxy learn backend anyllm region
      parse_wrap_args known_args host_args port no_rtk no_proxy learn backend anyllm region "$@"

      local proxy_args=()
      if [[ "${learn}" -eq 1 ]]; then
        proxy_args+=(--learn)
      fi
      if [[ -n "${backend}" ]]; then
        proxy_args+=(--backend "${backend}")
      fi
      if [[ -n "${anyllm}" ]]; then
        proxy_args+=(--anyllm-provider "${anyllm}")
      fi
      if [[ -n "${region}" ]]; then
        proxy_args+=(--region "${region}")
      fi

      case "${tool}" in
        claude|codex|aider|cursor)
          ;;
        openclaw)
          die "Docker-native install does not support 'headroom wrap openclaw' yet. Use a native Headroom install for OpenClaw plugin management."
          ;;
        *)
          die "Unsupported wrap target: ${tool}"
          ;;
      esac

      local container_name=""
      if [[ "${no_proxy}" -eq 0 ]]; then
        container_name="$(start_proxy_container "${port}" "${proxy_args[@]}")"
      fi
      trap 'stop_proxy_container "${container_name}"' EXIT INT TERM

      local prep_args=("${known_args[@]}")
      if [[ "${no_proxy}" -eq 0 ]]; then
        prep_args+=(--no-proxy)
      fi
      run_prepare_only "${tool}" "${prep_args[@]}"

      case "${tool}" in
        claude)
          if [[ "${no_rtk}" -eq 0 ]]; then
            run_claude_rtk_init
          fi
          ANTHROPIC_BASE_URL="http://127.0.0.1:${port}" run_host_tool claude "${host_args[@]}"
          ;;
        codex)
          OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" run_host_tool codex "${host_args[@]}"
          ;;
        aider)
          OPENAI_API_BASE="http://127.0.0.1:${port}/v1" \
          ANTHROPIC_BASE_URL="http://127.0.0.1:${port}" \
          run_host_tool aider "${host_args[@]}"
          ;;
        cursor)
          cat <<EOF
Headroom proxy is running for Cursor.

OpenAI base URL:     http://127.0.0.1:${port}/v1
Anthropic base URL:  http://127.0.0.1:${port}

Press Ctrl+C to stop the proxy.
EOF
          while true; do
            sleep 1
          done
          ;;
      esac
      ;;
    unwrap)
      if (($# >= 2)) && [[ "$2" == "openclaw" ]]; then
        die "Docker-native install does not support 'headroom unwrap openclaw' yet. Use a native Headroom install for OpenClaw plugin management."
      fi
      run_headroom "$@"
      ;;
    proxy)
      shift
      local port=8787
      local args=()
      args=(proxy)
      while (($#)); do
        case "$1" in
          --port|-p)
            port="$2"
            args+=("$1" "$2")
            shift 2
            ;;
          --port=*)
            port="${1#*=}"
            args+=("$1")
            shift
            ;;
          *)
            args+=("$1")
            shift
            ;;
        esac
      done
      local run_args=()
      run_args=(docker run --rm)
      append_tty_args run_args
      append_common_container_args run_args
      run_args+=(-p "${port}:${port}")
      run_args+=(--entrypoint headroom "${HEADROOM_IMAGE}" "${args[@]}")
      "${run_args[@]}"
      ;;
    *)
      run_headroom "$@"
      ;;
  esac
}

main "$@"
WRAPPER

  chmod +x "${wrapper_path}"
}

main() {
  require_cmd docker
  docker version >/dev/null 2>&1 || die "Docker is installed but not available to the current user"

  mkdir -p "${INSTALL_DIR}"
  write_wrapper

  append_path_block "${HOME}/.bashrc"
  append_path_block "${HOME}/.zshrc"
  append_path_block "${HOME}/.profile"

  info "Pulling ${IMAGE_DEFAULT}"
  docker pull "${IMAGE_DEFAULT}" >/dev/null

  cat <<EOF

Headroom Docker-native install complete.

Installed wrapper:
  ${INSTALL_DIR}/headroom

Next steps:
  1. Restart your shell or run: export PATH="${INSTALL_DIR}:\$PATH"
  2. Try: headroom proxy
  3. Docs: https://github.com/chopratejas/headroom/blob/main/docs/docker-install.md
EOF
}

main "$@"
