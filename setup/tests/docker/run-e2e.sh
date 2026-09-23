#!/usr/bin/env bash
set -euo pipefail

mode="${MLFB_MODE:-venv}"
run_notebooks="${MLFB_RUN_NOTEBOOKS:-0}"
second_run="${MLFB_SECOND_RUN:-1}"
remove_after="${MLFB_REMOVE_AFTER:-1}"

case "${mode}" in
    anaconda|miniconda|venv) ;;
    *)
        echo "MLFB_MODE must be anaconda, miniconda, or venv: ${mode}" >&2
        exit 2
        ;;
esac

validate_boolean() {
    value_name="$1"
    value="$2"
    case "${value}" in
        0|1) ;;
        *)
            echo "${value_name} must be 0 or 1: ${value}" >&2
            exit 2
            ;;
    esac
}

validate_boolean MLFB_RUN_NOTEBOOKS "${run_notebooks}"
validate_boolean MLFB_SECOND_RUN "${second_run}"
validate_boolean MLFB_REMOVE_AFTER "${remove_after}"

if [ "$(id -u)" -eq 0 ]; then
    echo "Docker E2E must run as a non-root user." >&2
    exit 1
fi
if command -v apt-get >/dev/null 2>&1; then
    sudo_command="apt-get"
elif command -v dnf >/dev/null 2>&1; then
    sudo_command="dnf"
else
    echo "Supported package manager not found." >&2
    exit 1
fi
if ! sudo -n "${sudo_command}" --version >/dev/null; then
    echo "Passwordless sudo is not available for ${sudo_command}." >&2
    exit 1
fi

export MLFB_TEST_ISOLATE=1

setup_args=(
    --mode "${mode}"
    --yes
    --skip-data-download
)
if [ "${mode}" = "anaconda" ]; then
    setup_args+=(--accept-anaconda-tos)
fi
if [ "${run_notebooks}" = "1" ]; then
    setup_args+=(--run-notebooks)
fi

bash /workspace/setup/setup-linux.sh "${setup_args[@]}"
if [ "${second_run}" = "1" ]; then
    bash /workspace/setup/setup-linux.sh "${setup_args[@]}"
fi

if [ "${run_notebooks}" = "1" ]; then
    isolate="/tmp/mlfb-test-home-$(id -u)-workspace"
    if [ "${mode}" = "venv" ]; then
        /workspace/.venv/bin/python \
            /workspace/setup/tools/check_results.py --root /workspace
    else
        "${isolate}/${mode}3/bin/conda" run --no-capture-output \
            --name mlfb-mnist \
            python /workspace/setup/tools/check_results.py --root /workspace
    fi
fi

if [ "${remove_after}" = "1" ]; then
    bash /workspace/setup/setup-linux.sh --remove "${mode}" --yes
fi
