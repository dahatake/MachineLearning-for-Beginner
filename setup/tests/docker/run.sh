#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash setup/tests/docker/run.sh DISTRO MODE [options]

DISTRO:
  ubuntu-22.04 | ubuntu-24.04 | ubuntu-26.04
  debian-12 | debian-13 | fedora-latest | all

MODE:
  anaconda | miniconda | venv | all

Options:
  --run-notebooks  Execute notebooks and check their stable results.
  --single-run     Do not repeat setup to test idempotency.
  --keep           Do not run setup-linux.sh --remove after validation.
EOF
}

[ "$#" -ge 2 ] || {
    usage >&2
    exit 2
}

distro="$1"
mode="$2"
shift 2

run_notebooks=0
second_run=1
remove_after=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-notebooks)
            run_notebooks=1
            ;;
        --single-run)
            second_run=0
            ;;
        --keep)
            remove_after=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

case "${distro}" in
    ubuntu-22.04|ubuntu-24.04|ubuntu-26.04|debian-12|debian-13|fedora-latest|all) ;;
    *)
        usage >&2
        echo "Unknown distro: ${distro}" >&2
        exit 2
        ;;
esac

case "${mode}" in
    anaconda|miniconda|venv|all) ;;
    *)
        usage >&2
        echo "Unknown mode: ${mode}" >&2
        exit 2
        ;;
esac

if [ "${distro}" = "all" ]; then
    distros="ubuntu-22.04 ubuntu-24.04 ubuntu-26.04 debian-12 debian-13 fedora-latest"
else
    distros="${distro}"
fi

if [ "${mode}" = "all" ]; then
    modes="anaconda miniconda venv"
else
    modes="${mode}"
fi

for current_distro in ${distros}; do
    case "${current_distro}" in
        ubuntu-22.04)
            dockerfile="${script_dir}/Dockerfile.debian"
            base_image="ubuntu:22.04"
            ;;
        ubuntu-24.04)
            dockerfile="${script_dir}/Dockerfile.debian"
            base_image="ubuntu:24.04"
            ;;
        ubuntu-26.04)
            dockerfile="${script_dir}/Dockerfile.debian"
            base_image="ubuntu:26.04"
            ;;
        debian-12)
            dockerfile="${script_dir}/Dockerfile.debian"
            base_image="debian:12"
            ;;
        debian-13)
            dockerfile="${script_dir}/Dockerfile.debian"
            base_image="debian:13"
            ;;
        fedora-latest)
            dockerfile="${script_dir}/Dockerfile.fedora"
            base_image="fedora:latest"
            ;;
    esac

    image="mlfb-setup-e2e:${current_distro}"
    docker build \
        --file "${dockerfile}" \
        --build-arg "BASE_IMAGE=${base_image}" \
        --tag "${image}" \
        "${repository_root}"

    for current_mode in ${modes}; do
        echo "Running ${current_distro} / ${current_mode}"
        docker run --rm \
            --env "MLFB_MODE=${current_mode}" \
            --env "MLFB_RUN_NOTEBOOKS=${run_notebooks}" \
            --env "MLFB_SECOND_RUN=${second_run}" \
            --env "MLFB_REMOVE_AFTER=${remove_after}" \
            "${image}"
    done
done
