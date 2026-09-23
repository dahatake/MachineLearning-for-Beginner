#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLFB_SETUP_OS="${MLFB_SETUP_OS:-$(uname -s | tr '[:upper:]' '[:lower:]')}"
case "$MLFB_SETUP_OS" in
    darwin) MLFB_SETUP_OS="macos" ;;
esac
export MLFB_SETUP_OS

# shellcheck source=setup/lib/common.sh
. "${SCRIPT_DIR}/lib/common.sh"

mlfb_start_jupyter_main "$@"
