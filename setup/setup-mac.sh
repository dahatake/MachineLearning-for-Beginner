#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLFB_SETUP_OS="macos"
export MLFB_SETUP_OS

# shellcheck source=setup/lib/common.sh
. "${SCRIPT_DIR}/lib/common.sh"

mlfb_setup_main "$@"
