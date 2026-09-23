# shellcheck shell=bash
# Shared macOS/Linux setup implementation for MachineLearning-for-Beginner.
# Bash 3.2 compatible: no associative arrays, mapfile, or newer test syntax.

set -euo pipefail

MLFB_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLFB_SETUP_DIR="$(cd "${MLFB_COMMON_DIR}/.." && pwd)"
MLFB_REPOSITORY_ROOT="$(cd "${MLFB_SETUP_DIR}/.." && pwd)"
MLFB_ENVIRONMENT_NAME="mlfb-mnist"
MLFB_VENV_KERNEL_NAME="mlfb-venv"
MLFB_ENVIRONMENT_FILE="${MLFB_SETUP_DIR}/envs/environment.yml"
MLFB_CONFIG_FILE="${MLFB_SETUP_DIR}/config/versions.conf"
MLFB_REQUIREMENTS_BASE="${MLFB_SETUP_DIR}/requirements/venv-base.txt"
MLFB_REQUIREMENTS_TORCH="${MLFB_SETUP_DIR}/requirements/venv-torch.txt"
MLFB_VERIFY_TOOL="${MLFB_SETUP_DIR}/tools/verify_env.py"
MLFB_RUN_NOTEBOOKS_TOOL="${MLFB_SETUP_DIR}/tools/run_notebooks.py"
MLFB_LOG_DIR="${MLFB_SETUP_DIR}/logs"
MLFB_STATE_DIR="${MLFB_SETUP_DIR}/.state"
MLFB_DATA_DIR="${MLFB_REPOSITORY_ROOT}/data"
MLFB_VENV_DIR="${MLFB_REPOSITORY_ROOT}/.venv"
MLFB_TEMP_DIR=""
MLFB_LOG_FILE=""
MLFB_LOGGING_STARTED=0
MLFB_INSTALLED_CONDA=""

MLFB_MODE=""
MLFB_RUN_NOTEBOOKS=0
MLFB_SKIP_DATA_DOWNLOAD=0
MLFB_YES=0
MLFB_ACCEPT_ANACONDA_TOS=0
MLFB_INSTALL_ROOT="${HOME}"
MLFB_CONDA_PATH="${CONDA_EXE:-}"
MLFB_INIT_SHELL=0
MLFB_CHECK=0
MLFB_DRY_RUN=0
MLFB_REMOVE_MODE=""
MLFB_TEST_ISOLATE="${MLFB_TEST_ISOLATE:-0}"
MLFB_ISOLATED_HOME=""

mlfb_usage() {
    cat <<'EOF'
使い方:
  bash setup/setup-mac.sh [--mode anaconda|miniconda|venv|all] [options]
  bash setup/setup-linux.sh [--mode anaconda|miniconda|venv|all] [options]

主なオプション:
  --mode MODE                 anaconda, miniconda, venv, all
  --run-notebooks             セットアップ後に全 Notebook を実行する
  --skip-data-download        MNIST データの事前ダウンロードを省略する
  --yes                       確認を省略する（ToS 同意は含みません）
  --accept-anaconda-tos       Anaconda の利用規約に明示的に同意する
  --install-root PATH         Anaconda/Miniconda を入れる親フォルダー
  --conda PATH                既存の conda 実行ファイルを使う
  --init-shell                conda init を実行する
  --check                     診断だけを行い、何も変更しない
  --dry-run                   実行予定だけを表示する
  --remove MODE               作成した環境だけを削除する
  -h, --help                  このヘルプを表示する
EOF
}

mlfb_enable_test_isolation() {
    [ "${MLFB_TEST_ISOLATE}" = "1" ] || return 0

    MLFB_ISOLATED_HOME="${TMPDIR:-/tmp}/mlfb-test-home-$(id -u)-$(basename "${MLFB_REPOSITORY_ROOT}")"
    mkdir -p "${MLFB_ISOLATED_HOME}"
    export HOME="${MLFB_ISOLATED_HOME}"
    unset CONDA_EXE CONDA_PREFIX CONDA_DEFAULT_ENV
    MLFB_CONDA_PATH=""
    MLFB_INSTALL_ROOT="${HOME}"
    mlfb_info "テスト隔離モード: ${MLFB_ISOLATED_HOME}"
}

mlfb_start_jupyter_usage() {
    cat <<'EOF'
使い方:
  bash setup/start-jupyter.sh [--mode anaconda|miniconda|venv] [--navigator] [--conda PATH]

Jupyter Notebook をリポジトリのルートで起動します。
--navigator を付けると Anaconda Navigator の起動を試します。
EOF
}

mlfb_die() {
    echo "エラー: $*" >&2
    if [ -n "${MLFB_LOG_FILE}" ]; then
        echo "ログ: ${MLFB_LOG_FILE}" >&2
    fi
    exit 1
}

mlfb_info() {
    printf '%s\n' "$*"
}

mlfb_step() {
    mlfb_info "[$1/7] $2（$3）"
}

mlfb_cleanup() {
    if [ -n "${MLFB_TEMP_DIR}" ] && [ -d "${MLFB_TEMP_DIR}" ]; then
        rm -rf "${MLFB_TEMP_DIR}"
    fi
}

trap mlfb_cleanup EXIT

mlfb_start_logging() {
    if [ "${MLFB_LOGGING_STARTED}" -eq 1 ] || [ "${MLFB_CHECK}" -eq 1 ]; then
        return 0
    fi
    mkdir -p "${MLFB_LOG_DIR}"
    MLFB_LOG_FILE="${MLFB_LOG_DIR}/setup-${MLFB_SETUP_OS}-$(date '+%Y%m%d-%H%M%S').log"
    touch "${MLFB_LOG_FILE}"
    exec > >(tee -a "${MLFB_LOG_FILE}") 2>&1
    MLFB_LOGGING_STARTED=1
    mlfb_info "ログ: ${MLFB_LOG_FILE}"
}

mlfb_load_config() {
    [ -f "${MLFB_CONFIG_FILE}" ] || mlfb_die "設定ファイルが見つかりません: ${MLFB_CONFIG_FILE}"
    # versions.conf is maintained as simple key=value assignments for setup scripts.
    # shellcheck disable=SC1090
    . "${MLFB_CONFIG_FILE}"
}

mlfb_parse_setup_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --mode)
                [ "$#" -ge 2 ] || mlfb_die "--mode には値が必要です。"
                MLFB_MODE="$2"
                shift 2
                ;;
            --run-notebooks)
                MLFB_RUN_NOTEBOOKS=1
                shift
                ;;
            --skip-data-download)
                MLFB_SKIP_DATA_DOWNLOAD=1
                shift
                ;;
            --yes)
                MLFB_YES=1
                shift
                ;;
            --accept-anaconda-tos)
                MLFB_ACCEPT_ANACONDA_TOS=1
                shift
                ;;
            --install-root)
                [ "$#" -ge 2 ] || mlfb_die "--install-root にはパスが必要です。"
                MLFB_INSTALL_ROOT="$2"
                shift 2
                ;;
            --conda)
                [ "$#" -ge 2 ] || mlfb_die "--conda にはパスが必要です。"
                MLFB_CONDA_PATH="$2"
                shift 2
                ;;
            --init-shell)
                MLFB_INIT_SHELL=1
                shift
                ;;
            --check)
                MLFB_CHECK=1
                shift
                ;;
            --dry-run)
                MLFB_DRY_RUN=1
                shift
                ;;
            --remove)
                [ "$#" -ge 2 ] || mlfb_die "--remove には mode が必要です。"
                MLFB_REMOVE_MODE="$2"
                shift 2
                ;;
            -h|--help)
                mlfb_usage
                exit 0
                ;;
            *)
                mlfb_usage >&2
                mlfb_die "不明なオプションです: $1"
                ;;
        esac
    done
}

mlfb_validate_mode() {
    case "$1" in
        anaconda|miniconda|venv|all) return 0 ;;
        *) mlfb_die "mode は anaconda, miniconda, venv, all のいずれかです: $1" ;;
    esac
}

mlfb_prompt_yes_no() {
    prompt="$1"
    if [ "${MLFB_YES}" -eq 1 ]; then
        return 1
    fi
    printf '%s [y/N]: ' "${prompt}"
    read -r answer || mlfb_die "入力を読み取れません。非対話実行では --yes と必要なオプションを指定してください。"
    case "${answer}" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

mlfb_choose_mode() {
    if [ -n "${MLFB_MODE}" ]; then
        mlfb_validate_mode "${MLFB_MODE}"
        return 0
    fi
    if [ "${MLFB_YES}" -eq 1 ]; then
        MLFB_MODE="anaconda"
        return 0
    fi
    cat <<'EOF'
セットアップ方法を選んでください。
  1: Anaconda（おすすめ。GUI の Navigator 付き。容量は大きめ）
  2: Miniconda（軽い conda）
  3: venv（Python 標準）
  4: すべて
EOF
    printf '番号を入力してください [1]: '
    read -r choice || mlfb_die "入力を読み取れません。非対話実行では --mode と --yes を指定してください。"
    case "${choice:-1}" in
        1) MLFB_MODE="anaconda" ;;
        2) MLFB_MODE="miniconda" ;;
        3) MLFB_MODE="venv" ;;
        4) MLFB_MODE="all" ;;
        *) mlfb_die "有効な番号を入力してください。" ;;
    esac
}

mlfb_command_exists() {
    command -v "$1" >/dev/null 2>&1
}

mlfb_sha256_file() {
    file_path="$1"
    if mlfb_command_exists sha256sum; then
        sha256sum "${file_path}" | awk '{print $1}'
    elif mlfb_command_exists shasum; then
        shasum -a 256 "${file_path}" | awk '{print $1}'
    else
        mlfb_die "SHA-256 検証に必要な sha256sum または shasum が見つかりません。"
    fi
}

mlfb_download_file() {
    url="$1"
    output="$2"
    if [ "${MLFB_DRY_RUN}" -eq 1 ]; then
        mlfb_info "[dry-run] download ${url} -> ${output}"
        return 0
    fi
    if mlfb_command_exists curl; then
        curl --fail --location --show-error --progress-bar "${url}" --output "${output}"
    elif mlfb_command_exists wget; then
        wget --progress=dot:giga -O "${output}" "${url}"
    else
        mlfb_die "curl または wget が必要です。"
    fi
}

mlfb_run() {
    description="$1"
    shift
    if [ "${MLFB_DRY_RUN}" -eq 1 ]; then
        printf '[dry-run] %s:' "${description}"
        for arg in "$@"; do
            printf ' %s' "${arg}"
        done
        printf '\n'
        return 0
    fi
    "$@"
}

mlfb_warn_path() {
    case "${MLFB_REPOSITORY_ROOT}" in
        *OneDrive*|*" "*)
            mlfb_info "注意: リポジトリのパスに OneDrive または空白が含まれます。問題が出たら ~/Work などへ移動してください。"
            ;;
    esac
}

mlfb_free_space_gb() {
    df -Pk "${MLFB_REPOSITORY_ROOT}" | awk 'NR==2 {printf "%.1f", $4 / 1024 / 1024}'
}

mlfb_preflight() {
    mlfb_step 1 "事前チェックをしています" "1 分"
    [ -f "${MLFB_ENVIRONMENT_FILE}" ] || mlfb_die "環境定義が見つかりません: ${MLFB_ENVIRONMENT_FILE}"
    [ -f "${MLFB_REQUIREMENTS_BASE}" ] || mlfb_die "requirements が見つかりません: ${MLFB_REQUIREMENTS_BASE}"
    [ -f "${MLFB_REQUIREMENTS_TORCH}" ] || mlfb_die "requirements が見つかりません: ${MLFB_REQUIREMENTS_TORCH}"
    [ -f "${MLFB_VERIFY_TOOL}" ] || mlfb_die "検証ツールが見つかりません: ${MLFB_VERIFY_TOOL}"

    case "${MLFB_SETUP_OS}" in
        macos)
            [ "$(uname -s)" = "Darwin" ] || mlfb_die "このスクリプトは macOS 用です。Linux では setup-linux.sh を使ってください。"
            ;;
        linux)
            [ "$(uname -s)" = "Linux" ] || mlfb_die "このスクリプトは Linux 用です。macOS では setup-mac.sh を使ってください。"
            ;;
        *)
            mlfb_die "未対応の OS です: ${MLFB_SETUP_OS}"
            ;;
    esac

    arch="$(uname -m)"
    case "${arch}" in
        x86_64|amd64|arm64|aarch64) ;;
        *) mlfb_die "未対応のアーキテクチャです: ${arch}" ;;
    esac
    mlfb_info "OS: ${MLFB_SETUP_OS}"
    mlfb_info "Architecture: ${arch}"
    mlfb_info "Repository: ${MLFB_REPOSITORY_ROOT}"
    mlfb_info "空き容量: $(mlfb_free_space_gb) GB"
    mlfb_warn_path

    if mlfb_command_exists curl; then
        mlfb_info "Network tool: curl"
    elif mlfb_command_exists wget; then
        mlfb_info "Network tool: wget"
    else
        mlfb_info "Network tool: なし（conda のインストーラー取得には curl または wget が必要です）"
    fi
}

mlfb_platform_key() {
    arch="$(uname -m)"
    case "${MLFB_SETUP_OS}:${arch}" in
        macos:arm64|macos:aarch64) printf '%s\n' "macos_arm64" ;;
        macos:x86_64|macos:amd64) printf '%s\n' "macos_intel" ;;
        linux:x86_64|linux:amd64) printf '%s\n' "linux_x64" ;;
        linux:arm64|linux:aarch64) printf '%s\n' "linux_arm64" ;;
        *) mlfb_die "未対応の OS/アーキテクチャです: ${MLFB_SETUP_OS}/$(uname -m)" ;;
    esac
}

mlfb_config_value() {
    key="$1"
    eval "value=\${${key}:-}"
    [ -n "${value}" ] || mlfb_die "versions.conf に ${key} がありません。"
    printf '%s\n' "${value}"
}

mlfb_default_conda_prefix() {
    selected_mode="$1"
    if [ "${selected_mode}" = "anaconda" ]; then
        printf '%s\n' "${MLFB_INSTALL_ROOT}/anaconda3"
    else
        printf '%s\n' "${MLFB_INSTALL_ROOT}/miniconda3"
    fi
}

mlfb_find_conda() {
    selected_mode="$1"
    if [ -n "${MLFB_CONDA_PATH}" ]; then
        if [ -x "${MLFB_CONDA_PATH}" ]; then
            printf '%s\n' "${MLFB_CONDA_PATH}"
            return 0
        fi
        if mlfb_command_exists "${MLFB_CONDA_PATH}"; then
            command -v "${MLFB_CONDA_PATH}"
            return 0
        fi
        return 1
    fi

    prefix="$(mlfb_default_conda_prefix "${selected_mode}")"
    if [ "${selected_mode}" = "anaconda" ]; then
        candidates="${prefix}/bin/conda
${HOME}/anaconda3/bin/conda
${HOME}/opt/anaconda3/bin/conda
/opt/anaconda3/bin/conda"
    else
        candidates="${prefix}/bin/conda
${HOME}/miniconda3/bin/conda
${HOME}/miniforge3/bin/conda
/opt/miniconda3/bin/conda
/opt/miniforge3/bin/conda"
    fi

    old_ifs="${IFS}"
    IFS='
'
    for candidate in ${candidates}; do
        IFS="${old_ifs}"
        if [ -x "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    IFS="${old_ifs}"

    if [ "${MLFB_TEST_ISOLATE}" != "1" ] && mlfb_command_exists conda; then
        command -v conda
        return 0
    fi
    return 1
}

mlfb_install_conda() {
    selected_mode="$1"
    platform_key="$(mlfb_platform_key)"
    name="$(mlfb_config_value "${selected_mode}_${platform_key}_name")"
    expected_sha="$(mlfb_config_value "${selected_mode}_${platform_key}_sha256")"
    prefix="$(mlfb_default_conda_prefix "${selected_mode}")"

    if [ "${selected_mode}" = "anaconda" ]; then
        url="https://repo.anaconda.com/archive/${name}"
    else
        url="https://repo.anaconda.com/miniconda/${name}"
    fi

    if [ -e "${prefix}" ] && [ ! -x "${prefix}/bin/conda" ]; then
        mlfb_die "Conda を含まない既存のパスには上書きしません: ${prefix}"
    fi

    mlfb_step 3 "${selected_mode} をインストールしています" "10〜20 分"
    MLFB_TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlfb-setup.XXXXXX")"
    installer="${MLFB_TEMP_DIR}/${name}"
    mlfb_download_file "${url}" "${installer}"

    if [ "${MLFB_DRY_RUN}" -eq 0 ]; then
        actual_sha="$(mlfb_sha256_file "${installer}")"
        if [ "${actual_sha}" != "${expected_sha}" ]; then
            rm -f "${installer}"
            mlfb_die "インストーラーの SHA-256 が一致しません。期待値: ${expected_sha} / 実測値: ${actual_sha}"
        fi
        mlfb_info "SHA-256 を検証しました: ${name}"
        mkdir -p "$(dirname "${prefix}")"
    fi

    case "${name}" in
        *.pkg)
            [ "${MLFB_SETUP_OS}" = "macos" ] || mlfb_die ".pkg インストーラーは macOS でだけ使えます: ${name}"
            if [ "${prefix}" != "${HOME}/anaconda3" ]; then
                mlfb_info "注意: macOS の Anaconda .pkg はユーザー領域の既定パスにインストールします。"
            fi
            mlfb_run "${selected_mode} pkg installer" installer -pkg "${installer}" -target CurrentUserHomeDirectory
            if [ "${MLFB_DRY_RUN}" -eq 1 ]; then
                MLFB_INSTALLED_CONDA="${HOME}/opt/anaconda3/bin/conda"
            elif MLFB_INSTALLED_CONDA="$(mlfb_find_conda "${selected_mode}")"; then
                :
            else
                mlfb_die "macOS の Anaconda インストール後に conda を見つけられませんでした。"
            fi
            ;;
        *.sh)
            mlfb_run "${selected_mode} shell installer" bash "${installer}" -b -p "${prefix}"
            MLFB_INSTALLED_CONDA="${prefix}/bin/conda"
            ;;
        *)
            mlfb_die "未対応のインストーラー形式です: ${name}"
            ;;
    esac
}

mlfb_accept_anaconda_tos() {
    conda="$1"
    if [ "${MLFB_DRY_RUN}" -eq 1 ]; then
        mlfb_info "[dry-run] Anaconda ToS の同意確認と conda tos accept を表示します。"
        return 0
    fi
    if [ "${MLFB_ACCEPT_ANACONDA_TOS}" -ne 1 ]; then
        if [ "${MLFB_YES}" -eq 1 ]; then
            mlfb_die "Anaconda の利用規約へ同意する場合は --accept-anaconda-tos を明示してください。--yes だけでは同意しません。"
        fi
        cat <<'EOF'
Anaconda の defaults チャンネルを使うには、Anaconda の利用規約への同意が必要な場合があります。
個人や小規模な学習用途では無料で使える場合がありますが、学校や会社のルールも確認してください。
公式情報: https://www.anaconda.com/legal
EOF
        if mlfb_prompt_yes_no "Anaconda の利用規約に同意しますか？"; then
            MLFB_ACCEPT_ANACONDA_TOS=1
        else
            mlfb_die "Anaconda の利用規約に同意しなかったため中止します。"
        fi
    fi

    mlfb_info "Anaconda ToS への同意を conda に記録します。"
    "${conda}" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || "${conda}" tos accept >/dev/null 2>&1 || true
    "${conda}" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
}

mlfb_conda_platform_expected() {
    case "$(mlfb_platform_key)" in
        macos_arm64) printf '%s\n' "osx-arm64" ;;
        macos_intel) printf '%s\n' "osx-64" ;;
        linux_x64) printf '%s\n' "linux-64" ;;
        linux_arm64) printf '%s\n' "linux-aarch64" ;;
    esac
}

mlfb_validate_conda_platform() {
    conda="$1"
    [ "${MLFB_DRY_RUN}" -eq 1 ] && return 0
    expected="$(mlfb_conda_platform_expected)"
    platform="$("${conda}" info --json | sed -n 's/^[[:space:]]*"platform":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
    [ "${platform}" = "${expected}" ] || mlfb_die "ネイティブ Conda が必要です。期待値: ${expected} / 検出値: ${platform:-不明}"
    mlfb_info "$("${conda}" --version) platform=${platform}"
}

mlfb_conda_env_exists() {
    conda="$1"
    [ "${MLFB_DRY_RUN}" -eq 1 ] && return 1
    "${conda}" run --name "${MLFB_ENVIRONMENT_NAME}" python -c "pass" >/dev/null 2>&1
}

mlfb_setup_conda_mode() {
    selected_mode="$1"
    conda=""
    if conda="$(mlfb_find_conda "${selected_mode}")"; then
        mlfb_info "Conda: ${conda}"
    else
        mlfb_install_conda "${selected_mode}"
        conda="${MLFB_INSTALLED_CONDA}"
    fi

    mlfb_validate_conda_platform "${conda}"
    if [ "${selected_mode}" = "anaconda" ]; then
        mlfb_accept_anaconda_tos "${conda}"
    fi

    mlfb_step 4 "Conda 環境を作成または更新しています" "5〜15 分"
    if mlfb_conda_env_exists "${conda}"; then
        CONDA_CHANNEL_PRIORITY=flexible mlfb_run "conda env update" \
            "${conda}" env update --name "${MLFB_ENVIRONMENT_NAME}" --file "${MLFB_ENVIRONMENT_FILE}" --prune
    else
        CONDA_CHANNEL_PRIORITY=flexible mlfb_run "conda env create" \
            "${conda}" env create --name "${MLFB_ENVIRONMENT_NAME}" --file "${MLFB_ENVIRONMENT_FILE}" --yes
    fi

    if [ "${MLFB_INIT_SHELL}" -eq 1 ]; then
        mlfb_run "conda init" "${conda}" init "$(basename "${SHELL:-bash}")"
    fi

    MLFB_LAST_PYTHON_KIND="conda"
    MLFB_LAST_PYTHON_RUNNER="${conda}"
}

mlfb_python_version_ok() {
    python_cmd="$1"
    "${python_cmd}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)' >/dev/null 2>&1
}

mlfb_find_python() {
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        if mlfb_command_exists "${candidate}" && mlfb_python_version_ok "${candidate}"; then
            command -v "${candidate}"
            return 0
        fi
    done
    return 1
}

mlfb_install_isolated_macos_python() {
    [ "${MLFB_TEST_ISOLATE}" = "1" ] || return 0
    [ "${MLFB_SETUP_OS}" = "macos" ] || return 0

    version="$(mlfb_config_value python_version)"
    installer_name="python-${version}-macos11.pkg"
    python_path="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
    expected_sha="$(mlfb_config_value python_macos_universal2_sha256)"
    url="https://www.python.org/ftp/python/${version}/${installer_name}"

    if [ "${MLFB_DRY_RUN}" -eq 1 ]; then
        mlfb_info "[dry-run] ${url} を ${python_path} へインストールします。"
        printf '%s\n' "${python_path}"
        return 0
    fi
    if [ -x "${python_path}" ] && mlfb_python_version_ok "${python_path}"; then
        printf '%s\n' "${python_path}"
        return 0
    fi

    mlfb_info "テスト隔離用の Python ${version} をインストールします。"
    if [ -z "${MLFB_TEMP_DIR}" ]; then
        MLFB_TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlfb-setup.XXXXXX")"
    fi
    installer="${MLFB_TEMP_DIR}/${installer_name}"
    mlfb_download_file "${url}" "${installer}"
    actual_sha="$(mlfb_sha256_file "${installer}")"
    [ "${actual_sha}" = "${expected_sha}" ] || mlfb_die "Python インストーラーの SHA-256 が一致しません。期待値: ${expected_sha} / 実測値: ${actual_sha}"
    mlfb_run_privileged "test Python installer" installer -pkg "${installer}" -target /
    [ -x "${python_path}" ] || mlfb_die "テスト用 Python が見つかりません: ${python_path}"
    printf '%s\n' "${python_path}"
}

mlfb_sudo_prefix() {
    if [ "$(id -u)" -eq 0 ]; then
        printf '%s\n' ""
    elif mlfb_command_exists sudo; then
        printf '%s\n' "sudo"
    else
        return 1
    fi
}

mlfb_run_privileged() {
    description="$1"
    shift
    if [ "$(id -u)" -eq 0 ]; then
        mlfb_run "${description}" "$@"
    elif mlfb_command_exists sudo; then
        mlfb_run "${description}" sudo "$@"
    else
        mlfb_die "${description} には sudo が必要ですが、sudo が見つかりません。"
    fi
}

mlfb_confirm_privileged() {
    message="$1"
    mlfb_info "${message}"
    if [ "${MLFB_YES}" -eq 1 ]; then
        return 0
    fi
    if mlfb_prompt_yes_no "このコマンドを実行してよいですか？"; then
        return 0
    fi
    mlfb_die "必要なシステムパッケージをインストールしなかったため中止します。"
}

mlfb_linux_install_venv_support() {
    [ "${MLFB_SETUP_OS}" = "linux" ] || return 0
    python_cmd="$1"
    if "${python_cmd}" -m venv --help >/dev/null 2>&1; then
        return 0
    fi

    # shellcheck disable=SC1091
    . /etc/os-release 2>/dev/null || true
    os_ids="${ID:-} ${ID_LIKE:-}"

    if echo "${os_ids}" | grep -Eq 'debian|ubuntu'; then
        mlfb_confirm_privileged "python3-venv をインストールします（venv 作成に必要です）。実行するコマンド: apt-get update; apt-get install -y python3-venv"
        mlfb_run_privileged "apt update" apt-get update
        mlfb_run_privileged "apt install python3-venv" apt-get install -y python3-venv
    elif echo "${os_ids}" | grep -Eq 'fedora|rhel|centos'; then
        mlfb_confirm_privileged "python3 と python3-pip を確認します。実行するコマンド: dnf install -y python3 python3-pip"
        mlfb_run_privileged "dnf install python3" dnf install -y python3 python3-pip
    else
        mlfb_die "python -m venv が使えません。お使いの Linux で python3-venv 相当のパッケージをインストールしてください。"
    fi
}

mlfb_setup_venv_mode() {
    mlfb_step 3 "Python venv を作成しています" "5〜15 分"
    python_cmd="$(mlfb_install_isolated_macos_python || true)"
    if [ -z "${python_cmd}" ]; then
        python_cmd="$(mlfb_find_python || true)"
    fi
    [ -n "${python_cmd}" ] || mlfb_die "Python 3.10〜3.14 が見つかりません。venv モードには Python が必要です。"
    mlfb_linux_install_venv_support "${python_cmd}"

    mlfb_run "venv create" "${python_cmd}" -m venv "${MLFB_VENV_DIR}"
    venv_python="${MLFB_VENV_DIR}/bin/python"
    [ "${MLFB_DRY_RUN}" -eq 1 ] || [ -x "${venv_python}" ] || mlfb_die "venv の Python が見つかりません: ${venv_python}"

    mlfb_run "pip upgrade" "${venv_python}" -m pip install --upgrade pip
    mlfb_run "pip install base requirements" "${venv_python}" -m pip install -r "${MLFB_REQUIREMENTS_BASE}"
    if [ "${MLFB_SETUP_OS}" = "linux" ]; then
        mlfb_run "pip install torch CPU requirements" "${venv_python}" -m pip install --index-url https://download.pytorch.org/whl/cpu -r "${MLFB_REQUIREMENTS_TORCH}"
    else
        mlfb_run "pip install torch requirements" "${venv_python}" -m pip install -r "${MLFB_REQUIREMENTS_TORCH}"
    fi
    mlfb_run "register jupyter kernel" "${venv_python}" -m ipykernel install --sys-prefix --name "${MLFB_VENV_KERNEL_NAME}" --display-name "Python (${MLFB_VENV_KERNEL_NAME})"

    MLFB_LAST_PYTHON_KIND="venv"
    MLFB_LAST_PYTHON_RUNNER="${venv_python}"
}

mlfb_verify_current_mode() {
    selected_mode="$1"
    mlfb_step 5 "環境を検証しています" "1 分"
    download_arg=""
    if [ "${MLFB_SKIP_DATA_DOWNLOAD}" -eq 0 ]; then
        download_arg="--download-mnist"
    fi
    if [ "${MLFB_LAST_PYTHON_KIND}" = "conda" ]; then
        if [ -n "${download_arg}" ]; then
            mlfb_run "verify conda environment" "${MLFB_LAST_PYTHON_RUNNER}" run --no-capture-output --name "${MLFB_ENVIRONMENT_NAME}" python "${MLFB_VERIFY_TOOL}" --data-dir "${MLFB_DATA_DIR}" "${download_arg}"
        else
            mlfb_run "verify conda environment" "${MLFB_LAST_PYTHON_RUNNER}" run --no-capture-output --name "${MLFB_ENVIRONMENT_NAME}" python "${MLFB_VERIFY_TOOL}" --data-dir "${MLFB_DATA_DIR}"
        fi
    else
        if [ -n "${download_arg}" ]; then
            mlfb_run "verify venv environment" "${MLFB_LAST_PYTHON_RUNNER}" "${MLFB_VERIFY_TOOL}" --data-dir "${MLFB_DATA_DIR}" "${download_arg}"
        else
            mlfb_run "verify venv environment" "${MLFB_LAST_PYTHON_RUNNER}" "${MLFB_VERIFY_TOOL}" --data-dir "${MLFB_DATA_DIR}"
        fi
    fi
    mlfb_info "検証完了: ${selected_mode}"
}

mlfb_run_notebooks_current_mode() {
    [ "${MLFB_RUN_NOTEBOOKS}" -eq 1 ] || return 0
    [ -f "${MLFB_RUN_NOTEBOOKS_TOOL}" ] || mlfb_die "Notebook 実行ツールが見つかりません: ${MLFB_RUN_NOTEBOOKS_TOOL}"
    mlfb_step 6 "Notebook を実行しています" "数分〜数十分"
    if [ "${MLFB_LAST_PYTHON_KIND}" = "conda" ]; then
        mlfb_run "run notebooks with conda" "${MLFB_LAST_PYTHON_RUNNER}" run --no-capture-output --name "${MLFB_ENVIRONMENT_NAME}" python "${MLFB_RUN_NOTEBOOKS_TOOL}" --root "${MLFB_REPOSITORY_ROOT}"
    else
        mlfb_run "run notebooks with venv" "${MLFB_LAST_PYTHON_RUNNER}" "${MLFB_RUN_NOTEBOOKS_TOOL}" --root "${MLFB_REPOSITORY_ROOT}" --kernel-name "${MLFB_VENV_KERNEL_NAME}"
    fi
}

mlfb_remove_envs() {
    selected="$1"
    removed_conda=""
    mlfb_validate_mode "${selected}"
    if [ "${selected}" = "venv" ] || [ "${selected}" = "all" ]; then
        if [ -d "${MLFB_VENV_DIR}" ]; then
            mlfb_run "remove venv" rm -rf "${MLFB_VENV_DIR}"
        else
            mlfb_info "venv は見つかりません: ${MLFB_VENV_DIR}"
        fi
    fi
    if [ "${selected}" = "anaconda" ] || [ "${selected}" = "miniconda" ] || [ "${selected}" = "all" ]; then
        for conda_mode in anaconda miniconda; do
            if [ "${selected}" != "all" ] && [ "${selected}" != "${conda_mode}" ]; then
                continue
            fi
            conda="$(mlfb_find_conda "${conda_mode}" || true)"
            if [ -n "${conda}" ]; then
                if [ "${conda}" = "${removed_conda}" ]; then
                    continue
                fi
                mlfb_run "remove conda environment" "${conda}" env remove --name "${MLFB_ENVIRONMENT_NAME}" --yes
                removed_conda="${conda}"
            else
                mlfb_info "Conda は見つかりません: ${conda_mode}"
            fi
        done
    fi
    if [ "${MLFB_TEST_ISOLATE}" = "1" ] && [ -n "${MLFB_ISOLATED_HOME}" ]; then
        mlfb_run "remove isolated test home" rm -rf "${MLFB_ISOLATED_HOME}"
    fi
}

mlfb_modes_to_run() {
    if [ "${MLFB_MODE}" = "all" ]; then
        printf '%s\n' anaconda miniconda venv
    else
        printf '%s\n' "${MLFB_MODE}"
    fi
}

mlfb_save_state() {
    [ "${MLFB_DRY_RUN}" -eq 1 ] && return 0
    mkdir -p "${MLFB_STATE_DIR}"
    printf '%s\n' "$1" > "${MLFB_STATE_DIR}/last-mode.txt"
}

mlfb_setup_main() {
    mlfb_parse_setup_args "$@"
    mlfb_enable_test_isolation
    mlfb_load_config
    mlfb_preflight
    if [ "${MLFB_CHECK}" -eq 1 ]; then
        mlfb_info "診断が完了しました。--check のため変更は行っていません。"
        return 0
    fi
    mlfb_start_logging
    if [ -n "${MLFB_REMOVE_MODE}" ]; then
        mlfb_remove_envs "${MLFB_REMOVE_MODE}"
        mlfb_info "削除処理が完了しました。"
        return 0
    fi

    mlfb_choose_mode
    mlfb_step 2 "モードを決めています: ${MLFB_MODE}" "1 分"

    for selected_mode in $(mlfb_modes_to_run); do
        if [ "${selected_mode}" = "venv" ]; then
            mlfb_setup_venv_mode
        else
            mlfb_setup_conda_mode "${selected_mode}"
        fi
        mlfb_verify_current_mode "${selected_mode}"
        mlfb_run_notebooks_current_mode
    done

    mlfb_save_state "${MLFB_MODE}"
    mlfb_step 7 "完了しました" "完了"
    mlfb_info "Jupyter Notebook を起動するには:"
    mlfb_info "  bash '${MLFB_SETUP_DIR}/start-jupyter.sh' --mode ${MLFB_MODE}"
}

mlfb_parse_start_jupyter_args() {
    START_MODE=""
    START_NAVIGATOR=0
    START_CONDA="${CONDA_EXE:-}"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --mode)
                [ "$#" -ge 2 ] || mlfb_die "--mode には値が必要です。"
                START_MODE="$2"
                shift 2
                ;;
            --navigator)
                START_NAVIGATOR=1
                shift
                ;;
            --conda)
                [ "$#" -ge 2 ] || mlfb_die "--conda にはパスが必要です。"
                START_CONDA="$2"
                shift 2
                ;;
            -h|--help)
                mlfb_start_jupyter_usage
                exit 0
                ;;
            *)
                mlfb_start_jupyter_usage >&2
                mlfb_die "不明なオプションです: $1"
                ;;
        esac
    done
}

mlfb_detect_start_mode() {
    if [ -n "${START_MODE}" ]; then
        case "${START_MODE}" in
            anaconda|miniconda|venv) return 0 ;;
            all) START_MODE="anaconda"; return 0 ;;
            *) mlfb_die "mode は anaconda, miniconda, venv のいずれかです: ${START_MODE}" ;;
        esac
    fi
    if [ -f "${MLFB_STATE_DIR}/last-mode.txt" ]; then
        START_MODE="$(sed -n '1p' "${MLFB_STATE_DIR}/last-mode.txt")"
        [ "${START_MODE}" = "all" ] && START_MODE="anaconda"
    elif [ -x "${MLFB_VENV_DIR}/bin/python" ]; then
        START_MODE="venv"
    else
        START_MODE="anaconda"
    fi
}

mlfb_start_jupyter_main() {
    mlfb_parse_start_jupyter_args "$@"
    mlfb_detect_start_mode
    if [ "${START_MODE}" = "venv" ]; then
        venv_python="${MLFB_VENV_DIR}/bin/python"
        [ -x "${venv_python}" ] || mlfb_die "venv が見つかりません。先に setup を実行してください: ${MLFB_VENV_DIR}"
        mlfb_info "Jupyter Notebook を起動します: ${MLFB_REPOSITORY_ROOT}"
        exec "${venv_python}" -m notebook "${MLFB_REPOSITORY_ROOT}"
    fi

    old_conda_path="${MLFB_CONDA_PATH}"
    MLFB_CONDA_PATH="${START_CONDA}"
    conda="$(mlfb_find_conda "${START_MODE}" || true)"
    MLFB_CONDA_PATH="${old_conda_path}"
    [ -n "${conda}" ] || mlfb_die "Conda が見つかりません。先に setup を実行するか --conda を指定してください。"

    if [ "${START_NAVIGATOR}" -eq 1 ]; then
        mlfb_info "Anaconda Navigator の起動を試します。"
        exec "${conda}" run --no-capture-output --name base anaconda-navigator
    fi
    mlfb_info "Jupyter Notebook を起動します: ${MLFB_REPOSITORY_ROOT}"
    exec "${conda}" run --no-capture-output --name "${MLFB_ENVIRONMENT_NAME}" jupyter notebook "${MLFB_REPOSITORY_ROOT}"
}
