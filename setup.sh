#!/usr/bin/env bash
# Sets up and optionally executes every notebook in this repository on macOS.
#
# Official sources:
# - Miniforge installation and platform support:
#   https://github.com/conda-forge/miniforge
# - Miniforge release metadata and SHA-256 digests:
#   https://api.github.com/repos/conda-forge/miniforge/releases/tags/26.5.3-0
# - Conda environment management:
#   https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
# - nbconvert notebook execution:
#   https://nbconvert.readthedocs.io/en/latest/execute_api.html
# - nbconvert execution configuration:
#   https://nbconvert.readthedocs.io/en/latest/config_options.html

set -Eeuo pipefail

readonly ENVIRONMENT_NAME="mlfb-mnist"
readonly MINIFORGE_VERSION="26.5.3-0"

RUN_NOTEBOOKS=0
SKIP_DATA_DOWNLOAD=0
CONDA_EXECUTABLE="${CONDA_EXE:-}"
CONDA_ARGUMENT_WAS_SET=0
MINIFORGE_PREFIX="${MINIFORGE_PREFIX:-${HOME}/miniforge3}"
TEMPORARY_DIRECTORY=""

cleanup() {
    if [[ -n "$TEMPORARY_DIRECTORY" && -d "$TEMPORARY_DIRECTORY" ]]; then
        rm -rf "$TEMPORARY_DIRECTORY"
    fi
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
Usage: bash setup.sh [options]

Options:
  --run-notebooks              Execute every notebook after setup.
  --skip-data-download         Do not pre-download torchvision MNIST data.
  --conda PATH                 Use a specific conda executable.
  --miniforge-prefix PATH      Miniforge install path when conda is absent.
  -h, --help                   Show this help.
EOF
}

while (($# > 0)); do
    case "$1" in
        --run-notebooks)
            RUN_NOTEBOOKS=1
            shift
            ;;
        --skip-data-download)
            SKIP_DATA_DOWNLOAD=1
            shift
            ;;
        --conda)
            if (($# < 2)); then
                echo "--conda にはパスが必要です。" >&2
                exit 2
            fi
            CONDA_EXECUTABLE="$2"
            CONDA_ARGUMENT_WAS_SET=1
            shift 2
            ;;
        --miniforge-prefix)
            if (($# < 2)); then
                echo "--miniforge-prefix にはパスが必要です。" >&2
                exit 2
            fi
            MINIFORGE_PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "不明なオプションです: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "setup.sh は macOS 用です。Windows では setup.ps1 を実行してください。" >&2
    exit 1
fi

SCRIPT_DIRECTORY="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$SCRIPT_DIRECTORY"
readonly ENVIRONMENT_FILE="${REPOSITORY_ROOT}/environment.yml"
readonly DATA_DIRECTORY="${REPOSITORY_ROOT}/data"
readonly EXECUTED_NOTEBOOK_DIRECTORY="${REPOSITORY_ROOT}/executed-notebooks"

if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
    echo "環境定義が見つかりません: $ENVIRONMENT_FILE" >&2
    exit 1
fi

resolve_conda() {
    if [[ -n "$CONDA_EXECUTABLE" ]]; then
        if [[ -x "$CONDA_EXECUTABLE" ]]; then
            printf '%s\n' "$CONDA_EXECUTABLE"
            return 0
        fi
        if command -v "$CONDA_EXECUTABLE" >/dev/null 2>&1; then
            command -v "$CONDA_EXECUTABLE"
            return 0
        fi
        if ((CONDA_ARGUMENT_WAS_SET == 1)); then
            echo "指定された Conda 実行ファイルが見つかりません: $CONDA_EXECUTABLE" >&2
            return 2
        fi
        echo "警告: CONDA_EXE が無効なため、他の Conda を検索します: $CONDA_EXECUTABLE" >&2
    fi

    local discovered
    discovered="$(command -v conda 2>/dev/null || true)"
    if [[ -n "$discovered" ]]; then
        printf '%s\n' "$discovered"
        return 0
    fi

    local candidate
    for candidate in \
        "${MINIFORGE_PREFIX}/bin/conda" \
        "${HOME}/miniforge3/bin/conda" \
        "${HOME}/Miniforge3/bin/conda" \
        "${HOME}/miniconda3/bin/conda" \
        "${HOME}/anaconda3/bin/conda" \
        "/opt/miniforge3/bin/conda" \
        "/opt/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

install_miniforge() {
    local architecture installer_name installer_url expected_sha256
    architecture="$(uname -m)"

    case "$architecture" in
        arm64)
            installer_name="Miniforge3-${MINIFORGE_VERSION}-MacOSX-arm64.sh"
            expected_sha256="0d765919d3ccfd1f89147aa1cf8133bfc55b3a3c13f5bacdcc091c33132fddd2"
            ;;
        x86_64)
            installer_name="Miniforge3-${MINIFORGE_VERSION}-MacOSX-x86_64.sh"
            expected_sha256="0266a7bfeb12165286133145717bef0d88070f1b76710beb6a62fec4e88371a1"
            ;;
        *)
            echo "未対応の macOS アーキテクチャです: $architecture" >&2
            return 1
            ;;
    esac

    if [[ -e "$MINIFORGE_PREFIX" ]]; then
        if [[ -d "$MINIFORGE_PREFIX" && -z "$(ls -A "$MINIFORGE_PREFIX")" ]]; then
            rmdir "$MINIFORGE_PREFIX"
        else
            echo "Conda を含まない既存パスには上書きしません: $MINIFORGE_PREFIX" >&2
            return 1
        fi
    fi

    installer_url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${installer_name}"

    local installer_path actual_sha256
    TEMPORARY_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/mlfb-setup.XXXXXX")"
    installer_path="${TEMPORARY_DIRECTORY}/${installer_name}"

    echo "==> Miniforge ${MINIFORGE_VERSION} をダウンロードしています"
    curl --fail --silent --show-error --location "$installer_url" --output "$installer_path"

    actual_sha256="$(shasum -a 256 "$installer_path" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "Miniforge インストーラーの SHA-256 が公式値と一致しません。" >&2
        echo "期待値: $expected_sha256" >&2
        echo "実測値: $actual_sha256" >&2
        return 1
    fi
    echo "==> Miniforge インストーラーの SHA-256 を検証しました"

    bash "$installer_path" -b -p "$MINIFORGE_PREFIX"

    local installed_conda="${MINIFORGE_PREFIX}/bin/conda"
    if [[ ! -x "$installed_conda" ]]; then
        echo "インストール完了後に Conda が見つかりません: $installed_conda" >&2
        return 1
    fi

    rm -rf "$TEMPORARY_DIRECTORY"
    TEMPORARY_DIRECTORY=""
    RESOLVED_CONDA_EXECUTABLE="$installed_conda"
}

run_step() {
    local description="$1"
    shift
    echo "==> $description"
    "$@"
}

set +e
discovered_conda="$(resolve_conda)"
resolve_status=$?
set -e
if ((resolve_status == 0)); then
    RESOLVED_CONDA_EXECUTABLE="$discovered_conda"
elif ((resolve_status == 2)); then
    exit 1
else
    install_miniforge
fi
readonly RESOLVED_CONDA_EXECUTABLE
echo "==> Conda: $RESOLVED_CONDA_EXECUTABLE"

case "$(uname -m)" in
    arm64)
        expected_conda_platform="osx-arm64"
        ;;
    x86_64)
        expected_conda_platform="osx-64"
        ;;
    *)
        echo "未対応の macOS アーキテクチャです: $(uname -m)" >&2
        exit 1
        ;;
esac
conda_info_json="$("$RESOLVED_CONDA_EXECUTABLE" info --json)"
conda_platform="$(printf '%s\n' "$conda_info_json" | sed -n 's/^[[:space:]]*"platform":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
conda_version="$("$RESOLVED_CONDA_EXECUTABLE" --version)"
if [[ "$conda_platform" != "$expected_conda_platform" ]]; then
    echo "ネイティブ macOS 用 Conda が必要です。期待値: ${expected_conda_platform} / 検出値: ${conda_platform:-不明}" >&2
    exit 1
fi
echo "==> ${conda_version} platform=${conda_platform}"

if "$RESOLVED_CONDA_EXECUTABLE" run --name "$ENVIRONMENT_NAME" python -c 'pass' >/dev/null 2>&1; then
    run_step "Conda 環境 '${ENVIRONMENT_NAME}' を更新しています" \
        env CONDA_CHANNEL_PRIORITY=flexible \
        "$RESOLVED_CONDA_EXECUTABLE" env update \
        --name "$ENVIRONMENT_NAME" \
        --file "$ENVIRONMENT_FILE" \
        --prune
else
    run_step "Conda 環境 '${ENVIRONMENT_NAME}' を作成しています" \
        env CONDA_CHANNEL_PRIORITY=flexible \
        "$RESOLVED_CONDA_EXECUTABLE" env create \
        --name "$ENVIRONMENT_NAME" \
        --file "$ENVIRONMENT_FILE" \
        --yes
fi

verification_code='import sys, matplotlib, sklearn, torch, torchvision; from sklearn.datasets import load_digits; digits = load_digits(); assert digits.data.shape == (1797, 64); mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available(); print(f"Python={sys.version.split()[0]} matplotlib={matplotlib.__version__} scikit-learn={sklearn.__version__} torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.cuda.is_available()} mps={mps}")'
run_step "Python 依存関係と scikit-learn Digits データを検証しています" \
    "$RESOLVED_CONDA_EXECUTABLE" run --no-capture-output \
    --name "$ENVIRONMENT_NAME" \
    python -c "$verification_code"

if ((SKIP_DATA_DOWNLOAD == 0)); then
    mkdir -p "$DATA_DIRECTORY"
    download_code='import sys; from torchvision import datasets; root = sys.argv[1]; train = datasets.MNIST(root, train=True, download=True); test = datasets.MNIST(root, train=False, download=True); assert len(train) > 0 and len(test) > 0; print(f"MNIST train={len(train)} test={len(test)} root={root}")'
    run_step "torchvision の MNIST データを準備しています" \
        "$RESOLVED_CONDA_EXECUTABLE" run --no-capture-output \
        --name "$ENVIRONMENT_NAME" \
        python -c "$download_code" "$DATA_DIRECTORY"
fi

if ((RUN_NOTEBOOKS == 1)); then
    notebooks=()
    while IFS= read -r notebook; do
        notebooks+=("$notebook")
    done < <(
        find "$REPOSITORY_ROOT" -type f -name '*.ipynb' \
            ! -path "${EXECUTED_NOTEBOOK_DIRECTORY}/*" \
            ! -path '*/.ipynb_checkpoints/*' \
            -print | LC_ALL=C sort
    )
    if ((${#notebooks[@]} == 0)); then
        echo "実行対象の Notebook が見つかりません。" >&2
        exit 1
    fi

    for notebook in "${notebooks[@]}"; do
        relative_path="${notebook#"${REPOSITORY_ROOT}/"}"
        relative_directory="$(dirname "$relative_path")"
        notebook_directory="$(dirname "$notebook")"
        notebook_name="$(basename "$notebook")"
        output_directory="${EXECUTED_NOTEBOOK_DIRECTORY}/${relative_directory}"
        output_name="${notebook_name%.ipynb}.executed.ipynb"
        mkdir -p "$output_directory"

        pushd "$notebook_directory" >/dev/null
        run_step "${relative_path} の全セルを実行しています" \
            "$RESOLVED_CONDA_EXECUTABLE" run --no-capture-output \
            --name "$ENVIRONMENT_NAME" \
            jupyter nbconvert \
            --to notebook \
            --execute "$notebook_name" \
            --output "$output_name" \
            --output-dir "$output_directory" \
            --ExecutePreprocessor.timeout=-1 \
            --ExecutePreprocessor.kernel_name=python3 \
            --ExecutePreprocessor.shutdown_kernel=immediate
        popd >/dev/null
    done

    echo "==> 実行済みノートブック: $EXECUTED_NOTEBOOK_DIRECTORY"
fi

echo
echo "セットアップが完了しました。"
echo "Jupyter Notebook を起動する場合:"
printf "  '%s' run --no-capture-output --name %s jupyter notebook '%s'\n" \
    "$RESOLVED_CONDA_EXECUTABLE" "$ENVIRONMENT_NAME" "$REPOSITORY_ROOT"
if ((RUN_NOTEBOOKS == 0)); then
    echo "全ノートブックを自動実行する場合:"
    printf "  bash '%s' --run-notebooks\n" "${REPOSITORY_ROOT}/setup.sh"
fi
