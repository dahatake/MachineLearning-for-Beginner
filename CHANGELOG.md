# Changelog

このプロジェクトの注目すべき変更は、このファイルに記録します。

形式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に基づきます。

## [Unreleased]

scikit-learnのDigitsデータセットで学習したSVCを、PCまたはMacのブラウザーから手書き入力で試せるようにしました。また、Windows・macOS・Linuxのいずれでも、1つのスクリプトを実行するだけでNotebook実行環境（Anaconda / Miniconda / venv）を構築できるようにしました。

### Added

- マウス、タッチ、ペンの描画を8×8の入力へ変換し、ブラウザー内でRBF SVC推論を行う単一HTMLアプリを追加しました。
- 予測数字、one-vs-one投票、推論時間、8×8入力、Notebookのテスト評価を表示する統計パネルを追加しました。
- モデルの欠落・不正を起動時に検出し、Notebookの全コード実行手順と再読み込み操作を案内する画面を追加しました。
- ブラウザー推論アプリの要求、対象外、受入条件を記録する要求定義書を追加しました。
- `setup/setup-windows.cmd` と `setup/setup-windows.ps1`（PowerShell 5.1 / 7 両対応）、`setup/setup-mac.sh`、`setup/setup-linux.sh`、共通処理 `setup/lib/common.sh` を追加し、Anaconda（既定）/ Miniconda / venv の3方式でセットアップできるようにしました。Anaconda / Miniconda は Python 本体も含めて導入します。venv は既存の Python（3.10〜3.14）があればそれを使い、なければ python.org の Python 3.12.10 をユーザー単位（管理者権限不要）で自動的に導入します。
- `setup/start-jupyter-windows.cmd` と `setup/start-jupyter.sh` を追加し、リポジトリのフォルダーを開いた状態で Jupyter（Anaconda では Navigator も）を起動できるようにしました。
- `.github/workflows/setup-e2e.yml` を追加し、Windows・macOS（Apple Silicon / Intel）・Linux（x64 / arm64）の実機・仮想環境ランナーで3方式それぞれの初回実行・冪等性（2回目実行）・Notebook実行結果の検証・クリーンアップを継続的に検証するとともに、Ubuntu 22.04〜26.04・Debian 12/13・Fedora のコンテナでも3方式の初回実行・冪等性・クリーンアップを検証する CI を追加しました。

### Changed

- `plot_digits_classification.ipynb` の最後のセルで、学習済みSVCと評価指標を同じモデルから再現可能な内容でブラウザー推論アプリへ埋め込むようにしました。
- ルート直下にあった単一の `setup.ps1` / `setup.sh` を、OS別の `setup/setup-windows.ps1` / `setup/setup-mac.sh` / `setup/setup-linux.sh` に置き換え、`environment.yml` を `setup/envs/environment.yml` へ移動しました。`README.md` と `SETUP.md` を、ブラウザー推論の利用方法・モデル更新・復旧手順・検証範囲、および新しいセットアップ手順（3方式の選び方、Anaconda利用規約への同意手順）に合わせて更新しました。

### Fixed

- Conda の `defaults` チャンネルに対する利用規約（ToS）未同意が原因で、macOS/Linux 共通のセットアップ処理（`setup/lib/common.sh`）における環境の存在確認・作成・更新・削除の各操作が非対話環境で `CondaToSNonInteractiveError` により失敗していた問題を修正しました。`conda-forge` / `pytorch` チャンネルのみを使う一時 `condarc` をこれらの操作に一貫して適用し、`--dry-run` 時のプレビュー表示は従来どおり維持しました。
- `setup/setup-windows.ps1` の venv モードで、既存の Python（3.10〜3.14）が見つからない場合にエラーで停止していた問題を修正しました。python.org の Python 3.12.10 インストーラーをダウンロード・SHA-256照合したうえで、`%LOCALAPPDATA%\Programs\Python\Python312` へユーザー単位（管理者権限不要）で自動的に導入するようにしました。
