"""Execute source notebooks and save deterministic output paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def source_notebooks(root: Path) -> list[Path]:
    executed = root / "executed-notebooks"
    return sorted(
        path
        for path in root.rglob("*.ipynb")
        if executed not in path.parents and ".ipynb_checkpoints" not in path.parts
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kernel-name")
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = root / "executed-notebooks"

    for source in source_notebooks(root):
        relative = source.relative_to(root)
        output = output_root / relative.with_suffix(".executed.ipynb")
        output.parent.mkdir(parents=True, exist_ok=True)
        notebook = nbformat.read(source, as_version=4)
        processor = ExecutePreprocessor(timeout=None, shutdown_kernel="immediate")
        if args.kernel_name:
            processor.kernel_name = args.kernel_name
        processor.preprocess(
            notebook, {"metadata": {"path": str(source.parent)}}
        )
        nbformat.write(notebook, output)
        print(f"Executed: {relative}")


if __name__ == "__main__":
    main()
