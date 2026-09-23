"""Check the stable result expected from the Digits classification notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    notebook_path = root / "executed-notebooks" / "mnist" / "plot_digits_classification.executed.ipynb"
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Executed notebook not found: {notebook_path}")

    notebook = nbformat.read(notebook_path, as_version=4)
    text = "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    if "871" not in text or "899" not in text:
        raise RuntimeError("Expected Digits result (871 correct of 899) was not found.")

    html_path = root / "mnist" / "plot_digits_predition.html"
    html = html_path.read_text(encoding="utf-8")
    marker = 'id="plot-digits-model"'
    if marker not in html:
        raise RuntimeError("Embedded browser model data was not found.")
    start = html.index(">", html.index(marker)) + 1
    end = html.index("</script>", start)
    model = json.loads(html[start:end].strip())
    metrics = model.get("metrics", {})
    if metrics.get("sampleCount") != 899 or metrics.get("correctCount") != 871:
        raise RuntimeError("Embedded model metrics do not match 871/899.")

    print("Notebook and embedded model results are valid: 871/899")


if __name__ == "__main__":
    main()
