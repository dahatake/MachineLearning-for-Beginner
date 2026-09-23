"""Verify the packages and data shape required by this repository's notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-mnist", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    import matplotlib  # noqa: F401
    import nbconvert  # noqa: F401
    import notebook  # noqa: F401
    import torch
    import torchvision
    from sklearn.datasets import load_digits

    digits = load_digits()
    if digits.data.shape != (1797, 64):
        raise RuntimeError(f"Unexpected Digits shape: {digits.data.shape}")

    if args.download_mnist:
        torchvision.datasets.MNIST(
            root=str(args.data_dir),
            train=True,
            download=True,
        )
        torchvision.datasets.MNIST(
            root=str(args.data_dir),
            train=False,
            download=True,
        )

    print(f"Digits shape: {digits.data.shape}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(
        "MPS available: "
        f"{bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}"
    )


if __name__ == "__main__":
    main()
