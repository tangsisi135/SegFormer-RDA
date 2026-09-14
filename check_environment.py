"""Check the Python packages and CUDA visibility required by this project."""

import importlib
import platform
import sys


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    required = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "cv2": "OpenCV",
        "matplotlib": "Matplotlib",
        "scipy": "SciPy",
        "tqdm": "tqdm",
        "tensorboard": "TensorBoard",
    }
    failed = False
    for module_name, display_name in required.items():
        try:
            module = importlib.import_module(module_name)
            print(f"{display_name}: OK ({getattr(module, '__version__', 'installed')})")
        except ImportError as exc:
            failed = True
            print(f"{display_name}: MISSING ({exc})")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    if failed:
        raise SystemExit("One or more required packages are missing.")
    print("Environment check passed.")


if __name__ == "__main__":
    main()

